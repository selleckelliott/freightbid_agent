"""Phase 7.1 import contract: read external **loads** and **brokers** from CSV or JSON and
validate them row-by-row into domain objects.

Semantics are **reject-row-not-batch**: every row is validated independently, a good row maps to
a :class:`~domain.models.load.Load` / :class:`~application.ingestion.real_broker_schema.BrokerReference`,
and a bad row becomes a structured :class:`~application.ingestion.errors.RowValidationError` in the
report - one malformed row never sinks the whole feed. A *structural* problem (not a list of
objects, an empty/headerless CSV) raises :class:`~application.ingestion.errors.SchemaContractError`,
because there are no rows to salvage.

The same ``validate_*`` entry points accept any iterable of mappings, so the Phase 7.2 sandbox /
replay load-board adapters can feed raw dicts straight in without writing a file first.
"""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from application.ingestion.errors import CODE_STRUCTURE, RowValidationError, SchemaContractError
from application.ingestion.real_broker_schema import BrokerReference, broker_reference_from_mapping
from application.ingestion.real_load_schema import domain_load_from_mapping
from domain.models.load import Load

_LOAD_ROOT_KEYS = ("loads", "data", "items", "records", "results")
_BROKER_ROOT_KEYS = ("brokers", "data", "items", "records", "results")


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
@dataclass
class IngestResult:
    """Accepted domain loads + per-row errors from one import."""

    loads: List[Load] = field(default_factory=list)
    errors: List[RowValidationError] = field(default_factory=list)
    total_rows: int = 0

    @property
    def accepted(self) -> int:
        return len(self.loads)

    @property
    def rejected(self) -> int:
        return len(self.errors)

    def error_report(self) -> List[dict]:
        return [e.as_dict() for e in self.errors]

    def summary(self) -> dict:
        return {
            "total_rows": self.total_rows,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "errors": self.error_report(),
        }


@dataclass
class BrokerResult:
    """Accepted redacted broker references + per-row errors from one import."""

    brokers: List[BrokerReference] = field(default_factory=list)
    errors: List[RowValidationError] = field(default_factory=list)
    total_rows: int = 0

    @property
    def accepted(self) -> int:
        return len(self.brokers)

    @property
    def rejected(self) -> int:
        return len(self.errors)

    def by_id(self) -> Dict[str, BrokerReference]:
        return {b.broker_id: b for b in self.brokers}

    def error_report(self) -> List[dict]:
        return [e.as_dict() for e in self.errors]

    def summary(self) -> dict:
        return {
            "total_rows": self.total_rows,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "errors": self.error_report(),
        }


@dataclass
class FeedImport:
    """A combined loads (+ optional brokers) import, for the live pull / demo path."""

    loads: IngestResult
    brokers: Optional[BrokerResult] = None


# --------------------------------------------------------------------------- #
# Row extraction (CSV / JSON -> list of mappings)
# --------------------------------------------------------------------------- #
def _rows_from_json(text: str, root_keys: Sequence[str]) -> List[dict]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SchemaContractError(f"invalid JSON: {exc}") from exc
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = None
        for key in root_keys:
            if isinstance(data.get(key), list):
                rows = data[key]
                break
        if rows is None:
            list_values = [v for v in data.values() if isinstance(v, list)]
            if len(list_values) == 1:
                rows = list_values[0]
        if rows is None:
            raise SchemaContractError(
                f"JSON object has no array under any of {list(root_keys)}"
            )
    else:
        raise SchemaContractError("JSON must be a list of objects or an object wrapping one")
    if not all(isinstance(r, dict) for r in rows):
        raise SchemaContractError("every JSON row must be an object")
    return list(rows)


def _rows_from_csv(text: str) -> List[dict]:
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None or all((h or "").strip() == "" for h in reader.fieldnames):
        raise SchemaContractError("CSV has no header row")
    rows: List[dict] = []
    for raw in reader:
        rows.append({k: v for k, v in raw.items() if k is not None})
    return rows


def _infer_format(path: Path, fmt: Optional[str]) -> str:
    if fmt:
        return fmt.lower().lstrip(".")
    suffix = path.suffix.lower().lstrip(".")
    if suffix in ("json", "csv"):
        return suffix
    raise SchemaContractError(f"cannot infer format from '{path.name}'; pass fmt='csv'|'json'")


def read_rows_from_text(text: str, fmt: str, *, root_keys: Sequence[str] = _LOAD_ROOT_KEYS) -> List[dict]:
    fmt = fmt.lower().lstrip(".")
    if fmt == "json":
        return _rows_from_json(text, root_keys)
    if fmt == "csv":
        return _rows_from_csv(text)
    raise SchemaContractError(f"unsupported format '{fmt}'")


def read_rows_from_file(path: str | Path, fmt: Optional[str] = None,
                        *, root_keys: Sequence[str] = _LOAD_ROOT_KEYS) -> List[dict]:
    path = Path(path)
    fmt = _infer_format(path, fmt)
    text = path.read_text(encoding="utf-8")
    return read_rows_from_text(text, fmt, root_keys=root_keys)


# --------------------------------------------------------------------------- #
# Row validation (mappings -> domain objects + structured errors)
# --------------------------------------------------------------------------- #
def validate_loads(rows: Iterable[Mapping[str, Any]]) -> IngestResult:
    result = IngestResult()
    for index, row in enumerate(rows):
        result.total_rows += 1
        try:
            result.loads.append(domain_load_from_mapping(row, index))
        except RowValidationError as exc:
            result.errors.append(exc)
    return result


def validate_brokers(rows: Iterable[Mapping[str, Any]]) -> BrokerResult:
    result = BrokerResult()
    for index, row in enumerate(rows):
        result.total_rows += 1
        try:
            result.brokers.append(broker_reference_from_mapping(row, index))
        except RowValidationError as exc:
            result.errors.append(exc)
    return result


# --------------------------------------------------------------------------- #
# File entry points
# --------------------------------------------------------------------------- #
def import_loads(path: str | Path, fmt: Optional[str] = None) -> IngestResult:
    return validate_loads(read_rows_from_file(path, fmt, root_keys=_LOAD_ROOT_KEYS))


def import_brokers(path: str | Path, fmt: Optional[str] = None) -> BrokerResult:
    return validate_brokers(read_rows_from_file(path, fmt, root_keys=_BROKER_ROOT_KEYS))


def import_feed(loads_path: str | Path, brokers_path: Optional[str | Path] = None,
                *, loads_fmt: Optional[str] = None, brokers_fmt: Optional[str] = None) -> FeedImport:
    """Import a loads file plus an optional brokers file (the live pull / demo convenience)."""
    loads = import_loads(loads_path, loads_fmt)
    brokers = import_brokers(brokers_path, brokers_fmt) if brokers_path is not None else None
    return FeedImport(loads=loads, brokers=brokers)
