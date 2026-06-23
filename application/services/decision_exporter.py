"""Phase 7.3 — build + export :class:`~domain.models.decision_record.DecisionRecord`s (no database).

Three pieces:

* :func:`gather_provenance` / :func:`build_decision_record` — stamp model/config provenance and map a
  persisted :class:`~domain.models.bid_draft.BidDraft` (recommendation snapshot + lifecycle audit
  trail) into an auditable :class:`DecisionRecord`, optionally carrying a **redacted**
  :class:`~application.ingestion.real_broker_schema.BrokerReference`.
* :class:`DecisionLog` — the inbound-facing service: reads the live bid drafts (reusing the bid
  approval service's lazy-expiry + status filter), stamps one provenance per snapshot, and returns
  records or an API payload. Records are built **on demand** — nothing is persisted to a DB.
* :class:`DecisionExporter` — pure stdlib writers that turn records (domain objects **or** their JSON
  dicts, so the CLI can export what it fetched) into ``decisions.jsonl`` / ``decisions.csv`` / an audit
  **bundle** folder (jsonl + csv + ``manifest.json``) *outside the process*.

No new dependencies, no Postgres: JSON/CSV + a folder is the whole durability story.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from application.bid_approval_service import BidApprovalService
from application.ingestion.real_broker_schema import BrokerReference
from domain.enums.bid_approval_status import BidApprovalStatus
from domain.models.bid_draft import BidDraft
from domain.models.decision_record import (
    CSV_COLUMNS,
    DECISION_RECORD_SCHEMA_VERSION,
    WARNING_NEGATIVE_EXPECTED_VALUE,
    DecisionRecord,
    Provenance,
    flatten_decision,
)
from ports.clock import ClockPort

ROOT = Path(__file__).resolve().parents[2]

# Mirrors ``ml.workflows.teacher_trace_generator.SOURCE_POLICY_VERSION``. Duplicated as a constant so
# the export path stays decoupled from the ML package (Phase 7 is source-engine hardening only).
SOURCE_POLICY_VERSION = "phase-5.5-full-risk-aware"


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
def _git(args: List[str]) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=str(ROOT), capture_output=True, text=True, timeout=5
        )
        return (out.stdout or "").strip() or "unknown"
    except Exception:  # pragma: no cover - provenance is best-effort
        return "unknown"


def git_commit() -> str:
    return _git(["rev-parse", "HEAD"])


def git_describe() -> str:
    return _git(["describe", "--tags", "--always", "--dirty"])


def config_hash(payload: Mapping[str, Any]) -> str:
    """Stable short hash of the decision-relevant config (sorted-key JSON ``sha256`` truncated)."""
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def gather_provenance(
    *,
    source_policy_version: str,
    model_artifact_ids: Mapping[str, str],
    feature_manifest_hash: Optional[str],
    config_payload: Mapping[str, Any],
    now: datetime,
    git_commit_value: Optional[str] = None,
    git_describe_value: Optional[str] = None,
    schema_version: str = DECISION_RECORD_SCHEMA_VERSION,
) -> Provenance:
    """Assemble a :class:`Provenance`. ``git_*_value`` let a caller inject cached git output (so a
    hot path does not shell out on every call) or pin values in tests."""
    return Provenance(
        source_policy_version=source_policy_version,
        git_commit=git_commit_value if git_commit_value is not None else git_commit(),
        git_describe=git_describe_value if git_describe_value is not None else git_describe(),
        config_hash=config_hash(config_payload),
        model_artifact_ids=dict(model_artifact_ids),
        feature_manifest_hash=feature_manifest_hash,
        schema_version=schema_version,
        generated_at=now,
    )


# --------------------------------------------------------------------------- #
# BidDraft -> DecisionRecord
# --------------------------------------------------------------------------- #
def _derive_warnings(draft: BidDraft) -> List[str]:
    """The minimal, honest warning set derivable from a draft's persisted snapshot.

    The bid draft does not store the full workflow warning list, so the only signal we can derive
    faithfully is a negative expected value. Callers with richer context pass ``warnings`` explicitly.
    """
    warnings: List[str] = []
    if draft.expected_value is not None and draft.expected_value < 0:
        warnings.append(WARNING_NEGATIVE_EXPECTED_VALUE)
    return warnings


def build_decision_record(
    draft: BidDraft,
    *,
    provenance: Provenance,
    broker_reference: Optional[BrokerReference] = None,
    warnings: Optional[Iterable[str]] = None,
) -> DecisionRecord:
    """Map a :class:`BidDraft` (+ provenance, + optional redacted broker) into a :class:`DecisionRecord`.

    The broker reference is serialized through :meth:`BrokerReference.as_dict` — already redacted at the
    7.1 chokepoint, so no raw contact PII can enter the record.
    """
    resolved = list(warnings) if warnings is not None else _derive_warnings(draft)
    broker = broker_reference.as_dict() if broker_reference is not None else None
    return DecisionRecord(
        decision_id=draft.bid_id,
        load_id=draft.load_id,
        truck_id=draft.truck_id,
        status=draft.status.value,
        recommended_amount=draft.recommended_amount,
        recommended_rate_per_mile=draft.recommended_rate_per_mile,
        current_amount=draft.current_amount,
        delta_from_recommended=draft.delta_from_recommended,
        delta_percent=draft.delta_percent,
        rationale=draft.rationale,
        winnability_available=draft.winnability_available,
        win_probability=draft.win_probability,
        expected_value=draft.expected_value,
        ev_recommended_label=draft.ev_recommended_label,
        ev_recommended_bid=draft.ev_recommended_bid,
        warnings=resolved,
        broker_reference=broker,
        submission_ref=draft.submission_ref,
        created_at=draft.created_at,
        updated_at=draft.updated_at,
        expires_at=draft.expires_at,
        audit=list(draft.audit),
        provenance=provenance,
    )


# --------------------------------------------------------------------------- #
# DecisionLog (inbound service)
# --------------------------------------------------------------------------- #
class DecisionLog:
    """Builds decision records on demand from the live bid drafts, stamping shared provenance.

    Git output + config hash inputs are captured once at construction (process-stable); only
    ``generated_at`` is re-stamped from the clock on each snapshot.
    """

    def __init__(
        self,
        bid_approval_service: BidApprovalService,
        clock: ClockPort,
        *,
        source_policy_version: str,
        model_artifact_ids: Mapping[str, str],
        feature_manifest_hash: Optional[str],
        config_payload: Mapping[str, Any],
    ) -> None:
        self._svc = bid_approval_service
        self._clock = clock
        self._source_policy_version = source_policy_version
        self._model_artifact_ids = dict(model_artifact_ids)
        self._feature_manifest_hash = feature_manifest_hash
        self._config_payload = dict(config_payload)
        self._git_commit = git_commit()
        self._git_describe = git_describe()

    def _provenance(self) -> Provenance:
        return gather_provenance(
            source_policy_version=self._source_policy_version,
            model_artifact_ids=self._model_artifact_ids,
            feature_manifest_hash=self._feature_manifest_hash,
            config_payload=self._config_payload,
            now=self._clock.now(),
            git_commit_value=self._git_commit,
            git_describe_value=self._git_describe,
        )

    def snapshot(
        self, status: Optional[BidApprovalStatus] = None
    ) -> Tuple[Provenance, List[DecisionRecord]]:
        provenance = self._provenance()
        drafts = self._svc.list_drafts(status)
        return provenance, [build_decision_record(d, provenance=provenance) for d in drafts]

    def records(self, status: Optional[BidApprovalStatus] = None) -> List[DecisionRecord]:
        return self.snapshot(status)[1]

    def payload(self, status: Optional[BidApprovalStatus] = None) -> Dict[str, Any]:
        """The read-only API shape: shared provenance + every decision's nested dict."""
        provenance, records = self.snapshot(status)
        return {
            "provenance": provenance.to_dict(),
            "decisions": [r.to_dict() for r in records],
        }


# --------------------------------------------------------------------------- #
# DecisionExporter (pure file writers)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExportBundleReport:
    """What an audit-bundle export wrote: the folder, its files, and per-status counts."""

    out_dir: str
    jsonl_path: str
    csv_path: str
    manifest_path: str
    decision_count: int
    status_counts: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "out_dir": self.out_dir,
            "jsonl_path": self.jsonl_path,
            "csv_path": self.csv_path,
            "manifest_path": self.manifest_path,
            "decision_count": self.decision_count,
            "status_counts": dict(self.status_counts),
        }


def _as_record_dict(record: Any) -> Dict[str, Any]:
    """Accept a :class:`DecisionRecord` or its already-serialized JSON dict (so the CLI can export
    records it fetched from the API without rebuilding domain objects)."""
    if isinstance(record, DecisionRecord):
        return record.to_dict()
    return dict(record)


class DecisionExporter:
    """Stdlib-only writers: JSONL (full nested) + CSV (flat) + an audit bundle folder."""

    @staticmethod
    def write_jsonl(records: Iterable[Any], path: Path | str) -> int:
        rows = [_as_record_dict(r) for r in records]
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="\n") as fh:
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True, default=str))
                fh.write("\n")
        return len(rows)

    @staticmethod
    def write_csv(records: Iterable[Any], path: Path | str) -> int:
        rows = [flatten_decision(_as_record_dict(r)) for r in records]
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return len(rows)

    @staticmethod
    def write_bundle(
        records: Iterable[Any],
        out_dir: Path | str,
        *,
        provenance: Any = None,
    ) -> ExportBundleReport:
        """Write ``decisions.jsonl`` + ``decisions.csv`` + ``manifest.json`` into ``out_dir`` (created
        if needed). ``provenance`` may be a :class:`Provenance` or its dict (e.g. fetched as JSON)."""
        rows = [_as_record_dict(r) for r in records]
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        jsonl_path = out / "decisions.jsonl"
        csv_path = out / "decisions.csv"
        manifest_path = out / "manifest.json"

        DecisionExporter.write_jsonl(rows, jsonl_path)
        DecisionExporter.write_csv(rows, csv_path)

        status_counts: Dict[str, int] = {}
        for row in rows:
            key = str(row.get("status"))
            status_counts[key] = status_counts.get(key, 0) + 1

        if isinstance(provenance, Provenance):
            prov_dict = provenance.to_dict()
        else:
            prov_dict = dict(provenance) if provenance is not None else {}

        manifest = {
            "schema_version": DECISION_RECORD_SCHEMA_VERSION,
            "exported_at": prov_dict.get("generated_at"),
            "decision_count": len(rows),
            "status_counts": status_counts,
            "provenance": prov_dict,
            "files": {"jsonl": "decisions.jsonl", "csv": "decisions.csv"},
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )

        return ExportBundleReport(
            out_dir=str(out),
            jsonl_path=str(jsonl_path),
            csv_path=str(csv_path),
            manifest_path=str(manifest_path),
            decision_count=len(rows),
            status_counts=status_counts,
        )
