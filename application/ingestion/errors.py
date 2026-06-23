"""Typed ingestion errors for the Phase 7.1 real-world data contracts.

The import contract is **reject-row-not-batch**: one malformed row produces a structured
:class:`RowValidationError` (row index, best-effort identifier, and the per-field problems)
and is dropped, while the remaining valid rows still map into domain objects. Callers get an
:class:`~application.ingestion.import_contract.IngestResult` carrying both the accepted objects
and every error - never a raw ``pydantic.ValidationError`` and never a half-built object.

``SchemaContractError`` is the harder, structural failure (the file is not the shape the
contract expects at all - e.g. JSON that is not a list of objects, or a CSV with no header).
It aborts the whole read because there are no rows to salvage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

# Stable error codes (machine-readable; surfaced in reports / API problem details).
CODE_MISSING = "missing"          # a required field was absent or empty
CODE_TYPE = "type"                # a value could not be coerced to the expected type
CODE_VALUE = "value"              # a value parsed but failed a domain rule (range/enum/cross-field)
CODE_STRUCTURE = "structure"      # the source is not the expected container shape


class IngestionError(Exception):
    """Base class for every ingestion failure."""


class SchemaContractError(IngestionError):
    """The source is structurally unusable (wrong container shape, no header, unreadable).

    Unlike a row error this is not recoverable per-row, so the read aborts.
    """


@dataclass(frozen=True)
class FieldError:
    """One field-level problem within a row (which field, a stable code, a human message)."""

    field: str
    code: str
    message: str

    def as_dict(self) -> dict:
        return {"field": self.field, "code": self.code, "message": self.message}


@dataclass(frozen=True)
class RowValidationError(IngestionError):
    """All field problems for a single rejected row, with locating context.

    ``identifier`` is a best-effort, redaction-safe handle (the row's ``load_id`` /
    ``reference`` / ``broker_id`` when parseable) so a report can point a human at the
    offending record without echoing PII.
    """

    row_index: int
    identifier: Optional[str]
    errors: List[FieldError] = field(default_factory=list)

    def __str__(self) -> str:  # pragma: no cover - convenience
        who = self.identifier if self.identifier is not None else f"row {self.row_index}"
        parts = "; ".join(f"{e.field}: {e.message}" for e in self.errors)
        return f"{who}: {parts}"

    def as_dict(self) -> dict:
        return {
            "row_index": self.row_index,
            "identifier": self.identifier,
            "errors": [e.as_dict() for e in self.errors],
        }
