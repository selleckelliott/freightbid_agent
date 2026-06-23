"""Phase 7.3 — durable, exportable **decision record** + provenance (no database).

A :class:`DecisionRecord` is the auditable snapshot of one human-in-the-loop bid decision: the
recommendation the source engine produced, any **warnings** attached to it, the bid draft's full
lifecycle **audit trail**, an optional **redacted** broker reference (Phase 7.1), and the
**model/config provenance** under which it was produced. It is a pure value object — constructing or
serializing one performs no IO. The
:class:`~application.services.decision_exporter.DecisionExporter` turns a list of records into JSONL /
CSV / an audit *bundle* **outside the process**, so decisions stay reviewable without standing up a
database.

Two serializations, one source of truth:

* :meth:`DecisionRecord.to_dict` — the full, nested form (recommendation snapshot + every audit event
  + provenance) written one-object-per-line to ``decisions.jsonl``.
* :meth:`DecisionRecord.to_row` — a flat, scalar-only row (the audit trail collapsed to a count +
  last action/actor) for ``decisions.csv`` / spreadsheet review. The column order is
  :data:`CSV_COLUMNS`. It is derived from :meth:`to_dict` via the pure :func:`flatten_decision`, so the
  CLI can flatten records it fetched as JSON without reconstructing domain objects.

The broker reference, when present, is the already-redacted :class:`~application.ingestion.
real_broker_schema.BrokerReference` mapping from the 7.1 redaction chokepoint — a record **never**
carries raw contact PII. ``broker_reference`` is typed as a plain mapping here so the domain layer
stays decoupled from the ingestion package.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional

from domain.models.bid_draft import BidAuditEvent

DECISION_RECORD_SCHEMA_VERSION = "1.0.0"

# Decision-record warning vocabulary (source-engine only — no ML coupling). Kept deliberately small;
# callers may pass their own warning list to :func:`build_decision_record`.
WARNING_NEGATIVE_EXPECTED_VALUE = "negative_expected_value"


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value is not None else None


def _audit_event_to_dict(event: BidAuditEvent) -> Dict[str, Any]:
    return {
        "at": _iso(event.at),
        "action": event.action,
        "actor_id": event.actor_id,
        "from_status": event.from_status.value if event.from_status is not None else None,
        "to_status": event.to_status.value,
        "note": event.note,
        "amount_before": event.amount_before,
        "amount_after": event.amount_after,
    }


@dataclass(frozen=True)
class Provenance:
    """Model/config provenance stamped on a batch of exported decisions.

    ``generated_at`` is the snapshot time (injected clock); everything else is stable for the running
    process. ``model_artifact_ids`` maps a logical model name (``winnability`` / ``payment_risk`` /
    ``compiled_dispatcher``) to an artifact identifier, and only includes models that are actually
    wired — an empty dict means the source engine ran on rules alone.
    """

    source_policy_version: str
    git_commit: str
    git_describe: str
    config_hash: str
    model_artifact_ids: Dict[str, str]
    feature_manifest_hash: Optional[str]
    schema_version: str
    generated_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_policy_version": self.source_policy_version,
            "git_commit": self.git_commit,
            "git_describe": self.git_describe,
            "config_hash": self.config_hash,
            "model_artifact_ids": dict(self.model_artifact_ids),
            "feature_manifest_hash": self.feature_manifest_hash,
            "schema_version": self.schema_version,
            "generated_at": _iso(self.generated_at),
        }


@dataclass(frozen=True)
class DecisionRecord:
    """An auditable, exportable snapshot of one bid decision + its provenance."""

    decision_id: int
    load_id: int
    truck_id: int
    status: str
    # -- recommendation snapshot ----------------------------------------------
    recommended_amount: float
    recommended_rate_per_mile: float
    current_amount: float
    delta_from_recommended: float
    delta_percent: float
    rationale: str
    winnability_available: Optional[bool]
    win_probability: Optional[float]
    expected_value: Optional[float]
    ev_recommended_label: Optional[str]
    ev_recommended_bid: Optional[float]
    # -- attached signals -----------------------------------------------------
    warnings: List[str]
    broker_reference: Optional[Dict[str, Any]]  # already redacted (Phase 7.1) — never raw PII
    submission_ref: Optional[str]
    # -- lifecycle ------------------------------------------------------------
    created_at: datetime
    updated_at: datetime
    expires_at: datetime
    audit: List[BidAuditEvent]
    provenance: Provenance

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.provenance.schema_version,
            "decision_id": self.decision_id,
            "load_id": self.load_id,
            "truck_id": self.truck_id,
            "status": self.status,
            "recommendation": {
                "recommended_amount": self.recommended_amount,
                "recommended_rate_per_mile": self.recommended_rate_per_mile,
                "current_amount": self.current_amount,
                "delta_from_recommended": self.delta_from_recommended,
                "delta_percent": self.delta_percent,
                "rationale": self.rationale,
                "winnability_available": self.winnability_available,
                "win_probability": self.win_probability,
                "expected_value": self.expected_value,
                "ev_recommended_label": self.ev_recommended_label,
                "ev_recommended_bid": self.ev_recommended_bid,
            },
            "warnings": list(self.warnings),
            "broker_reference": (
                dict(self.broker_reference) if self.broker_reference is not None else None
            ),
            "submission_ref": self.submission_ref,
            "created_at": _iso(self.created_at),
            "updated_at": _iso(self.updated_at),
            "expires_at": _iso(self.expires_at),
            "audit": [_audit_event_to_dict(e) for e in self.audit],
            "provenance": self.provenance.to_dict(),
        }

    def to_row(self) -> Dict[str, Any]:
        """Flat, scalar-only CSV row (keys == :data:`CSV_COLUMNS`)."""
        return flatten_decision(self.to_dict())


# Fixed CSV column order for ``decisions.csv`` / spreadsheet review.
CSV_COLUMNS = (
    "decision_id",
    "load_id",
    "truck_id",
    "status",
    "recommended_amount",
    "current_amount",
    "delta_from_recommended",
    "delta_percent",
    "recommended_rate_per_mile",
    "winnability_available",
    "win_probability",
    "expected_value",
    "ev_recommended_label",
    "ev_recommended_bid",
    "warnings",
    "broker_id",
    "submission_ref",
    "created_at",
    "updated_at",
    "expires_at",
    "audit_event_count",
    "last_action",
    "last_actor",
    "last_action_at",
    "source_policy_version",
    "git_commit",
    "config_hash",
    "feature_manifest_hash",
    "generated_at",
)


def flatten_decision(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Flatten a :meth:`DecisionRecord.to_dict` mapping into a scalar-only CSV row.

    Pure and shape-tolerant: the audit trail collapses to a count + last event, ``warnings`` join with
    ``;``, the broker reference reduces to its ``broker_id``, and the provenance fields surface as
    top-level columns. Works on records reconstructed from JSON (e.g. fetched by the CLI).
    """
    rec = record.get("recommendation", {}) or {}
    prov = record.get("provenance", {}) or {}
    audit = record.get("audit", []) or []
    last = audit[-1] if audit else {}
    broker = record.get("broker_reference") or {}
    warnings = record.get("warnings") or []
    return {
        "decision_id": record.get("decision_id"),
        "load_id": record.get("load_id"),
        "truck_id": record.get("truck_id"),
        "status": record.get("status"),
        "recommended_amount": rec.get("recommended_amount"),
        "current_amount": rec.get("current_amount"),
        "delta_from_recommended": rec.get("delta_from_recommended"),
        "delta_percent": rec.get("delta_percent"),
        "recommended_rate_per_mile": rec.get("recommended_rate_per_mile"),
        "winnability_available": rec.get("winnability_available"),
        "win_probability": rec.get("win_probability"),
        "expected_value": rec.get("expected_value"),
        "ev_recommended_label": rec.get("ev_recommended_label"),
        "ev_recommended_bid": rec.get("ev_recommended_bid"),
        "warnings": ";".join(str(w) for w in warnings),
        "broker_id": broker.get("broker_id"),
        "submission_ref": record.get("submission_ref"),
        "created_at": record.get("created_at"),
        "updated_at": record.get("updated_at"),
        "expires_at": record.get("expires_at"),
        "audit_event_count": len(audit),
        "last_action": last.get("action"),
        "last_actor": last.get("actor_id"),
        "last_action_at": last.get("at"),
        "source_policy_version": prov.get("source_policy_version"),
        "git_commit": prov.get("git_commit"),
        "config_hash": prov.get("config_hash"),
        "feature_manifest_hash": prov.get("feature_manifest_hash"),
        "generated_at": prov.get("generated_at"),
    }
