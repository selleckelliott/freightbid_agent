"""Phase 6.4 — compiled-dispatcher **shadow** port + DTOs.

One port, two adapters (mirrors ``ports/winnability.py`` / ``ports/payment_risk.py``):

* a **no-op** that reports the compiled dispatcher unavailable (the default wiring — flag off /
  no artifact), and
* a real **sklearn** adapter that wraps the frozen Phase 6.3 ``CompiledDispatcherModel`` artifact.

The compiled dispatcher runs **in shadow mode only**. It never owns the recommendation and it can
**not** draft, approve, or submit a bid — ``predict`` is pure. The comparison + fail-closed
orchestration lives in ``application/services/shadow_compiled_dispatcher_service.py``; this module
just defines the boundary (the port ABC) and the additive, read-only DTOs the service emits.

The prediction DTO itself is the Phase 6.3 :class:`CompiledDispatcherPrediction`, re-exported here so
callers depend on the *port*, not the ml model module. The **source** engine's decision is normalized
into the same prediction shape (:func:`source_prediction_from_targets`) so source-vs-compiled can be
compared field-for-field.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from ml.data.compiled_agent_trace_schema import (
    DECISION_APPROVAL_REQUIRED,
    DECISION_BID,
    DECISION_NO_BID,
)
from ml.models.compiled_dispatcher_model import CompiledDispatcherPrediction
from ml.workflows.freightbid_workflow_graph import (
    WARN_CALIBRATION_ALERT,
    WARN_CALIBRATION_WATCH,
    WARN_NEGATIVE_RISK_ADJUSTED_EV,
    WARN_NO_FEASIBLE_BID,
    WARN_PAYMENT_RISK,
)

__all__ = [
    "CompiledDispatcherPrediction",
    "CompiledDispatcherAvailability",
    "CompiledDispatcherShadowComparison",
    "CompiledDispatcherUnavailable",
    "CompiledDispatcherPort",
    "source_prediction_from_targets",
    "REASON_DISABLED",
    "REASON_NO_ARTIFACT",
    "REASON_MANIFEST_MISMATCH",
    "REASON_INVALID_OUTPUT",
    "REASON_PREDICTION_ERROR",
    "SHADOW_ONLY",
    "VALID_ACTIONS",
    "KNOWN_WARNINGS",
]

# -- Fail-closed reason codes (stable strings — surfaced in the comparison + logs) ----------
REASON_DISABLED = "disabled"
REASON_NO_ARTIFACT = "no_artifact"
REASON_MANIFEST_MISMATCH = "manifest_mismatch"
REASON_INVALID_OUTPUT = "invalid_output"
REASON_PREDICTION_ERROR = "prediction_error"

# In Phase 6.4 the compiled dispatcher is *always* shadow-only — there is no active mode yet.
SHADOW_ONLY = True

# The action vocabulary + warning codes the validator accepts (anything else ⇒ invalid output).
VALID_ACTIONS = frozenset({DECISION_BID, DECISION_NO_BID, DECISION_APPROVAL_REQUIRED})
KNOWN_WARNINGS = frozenset({
    WARN_PAYMENT_RISK,
    WARN_CALIBRATION_ALERT,
    WARN_CALIBRATION_WATCH,
    WARN_NEGATIVE_RISK_ADJUSTED_EV,
    WARN_NO_FEASIBLE_BID,
})


class CompiledDispatcherUnavailable(RuntimeError):
    """Raised by an adapter's ``predict`` when no compiled model can serve (fail-closed).

    Carries a ``REASON_*`` code so the shadow service can record *why* it fell back.
    """

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class CompiledDispatcherAvailability:
    """Whether a compiled dispatcher artifact is loaded and serveable — and, if not, why."""

    available: bool
    reason: Optional[str] = None  # None iff available; else a REASON_* code
    artifact_path: Optional[str] = None
    feature_manifest_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason,
            "artifact_path": self.artifact_path,
            "feature_manifest_hash": self.feature_manifest_hash,
        }


@dataclass(frozen=True)
class CompiledDispatcherShadowComparison:
    """An additive, **read-only** comparison of the source engine vs the compiled dispatcher.

    ``shadow_only`` is **always True** in Phase 6.4: the source engine owns the decision and this
    object is pure metadata. When the compiled model cannot serve, ``compiled_available`` is False,
    every ``compiled_*`` field is ``None``, and ``fallback_reason`` explains why — the source-side
    fields stay populated so the comparison is always renderable.
    """

    compiled_available: bool
    shadow_only: bool
    # -- action ---------------------------------------------------------------
    source_action: Optional[str]
    compiled_action: Optional[str]
    action_agrees: Optional[bool]
    # -- bid amount -----------------------------------------------------------
    source_bid: Optional[float]
    compiled_bid: Optional[float]
    bid_delta: Optional[float]
    bid_delta_percent: Optional[float]
    # -- approval -------------------------------------------------------------
    source_approval_required: Optional[bool]
    compiled_approval_required: Optional[bool]
    approval_agrees: Optional[bool]
    # -- warnings -------------------------------------------------------------
    source_warnings: List[str]
    compiled_warnings: Optional[List[str]]
    warning_agreement: Optional[float]
    # -- risk-adjusted EV -----------------------------------------------------
    source_risk_adjusted_ev: Optional[float]
    compiled_risk_adjusted_ev: Optional[float]
    ev_delta: Optional[float]
    # -- diagnostics ----------------------------------------------------------
    compiled_latency_ms: Optional[float]
    fallback_reason: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "compiled_available": self.compiled_available,
            "shadow_only": self.shadow_only,
            "source_action": self.source_action,
            "compiled_action": self.compiled_action,
            "action_agrees": self.action_agrees,
            "source_bid": self.source_bid,
            "compiled_bid": self.compiled_bid,
            "bid_delta": self.bid_delta,
            "bid_delta_percent": self.bid_delta_percent,
            "source_approval_required": self.source_approval_required,
            "compiled_approval_required": self.compiled_approval_required,
            "approval_agrees": self.approval_agrees,
            "source_warnings": list(self.source_warnings),
            "compiled_warnings": (
                list(self.compiled_warnings) if self.compiled_warnings is not None else None
            ),
            "warning_agreement": self.warning_agreement,
            "source_risk_adjusted_ev": self.source_risk_adjusted_ev,
            "compiled_risk_adjusted_ev": self.compiled_risk_adjusted_ev,
            "ev_delta": self.ev_delta,
            "compiled_latency_ms": self.compiled_latency_ms,
            "fallback_reason": self.fallback_reason,
        }


def source_prediction_from_targets(targets: Mapping[str, Any]) -> CompiledDispatcherPrediction:
    """Normalize a teacher/source ``targets`` dict (Phase 6.2 structured row) into the shared
    :class:`CompiledDispatcherPrediction` shape, so the source engine's decision and the compiled
    model's decision can be compared field-for-field. Pure / read-only — does not mutate ``targets``.
    """
    decision = targets["decision"]
    return CompiledDispatcherPrediction(
        recommended_load_id=targets.get("recommended_load_id"),
        decision=decision,
        recommended_bid=targets.get("recommended_bid_amount"),
        recommended_bid_rpm=targets.get("recommended_bid_rpm"),
        bid_ratio=None,
        risk_adjusted_ev=targets.get("risk_adjusted_ev"),
        approval_required=(decision == DECISION_APPROVAL_REQUIRED),
        warnings=list(targets.get("warnings") or []),
    )


class CompiledDispatcherPort(ABC):
    """Outbound port: the shadow service's view of a (possibly absent) compiled dispatcher."""

    @abstractmethod
    def availability(self) -> CompiledDispatcherAvailability:
        """Cheap, side-effect-free: can this adapter serve right now, and if not, why."""

    @abstractmethod
    def predict(self, features: Mapping[str, Any]) -> CompiledDispatcherPrediction:
        """Return a compiled prediction, or raise :class:`CompiledDispatcherUnavailable`.

        Implementations must **never** draft, approve, or submit a bid — ``predict`` is pure.
        """
