"""Phase 6.4 — the shadow compiled-dispatcher service.

Runs the compiled dispatcher **beside** the source engine and reports an additive, read-only
:class:`CompiledDispatcherShadowComparison`. The source engine still owns the recommendation; this
service only *observes*. It is the safety boundary for the whole phase:

* **Shadow only.** ``shadow_only`` is always True; there is no active mode.
* **Fails closed.** Any unavailability / invalid output / exception yields a comparison with
  ``compiled_available=False`` and a ``fallback_reason`` — it never raises into the source path.
* **Never mutates the source.** ``compare`` only *reads* the source decision; the object the caller
  passes in is byte-identical afterwards.
* **No bid authority.** The service is constructed with a single read-only port and holds **no**
  reference to the bid-approval repository/service, so it structurally cannot draft, approve, or
  submit a bid.
"""
from __future__ import annotations

import time
from typing import Any, List, Mapping, Optional

from ports.compiled_dispatcher import (
    KNOWN_WARNINGS,
    REASON_INVALID_OUTPUT,
    REASON_PREDICTION_ERROR,
    SHADOW_ONLY,
    VALID_ACTIONS,
    CompiledDispatcherAvailability,
    CompiledDispatcherPort,
    CompiledDispatcherPrediction,
    CompiledDispatcherShadowComparison,
    CompiledDispatcherUnavailable,
)


def _jaccard(a: List[str], b: List[str]) -> float:
    """Set agreement between two warning lists: ``|A∩B| / |A∪B|`` (empty/empty ⇒ 1.0)."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    return len(sa & sb) / len(union)


def _validate_prediction(pred: CompiledDispatcherPrediction) -> Optional[str]:
    """Return ``None`` if the compiled output is well-formed, else a fail-closed reason code.

    Guards the additive surface against a future/mock adapter (or a corrupted artifact) emitting an
    out-of-contract decision, a non-boolean approval, a junk bid, or an unknown warning code.
    """
    if pred is None:
        return REASON_INVALID_OUTPUT
    if pred.decision not in VALID_ACTIONS:
        return REASON_INVALID_OUTPUT
    if not isinstance(pred.approval_required, bool):
        return REASON_INVALID_OUTPUT
    warnings = pred.warnings
    if not isinstance(warnings, (list, tuple)) or any(w not in KNOWN_WARNINGS for w in warnings):
        return REASON_INVALID_OUTPUT
    bid = pred.recommended_bid
    if bid is not None and (not isinstance(bid, (int, float)) or bid < 0):
        return REASON_INVALID_OUTPUT
    ev = pred.risk_adjusted_ev
    if ev is not None and not isinstance(ev, (int, float)):
        return REASON_INVALID_OUTPUT
    return None


class ShadowCompiledDispatcherService:
    """Compares the source decision with the compiled dispatcher's, fail-closed and read-only."""

    def __init__(self, port: CompiledDispatcherPort):
        self._port = port

    def availability(self) -> CompiledDispatcherAvailability:
        return self._port.availability()

    def compare(
        self,
        source: CompiledDispatcherPrediction,
        features: Mapping[str, Any],
    ) -> CompiledDispatcherShadowComparison:
        """Run the compiled model beside the source decision and return the comparison.

        ``source`` is the source engine's decision (normalized into the shared prediction shape);
        ``features`` is the ``inference_context`` feature dict for the same load. The source object
        is only read — never mutated — and the comparison is pure metadata.
        """
        avail = self._port.availability()
        if not avail.available:
            return self._fallback(source, avail.reason or REASON_PREDICTION_ERROR)

        start = time.perf_counter()
        try:
            compiled = self._port.predict(features)
        except CompiledDispatcherUnavailable as exc:
            return self._fallback(source, exc.reason)
        except Exception:  # noqa: BLE001 — never let a compiled failure escape into the source path
            return self._fallback(source, REASON_PREDICTION_ERROR)
        latency_ms = round((time.perf_counter() - start) * 1000.0, 4)

        invalid = _validate_prediction(compiled)
        if invalid is not None:
            return self._fallback(source, invalid, latency_ms=latency_ms)

        return self._compare(source, compiled, latency_ms)

    # ----- internals (all read-only w.r.t. ``source``) ---------------------- #
    def _compare(
        self,
        source: CompiledDispatcherPrediction,
        compiled: CompiledDispatcherPrediction,
        latency_ms: float,
    ) -> CompiledDispatcherShadowComparison:
        source_bid = source.recommended_bid
        compiled_bid = compiled.recommended_bid
        bid_delta = (
            compiled_bid - source_bid
            if source_bid is not None and compiled_bid is not None
            else None
        )
        bid_delta_percent = (
            (bid_delta / source_bid * 100.0)
            if bid_delta is not None and source_bid
            else None
        )
        source_ev = source.risk_adjusted_ev
        compiled_ev = compiled.risk_adjusted_ev
        ev_delta = (
            compiled_ev - source_ev
            if source_ev is not None and compiled_ev is not None
            else None
        )
        source_warnings = list(source.warnings or [])
        compiled_warnings = list(compiled.warnings or [])
        return CompiledDispatcherShadowComparison(
            compiled_available=True,
            shadow_only=SHADOW_ONLY,
            source_action=source.decision,
            compiled_action=compiled.decision,
            action_agrees=source.decision == compiled.decision,
            source_bid=source_bid,
            compiled_bid=compiled_bid,
            bid_delta=(round(bid_delta, 4) if bid_delta is not None else None),
            bid_delta_percent=(
                round(bid_delta_percent, 4) if bid_delta_percent is not None else None
            ),
            source_approval_required=source.approval_required,
            compiled_approval_required=compiled.approval_required,
            approval_agrees=source.approval_required == compiled.approval_required,
            source_warnings=source_warnings,
            compiled_warnings=compiled_warnings,
            warning_agreement=round(_jaccard(source_warnings, compiled_warnings), 4),
            source_risk_adjusted_ev=source_ev,
            compiled_risk_adjusted_ev=compiled_ev,
            ev_delta=(round(ev_delta, 4) if ev_delta is not None else None),
            compiled_latency_ms=latency_ms,
            fallback_reason=None,
        )

    def _fallback(
        self,
        source: CompiledDispatcherPrediction,
        reason: str,
        *,
        latency_ms: Optional[float] = None,
    ) -> CompiledDispatcherShadowComparison:
        """Compiled side is None; the source decision is echoed unchanged with a reason."""
        return CompiledDispatcherShadowComparison(
            compiled_available=False,
            shadow_only=SHADOW_ONLY,
            source_action=source.decision,
            compiled_action=None,
            action_agrees=None,
            source_bid=source.recommended_bid,
            compiled_bid=None,
            bid_delta=None,
            bid_delta_percent=None,
            source_approval_required=source.approval_required,
            compiled_approval_required=None,
            approval_agrees=None,
            source_warnings=list(source.warnings or []),
            compiled_warnings=None,
            warning_agreement=None,
            source_risk_adjusted_ev=source.risk_adjusted_ev,
            compiled_risk_adjusted_ev=None,
            ev_delta=None,
            compiled_latency_ms=latency_ms,
            fallback_reason=reason,
        )
