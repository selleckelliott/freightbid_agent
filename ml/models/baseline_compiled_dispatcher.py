"""Non-ML baseline for the compiled dispatcher (Phase 6.3).

The learned multi-head model only earns its keep if it beats a trivial predictor built from the
same training labels. :class:`MajorityCompiledDispatcherBaseline` predicts the **majority decision**
for every load, the **mean** bid-ratio and risk-adjusted EV, the majority approval flag, and each
warning's majority label. It exposes the same ``predict_batch`` / ``predict_dto`` surface as
:class:`~ml.models.compiled_dispatcher_model.CompiledDispatcherModel`, so the evaluator can score
both through one code path — and the headline test is that the model beats this baseline on action
macro-F1 (a majority predictor scores poorly there by construction, since the minority no-bid and
approval classes get zero recall).
"""
from __future__ import annotations

from collections import Counter
from typing import Any, List, Mapping, Optional, Sequence

from ml.data.compiled_agent_trace_schema import DECISION_NO_BID
from ml.models.compiled_dispatcher_model import (
    WARNING_HEADS,
    CompiledDispatcherPrediction,
    derive_action,
    derive_approval_required,
    derive_bid_ratio,
    derive_ev,
    derive_warnings,
)


def _majority(values: Sequence[Any], default: Any) -> Any:
    counts = Counter(values)
    if not counts:
        return default
    return counts.most_common(1)[0][0]


def _mean(values: Sequence[float], default: float = 0.0) -> float:
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else default


class MajorityCompiledDispatcherBaseline:
    def __init__(self) -> None:
        self.majority_action: str = ""
        self.mean_bid_ratio: float = 0.0
        self.mean_ev: float = 0.0
        self.majority_approval: int = 0
        self.majority_warnings: List[str] = []

    def fit(self, rows: Sequence[Mapping[str, Any]]) -> "MajorityCompiledDispatcherBaseline":
        self.majority_action = _majority([derive_action(r) for r in rows], default="")
        self.mean_bid_ratio = _mean([derive_bid_ratio(r) for r in rows], default=1.0)
        self.mean_ev = _mean([derive_ev(r) for r in rows], default=0.0)
        self.majority_approval = int(
            _majority([derive_approval_required(r) for r in rows], default=0)
        )
        self.majority_warnings = [
            w for w in WARNING_HEADS
            if _majority([derive_warnings(r)[w] for r in rows], default=0) == 1
        ]
        return self

    def predict_raw(self, feature_dicts: Sequence[Mapping[str, Any]]) -> dict:
        """Constant per-head outputs, matching the model's evaluation surface."""
        import numpy as np

        n = len(feature_dicts)
        return {
            "action": np.array([self.majority_action] * n),
            "bid_ratio": np.full(n, self.mean_bid_ratio, dtype=float),
            "risk_adjusted_ev": np.full(n, self.mean_ev, dtype=float),
            "warnings": {
                w: np.full(n, int(w in self.majority_warnings)) for w in WARNING_HEADS
            },
            "approval_required": np.full(n, self.majority_approval),
        }

    def predict_batch(
        self, feature_dicts: Sequence[Mapping[str, Any]]
    ) -> List[CompiledDispatcherPrediction]:
        out: List[CompiledDispatcherPrediction] = []
        for fd in feature_dicts:
            decision = self.majority_action
            market = fd.get("market_rate")
            miles = fd.get("loaded_miles")
            if decision == DECISION_NO_BID or market is None or miles is None:
                rpm: Optional[float] = None
                amount: Optional[float] = None
                ratio_out: Optional[float] = None
            else:
                rpm = self.mean_bid_ratio * float(market)
                amount = rpm * float(miles)
                ratio_out = self.mean_bid_ratio
            out.append(
                CompiledDispatcherPrediction(
                    recommended_load_id=fd.get("load_id"),
                    decision=decision,
                    recommended_bid=amount,
                    recommended_bid_rpm=rpm,
                    bid_ratio=ratio_out,
                    risk_adjusted_ev=self.mean_ev,
                    approval_required=bool(self.majority_approval == 1),
                    warnings=list(self.majority_warnings),
                )
            )
        return out

    def predict_dto(self, features: Mapping[str, Any]) -> CompiledDispatcherPrediction:
        return self.predict_batch([features])[0]
