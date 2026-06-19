"""Model-backed payment-risk adapter (Phase 5.2).

Wraps the trained Phase 5.2 ``SklearnPaymentRiskModel`` and scores one board load's
broker. Features are built with ``build_payment_features`` — the exact training-time,
ask-free builder — so there is no train/serve skew. ``estimate`` returns ``P(default)``,
its complement ``p_collect``, and (when the model carries a pay-days head) the
``E[pay_days]`` slow-pay estimate.

``from_config`` / ``from_artifact`` return ``None`` when no artifact is present, so a
caller can wire the no-op fallback and get the same risk-blind behavior — the Phase 4.3
"model is optional" contract, applied to payment.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from ml.features.payment_features import build_payment_features
from ml.features.winnability_features import BidQuery
from ports.payment_risk import PaymentEstimate, PaymentRiskPort


class ModelPaymentRiskAdapter(PaymentRiskPort):
    def __init__(self, model) -> None:
        self._model = model

    @classmethod
    def from_artifact(cls, model_path: str | Path) -> Optional["ModelPaymentRiskAdapter"]:
        """Load the trained model, or ``None`` when the artifact is missing."""
        path = Path(model_path)
        if not path.exists():
            return None
        from ml.models.sklearn_payment_risk_model import SklearnPaymentRiskModel

        return cls(SklearnPaymentRiskModel.load(path))

    @classmethod
    def from_config(cls, cfg) -> Optional["ModelPaymentRiskAdapter"]:
        """Build from an ``MLConfig``, or ``None`` when the artifact is missing."""
        from ml.training.payment_risk_dataset import resolve_path

        return cls.from_artifact(resolve_path(cfg.payment_risk.model_path))

    def estimate(self, query: BidQuery) -> Optional[PaymentEstimate]:
        frame = pd.DataFrame([build_payment_features(query)])
        p_default = float(self._model.predict_proba(frame)[0])
        expected_pay_days: Optional[float] = None
        if self._model.has_pay_days:
            expected_pay_days = float(self._model.predict_pay_days(frame)[0])
        return PaymentEstimate(
            p_default=p_default,
            p_collect=1.0 - p_default,
            expected_pay_days=expected_pay_days,
        )
