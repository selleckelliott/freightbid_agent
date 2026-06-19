"""Gradient-boosted broker payment-risk model (Phase 5.2).

Estimates ``P(default | load, broker, market)`` — the probability the broker never
pays — with a ``HistGradientBoostingClassifier`` over the **ask-free** observable
feature frame from ``ml/features/payment_features.py``. Default is the catastrophic,
total-loss outcome and the minority class, so this is a calibration-first model: the
Phase 5.1 risk-adjusted EV will multiply expected margin by ``p_collect = 1 - p_default``,
and a predicted 0.05 has to mean roughly a 5% loss rate or the EV math is wrong. The
wrapper mirrors :class:`~ml.models.sklearn_winnability_model.SklearnWinnabilityModel`
(same HGB hyperparameters, the same prefit-calibration discipline) and reuses its
version-robust :func:`~ml.models.sklearn_winnability_model.calibrate_prefit`.

Categorical columns (equipment, mode, commodity, credit bucket, origin zone,
load-views) pass through as pandas ``category`` dtype; ``NaN`` in numeric columns (an
unknown-credit broker's missing pay-days) is handled natively, so **missingness stays a
signal**.

A second, optional head estimates ``E[pay_days]`` with a ``HistGradientBoostingRegressor``
trained **only on non-default rows** (where realized pay-days exist). It is a lightweight
slow-pay discount input for 5.1's EV, not a core target, so it travels as an *optional*
field in the same save/load payload and is simply absent when never fit. ``predict_proba``
still returns ``P(default)``; ``predict_pay_days`` returns the day estimate.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

from ml.features.payment_features import PAYMENT_CATEGORICAL_COLUMNS
from ml.models.sklearn_winnability_model import calibrate_prefit

MODEL_NAME = "HistGradientBoostingPaymentRisk"


class SklearnPaymentRiskModel:
    def __init__(
        self,
        feature_columns: Sequence[str],
        categorical_columns: Sequence[str] = PAYMENT_CATEGORICAL_COLUMNS,
        random_state: int = 45,
    ) -> None:
        self.feature_columns: List[str] = list(feature_columns)
        self.categorical_columns: List[str] = [
            c for c in categorical_columns if c in self.feature_columns
        ]
        self.random_state = random_state
        self.classifier = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=400,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=0.1,
            early_stopping=True,
            validation_fraction=0.15,
            categorical_features="from_dtype",
            random_state=random_state,
        )
        self.calibrator: Optional[CalibratedClassifierCV] = None
        self.calibration_method: Optional[str] = None
        # Optional secondary head: E[pay_days] on non-default rows. None until fit.
        self.pay_days_regressor: Optional[HistGradientBoostingRegressor] = None

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X[self.feature_columns].copy()
        for col in self.categorical_columns:
            frame[col] = frame[col].astype("category")
        return frame

    def fit(self, X: pd.DataFrame, y) -> "SklearnPaymentRiskModel":
        frame = self._prepare(X)
        self.classifier.fit(frame, np.asarray(y, dtype=int))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return ``P(default)`` (1-D). Uses the calibrator when one is attached."""
        frame = self._prepare(X)
        estimator = self.calibrator if self.calibrator is not None else self.classifier
        return estimator.predict_proba(frame)[:, 1]

    def fit_pay_days(self, X: pd.DataFrame, y_days) -> "SklearnPaymentRiskModel":
        """Fit the optional ``E[pay_days]`` regressor on (non-default) rows.

        Same categorical handling as the classifier. Caller is responsible for passing
        only rows with a realized pay-days target (defaulted loads have none).
        """
        frame = self._prepare(X)
        regressor = HistGradientBoostingRegressor(
            learning_rate=0.08,
            max_iter=400,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=0.1,
            early_stopping=True,
            validation_fraction=0.15,
            categorical_features="from_dtype",
            random_state=self.random_state,
        )
        regressor.fit(frame, np.asarray(y_days, dtype=float))
        self.pay_days_regressor = regressor
        return self

    def predict_pay_days(self, X: pd.DataFrame) -> np.ndarray:
        """Return ``E[pay_days]`` (1-D). Raises if the regressor was never fit."""
        if self.pay_days_regressor is None:
            raise RuntimeError("pay-days regressor was not fit on this model")
        frame = self._prepare(X)
        return self.pay_days_regressor.predict(frame)

    @property
    def is_calibrated(self) -> bool:
        return self.calibrator is not None

    @property
    def has_pay_days(self) -> bool:
        return self.pay_days_regressor is not None

    def make_calibrated(
        self, X_cal: pd.DataFrame, y_cal, method: str
    ) -> "SklearnPaymentRiskModel":
        """Return a sibling model sharing this base classifier + a prefit calibrator.

        The base classifier is reused (not refit); only the calibration map is learned
        on ``(X_cal, y_cal)``. The pay-days regressor (if any) is carried by reference,
        so the calibrated sibling can still emit slow-pay estimates. Used by the trainer
        to build isotonic/sigmoid candidates and compare them on validation.
        """
        frame = self._prepare(X_cal)
        calibrator = calibrate_prefit(
            self.classifier, frame, np.asarray(y_cal, dtype=int), method
        )
        clone = SklearnPaymentRiskModel(
            self.feature_columns, self.categorical_columns, self.random_state
        )
        clone.classifier = self.classifier
        clone.calibrator = calibrator
        clone.calibration_method = method
        clone.pay_days_regressor = self.pay_days_regressor
        return clone

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "classifier": self.classifier,
                "calibrator": self.calibrator,
                "calibration_method": self.calibration_method,
                "pay_days_regressor": self.pay_days_regressor,
                "feature_columns": self.feature_columns,
                "categorical_columns": self.categorical_columns,
                "random_state": self.random_state,
                "model_name": MODEL_NAME,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "SklearnPaymentRiskModel":
        payload = joblib.load(Path(path))
        obj = cls(
            feature_columns=payload["feature_columns"],
            categorical_columns=payload["categorical_columns"],
            random_state=payload.get("random_state", 45),
        )
        obj.classifier = payload["classifier"]
        obj.calibrator = payload.get("calibrator")
        obj.calibration_method = payload.get("calibration_method")
        obj.pay_days_regressor = payload.get("pay_days_regressor")
        return obj
