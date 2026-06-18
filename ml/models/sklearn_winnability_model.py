"""Gradient-boosted bid-winnability classifier (Phase 4.2).

Estimates ``P(win | load, broker, market, ask)`` with a
``HistGradientBoostingClassifier`` that consumes the decision-time feature frame from
``ml/features/winnability_features.py``. Categorical columns (equipment, mode,
commodity, credit bucket, origin zone, load-views) are passed through as pandas
``category`` dtype for native handling (no one-hot); ``NaN`` in numeric columns (a
missing posted-rate ratio, an unknown broker's pay days) is handled natively too, so
**missingness stays a signal** rather than being imputed away.

Calibration is **optional and decided upstream** (on the validation slice — never on
test). When the base model is poorly calibrated the trainer attaches a prefit
calibrator via :func:`calibrate_prefit`; ``predict_proba`` then transparently serves the
calibrated probability. The same wrapper, calibrated or not, is what Phase 4.3 loads.

``calibrate_prefit`` is version-robust: scikit-learn 1.6 deprecated
``CalibratedClassifierCV(cv="prefit")`` in favor of wrapping the fitted estimator in
``sklearn.frozen.FrozenEstimator``. We prefer the new path and fall back to ``cv="prefit"``
on older installs, so the project runs across scikit-learn versions.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier

from ml.features.winnability_features import CATEGORICAL_COLUMNS

MODEL_NAME = "HistGradientBoostingWinnability"


def calibrate_prefit(estimator, X_cal, y_cal, method: str) -> CalibratedClassifierCV:
    """Calibrate an already-fitted ``estimator`` on a held-out slice.

    Prefers ``FrozenEstimator`` (scikit-learn ≥ 1.6) and falls back to the deprecated
    ``cv="prefit"`` path on older versions.
    """
    try:
        from sklearn.frozen import FrozenEstimator

        return CalibratedClassifierCV(
            estimator=FrozenEstimator(estimator), method=method
        ).fit(X_cal, y_cal)
    except ImportError:  # pragma: no cover - exercised only on sklearn < 1.6
        return CalibratedClassifierCV(
            estimator=estimator, method=method, cv="prefit"
        ).fit(X_cal, y_cal)


class SklearnWinnabilityModel:
    def __init__(
        self,
        feature_columns: Sequence[str],
        categorical_columns: Sequence[str] = CATEGORICAL_COLUMNS,
        random_state: int = 42,
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

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X[self.feature_columns].copy()
        for col in self.categorical_columns:
            frame[col] = frame[col].astype("category")
        return frame

    def fit(self, X: pd.DataFrame, y) -> "SklearnWinnabilityModel":
        frame = self._prepare(X)
        self.classifier.fit(frame, np.asarray(y, dtype=int))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return ``P(win)`` (1-D). Uses the calibrator when one is attached."""
        frame = self._prepare(X)
        estimator = self.calibrator if self.calibrator is not None else self.classifier
        return estimator.predict_proba(frame)[:, 1]

    @property
    def is_calibrated(self) -> bool:
        return self.calibrator is not None

    def make_calibrated(
        self, X_cal: pd.DataFrame, y_cal, method: str
    ) -> "SklearnWinnabilityModel":
        """Return a sibling model sharing this base classifier + a prefit calibrator.

        The base classifier is reused (not refit); only the calibration map is learned
        on ``(X_cal, y_cal)``. Used by the trainer to build isotonic/sigmoid candidates
        and compare them on the validation slice before choosing what to serve.
        """
        frame = self._prepare(X_cal)
        calibrator = calibrate_prefit(self.classifier, frame, np.asarray(y_cal, dtype=int), method)
        clone = SklearnWinnabilityModel(
            self.feature_columns, self.categorical_columns, self.random_state
        )
        clone.classifier = self.classifier
        clone.calibrator = calibrator
        clone.calibration_method = method
        return clone

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "classifier": self.classifier,
                "calibrator": self.calibrator,
                "calibration_method": self.calibration_method,
                "feature_columns": self.feature_columns,
                "categorical_columns": self.categorical_columns,
                "random_state": self.random_state,
                "model_name": MODEL_NAME,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "SklearnWinnabilityModel":
        payload = joblib.load(Path(path))
        obj = cls(
            feature_columns=payload["feature_columns"],
            categorical_columns=payload["categorical_columns"],
            random_state=payload.get("random_state", 42),
        )
        obj.classifier = payload["classifier"]
        obj.calibrator = payload.get("calibrator")
        obj.calibration_method = payload.get("calibration_method")
        return obj
