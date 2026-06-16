"""Gradient-boosted destination-desirability model (Phase 3.1).

The label ``expected_next_deadhead_miles`` is *censored*: when no viable next
load exists within the search window it is pinned to ``max_deadhead_cap_miles``
(300 mi). In practice that produces a spike of cap values (~10-12% of rows) on
top of a continuous bulk of short real deadheads. A single regressor cannot
serve both well — squared-error loss is dragged toward the mean by the cap
spike (good RMSE/R2, poor on the bulk), while absolute-error loss predicts the
bulk median and ignores the spike (good MAE, R2 ~ 0).

So this is a **hurdle model**:

* a classifier estimates ``p = P(no viable next load -> label == cap)``;
* a regressor (squared-error -> conditional mean) estimates the deadhead
  *given that a viable load exists*, trained only on non-censored rows;
* the served expectation recombines them::

      E[deadhead] = p * cap + (1 - p) * E[deadhead | load exists]

Both sub-models are ``HistGradientBoosting*`` with native categorical handling
(no one-hot): ``destination_zone``, ``destination_state``, ``equipment_type``
and ``mode`` are passed through as pandas ``category`` columns. The class
owns its feature selection, persistence (joblib), and reload, and keeps the
plain ``fit``/``predict`` interface the rest of the pipeline expects.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

from ml.features.destination_features import CATEGORICAL_COLUMNS

MODEL_NAME = "HurdleHistGradientBoosting"

# Below this many non-censored training rows the regressor is not trustworthy;
# fall back to the cap-driven part only.
_MIN_REG_ROWS = 20


class SklearnDestinationModel:
    def __init__(
        self,
        feature_columns: Sequence[str],
        categorical_columns: Sequence[str] = CATEGORICAL_COLUMNS,
        random_state: int = 42,
        cap: float = 300.0,
        bulk_loss: str = "absolute_error",
    ) -> None:
        self.feature_columns: List[str] = list(feature_columns)
        self.categorical_columns: List[str] = [
            c for c in categorical_columns if c in self.feature_columns
        ]
        self.random_state = random_state
        self.cap = float(cap)
        self.bulk_loss = bulk_loss

        self.classifier: Optional[HistGradientBoostingClassifier] = (
            HistGradientBoostingClassifier(
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
        )
        self.regressor: Optional[HistGradientBoostingRegressor] = (
            HistGradientBoostingRegressor(
                loss=bulk_loss,
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
        )
        # Set when one side is degenerate (e.g. no censored rows at all).
        self._const_cap_prob: Optional[float] = None

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X[self.feature_columns].copy()
        for col in self.categorical_columns:
            frame[col] = frame[col].astype("category")
        return frame

    def _is_cap(self, y: np.ndarray) -> np.ndarray:
        return y >= self.cap - 1e-9

    def fit(self, X: pd.DataFrame, y) -> "SklearnDestinationModel":
        frame = self._prepare(X)
        y = np.asarray(y, dtype=float)
        is_cap = self._is_cap(y)
        n_cap = int(is_cap.sum())
        n_real = int((~is_cap).sum())

        # Classifier: only meaningful with both classes present.
        if n_cap == 0:
            self.classifier = None
            self._const_cap_prob = 0.0
        elif n_real == 0:
            self.classifier = None
            self._const_cap_prob = 1.0
        else:
            self.classifier.fit(frame, is_cap.astype(int))
            self._const_cap_prob = None

        # Regressor: conditional mean of the *non-censored* bulk.
        if n_real >= _MIN_REG_ROWS:
            self.regressor.fit(frame[~is_cap], y[~is_cap])
        else:
            self.regressor = None

        return self

    def _cap_prob(self, frame: pd.DataFrame) -> np.ndarray:
        if self.classifier is None:
            prob = 0.0 if self._const_cap_prob is None else self._const_cap_prob
            return np.full(len(frame), prob, dtype=float)
        return self.classifier.predict_proba(frame)[:, 1]

    def _bulk_pred(self, frame: pd.DataFrame) -> np.ndarray:
        if self.regressor is None:
            # No trustworthy bulk estimate; fall back to the cap.
            return np.full(len(frame), self.cap, dtype=float)
        return np.clip(self.regressor.predict(frame), 0.0, self.cap)

    def predict(self, X: pd.DataFrame):
        frame = self._prepare(X)
        p_cap = self._cap_prob(frame)
        bulk = self._bulk_pred(frame)
        combined = p_cap * self.cap + (1.0 - p_cap) * bulk
        return np.clip(combined, 0.0, self.cap)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "classifier": self.classifier,
                "regressor": self.regressor,
                "const_cap_prob": self._const_cap_prob,
                "cap": self.cap,
                "bulk_loss": self.bulk_loss,
                "feature_columns": self.feature_columns,
                "categorical_columns": self.categorical_columns,
                "random_state": self.random_state,
                "model_name": MODEL_NAME,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "SklearnDestinationModel":
        payload = joblib.load(Path(path))
        obj = cls(
            feature_columns=payload["feature_columns"],
            categorical_columns=payload["categorical_columns"],
            random_state=payload.get("random_state", 42),
            cap=payload.get("cap", 300.0),
            bulk_loss=payload.get("bulk_loss", "absolute_error"),
        )
        obj.classifier = payload["classifier"]
        obj.regressor = payload["regressor"]
        obj._const_cap_prob = payload.get("const_cap_prob")
        return obj
