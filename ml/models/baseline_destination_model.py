"""Non-ML baselines for the destination model (Phase 3.1).

A model only earns its keep if it beats a dumb predictor. Two baselines:

* ``GlobalMeanModel`` — predicts the training-set mean for everything.
* ``ZoneDaypartBaseline`` — predicts the mean label for the destination zone and
  arrival daypart, with a fallback chain ``zone+daypart -> zone -> global`` so
  thin buckets degrade gracefully.

Both accept a pandas ``DataFrame`` of features so they are interchangeable with
the scikit-learn model at the call site.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from ml.features.destination_features import daypart


class GlobalMeanModel:
    def __init__(self) -> None:
        self.mean_: float = 0.0

    def fit(self, X: pd.DataFrame, y) -> "GlobalMeanModel":
        self.mean_ = float(np.asarray(y, dtype=float).mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.mean_, dtype=float)


class ZoneDaypartBaseline:
    def __init__(self, min_count: int = 5) -> None:
        self.min_count = min_count
        self.global_: float = 0.0
        self.zone_: Dict[str, float] = {}
        self.zone_daypart_: Dict[Tuple[str, str], float] = {}

    def fit(self, X: pd.DataFrame, y) -> "ZoneDaypartBaseline":
        y = np.asarray(y, dtype=float)
        self.global_ = float(y.mean())
        frame = pd.DataFrame(
            {
                "zone": X["destination_zone"].to_numpy(),
                "daypart": X["arrival_hour"].map(daypart).to_numpy(),
                "y": y,
            }
        )
        for zone, sub in frame.groupby("zone"):
            if len(sub) >= self.min_count:
                self.zone_[zone] = float(sub["y"].mean())
        for (zone, dp), sub in frame.groupby(["zone", "daypart"]):
            if len(sub) >= self.min_count:
                self.zone_daypart_[(zone, dp)] = float(sub["y"].mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        dparts = X["arrival_hour"].map(daypart)
        out = np.empty(len(X), dtype=float)
        for i, (zone, dp) in enumerate(zip(X["destination_zone"], dparts)):
            if (zone, dp) in self.zone_daypart_:
                out[i] = self.zone_daypart_[(zone, dp)]
            elif zone in self.zone_:
                out[i] = self.zone_[zone]
            else:
                out[i] = self.global_
        return out
