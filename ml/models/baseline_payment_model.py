"""Non-ML baselines for the broker payment-risk model (Phase 5.2).

A learned default model only earns its keep if it beats simple, explainable predictors
of ``P(default)`` built from the same observable board columns. None of these touches
the ask — payment is broker-driven. Three baselines, increasing in sophistication:

* ``GlobalDefaultRateModel`` — predicts the training base default rate for everything.
  The floor any model must beat on log loss / Brier.
* ``CreditBucketBaseline`` — default rate grouped by ``broker_credit_bucket`` (A/B/C/
  ``unknown``), the single column a dispatcher reads first. Thin or unseen buckets fall
  back to the global rate so the estimate never overfits a handful of rows.
* ``BondedQuickPayBaseline`` — default rate grouped by the ``(broker_bonded,
  broker_quick_pay_available)`` cell: the "who actually pays" board heuristic, since a
  bonded broker offering quick-pay is structurally far less likely to stiff a carrier.
  Unseen/thin cells fall back to global.

All three expose ``fit(X, y)`` + ``predict_proba(X) → P(default)`` (a 1-D array), so they
are drop-in swappable with the scikit-learn model at the evaluation call site.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

_EPS = 1e-6


def _bonded_quick_cell(X: pd.DataFrame) -> np.ndarray:
    """Map each row to its ``(bonded, quick_pay)`` cell key.

    ``NaN`` (an unknown flag) is preserved as the string ``"nan"`` so unknown-flag rows
    form their own cell rather than silently merging with ``False``.
    """
    bonded = X["broker_bonded"].to_numpy(dtype=float)
    quick = X["broker_quick_pay_available"].to_numpy(dtype=float)
    return np.array(
        [f"{_flag(b)}|{_flag(q)}" for b, q in zip(bonded, quick)], dtype=object
    )


def _flag(value: float) -> str:
    if np.isnan(value):
        return "nan"
    return "1" if value > 0.5 else "0"


class GlobalDefaultRateModel:
    def __init__(self) -> None:
        self.default_rate_: float = 0.0

    def fit(self, X: pd.DataFrame, y) -> "GlobalDefaultRateModel":
        self.default_rate_ = float(np.asarray(y, dtype=float).mean())
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.default_rate_, dtype=float)


class CreditBucketBaseline:
    """Default rate grouped by broker credit bucket, with a global fallback."""

    def __init__(self, min_count: int = 25) -> None:
        self.min_count = min_count
        self.global_: float = 0.0
        self.bucket_rate_: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y) -> "CreditBucketBaseline":
        y = np.asarray(y, dtype=float)
        self.global_ = float(y.mean())
        frame = pd.DataFrame(
            {"bucket": X["broker_credit_bucket"].to_numpy(), "y": y}
        )
        for bucket, sub in frame.groupby("bucket"):
            if len(sub) >= self.min_count:
                self.bucket_rate_[str(bucket)] = float(sub["y"].mean())
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        buckets = X["broker_credit_bucket"].to_numpy()
        out = np.array(
            [self.bucket_rate_.get(str(b), self.global_) for b in buckets],
            dtype=float,
        )
        return np.clip(out, _EPS, 1.0 - _EPS)


class BondedQuickPayBaseline:
    """Default rate grouped by ``(bonded, quick_pay)`` cell, with a global fallback."""

    def __init__(self, min_count: int = 25) -> None:
        self.min_count = min_count
        self.global_: float = 0.0
        self.cell_rate_: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y) -> "BondedQuickPayBaseline":
        y = np.asarray(y, dtype=float)
        self.global_ = float(y.mean())
        frame = pd.DataFrame({"cell": _bonded_quick_cell(X), "y": y})
        for cell, sub in frame.groupby("cell"):
            if len(sub) >= self.min_count:
                self.cell_rate_[str(cell)] = float(sub["y"].mean())
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        cells = _bonded_quick_cell(X)
        out = np.array(
            [self.cell_rate_.get(str(c), self.global_) for c in cells],
            dtype=float,
        )
        return np.clip(out, _EPS, 1.0 - _EPS)
