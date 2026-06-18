"""Non-ML baselines for the bid-winnability model (Phase 4.2).

A learned win model only earns its keep if it beats simple, explainable predictors of
``P(win)``. Three baselines, increasing in sophistication:

* ``GlobalWinRateModel`` — predicts the training base win rate for everything. The
  floor any model must beat on log loss / Brier.
* ``AskRatioHeuristicModel`` — the "smart dispatcher rule of thumb": win probability
  is a function of how aggressive the ask is. It bins the ask **relative to the posted
  rate** (falling back to the **market** rate on no-rate loads) and predicts each bin's
  training win rate. Captures the dominant monotone "ask higher → win less" effect with
  no broker/market knowledge.
* ``BrokerMarketGroupedBaseline`` — win rate by ``(origin_zone, credit_bucket,
  ask-ratio bin)`` with a graceful fallback chain so thin buckets degrade to coarser
  ones instead of overfitting. Adds the "who and where" the heuristic ignores.

All three expose ``fit(X, y)`` + ``predict_proba(X) → P(win)`` (a 1-D array), so they
are drop-in swappable with the scikit-learn model at the evaluation call site.
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd

_EPS = 1e-6


def effective_ask_ratio(X: pd.DataFrame) -> np.ndarray:
    """Ask relative to the posted rate where visible, else relative to market.

    This is the anchor a dispatcher actually reasons about: bid-vs-posted when the
    broker shows a rate, bid-vs-market on "call for rate" loads.
    """
    posted = X["ask_to_posted_ratio"].to_numpy(dtype=float)
    market = X["ask_to_market_ratio"].to_numpy(dtype=float)
    has_posted = X["has_posted_rate"].to_numpy(dtype=float) > 0.5
    use_posted = has_posted & ~np.isnan(posted)
    return np.where(use_posted, posted, market)


def _bucket(ratios: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    """Map ratios to bin indices ``0..len(edges)-2`` via the interior edges.

    ``NaN`` ratios fall into the last bin (defensive; effective ratios are finite).
    """
    interior = np.asarray(edges[1:-1], dtype=float)
    filled = np.where(np.isnan(ratios), np.inf, ratios)
    return np.digitize(filled, interior).astype(int)


class GlobalWinRateModel:
    def __init__(self) -> None:
        self.win_rate_: float = 0.0

    def fit(self, X: pd.DataFrame, y) -> "GlobalWinRateModel":
        self.win_rate_ = float(np.asarray(y, dtype=float).mean())
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.win_rate_, dtype=float)


class AskRatioHeuristicModel:
    def __init__(self, edges: Sequence[float], min_count: int = 25) -> None:
        self.edges = list(edges)
        self.min_count = min_count
        self.global_: float = 0.0
        self.bin_rate_: Dict[int, float] = {}

    def fit(self, X: pd.DataFrame, y) -> "AskRatioHeuristicModel":
        y = np.asarray(y, dtype=float)
        self.global_ = float(y.mean())
        bins = _bucket(effective_ask_ratio(X), self.edges)
        for b in np.unique(bins):
            mask = bins == b
            if int(mask.sum()) >= self.min_count:
                self.bin_rate_[int(b)] = float(y[mask].mean())
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        bins = _bucket(effective_ask_ratio(X), self.edges)
        out = np.empty(len(X), dtype=float)
        for i, b in enumerate(bins):
            out[i] = self.bin_rate_.get(int(b), self.global_)
        return np.clip(out, _EPS, 1.0 - _EPS)


class BrokerMarketGroupedBaseline:
    """Win rate by ``(zone, credit_bucket, ask-bin)`` with a fallback chain."""

    def __init__(self, edges: Sequence[float], min_count: int = 25) -> None:
        self.edges = list(edges)
        self.min_count = min_count
        self.global_: float = 0.0
        self.bin_: Dict[int, float] = {}
        self.zone_bin_: Dict[Tuple[str, int], float] = {}
        self.zone_credit_bin_: Dict[Tuple[str, str, int], float] = {}

    def fit(self, X: pd.DataFrame, y) -> "BrokerMarketGroupedBaseline":
        y = np.asarray(y, dtype=float)
        self.global_ = float(y.mean())
        frame = pd.DataFrame(
            {
                "zone": X["origin_zone"].to_numpy(),
                "credit": X["broker_credit_bucket"].to_numpy(),
                "bin": _bucket(effective_ask_ratio(X), self.edges),
                "y": y,
            }
        )
        for b, sub in frame.groupby("bin"):
            if len(sub) >= self.min_count:
                self.bin_[int(b)] = float(sub["y"].mean())
        for (zone, b), sub in frame.groupby(["zone", "bin"]):
            if len(sub) >= self.min_count:
                self.zone_bin_[(zone, int(b))] = float(sub["y"].mean())
        for (zone, credit, b), sub in frame.groupby(["zone", "credit", "bin"]):
            if len(sub) >= self.min_count:
                self.zone_credit_bin_[(zone, credit, int(b))] = float(sub["y"].mean())
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        zones = X["origin_zone"].to_numpy()
        credits = X["broker_credit_bucket"].to_numpy()
        bins = _bucket(effective_ask_ratio(X), self.edges)
        out = np.empty(len(X), dtype=float)
        for i in range(len(X)):
            zone, credit, b = zones[i], credits[i], int(bins[i])
            if (zone, credit, b) in self.zone_credit_bin_:
                out[i] = self.zone_credit_bin_[(zone, credit, b)]
            elif (zone, b) in self.zone_bin_:
                out[i] = self.zone_bin_[(zone, b)]
            elif b in self.bin_:
                out[i] = self.bin_[b]
            else:
                out[i] = self.global_
        return np.clip(out, _EPS, 1.0 - _EPS)
