"""Evaluation metrics for the destination model (Phase 3.1).

Regression accuracy plus business-facing views: distance buckets, label
censoring rate, and a ranking hit-rate that mirrors how the planner will use the
signal (pick the best destination among the loads on the board).
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    err = yp - yt
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "median_ae": float(np.median(np.abs(err))),
        "r2": (1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0,
    }


def bucket_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    thresholds: Sequence[float] = (25.0, 50.0),
) -> Dict[str, float]:
    err = np.abs(np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float))
    return {f"within_{int(t)}mi": float(np.mean(err <= t)) for t in thresholds}


def censoring_rate(y: Sequence[float], cap: float) -> float:
    arr = np.asarray(y, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr >= cap))


def top_k_hit_rate(
    snapshot: Sequence,
    equipment: Sequence,
    y_true: Sequence[float],
    y_pred: Sequence[float],
    k: int = 3,
    min_candidates: int = 4,
) -> Dict[str, float]:
    """How often the truly-best destination lands in the model's top-k.

    Examples are grouped by (snapshot, equipment) — the set of destinations a
    dispatcher would choose among at one decision point. Lower predicted
    next-deadhead = more desirable. Groups smaller than ``min_candidates`` are
    skipped.
    """
    frame = pd.DataFrame(
        {
            "snap": np.asarray(snapshot),
            "eq": np.asarray(equipment),
            "true": np.asarray(y_true, dtype=float),
            "pred": np.asarray(y_pred, dtype=float),
        }
    )
    hits = 0
    groups = 0
    for _, sub in frame.groupby(["snap", "eq"], sort=False):
        if len(sub) < min_candidates:
            continue
        groups += 1
        top_idx = set(sub.nsmallest(k, "pred").index)
        if sub["true"].idxmin() in top_idx:
            hits += 1
    return {
        "top_k_hit_rate": float(hits / groups) if groups else float("nan"),
        "ranking_groups": float(groups),
        "k": float(k),
    }
