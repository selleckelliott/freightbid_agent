"""Calibration-first evaluation metrics for the winnability model (Phase 4.2).

For a bid model, **calibration matters more than ranking**: the Phase 4.3 EV optimizer
multiplies ``P(win)`` by margin, so a predicted 0.70 has to win about 70% of the time
or the expected-value math is wrong. So alongside the usual discrimination metrics
(ROC AUC, PR AUC) this module reports the probability-quality metrics that actually
gate downstream use:

* **Brier score** and **log loss** — proper scoring rules (lower is better).
* **Expected Calibration Error (ECE)** — the sample-weighted average gap between
  predicted probability and observed win rate across equal-width probability bins.
* a **reliability table** — per-bin count, mean predicted probability, and observed
  win rate; the tabular form of the calibration curve that gets charted.

Everything is pure ``numpy`` + a few ``sklearn.metrics`` scoring functions, so the same
helpers run on the validation slice (to decide calibration) and once on the test slice.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

_EPS = 1e-12


def _clip(p: Sequence[float]) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)


def classification_metrics(y_true: Sequence[int], y_prob: Sequence[float]) -> Dict[str, float]:
    """ROC AUC, PR AUC, Brier, and log loss for ``P(win)`` predictions.

    AUC metrics need both classes present; they return ``nan`` on a degenerate slice.
    """
    yt = np.asarray(y_true, dtype=int)
    yp = _clip(y_prob)
    both_classes = np.unique(yt).size > 1
    return {
        "roc_auc": float(roc_auc_score(yt, yp)) if both_classes else float("nan"),
        "pr_auc": float(average_precision_score(yt, yp)) if both_classes else float("nan"),
        "brier": float(brier_score_loss(yt, yp)),
        "log_loss": float(log_loss(yt, yp, labels=[0, 1])),
    }


def reliability_table(
    y_true: Sequence[int], y_prob: Sequence[float], n_bins: int = 10
) -> List[Dict[str, float]]:
    """Per-bin calibration view over ``n_bins`` equal-width probability bins.

    Each row reports the bin's probability interval, the count of predictions in it,
    the mean predicted probability, and the observed win rate. Empty bins are omitted.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    table: List[Dict[str, float]] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (yp >= lo) & (yp < hi) if i < n_bins - 1 else (yp >= lo) & (yp <= hi)
        count = int(mask.sum())
        if count == 0:
            continue
        table.append(
            {
                "bin_lower": float(lo),
                "bin_upper": float(hi),
                "count": count,
                "mean_predicted": float(yp[mask].mean()),
                "observed_rate": float(yt[mask].mean()),
            }
        )
    return table


def expected_calibration_error(
    y_true: Sequence[int], y_prob: Sequence[float], n_bins: int = 10
) -> float:
    """Sample-weighted mean ``|mean_predicted − observed_rate|`` across bins."""
    yt = np.asarray(y_true, dtype=float)
    n = len(yt)
    if n == 0:
        return float("nan")
    ece = 0.0
    for row in reliability_table(yt, y_prob, n_bins):
        weight = row["count"] / n
        ece += weight * abs(row["mean_predicted"] - row["observed_rate"])
    return float(ece)


def evaluate_probabilities(
    y_true: Sequence[int], y_prob: Sequence[float], n_bins: int = 10
) -> Dict[str, object]:
    """Full calibration-first report: scalar metrics + ECE + the reliability table."""
    metrics = classification_metrics(y_true, y_prob)
    metrics["ece"] = expected_calibration_error(y_true, y_prob, n_bins)
    metrics["n"] = int(len(y_true))
    metrics["positive_rate"] = float(np.asarray(y_true, dtype=float).mean()) if len(y_true) else float("nan")
    return {
        "metrics": metrics,
        "reliability_table": reliability_table(y_true, y_prob, n_bins),
    }
