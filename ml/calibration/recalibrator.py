"""Post-hoc probability recalibrators for Phase 5.4 (repair, not retrain).

When the Phase 5.3 monitor flags a market whose predicted ``P(win)`` no longer matches the
realized win rate, the base winnability model is **frozen** — we do not retrain the
``HistGradientBoostingClassifier``. Instead we fit a lightweight post-hoc map

    repaired_p = recalibrator(raw_p)

on recent labeled outcomes and wrap the frozen model. This module provides that map.

The default is a **sigmoid / Platt** recalibrator::

    repaired_p = sigmoid(a * logit(raw_p) + b)

chosen over isotonic for an operational recalibration window because it is monotonic,
two-parameter (so it is stable on the smaller / noisier windows recalibration runs on), and
easy to explain — clearly a thin *repair layer*, not a second model. ``a`` rescales the
model's confidence (``a < 1`` shrinks over-optimistic logits toward 0.5) and ``b`` shifts the
overall level. Isotonic is offered as an optional comparator (``method="isotonic"``) for
larger windows but is **not** the default.

A recalibrator only ever sees ``(raw_probability, realized_outcome)`` pairs, so it cannot leak
base-model features or training labels; fitting it is cheap and deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np

SIGMOID = "sigmoid"
ISOTONIC = "isotonic"
METHODS = (SIGMOID, ISOTONIC)

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


@dataclass(frozen=True)
class Recalibrator:
    """A fitted post-hoc probability map ``raw_p -> repaired_p``.

    Immutable and self-contained: ``sigmoid`` carries scalar ``(a, b)``; ``isotonic`` carries
    the fitted step function in ``iso``. :meth:`transform` is a pure, vectorized mapping whose
    output stays in ``[0, 1]``.
    """

    method: str
    n_fit: int
    a: Optional[float] = None          # sigmoid slope (None for isotonic)
    b: Optional[float] = None          # sigmoid intercept (None for isotonic)
    iso: Optional[Any] = None          # fitted IsotonicRegression (None for sigmoid)

    def transform(self, raw_p: Sequence[float]) -> np.ndarray:
        """Map raw probabilities to repaired probabilities (vectorized, in ``[0, 1]``)."""
        p = np.asarray(raw_p, dtype=float)
        if p.size == 0:
            return p.astype(float)
        if self.method == SIGMOID:
            return _sigmoid(self.a * _logit(p) + self.b)
        repaired = self.iso.predict(np.clip(p, 0.0, 1.0))
        return np.clip(np.asarray(repaired, dtype=float), 0.0, 1.0)

    def as_dict(self) -> Dict[str, Any]:
        """JSON-ready parameters (the isotonic step function is summarized, not dumped)."""
        out: Dict[str, Any] = {"method": self.method, "n_fit": self.n_fit}
        if self.method == SIGMOID:
            out["a"] = round(float(self.a), 6)
            out["b"] = round(float(self.b), 6)
        return out


def fit_recalibrator(
    raw_p: Sequence[float], y_true: Sequence[int], method: str = SIGMOID
) -> Recalibrator:
    """Fit a post-hoc recalibrator on ``(raw_p, y_true)``.

    ``sigmoid`` fits ``a, b`` by a 1-D logistic regression on ``logit(raw_p)`` (Platt scaling);
    ``isotonic`` fits a monotonic step function. Raises ``ValueError`` on empty input,
    mismatched shapes, a single-class target (nothing to calibrate against), or an unknown
    method. Deterministic given the inputs.
    """
    p = np.asarray(raw_p, dtype=float)
    y = np.asarray(y_true, dtype=float)
    if p.shape != y.shape:
        raise ValueError(f"raw_p and y_true must match shape, got {p.shape} vs {y.shape}")
    n = int(p.size)
    if n == 0:
        raise ValueError("cannot fit a recalibrator on an empty slice")
    if np.unique(y).size < 2:
        raise ValueError("cannot fit a recalibrator on a single-class target")

    if method == SIGMOID:
        from sklearn.linear_model import LogisticRegression

        z = _logit(p).reshape(-1, 1)
        # Large C ~ unregularized logistic (Platt) fit; lbfgs is deterministic.
        lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
        lr.fit(z, y.astype(int))
        return Recalibrator(
            method=SIGMOID, n_fit=n, a=float(lr.coef_[0, 0]), b=float(lr.intercept_[0])
        )
    if method == ISOTONIC:
        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(np.clip(p, 0.0, 1.0), y)
        return Recalibrator(method=ISOTONIC, n_fit=n, iso=iso)
    raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
