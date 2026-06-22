"""Operational recalibration workflow for Phase 5.4 (repair flagged win-prob drift).

Phase 5.3 *detects* when the frozen winnability model's predicted ``P(win)`` no longer matches
the realized win rate. Phase 5.4 *repairs* it without retraining the base model: fit a
lightweight post-hoc recalibrator (:mod:`ml.calibration.recalibrator`) on a **recent** window
of labeled outcomes, then prove calibration improves on a **later, disjoint** holdout window.
The base model stays frozen, so this tests *operational recalibration*, not retraining.

Two rules keep the claim honest:

1. **Never fit and evaluate on the same outcomes.** :func:`time_split` carves an early ``fit``
   window and a later ``eval`` window out of a world by calendar day; the recalibrator is fit
   on ``fit`` and judged only on ``eval``.
2. **Only promote a safer map.** :func:`decide_promotion` accepts a recalibrator only if, on
   the held-out ``eval`` window, it improves ECE, does not raise severity, and does not worsen
   the Brier score by more than ``max_brier_worsening``. Otherwise the base model is kept —
   recalibration can overcorrect.

The pure functions here (:func:`recalibrate`, :func:`decide_promotion`) take arrays /
reports, so they are exercised directly in tests; the benchmark
(``benchmarks/run_recalibration_workflow.py``) wires them over the Phase 4.5 worlds.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from ml.calibration.recalibrator import SIGMOID, Recalibrator, fit_recalibrator
from ml.monitoring.calibration_drift import (
    SEVERITY_ORDER,
    CalibrationReport,
    CalibrationThresholds,
    evaluate_calibration,
)


@dataclass(frozen=True)
class RecalibrationConfig:
    """Config-driven recalibration policy (``config/recalibration.yaml``).

    ``enabled`` gates wiring the recalibrated adapter into a live path (default off ⇒ no
    behavior change); the offline workflow always computes so drift can be studied.
    """

    enabled: bool = False
    method: str = SIGMOID
    min_samples: int = 500
    fit_days: int = 7
    eval_days: int = 14
    max_brier_worsening: float = 0.01
    require_ece_improvement: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RecalibrationResult:
    """Outcome of one world's recalibration attempt.

    ``pre`` is the frozen base model's calibration on the eval window; ``post`` is the
    recalibrated model's calibration on the *same* eval window (``None`` when no recalibrator
    could be fit). ``promoted`` is the guardrail decision; ``recalibrator`` carries the fitted
    map's parameters when one was fit.
    """

    label: str
    promoted: bool
    reason: str
    method: str
    n_fit: int
    n_eval: int
    pre: CalibrationReport
    post: Optional[CalibrationReport]
    recalibrator: Optional[Dict[str, Any]]


def load_recalibration_config(path: str | Path) -> RecalibrationConfig:
    """Parse the ``recalibration:`` block into a :class:`RecalibrationConfig`.

    Unknown keys are ignored and every field falls back to its dataclass default, so a partial
    (or empty / missing-block) config still yields a usable policy.
    """
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    block = doc.get("recalibration", doc) or {}
    d = RecalibrationConfig()
    return RecalibrationConfig(
        enabled=bool(block.get("enabled", d.enabled)),
        method=str(block.get("method", d.method)),
        min_samples=int(block.get("min_samples", d.min_samples)),
        fit_days=int(block.get("fit_days", d.fit_days)),
        eval_days=int(block.get("eval_days", d.eval_days)),
        max_brier_worsening=float(block.get("max_brier_worsening", d.max_brier_worsening)),
        require_ece_improvement=bool(
            block.get("require_ece_improvement", d.require_ece_improvement)
        ),
    )


def time_split(
    frame: pd.DataFrame, fit_days: int, eval_days: int, *, time_column: str = "snapshot_time"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``frame`` into an early ``fit`` window and a later, disjoint ``eval`` window.

    Day 0 is the first ``snapshot_time``; rows in days ``[0, fit_days)`` form the fit window
    and rows in days ``[fit_days, fit_days + eval_days)`` form the eval window. The windows
    never overlap, so a recalibrator fit on ``fit`` is judged on strictly later outcomes.
    """
    times = pd.to_datetime(frame[time_column])
    day0 = times.min().normalize()
    day_index = (times.dt.normalize() - day0).dt.days
    fit_mask = (day_index >= 0) & (day_index < fit_days)
    eval_mask = (day_index >= fit_days) & (day_index < fit_days + eval_days)
    fit_df = frame[fit_mask.to_numpy()].reset_index(drop=True)
    eval_df = frame[eval_mask.to_numpy()].reset_index(drop=True)
    return fit_df, eval_df


def decide_promotion(
    pre: CalibrationReport, post: CalibrationReport, config: RecalibrationConfig
) -> Tuple[bool, str]:
    """Guardrail: promote the recalibrator only if it is a safe, strict improvement.

    Promote iff, on the held-out eval window: ECE improves (strictly when
    ``require_ece_improvement``), severity does not worsen, and Brier rises by no more than
    ``max_brier_worsening``. Any missing metric (an empty / sub-``min_samples`` eval window)
    blocks promotion. Returns ``(promoted, reason)``.
    """
    if post.insufficient_data or pre.ece is None or post.ece is None or post.brier is None or pre.brier is None:
        return False, "insufficient_eval_samples"
    ece_improved = post.ece < pre.ece if config.require_ece_improvement else post.ece <= pre.ece
    if not ece_improved:
        return False, "no_ece_improvement"
    if SEVERITY_ORDER.get(post.severity, 0) > SEVERITY_ORDER.get(pre.severity, 0):
        return False, "severity_worsened"
    if post.brier > pre.brier + config.max_brier_worsening:
        return False, "brier_worsened"
    return True, "promoted"


def recalibrate(
    raw_fit: Sequence[float],
    y_fit: Sequence[int],
    raw_eval: Sequence[float],
    y_eval: Sequence[int],
    thresholds: CalibrationThresholds,
    config: RecalibrationConfig,
    *,
    label: str = "world",
) -> RecalibrationResult:
    """Fit a recalibrator on the fit window and judge it on the later eval window.

    Computes the frozen base model's calibration on the eval window (``pre``), fits the
    recalibrator on ``(raw_fit, y_fit)``, applies it to the eval window, recomputes calibration
    (``post``), and runs the :func:`decide_promotion` guardrail. When the fit window is below
    ``config.min_samples`` or single-class, no recalibrator is fit and the base model is kept.
    """
    raw_fit = np.asarray(raw_fit, dtype=float)
    y_fit = np.asarray(y_fit, dtype=float)
    raw_eval = np.asarray(raw_eval, dtype=float)
    y_eval = np.asarray(y_eval, dtype=float)

    pre = evaluate_calibration(y_eval, raw_eval, thresholds, label=f"{label}:pre")
    n_fit = int(raw_fit.size)
    n_eval = int(raw_eval.size)

    if n_fit < config.min_samples or np.unique(y_fit).size < 2:
        reason = "insufficient_fit_samples" if n_fit < config.min_samples else "single_class_fit"
        return RecalibrationResult(
            label=label, promoted=False, reason=reason, method=config.method,
            n_fit=n_fit, n_eval=n_eval, pre=pre, post=None, recalibrator=None,
        )

    recalibrator = fit_recalibrator(raw_fit, y_fit, method=config.method)
    repaired_eval = recalibrator.transform(raw_eval)
    post = evaluate_calibration(y_eval, repaired_eval, thresholds, label=f"{label}:post")
    promoted, reason = decide_promotion(pre, post, config)

    return RecalibrationResult(
        label=label, promoted=promoted, reason=reason, method=config.method,
        n_fit=n_fit, n_eval=n_eval, pre=pre, post=post, recalibrator=recalibrator.as_dict(),
    )
