"""Generic probability-calibration drift monitor (Phase 5.3).

Promotes calibration drift from an *offline observation* — the Phase 4.5 sweep noticed the
baseline-trained winnability model turning over-optimistic under reserve / win-curve shift —
into a *standing*, label-based monitor: given predicted probabilities and the realized binary
outcomes they were meant to predict, is the model still calibrated? Does a predicted ``0.70``
still win about 70% of the time?

The monitor is deliberately **generic** over ``(predicted_probability, realized_outcome)``.
It is instantiated first on the winnability model (``p_win`` vs ``bid_won``) because Phase 4.5
already proved drift there, but the same code serves the payment-default model
(``p_default`` vs ``payment_defaulted``) unchanged.

This phase **detects**; it does not repair (that is Phase 5.4) and it changes no recommender
behavior — it only reads predictions and outcomes.

Reported metrics are all **label-based** (never feature-distribution):

* **bias** — ``mean(predicted) − observed_event_rate`` (signed; ``+`` is over-optimistic).
* **ECE** — expected calibration error over equal-width probability bins.
* **Brier** score and **log loss** — proper scoring rules (lower is better).
* a **reliability table** — per-bin mean predicted vs observed rate (the charted curve).

Severity is config-driven (``config/calibration_monitor.yaml``):

* **ALERT** if ``ece >= ece_alert`` or ``abs(bias) >= bias_alert``.
* **WATCH** if ``ece >= ece_watch`` or ``abs(bias) >= bias_watch``.
* **OK** otherwise.

Below ``min_samples`` the verdict is gated to ``OK`` with ``insufficient_data`` set — a small,
noisy slice must not cry wolf. Metrics are still reported so a human can judge.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss

from ml.training.winnability_metrics import (
    expected_calibration_error,
    reliability_table,
)

# Severity levels, least → most severe.
OK = "OK"
WATCH = "WATCH"
ALERT = "ALERT"
SEVERITY_ORDER: Dict[str, int] = {OK: 0, WATCH: 1, ALERT: 2}

_EPS = 1e-12


@dataclass(frozen=True)
class CalibrationThresholds:
    """Config-driven severity bands for the calibration monitor.

    Defaults match ``config/calibration_monitor.yaml``; see :func:`classify_severity`.
    """

    ece_watch: float = 0.03
    ece_alert: float = 0.07
    bias_watch: float = 0.05
    bias_alert: float = 0.10
    min_samples: int = 500
    n_bins: int = 10

    def as_dict(self) -> Dict[str, float]:
        return {
            "ece_watch": self.ece_watch,
            "ece_alert": self.ece_alert,
            "bias_watch": self.bias_watch,
            "bias_alert": self.bias_alert,
            "min_samples": self.min_samples,
            "n_bins": self.n_bins,
        }


@dataclass(frozen=True)
class CalibrationReport:
    """Calibration verdict for one ``(predicted, realized)`` slice.

    Scalar metrics are ``None`` only for an empty slice; otherwise finite floats (never
    ``NaN``) so the report serializes to clean JSON.
    """

    label: str
    n: int
    severity: str
    insufficient_data: bool
    mean_predicted: Optional[float]
    observed_rate: Optional[float]
    bias: Optional[float]
    ece: Optional[float]
    brier: Optional[float]
    log_loss: Optional[float]
    reliability_table: List[Dict[str, float]]
    thresholds: Dict[str, float]


def classify_severity(ece: float, abs_bias: float, thresholds: CalibrationThresholds) -> str:
    """Pure severity rule: ``ALERT`` > ``WATCH`` > ``OK`` from ECE and ``|bias|``.

    ``NaN`` inputs compare ``False`` against every threshold and therefore yield ``OK`` — an
    empty/degenerate slice is gated to ``OK`` by :func:`evaluate_calibration` regardless.
    """
    if ece >= thresholds.ece_alert or abs_bias >= thresholds.bias_alert:
        return ALERT
    if ece >= thresholds.ece_watch or abs_bias >= thresholds.bias_watch:
        return WATCH
    return OK


def evaluate_calibration(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    thresholds: Optional[CalibrationThresholds] = None,
    *,
    label: str = "prediction",
) -> CalibrationReport:
    """Calibration report for predicted probabilities ``y_prob`` vs realized ``y_true``.

    Generic over any probability/outcome pair (winnability ``p_win``/``bid_won``, payment
    ``p_default``/``payment_defaulted``, …). Below ``thresholds.min_samples`` the severity is
    gated to ``OK`` and ``insufficient_data`` is set.
    """
    th = thresholds or CalibrationThresholds()
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_prob, dtype=float)
    if yt.shape != yp.shape:
        raise ValueError(
            f"y_true and y_prob must have the same shape, got {yt.shape} vs {yp.shape}"
        )
    n = int(yt.size)
    insufficient = n < th.min_samples

    if n == 0:
        return CalibrationReport(
            label=label,
            n=0,
            severity=OK,
            insufficient_data=True,
            mean_predicted=None,
            observed_rate=None,
            bias=None,
            ece=None,
            brier=None,
            log_loss=None,
            reliability_table=[],
            thresholds=th.as_dict(),
        )

    mean_pred = float(yp.mean())
    obs_rate = float(yt.mean())
    bias = mean_pred - obs_rate
    ece = float(expected_calibration_error(yt, yp, th.n_bins))
    yp_clipped = np.clip(yp, _EPS, 1.0 - _EPS)
    brier = float(brier_score_loss(yt, yp_clipped))
    ll = float(log_loss(yt, yp_clipped, labels=[0, 1]))
    table = reliability_table(yt, yp, th.n_bins)

    raw_severity = classify_severity(ece, abs(bias), th)
    severity = OK if insufficient else raw_severity

    return CalibrationReport(
        label=label,
        n=n,
        severity=severity,
        insufficient_data=insufficient,
        mean_predicted=round(mean_pred, 6),
        observed_rate=round(obs_rate, 6),
        bias=round(bias, 6),
        ece=round(ece, 6),
        brier=round(brier, 6),
        log_loss=round(ll, 6),
        reliability_table=table,
        thresholds=th.as_dict(),
    )


def calibration_drift(
    report: CalibrationReport, baseline: Optional[CalibrationReport]
) -> Dict[str, Optional[float]]:
    """Signed change in bias / ECE of ``report`` relative to a ``baseline`` report.

    Returns ``0.0`` deltas when ``report`` *is* the baseline (``baseline is None``); ``None``
    deltas when either side lacks the metric (an empty slice).
    """
    if baseline is None:
        return {"bias_drift": 0.0, "ece_drift": 0.0}

    def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is None or b is None:
            return None
        return round(a - b, 6)

    return {
        "bias_drift": _delta(report.bias, baseline.bias),
        "ece_drift": _delta(report.ece, baseline.ece),
    }


def worst_severity(severities: Sequence[str]) -> str:
    """The most severe level among ``severities`` (``OK`` for an empty input)."""
    return max(severities, key=lambda s: SEVERITY_ORDER.get(s, 0), default=OK)
