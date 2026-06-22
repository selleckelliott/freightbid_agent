"""Tests for the Phase 5.3 calibration drift monitor.

Fast, pure tests for the severity rule, the calibration metrics + binning, the
``min_samples`` gate, config-driven thresholds, determinism, and the drift helper — plus one
small seeded end-to-end smoke proving the baseline (training) world stays calibrated while a
reserve/win-curve shift trips the monitor.
"""
import json
from dataclasses import replace

import numpy as np
import pytest

from benchmarks.run_broker_quality_stress import Condition
from benchmarks.run_calibration_monitor import run_monitor
from ml.config import load_ml_config
from ml.monitoring.calibration_drift import (
    ALERT,
    OK,
    WATCH,
    CalibrationThresholds,
    calibration_drift,
    classify_severity,
    evaluate_calibration,
    worst_severity,
)
from ml.monitoring.calibration_report import (
    load_calibration_config,
    report_to_dict,
    severity_tally,
)

TH = CalibrationThresholds()  # defaults: ece 0.03/0.07, bias 0.05/0.10, min 500, bins 10


# --------------------------------------------------------------------------- #
# Severity rule
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "ece,abs_bias,expected",
    [
        (0.00, 0.00, OK),
        (0.029, 0.049, OK),
        (0.03, 0.00, WATCH),        # ECE inclusive WATCH edge
        (0.00, 0.05, WATCH),        # bias inclusive WATCH edge
        (0.069, 0.099, WATCH),
        (0.07, 0.00, ALERT),        # ECE inclusive ALERT edge
        (0.00, 0.10, ALERT),        # bias inclusive ALERT edge
        (0.5, 0.0, ALERT),
        (0.0, 0.6, ALERT),
    ],
)
def test_classify_severity_bands(ece, abs_bias, expected):
    assert classify_severity(ece, abs_bias, TH) == expected


def test_classify_severity_takes_worst_of_ece_or_bias():
    # ECE only WATCH, bias ALERT -> ALERT (max of the two signals).
    assert classify_severity(0.04, 0.12, TH) == ALERT
    # bias OK, ECE WATCH -> WATCH.
    assert classify_severity(0.05, 0.0, TH) == WATCH


# --------------------------------------------------------------------------- #
# evaluate_calibration: metrics + verdict
# --------------------------------------------------------------------------- #

def _well_calibrated(n_per_bin=400):
    # Two buckets whose observed rate equals the predicted probability.
    yp = np.concatenate([np.full(n_per_bin, 0.2), np.full(n_per_bin, 0.8)])
    yt = np.concatenate([
        np.array([1] * (n_per_bin // 5) + [0] * (n_per_bin - n_per_bin // 5)),  # 20% win
        np.array([1] * (4 * n_per_bin // 5) + [0] * (n_per_bin - 4 * n_per_bin // 5)),  # 80%
    ])
    return yt, yp


def test_well_calibrated_is_ok_with_tiny_ece_and_bias():
    yt, yp = _well_calibrated()
    rep = evaluate_calibration(yt, yp, TH, label="winnability")
    assert rep.severity == OK
    assert rep.ece < 1e-6
    assert abs(rep.bias) < 1e-6
    assert rep.insufficient_data is False
    assert rep.label == "winnability"
    assert rep.n == len(yt)


def test_over_optimistic_predictions_alert():
    # Predict 0.9 everywhere, but only half actually win -> bias +0.4, ECE 0.4.
    yp = np.full(1000, 0.9)
    yt = np.array([1] * 500 + [0] * 500)
    rep = evaluate_calibration(yt, yp, TH, label="p_win")
    assert rep.severity == ALERT
    assert rep.bias == pytest.approx(0.4, abs=1e-6)      # over-optimistic (signed +)
    assert rep.ece == pytest.approx(0.4, abs=1e-6)


def test_under_confident_predictions_also_alert_via_abs_bias():
    # Predict 0.1 but half win -> bias -0.4; severity uses |bias|.
    yp = np.full(1000, 0.1)
    yt = np.array([1] * 500 + [0] * 500)
    rep = evaluate_calibration(yt, yp, TH, label="p_win")
    assert rep.bias == pytest.approx(-0.4, abs=1e-6)
    assert rep.severity == ALERT


def test_reliability_table_binning_counts_and_rates():
    # 300 at 0.9 (half win) + 200 at 0.2 (all lose) -> two populated bins.
    yp = np.concatenate([np.full(300, 0.9), np.full(200, 0.2)])
    yt = np.concatenate([np.array([1] * 150 + [0] * 150), np.zeros(200)])
    rep = evaluate_calibration(yt, yp, TH)
    table = {round(b["bin_lower"], 1): b for b in rep.reliability_table}
    assert set(table) == {0.2, 0.9}
    assert table[0.9]["count"] == 300
    assert table[0.9]["observed_rate"] == pytest.approx(0.5, abs=1e-9)
    assert table[0.2]["count"] == 200
    assert table[0.2]["observed_rate"] == pytest.approx(0.0, abs=1e-9)
    # Populated-bin counts cover every sample.
    assert sum(b["count"] for b in rep.reliability_table) == len(yt)


def test_n_bins_is_config_driven():
    yp = np.linspace(0.0, 1.0, 1000)
    yt = (np.random.default_rng(0).random(1000) < yp).astype(int)
    five = evaluate_calibration(yt, yp, replace(TH, n_bins=5))
    twenty = evaluate_calibration(yt, yp, replace(TH, n_bins=20))
    assert len(five.reliability_table) <= 5
    assert len(twenty.reliability_table) <= 20
    assert len(twenty.reliability_table) > len(five.reliability_table)


# --------------------------------------------------------------------------- #
# min_samples gate + degenerate slices
# --------------------------------------------------------------------------- #

def test_insufficient_samples_gates_severity_to_ok():
    # Badly miscalibrated but below min_samples -> flagged, not alarmed.
    yp = np.full(100, 0.9)
    yt = np.zeros(100, dtype=int)
    rep = evaluate_calibration(yt, yp, replace(TH, min_samples=500))
    assert rep.insufficient_data is True
    assert rep.severity == OK
    # The raw metrics are still reported honestly.
    assert rep.bias == pytest.approx(0.9, abs=1e-6)
    assert rep.ece > 0.1


def test_sufficient_samples_are_not_flagged():
    yt, yp = _well_calibrated()
    rep = evaluate_calibration(yt, yp, replace(TH, min_samples=10))
    assert rep.insufficient_data is False


def test_empty_slice_is_safe_and_serializable():
    rep = evaluate_calibration([], [], TH, label="empty")
    assert rep.n == 0
    assert rep.severity == OK
    assert rep.insufficient_data is True
    assert rep.ece is None and rep.bias is None and rep.brier is None
    blob = json.dumps(report_to_dict(rep))  # no NaN / non-serializable values
    assert "NaN" not in blob


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError, match="same shape"):
        evaluate_calibration([1, 0, 1], [0.5, 0.5], TH)


def test_no_nan_in_serialized_report():
    yt, yp = _well_calibrated()
    rep = evaluate_calibration(yt, yp, TH)
    blob = json.dumps(report_to_dict(rep))
    assert "NaN" not in blob and "Infinity" not in blob


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #

def test_evaluate_calibration_is_deterministic():
    yt, yp = _well_calibrated()
    a = report_to_dict(evaluate_calibration(yt, yp, TH))
    b = report_to_dict(evaluate_calibration(yt, yp, TH))
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# --------------------------------------------------------------------------- #
# Drift + aggregation helpers
# --------------------------------------------------------------------------- #

def test_calibration_drift_deltas():
    yt, yp = _well_calibrated()
    base = evaluate_calibration(yt, yp, TH)
    worse = evaluate_calibration(np.zeros(1000, dtype=int), np.full(1000, 0.9), TH)
    drift = calibration_drift(worse, base)
    assert drift["bias_drift"] == pytest.approx(worse.bias - base.bias, abs=1e-6)
    assert drift["ece_drift"] > 0.5
    # Baseline-vs-None anchors at zero.
    assert calibration_drift(base, None) == {"bias_drift": 0.0, "ece_drift": 0.0}


def test_worst_severity_and_tally():
    yt, yp = _well_calibrated()
    ok = evaluate_calibration(yt, yp, TH)
    bad = evaluate_calibration(np.zeros(1000, dtype=int), np.full(1000, 0.9), TH)
    assert worst_severity([ok.severity, bad.severity]) == ALERT
    assert worst_severity([]) == OK
    tally = severity_tally([ok, bad])
    assert tally[OK] == 1 and tally[ALERT] == 1 and tally[WATCH] == 0


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #

def test_shipped_config_has_expected_defaults():
    thresholds, extras = load_calibration_config("config/calibration_monitor.yaml")
    assert thresholds.ece_watch == 0.03
    assert thresholds.ece_alert == 0.07
    assert thresholds.bias_watch == 0.05
    assert thresholds.bias_alert == 0.10
    assert thresholds.min_samples == 500
    assert thresholds.n_bins == 10
    assert extras["target"] == "winnability"
    assert extras["conditions"].endswith("broker_quality_stress.yaml")


def test_config_thresholds_are_applied(tmp_path):
    path = tmp_path / "cm.yaml"
    path.write_text(
        "calibration_monitor:\n"
        "  ece_watch_threshold: 0.5\n"
        "  ece_alert_threshold: 0.9\n"
        "  bias_watch_threshold: 0.5\n"
        "  bias_alert_threshold: 0.9\n"
        "  min_samples: 10\n",
        encoding="utf-8",
    )
    thresholds, _ = load_calibration_config(path)
    assert thresholds.ece_watch == 0.5 and thresholds.ece_alert == 0.9
    # An ECE of 0.4 is ALERT under defaults but OK under these loosened thresholds.
    yp = np.full(1000, 0.9)
    yt = np.array([1] * 500 + [0] * 500)
    assert evaluate_calibration(yt, yp, CalibrationThresholds()).severity == ALERT
    assert evaluate_calibration(yt, yp, thresholds).severity == OK


def test_partial_config_falls_back_to_defaults(tmp_path):
    path = tmp_path / "cm.yaml"
    path.write_text("calibration_monitor:\n  ece_watch_threshold: 0.02\n", encoding="utf-8")
    thresholds, extras = load_calibration_config(path)
    assert thresholds.ece_watch == 0.02
    assert thresholds.ece_alert == 0.07  # default preserved
    assert extras["target"] == "winnability"  # default preserved


# --------------------------------------------------------------------------- #
# End-to-end smoke: baseline stays calibrated, a reserve shift trips the monitor
# --------------------------------------------------------------------------- #

def _tiny_cfg():
    cfg = load_ml_config()
    return replace(
        cfg,
        synthetic_data=replace(cfg.synthetic_data, loads_per_snapshot_mean=12.0,
                               snapshots_per_day=4),
    )


def test_monitor_smoke_baseline_ok_and_reserve_shift_drifts():
    cfg = _tiny_cfg()
    thresholds = replace(CalibrationThresholds(), min_samples=30)
    conditions = [
        Condition("baseline", "", {}),
        Condition("tight_brokers", "reserve down",
                  {"outcomes": {"reservation_center_mult": 0.92}}),
    ]
    records = run_monitor(cfg, thresholds, conditions, days=3, max_rows=None)

    base = next(r for r in records if r["name"] == "baseline")
    shift = next(r for r in records if r["name"] == "tight_brokers")

    # Baseline is the anchored in-distribution reference (zero self-drift). Its *absolute*
    # calibration is asserted on the canonical run, not this deliberately tiny/noisy world.
    assert base["n"] >= 30
    assert base["ece_drift"] == 0.0 and base["bias_drift"] == 0.0
    assert base["lens"] == "reference"

    # The monitor's job: a reserve shift makes the fixed model over-optimistic, so it must
    # be detected as materially worse than baseline and escalated off OK.
    assert shift["severity"] in (WATCH, ALERT)
    assert shift["ece"] > base["ece"]            # more miscalibrated than baseline
    assert shift["ece_drift"] > 0.0
    assert shift["bias"] > base["bias"] + 0.02   # predicts higher than reality vs baseline

    # Records are JSON-serializable with the documented shape.
    for r in records:
        assert {"name", "lens", "severity", "ece", "bias", "ece_drift",
                "bias_drift", "reliability_table", "thresholds"} <= set(r)
    json.dumps(records)
