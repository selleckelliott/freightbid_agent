"""Tests for the Phase 5.4 recalibration workflow (repair flagged win-prob drift).

Fast, mostly-pure tests for the post-hoc recalibrator math, the time split's disjointness,
the promotion guardrail (including every rejection branch), the recalibrated adapter's
passthrough contract, and config loading — plus one small seeded end-to-end smoke proving a
reserve shift's over-optimistic base model is detected and repaired on a later holdout window
while the base model stays frozen.
"""
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from adapters.outbound.winnability.recalibrated_winnability_adapter import (
    RecalibratedWinnabilityAdapter,
)
from benchmarks.run_broker_quality_stress import Condition
from benchmarks.run_recalibration_workflow import run_recalibration
from ml.calibration.recalibration_workflow import (
    RecalibrationConfig,
    decide_promotion,
    load_recalibration_config,
    recalibrate,
    time_split,
)
from ml.calibration.recalibrator import (
    ISOTONIC,
    METHODS,
    SIGMOID,
    Recalibrator,
    fit_recalibrator,
)
from ml.config import load_ml_config
from ml.monitoring.calibration_drift import (
    ALERT,
    OK,
    WATCH,
    CalibrationReport,
    CalibrationThresholds,
)
from ports.winnability import WinnabilityPort

TH = CalibrationThresholds()  # ece 0.03/0.07, bias 0.05/0.10, min 500, bins 10


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def _overconfident(n: int, seed: int, k: float = 2.2):
    """Realized outcomes from a latent ``true_p`` paired with *over-confident* raw scores.

    ``raw = sigmoid(k * logit(true_p))`` pushes predictions toward the extremes (k>1), so the
    base scores are badly miscalibrated (high ECE) but a sigmoid recalibrator with ``a ~ 1/k``
    can undo it — the exact situation Phase 5.4 repairs.
    """
    rng = np.random.default_rng(seed)
    true_p = rng.uniform(0.05, 0.95, n)
    y = (rng.random(n) < true_p).astype(int)
    raw = _sigmoid(k * _logit(true_p))
    return raw, y


def _calibrated(n: int, seed: int):
    rng = np.random.default_rng(seed)
    true_p = rng.uniform(0.05, 0.95, n)
    y = (rng.random(n) < true_p).astype(int)
    return true_p, y


# --------------------------------------------------------------------------- #
# Recalibrator: fit + transform
# --------------------------------------------------------------------------- #

def test_sigmoid_recalibrator_repairs_overconfident_scores():
    raw, y = _overconfident(6000, seed=0)
    rec = fit_recalibrator(raw, y, method=SIGMOID)
    repaired = rec.transform(raw)

    # The fit shrinks confidence (a < 1 undoes the k>1 sharpening).
    assert rec.a < 1.0
    # Calibration error (binned) collapses after the map.
    from ml.training.winnability_metrics import expected_calibration_error

    pre = expected_calibration_error(y, raw, 10)
    post = expected_calibration_error(y, repaired, 10)
    assert pre > 0.07          # over-confident scores are ALERT-level miscalibrated
    assert post < pre          # repaired
    assert post < 0.03         # ...all the way back to OK
    assert repaired.min() >= 0.0 and repaired.max() <= 1.0


def test_sigmoid_recalibrator_is_near_identity_on_calibrated_data():
    raw, y = _calibrated(6000, seed=1)
    rec = fit_recalibrator(raw, y, method=SIGMOID)
    repaired = rec.transform(raw)
    # Already calibrated -> the map barely moves anything.
    assert float(np.mean(np.abs(repaired - raw))) < 0.03


def test_sigmoid_fit_is_deterministic():
    raw, y = _overconfident(3000, seed=2)
    a = fit_recalibrator(raw, y, method=SIGMOID).as_dict()
    b = fit_recalibrator(raw, y, method=SIGMOID).as_dict()
    assert a == b
    assert a["method"] == SIGMOID and {"a", "b"} <= set(a)


def test_isotonic_recalibrator_is_monotonic_and_bounded():
    raw, y = _overconfident(4000, seed=3)
    rec = fit_recalibrator(raw, y, method=ISOTONIC)
    grid = np.linspace(0.0, 1.0, 50)
    out = rec.transform(grid)
    assert np.all(np.diff(out) >= -1e-9)              # non-decreasing
    assert out.min() >= 0.0 and out.max() <= 1.0
    d = rec.as_dict()
    assert d["method"] == ISOTONIC and "a" not in d   # step function not dumped


def test_transform_on_empty_is_empty():
    rec = Recalibrator(method=SIGMOID, n_fit=10, a=0.5, b=0.0)
    out = rec.transform([])
    assert isinstance(out, np.ndarray) and out.size == 0


@pytest.mark.parametrize(
    "raw,y,match",
    [
        ([], [], "empty"),
        ([0.4, 0.6], [1, 1], "single-class"),
        ([0.4, 0.6, 0.7], [1, 0], "match shape"),
    ],
)
def test_fit_recalibrator_rejects_bad_input(raw, y, match):
    with pytest.raises(ValueError, match=match):
        fit_recalibrator(raw, y, method=SIGMOID)


def test_fit_recalibrator_rejects_unknown_method():
    raw, y = _overconfident(200, seed=4)
    with pytest.raises(ValueError, match="unknown method"):
        fit_recalibrator(raw, y, method="platt-ish")
    assert set(METHODS) == {SIGMOID, ISOTONIC}


# --------------------------------------------------------------------------- #
# time_split: disjoint early/late windows by calendar day
# --------------------------------------------------------------------------- #

def _day_frame(rows_per_day: int = 5, n_days: int = 6) -> pd.DataFrame:
    base = pd.Timestamp("2024-03-01T00:00:00")
    rows = []
    for d in range(n_days):
        for j in range(rows_per_day):
            ts = base + pd.Timedelta(days=d, hours=1 + j)
            rows.append({"snapshot_time": ts.isoformat(), "idx": d * 100 + j, "day": d})
    return pd.DataFrame(rows)


def test_time_split_carves_disjoint_day_windows():
    frame = _day_frame(rows_per_day=5, n_days=6)
    fit_df, eval_df = time_split(frame, fit_days=2, eval_days=3)

    assert sorted(fit_df["day"].unique()) == [0, 1]            # days [0, 2)
    assert sorted(eval_df["day"].unique()) == [2, 3, 4]        # days [2, 5)
    assert len(fit_df) == 10 and len(eval_df) == 15
    # Day 5 falls past fit_days+eval_days and is dropped; windows never overlap.
    assert set(fit_df["idx"]).isdisjoint(set(eval_df["idx"]))
    assert 5 not in set(fit_df["day"]) and 5 not in set(eval_df["day"])


def test_time_split_handles_empty_eval_window():
    frame = _day_frame(rows_per_day=4, n_days=2)
    fit_df, eval_df = time_split(frame, fit_days=2, eval_days=14)
    assert len(fit_df) == 8 and len(eval_df) == 0


# --------------------------------------------------------------------------- #
# decide_promotion: the guardrail (and every rejection branch)
# --------------------------------------------------------------------------- #

def _report(ece, brier, severity, *, insufficient=False, bias=0.0) -> CalibrationReport:
    return CalibrationReport(
        label="t", n=1000, severity=severity, insufficient_data=insufficient,
        mean_predicted=0.5, observed_rate=0.5, bias=bias, ece=ece, brier=brier,
        log_loss=0.5, reliability_table=[], thresholds={},
    )


CFG = RecalibrationConfig()  # max_brier_worsening 0.01, require_ece_improvement True


def test_promote_when_strictly_safer():
    pre = _report(0.10, 0.20, ALERT)
    post = _report(0.02, 0.20, OK)
    assert decide_promotion(pre, post, CFG) == (True, "promoted")


def test_reject_when_ece_does_not_improve():
    pre = _report(0.02, 0.20, OK)
    post = _report(0.03, 0.20, OK)
    assert decide_promotion(pre, post, CFG) == (False, "no_ece_improvement")


def test_reject_when_severity_worsens_even_if_ece_drops():
    # Lower ECE but bias pushes post to a worse severity band -> not promoted.
    pre = _report(0.05, 0.20, WATCH)
    post = _report(0.02, 0.20, ALERT)
    assert decide_promotion(pre, post, CFG) == (False, "severity_worsened")


def test_reject_when_brier_worsens_past_tolerance():
    pre = _report(0.05, 0.20, WATCH)
    post = _report(0.02, 0.25, WATCH)   # +0.05 Brier > 0.01 tolerance
    assert decide_promotion(pre, post, CFG) == (False, "brier_worsened")


def test_reject_when_eval_window_insufficient():
    pre = _report(0.05, 0.20, WATCH)
    post = _report(0.02, 0.20, OK, insufficient=True)
    assert decide_promotion(pre, post, CFG) == (False, "insufficient_eval_samples")


def test_require_ece_improvement_false_allows_equal_ece():
    cfg = replace(CFG, require_ece_improvement=False)
    pre = _report(0.02, 0.20, OK)
    post = _report(0.02, 0.20, OK)
    assert decide_promotion(pre, post, cfg) == (True, "promoted")


# --------------------------------------------------------------------------- #
# recalibrate: fit gates + the key acceptance behavior
# --------------------------------------------------------------------------- #

def test_recalibrate_keeps_base_when_fit_window_too_small():
    raw_fit, y_fit = _overconfident(20, seed=5)
    raw_eval, y_eval = _overconfident(800, seed=6)
    res = recalibrate(raw_fit, y_fit, raw_eval, y_eval, replace(TH, min_samples=200),
                      replace(CFG, min_samples=200), label="tiny")
    assert res.promoted is False
    assert res.reason == "insufficient_fit_samples"
    assert res.post is None and res.recalibrator is None


def test_recalibrate_keeps_base_on_single_class_fit_window():
    raw_eval, y_eval = _overconfident(800, seed=7)
    raw_fit = np.full(400, 0.7)
    y_fit = np.zeros(400, dtype=int)               # only one class present
    res = recalibrate(raw_fit, y_fit, raw_eval, y_eval, replace(TH, min_samples=200),
                      replace(CFG, min_samples=200), label="oneclass")
    assert res.promoted is False and res.reason == "single_class_fit"
    assert res.post is None


def test_recalibrate_repairs_drifted_world_on_holdout():
    # Over-confident base scores, judged on a DISJOINT eval draw -> detected then repaired.
    raw_fit, y_fit = _overconfident(4000, seed=10)
    raw_eval, y_eval = _overconfident(4000, seed=11)
    th = replace(TH, min_samples=200)
    cfg = replace(CFG, min_samples=200)
    res = recalibrate(raw_fit, y_fit, raw_eval, y_eval, th, cfg, label="drifted")

    assert res.pre.severity == ALERT                       # base is over-confident
    assert res.post is not None
    assert res.post.severity in (OK, WATCH)                # repaired
    assert res.post.ece < res.pre.ece
    assert res.promoted is True and res.reason == "promoted"
    assert res.recalibrator["method"] == SIGMOID and res.n_eval == 4000


# --------------------------------------------------------------------------- #
# RecalibratedWinnabilityAdapter: behavior-preserving passthrough
# --------------------------------------------------------------------------- #

class _FakeBase(WinnabilityPort):
    def __init__(self, probs):
        self._probs = probs

    def win_probabilities(self, query, bid_rpms):
        return None if self._probs is None else list(self._probs)


def test_adapter_applies_recalibrator():
    rec = Recalibrator(method=SIGMOID, n_fit=100, a=0.5, b=0.0)
    adapter = RecalibratedWinnabilityAdapter(_FakeBase([0.8, 0.9]), rec)
    out = adapter.win_probabilities(None, [7.0, 8.0])
    expected = [float(p) for p in _sigmoid(0.5 * _logit([0.8, 0.9]))]
    assert adapter.is_active is True
    assert out == pytest.approx(expected, abs=1e-9)
    assert out != [0.8, 0.9]


def test_adapter_passthrough_without_recalibrator():
    adapter = RecalibratedWinnabilityAdapter(_FakeBase([0.8, 0.9]), None)
    assert adapter.is_active is False
    assert adapter.win_probabilities(None, [7.0, 8.0]) == [0.8, 0.9]


def test_adapter_passthrough_when_base_returns_none():
    rec = Recalibrator(method=SIGMOID, n_fit=100, a=0.5, b=0.0)
    adapter = RecalibratedWinnabilityAdapter(_FakeBase(None), rec)
    assert adapter.win_probabilities(None, [7.0, 8.0]) is None


def test_adapter_passthrough_on_empty_base_list():
    rec = Recalibrator(method=SIGMOID, n_fit=100, a=0.5, b=0.0)
    adapter = RecalibratedWinnabilityAdapter(_FakeBase([]), rec)
    assert adapter.win_probabilities(None, []) == []


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #

def test_shipped_recalibration_config_defaults():
    cfg = load_recalibration_config("config/recalibration.yaml")
    assert cfg.enabled is False
    assert cfg.method == SIGMOID
    assert cfg.min_samples == 500
    assert cfg.fit_days == 7 and cfg.eval_days == 14
    assert cfg.max_brier_worsening == 0.01
    assert cfg.require_ece_improvement is True


def test_recalibration_config_overrides_and_partial_fallback(tmp_path):
    path = tmp_path / "rc.yaml"
    path.write_text(
        "recalibration:\n"
        "  enabled: true\n"
        "  method: isotonic\n"
        "  fit_days: 10\n",
        encoding="utf-8",
    )
    cfg = load_recalibration_config(path)
    assert cfg.enabled is True and cfg.method == ISOTONIC and cfg.fit_days == 10
    # Unspecified fields keep their defaults.
    assert cfg.eval_days == 14 and cfg.min_samples == 500
    assert cfg.require_ece_improvement is True


def test_recalibration_config_missing_block_is_all_defaults(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("other: {}\n", encoding="utf-8")
    cfg = load_recalibration_config(path)
    assert cfg == RecalibrationConfig()


# --------------------------------------------------------------------------- #
# End-to-end smoke: reserve shift drifts then is repaired on a later window
# --------------------------------------------------------------------------- #

def _tiny_cfg():
    cfg = load_ml_config()
    return replace(
        cfg,
        synthetic_data=replace(cfg.synthetic_data, loads_per_snapshot_mean=16.0,
                               snapshots_per_day=4),
    )


def test_recalibration_smoke_reserve_shift_detected_and_repaired():
    cfg = _tiny_cfg()
    thresholds = replace(CalibrationThresholds(), min_samples=40)
    config = replace(RecalibrationConfig(), fit_days=2, eval_days=3, min_samples=40)
    conditions = [
        Condition("baseline", "", {}),
        Condition("tight_brokers", "reserve down",
                  {"outcomes": {"reservation_center_mult": 0.92}}),
    ]
    records = run_recalibration(cfg, thresholds, config, conditions, days=6)

    base = next(r for r in records if r["name"] == "baseline")
    shift = next(r for r in records if r["name"] == "tight_brokers")

    # The reserve shift makes the frozen base model over-optimistic vs baseline...
    assert shift["severity_pre"] in (WATCH, ALERT)
    assert shift["ece_pre"] > base["ece_pre"]
    assert shift["pre"]["bias"] > base["pre"]["bias"]

    # ...and the recalibrator, fit on the early window, improves it on the LATER holdout.
    assert shift["ece_post"] is not None
    assert shift["ece_post"] < shift["ece_pre"]

    # Records are JSON-serializable with the documented shape.
    for r in records:
        assert {"name", "promoted", "reason", "severity_pre", "severity_post",
                "ece_pre", "ece_post", "n_fit", "n_eval", "pre"} <= set(r)
    json.dumps(records)
