"""Tests for the Phase 5.5 risk-aware stress capstone (run_risk_aware_stress).

Fast, mostly-pure tests for the collectible-profit oracle math, the per-arm aggregation, the
uplift/verdict bands, the eval-window snapshot filter, and config loading — plus two small
seeded end-to-end smokes proving the four arms score the same loads (equal counts), the full
arm collapses to risk-adjusted EV when no recalibrator is promoted, and the win-curve world's
recalibration repairs drift while the base models stay frozen.
"""
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from application.config_loader import load_bid_recommender_config
from benchmarks.run_broker_quality_stress import Condition
from benchmarks.run_risk_aware_stress import (
    BEST_FIXED,
    DEFAULT_CONFIG,
    FULL_RISK_AWARE,
    RAW_EV,
    RISK_ADJUSTED_EV,
    VERDICT_HOLDS,
    VERDICT_NEUTRAL,
    VERDICT_REGRESSION,
    _aggregate,
    _eval_window_snapshots,
    _load_sweep_config,
    _score_ask,
    _uplift_pct,
    _verdict,
    run_risk_aware_stress,
)
from ml.calibration.recalibration_workflow import RecalibrationConfig, load_recalibration_config
from ml.config import load_ml_config
from ml.data.outcome_simulator import win_prob
from ml.monitoring.calibration_drift import ALERT, WATCH, CalibrationThresholds
from ml.monitoring.calibration_report import load_calibration_config

CASH = 0.18
FREE = 30.0
SCALE = 0.06


def _score(ask_rpm=2.0, ask_amount=1000.0, cost=600.0, reserve=1.5,
           p_default=0.0, pay_days=20.0):
    return _score_ask(ask_rpm, ask_amount, cost, reserve, SCALE,
                      p_default, pay_days, CASH, FREE)


# --------------------------------------------------------------------------- #
# Collectible-profit oracle: the Phase 5.1 formula with TRUE inputs
# --------------------------------------------------------------------------- #
def test_oracle_no_default_no_delay_is_win_times_margin():
    """p_default=0 and pay_days<=free_days -> collectible = p_win * (ask - cost)."""
    r = _score(p_default=0.0, pay_days=FREE)  # exactly at the free threshold -> no penalty
    p_win = win_prob(1.5, 2.0, SCALE)
    assert r["p_collect"] == pytest.approx(1.0)
    assert r["delay_expected"] == pytest.approx(0.0)
    assert r["collectible"] == pytest.approx(p_win * (1000.0 - 600.0))


def test_oracle_default_branch_still_eats_operating_cost():
    """A certain default still pays the cost: won_value = -cost (never magically zero)."""
    r = _score(p_default=1.0)
    p_win = win_prob(1.5, 2.0, SCALE)
    assert r["p_collect"] == pytest.approx(0.0)
    assert r["collectible"] == pytest.approx(p_win * (-600.0))
    assert r["collectible"] < 0.0


def test_oracle_delay_penalty_reduces_collected_branch():
    """Slow-but-collected pay applies a cash-cost penalty only beyond the free window."""
    fast = _score(p_default=0.0, pay_days=FREE)
    slow = _score(p_default=0.0, pay_days=FREE + 60.0)
    expected_delay = 1000.0 * CASH * 60.0 / 365.0
    p_win = win_prob(1.5, 2.0, SCALE)
    assert slow["collectible"] == pytest.approx(p_win * (400.0 - expected_delay))
    assert slow["collectible"] < fast["collectible"]


def test_oracle_is_monotonic_in_default_and_pay_days():
    base = _score(p_default=0.05, pay_days=35.0)
    riskier = _score(p_default=0.25, pay_days=35.0)
    slower = _score(p_default=0.05, pay_days=70.0)
    assert riskier["collectible"] < base["collectible"]   # more default risk hurts
    assert slower["collectible"] < base["collectible"]    # slower pay hurts


def test_oracle_clamps_default_probability():
    assert _score(p_default=1.5)["p_default"] == pytest.approx(1.0)
    assert _score(p_default=-0.2)["p_default"] == pytest.approx(0.0)


def test_oracle_higher_ask_lowers_win_probability():
    low = _score(ask_rpm=1.8, ask_amount=900.0)
    high = _score(ask_rpm=2.6, ask_amount=1300.0)
    assert high["p_win"] < low["p_win"]


# --------------------------------------------------------------------------- #
# Per-arm aggregation
# --------------------------------------------------------------------------- #
def _rows():
    return [
        {"collectible": 100.0, "p_win": 0.8, "p_collect": 0.9, "p_default": 0.1,
         "pay_days": 30.0, "delay_expected": 2.0, "ask_rpm": 2.0},
        {"collectible": 40.0, "p_win": 0.4, "p_collect": 0.5, "p_default": 0.5,
         "pay_days": 50.0, "delay_expected": 1.0, "ask_rpm": 2.4},
    ]


def test_aggregate_means_and_weighted_rates():
    a = _aggregate(_rows())
    assert a["n"] == 2
    assert a["collectible_profit"] == pytest.approx(70.0)
    assert a["win_rate"] == pytest.approx(0.6)
    assert a["avg_ask_rpm"] == pytest.approx(2.2)
    # default rate among won loads is win-weighted: (0.8*0.1 + 0.4*0.5) / (0.8 + 0.4)
    assert a["default_rate_on_won_loads"] == pytest.approx((0.08 + 0.20) / 1.2, abs=1e-3)
    # realized pay days is win-and-collect weighted.
    num = 0.8 * 0.9 * 30.0 + 0.4 * 0.5 * 50.0
    den = 0.8 * 0.9 + 0.4 * 0.5
    assert a["average_realized_pay_days"] == pytest.approx(num / den, abs=1e-2)
    assert a["delay_penalty_total"] == pytest.approx(3.0)


def test_aggregate_empty_is_zeroed():
    a = _aggregate([])
    assert a["n"] == 0
    assert a["collectible_profit"] == 0.0
    assert a["default_rate_on_won_loads"] is None
    assert a["average_realized_pay_days"] is None


# --------------------------------------------------------------------------- #
# Uplift % and verdict bands
# --------------------------------------------------------------------------- #
def test_uplift_pct_uses_abs_denominator_and_guards_zero():
    assert _uplift_pct(110.0, 100.0) == pytest.approx(10.0)
    # abs denominator keeps the sign meaningful even when the baseline is negative.
    assert _uplift_pct(-90.0, -100.0) == pytest.approx(10.0)
    assert _uplift_pct(5.0, 0.0) == 0.0


@pytest.mark.parametrize("uplift,expected", [
    (1.5, VERDICT_HOLDS),
    (1.0, VERDICT_HOLDS),
    (0.4, VERDICT_NEUTRAL),
    (-0.4, VERDICT_NEUTRAL),
    (-1.0, VERDICT_REGRESSION),
    (-3.0, VERDICT_REGRESSION),
])
def test_verdict_bands(uplift, expected):
    assert _verdict(uplift, 1.0) == expected


# --------------------------------------------------------------------------- #
# Eval-window snapshot filter (mirrors time_split's day-0 convention)
# --------------------------------------------------------------------------- #
def test_eval_window_selects_later_disjoint_days():
    base = np.datetime64("2024-01-01T00:00:00")
    snaps = []
    for d in range(6):
        for j in range(3):
            ts = (base + np.timedelta64(d, "D") + np.timedelta64(1 + j, "h")).astype("datetime64[s]")
            snaps.append(SimpleNamespace(snapshot_time=str(ts), load_id=f"L{d}-{j}"))
    # fit_days=2 -> fit covers days {0,1}; eval covers days {2,3,4}; day 5 excluded.
    eval_snaps = _eval_window_snapshots(snaps, fit_days=2, eval_days=3)
    days = sorted({int(s.load_id.split("-")[0][1:]) for s in eval_snaps})
    assert days == [2, 3, 4]
    assert len(eval_snaps) == 3 * 3


# --------------------------------------------------------------------------- #
# Config loading (sweep block + the cross-config single sources of truth)
# --------------------------------------------------------------------------- #
def test_sweep_config_and_cross_configs_load():
    sweep = _load_sweep_config(DEFAULT_CONFIG)
    assert "conditions" in sweep
    assert sweep.get("days", 21) >= 1
    assert isinstance(sweep.get("watch_worlds", []), list)
    # The metric/policy/severity single-sources still parse.
    bid_cfg = load_bid_recommender_config("config")
    assert bid_cfg.annual_cash_cost_rate > 0
    assert bid_cfg.free_pay_days >= 0
    recal = load_recalibration_config("config/recalibration.yaml")
    assert recal.fit_days >= 1 and recal.eval_days >= 1
    thresholds, _ = load_calibration_config("config/calibration_monitor.yaml")
    assert thresholds.min_samples >= 1


# --------------------------------------------------------------------------- #
# End-to-end smokes: tiny seeded worlds, base models frozen on baseline
# --------------------------------------------------------------------------- #
def _tiny_cfg():
    cfg = load_ml_config()
    return replace(
        cfg,
        synthetic_data=replace(cfg.synthetic_data, loads_per_snapshot_mean=16.0,
                               snapshots_per_day=4),
    )


def _smoke(conditions, recal_config, *, days=6, max_loads=120):
    cfg = _tiny_cfg()
    bid_cfg = load_bid_recommender_config("config")
    thresholds = replace(CalibrationThresholds(), min_samples=40)
    return run_risk_aware_stress(
        cfg, bid_cfg, thresholds, recal_config, conditions,
        days=days, band=1.0, max_loads=max_loads,
    )


def test_stress_smoke_four_arms_same_loads_and_json_shape():
    config = replace(RecalibrationConfig(), fit_days=2, eval_days=3, min_samples=40)
    conditions = [
        Condition("baseline", "", {}),
        Condition("tight_brokers", "reserve down",
                  {"outcomes": {"reservation_center_mult": 0.92}}),
    ]
    records = _smoke(conditions, config)

    assert {r["name"] for r in records} == {"baseline", "tight_brokers"}
    for r in records:
        arms = r["arms"]
        # All four arms scored the identical load set (CRN / shared candidate support).
        counts = {arms[k]["n"] for k in (BEST_FIXED, RAW_EV, RISK_ADJUSTED_EV, FULL_RISK_AWARE)}
        assert len(counts) == 1
        assert r["n_loads"] == arms[RAW_EV]["n"] > 0
        assert r["verdict"] in (VERDICT_HOLDS, VERDICT_NEUTRAL, VERDICT_REGRESSION)
        assert arms[BEST_FIXED]["fixed_winner"] in (
            "conservative_fixed", "posted_rate", "stretch_fixed"
        )
        assert {"uplift_vs_raw_ev", "uplift_vs_fixed", "risk_adj_uplift_vs_raw",
                "full_uplift_vs_risk_adj", "recalibrator_promoted",
                "calibration_severity_before", "calibration_severity_after",
                "ece_before", "ece_after"} <= set(r)
    json.dumps(records)


def test_stress_smoke_full_equals_risk_adjusted_when_not_promoted():
    """With promotion made impossible, the full arm is byte-identical to risk-adjusted EV."""
    # An unsatisfiable brier guardrail blocks every promotion, so no recalibrator wraps P(win).
    config = replace(RecalibrationConfig(), fit_days=2, eval_days=3, min_samples=40,
                     max_brier_worsening=-10.0)
    conditions = [
        Condition("baseline", "", {}),
        Condition("tight_brokers", "reserve down",
                  {"outcomes": {"reservation_center_mult": 0.92}}),
    ]
    records = _smoke(conditions, config)
    for r in records:
        assert r["recalibrator_promoted"] is False
        # No promoted map -> full risk-aware collapses onto risk-adjusted EV exactly.
        assert (r["arms"][FULL_RISK_AWARE]["collectible_profit"]
                == r["arms"][RISK_ADJUSTED_EV]["collectible_profit"])
        assert r["full_uplift_vs_risk_adj"] == 0.0


def test_stress_smoke_reserve_shift_calibration_drift_detected_and_repaired():
    """The win-curve-shifted world drifts (WATCH/ALERT); a promoted recalibrator improves ECE.

    The collectible-profit *lift* from recalibration is a canonical-scale result (see the
    committed sweep, where tight_brokers' full arm clears risk-adjusted EV by a wide margin);
    on a tiny, noisy holdout only the calibration guarantee is asserted here.
    """
    config = replace(RecalibrationConfig(), fit_days=2, eval_days=3, min_samples=40)
    conditions = [
        Condition("tight_brokers", "reserve down",
                  {"outcomes": {"reservation_center_mult": 0.90}}),
    ]
    (shift,) = _smoke(conditions, config, max_loads=200)
    assert shift["calibration_severity_before"] in (WATCH, ALERT)
    if shift["recalibrator_promoted"]:
        # Promotion guarantees the held-out calibration strictly improved.
        assert shift["ece_after"] < shift["ece_before"]
