"""Tests for the Phase 6.5 compiled-vs-orchestrated stress capstone (run_compiled_dispatcher_stress).

Fast, mostly-pure tests for the collectible-profit regret math, the verdict bands (incl. the
safety-critical override), the three safety-critical miss cases, the not-bidding ⇒ zero-collectible
rule, the per-world win-curve scale lookup, and config loading — plus one tiny seeded end-to-end
smoke proving the three systems score the same loads through the shipped 6.4 shadow service, the
compiled model never owns the decision, and the summary is JSON-serializable.
"""
import json
from dataclasses import replace

import pytest

from application.config_loader import load_bid_recommender_config
from benchmarks.run_broker_quality_stress import Condition
from benchmarks.run_compiled_dispatcher_stress import (
    BASELINE,
    COMPILED,
    DEFAULT_CONFIG,
    SOURCE,
    VERDICT_FAIL,
    VERDICT_PASS,
    VERDICT_WATCH,
    _collectible,
    _load_sweep_config,
    _regret_pct,
    _safety_critical,
    _scale_for,
    _verdict,
    run_compiled_dispatcher_stress,
)
from ml.calibration.recalibration_workflow import RecalibrationConfig, load_recalibration_config
from ml.config import load_ml_config
from ml.data.compiled_agent_trace_schema import (
    DECISION_APPROVAL_REQUIRED,
    DECISION_BID,
    DECISION_NO_BID,
)
from ml.monitoring.calibration_drift import CalibrationThresholds
from ml.workflows.freightbid_workflow_graph import (
    WARN_CALIBRATION_ALERT,
    WARN_CALIBRATION_WATCH,
    WARN_NEGATIVE_RISK_ADJUSTED_EV,
    WARN_PAYMENT_RISK,
)

CASH = 0.18
FREE = 30.0
SCALE = 0.06


# --------------------------------------------------------------------------- #
# Collectible-profit regret (signed; positive ⇒ the arm is worse than source)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("src,arm,expected", [
    (100.0, 90.0, 10.0),     # arm leaves 10% on the table
    (100.0, 110.0, -10.0),   # arm beats source -> negative regret
    (100.0, 100.0, 0.0),     # identical
    (0.5, 0.4, 0.0),         # |source| < $1 -> ratio is noise, call it flat
    (-0.2, 5.0, 0.0),        # tiny source magnitude guard
])
def test_regret_pct(src, arm, expected):
    assert _regret_pct(src, arm) == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# Verdict bands: FAIL on safety-critical miss OR regret > watch; WATCH on minor / mild regret
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("regret,crit,minor,expected", [
    (1.0, 0, 0, VERDICT_PASS),       # within pass band, clean
    (-3.0, 0, 0, VERDICT_PASS),      # compiled better than source
    (3.0, 0, 0, VERDICT_WATCH),      # regret in (pass, watch]
    (0.5, 0, 1, VERDICT_WATCH),      # a minor miss downgrades to watch
    (6.0, 0, 0, VERDICT_FAIL),       # regret beyond watch
    (0.5, 1, 0, VERDICT_FAIL),       # a single safety-critical miss FAILs even at low regret
    (-9.0, 2, 0, VERDICT_FAIL),      # cheaper AND unsafe -> still FAIL (safety dominates)
])
def test_verdict_bands(regret, crit, minor, expected):
    assert _verdict(regret, crit, minor, pass_pct=2.0, watch_pct=5.0) == expected


# --------------------------------------------------------------------------- #
# Safety-critical classification: the three "override caution" cases
# --------------------------------------------------------------------------- #
def test_safety_critical_takes_a_load_source_rejected():
    assert _safety_critical(DECISION_NO_BID, DECISION_BID, set(), set(), risky=False) is True
    # approval_required still places a bid on the board -> also a "take rejected" miss.
    assert _safety_critical(
        DECISION_NO_BID, DECISION_APPROVAL_REQUIRED, set(), set(), risky=False
    ) is True


def test_safety_critical_auto_bids_what_needed_approval():
    assert _safety_critical(
        DECISION_APPROVAL_REQUIRED, DECISION_BID, set(), set(), risky=False
    ) is True


def test_safety_critical_suppresses_warning_only_in_risky_world():
    src_w, comp_w = {WARN_PAYMENT_RISK}, set()
    assert _safety_critical(DECISION_BID, DECISION_BID, src_w, comp_w, risky=True) is True
    # The same suppression in a calm world is not safety-critical (counts only as a minor miss).
    assert _safety_critical(DECISION_BID, DECISION_BID, src_w, comp_w, risky=False) is False


def test_safety_critical_ignores_noncritical_warning_suppression():
    # A non-critical warning (negative-EV) dropped in a risky world is not safety-critical.
    src_w = {WARN_NEGATIVE_RISK_ADJUSTED_EV}
    assert _safety_critical(DECISION_BID, DECISION_BID, src_w, set(), risky=True) is False


def test_safety_critical_ignores_calibration_watch_suppression():
    # The soft calibration_WATCH tier is NOT safety-critical: the compiled model has no watch
    # head, so it can never reproduce it — counting its suppression as critical would make the
    # metric a structural constant on watch-heavy worlds. It is a minor miss (warning_agreement),
    # not a FAIL-triggering caution override. Only ALERT / payment suppression is critical.
    assert _safety_critical(
        DECISION_BID, DECISION_BID, {WARN_CALIBRATION_WATCH}, set(), risky=True
    ) is False
    # ALERT in the same spot still IS safety-critical (sanity contrast).
    assert _safety_critical(
        DECISION_BID, DECISION_BID, {WARN_CALIBRATION_ALERT}, set(), risky=True
    ) is True


def test_safety_critical_false_when_compiled_more_cautious():
    # Compiled is *more* conservative than source -> never a safety-critical miss.
    assert _safety_critical(DECISION_BID, DECISION_NO_BID, set(), set(), risky=True) is False
    assert _safety_critical(
        DECISION_APPROVAL_REQUIRED, DECISION_APPROVAL_REQUIRED,
        {WARN_CALIBRATION_ALERT}, {WARN_CALIBRATION_ALERT}, risky=True,
    ) is False


# --------------------------------------------------------------------------- #
# Collectible profit of one arm's decision (0 unless it actually bids)
# --------------------------------------------------------------------------- #
def _coll(decision, ask_rpm=2.0, ask_amount=1000.0):
    return _collectible(
        decision, ask_rpm, ask_amount,
        cost=600.0, reserve=1.5, scale=SCALE, p_default=0.0, pay_days=FREE,
        cash_rate=CASH, free_days=FREE,
    )


def test_collectible_zero_when_not_bidding():
    assert _coll(DECISION_NO_BID) == 0.0
    # bidding decisions with a missing ask also collect nothing
    assert _collectible(
        DECISION_BID, None, None, cost=600.0, reserve=1.5, scale=SCALE,
        p_default=0.0, pay_days=FREE, cash_rate=CASH, free_days=FREE,
    ) == 0.0


def test_collectible_positive_when_bidding_above_reserve():
    bid = _coll(DECISION_BID)
    approval = _coll(DECISION_APPROVAL_REQUIRED)  # approval_required still bids
    assert bid > 0.0
    assert approval == pytest.approx(bid)  # same ask -> same collectible


# --------------------------------------------------------------------------- #
# Per-world win-curve scale lookup
# --------------------------------------------------------------------------- #
def test_scale_for_reads_override_else_baseline():
    cfg = load_ml_config()
    base = Condition("baseline", "", {})
    shifted = Condition("sharp", "", {"outcomes": {"win_logistic_scale_rpm": 0.025}})
    assert _scale_for(base, cfg) == pytest.approx(cfg.outcomes.win_logistic_scale_rpm)
    assert _scale_for(shifted, cfg) == pytest.approx(0.025)


# --------------------------------------------------------------------------- #
# Config loading (the sweep block + the cross-config single sources of truth)
# --------------------------------------------------------------------------- #
def test_sweep_config_and_cross_configs_load():
    sweep = _load_sweep_config(DEFAULT_CONFIG)
    assert "conditions" in sweep
    assert sweep.get("days", 21) >= 1
    assert float(sweep.get("regret_pass_pct", 2.0)) <= float(sweep.get("regret_watch_pct", 5.0))
    assert isinstance(sweep.get("risky_worlds", []), list)
    bid_cfg = load_bid_recommender_config("config")
    assert bid_cfg.annual_cash_cost_rate > 0
    recal = load_recalibration_config("config/recalibration.yaml")
    assert recal.fit_days >= 1 and recal.eval_days >= 1


# --------------------------------------------------------------------------- #
# End-to-end smoke: tiny seeded worlds, compiled frozen, scored through the shadow service
# --------------------------------------------------------------------------- #
def _tiny_cfg():
    cfg = load_ml_config()
    return replace(
        cfg,
        synthetic_data=replace(cfg.synthetic_data, loads_per_snapshot_mean=16.0,
                               snapshots_per_day=4),
    )


def test_stress_smoke_three_systems_same_loads_and_safe_shadow():
    cfg = _tiny_cfg()
    bid_cfg = load_bid_recommender_config("config")
    thresholds = replace(CalibrationThresholds(), min_samples=40)
    recal = replace(RecalibrationConfig(), fit_days=2, eval_days=3, min_samples=40)
    conditions = [
        Condition("baseline", "", {}),
        Condition("tight_brokers", "reserve down",
                  {"outcomes": {"reservation_center_mult": 0.92}}),
    ]
    records, prov = run_compiled_dispatcher_stress(
        cfg, bid_cfg, thresholds, recal, conditions,
        train_conditions=conditions, days=6, max_loads=80,
        risky_worlds=["tight_brokers"], pass_pct=2.0, watch_pct=5.0,
    )

    assert {r["name"] for r in records} == {"baseline", "tight_brokers"}
    assert prov["train_rows"] > 0
    assert prov["feature_count"] == 22
    for r in records:
        assert r["verdict"] in (VERDICT_PASS, VERDICT_WATCH, VERDICT_FAIL)
        assert r["n_served"] > 0
        # The compiled model never owns the decision (Phase 6.4 invariant, carried into 6.5).
        assert r["compiled_used_for_decision"] is False
        # Agreement + safety fields are present and well-formed.
        assert 0.0 <= r["action_agreement"] <= 1.0
        assert r["safety_critical_misses"] >= 0
        assert r["safety_critical_rate"] is None or 0.0 <= r["safety_critical_rate"] <= 1.0
        assert 0.0 <= (r["fallback_rate"] or 0.0) <= 1.0
        # A FAIL world must be justified: regret beyond watch OR a safety-critical miss.
        if r["verdict"] == VERDICT_FAIL:
            assert r["regret_pct"] > 5.0 or r["safety_critical_misses"] > 0
    json.dumps({"conditions": records})
