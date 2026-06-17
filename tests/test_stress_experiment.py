"""Tests for the Phase 3.4 stress harness (``simulation.experiment``):
condition determinism under Common Random Numbers, reconciliation of the baseline
condition with the shipped Phase 3.3 result, cost-override effect, equipment
pinning, verdict logic, and the artifact-missing graceful-degradation guard.

Runs use the real profit-aware planner but ``model_path=None`` so they never
depend on the gitignored model artifact; the destination trajectory is simply
absent (DEST_SKIPPED), which is exactly the guard we want to cover.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from adapters.inbound.api.container import build_container
from simulation.experiment import (
    VERDICT_DEST_SKIPPED,
    VERDICT_HOLDS,
    VERDICT_NEUTRAL,
    VERDICT_REGRESSION,
    ConditionSpec,
    WorldDefaults,
    classify_verdict,
    run_condition,
)
from simulation.snapshot_board import ROUND_TRIP_ML_EQUIPMENT

ROOT = Path(__file__).resolve().parents[1]
SHIPPED_SUMMARY = ROOT / "benchmarks" / "rolling_replay_summary.json"

_CONTAINER = None


def _container():
    global _CONTAINER
    if _CONTAINER is None:
        _CONTAINER = build_container(ROOT / "config")
    return _CONTAINER


def _small_world(horizon_days: int = 2) -> WorldDefaults:
    """A short, lean world so condition runs stay fast in the test suite."""
    return WorldDefaults(
        start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
        snapshots_per_day=8,
        loads_per_snapshot_mean=30.0,
        unposted_rate_fraction=0.15,
        max_post_age_hours=12.0,
        horizon_days=horizon_days,
        radius_mi=400.0,
        daily_drive_hours=11.0,
    )


def _profit_series(result, metric="total_profit"):
    return [row[metric] for row in result.per_episode["profit_aware"]]


def test_run_condition_is_deterministic_under_crn():
    base = _small_world()
    kw = dict(
        container=_container(),
        base=base,
        episode_count=2,
        base_seed=1000,
        solver_time_limit=0.1,
        model_path=None,
    )
    r1 = run_condition(ConditionSpec(name="baseline"), **kw)
    r2 = run_condition(ConditionSpec(name="baseline"), **kw)
    assert _profit_series(r1) == _profit_series(r2)
    assert _profit_series(r1, "total_deadhead_miles") == _profit_series(
        r2, "total_deadhead_miles"
    )
    assert [r["seed"] for r in r1.per_episode["profit_aware"]] == [1000, 1001]


@pytest.mark.skipif(
    not SHIPPED_SUMMARY.exists(), reason="shipped 3.3 summary not present"
)
def test_baseline_condition_reconciles_with_shipped_summary():
    """The ``baseline`` condition must reproduce the shipped Phase 3.3 numbers.

    Guards the refactor that put the 3.3 runner and the 3.4 sweep on one engine:
    a regression here means the stress baseline has silently drifted from the
    released rolling-replay result.
    """
    shipped = json.loads(SHIPPED_SUMMARY.read_text(encoding="utf-8"))
    cfg = shipped["config"]
    base = WorldDefaults(
        start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
        snapshots_per_day=int(cfg["snapshots_per_day"]),
        loads_per_snapshot_mean=float(cfg["loads_per_snapshot_mean"]),
        unposted_rate_fraction=0.15,
        max_post_age_hours=12.0,
        horizon_days=int(cfg["horizon_days"]),
        radius_mi=float(cfg["radius_mi"]),
        daily_drive_hours=float(cfg["daily_drive_hours"]),
    )
    res = run_condition(
        ConditionSpec(name="baseline"),
        container=_container(),
        base=base,
        episode_count=1,
        base_seed=int(cfg["base_seed"]),
        solver_time_limit=float(cfg["solver_time_limit_seconds"]),
        model_path=None,
    )
    got = res.per_episode["profit_aware"][0]
    want = shipped["per_episode"]["profit_aware"][0]
    assert got["seed"] == want["seed"]
    assert got["total_profit"] == pytest.approx(want["total_profit"], abs=0.01)
    assert got["total_deadhead_miles"] == pytest.approx(
        want["total_deadhead_miles"], abs=0.01
    )
    assert got["loads_completed"] == want["loads_completed"]


def test_cost_override_rebuilds_economics_and_changes_profit():
    # A richer, slightly longer world so the truck reliably completes loads and
    # the profit comparison is non-degenerate.
    base = WorldDefaults(
        start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
        snapshots_per_day=8,
        loads_per_snapshot_mean=60.0,
        unposted_rate_fraction=0.15,
        max_post_age_hours=12.0,
        horizon_days=3,
        radius_mi=400.0,
        daily_drive_hours=11.0,
    )
    kw = dict(
        container=_container(),
        base=base,
        episode_count=1,
        base_seed=1000,
        solver_time_limit=0.15,
        model_path=None,
    )
    baseline = run_condition(ConditionSpec(name="baseline"), **kw)
    pricey = run_condition(
        ConditionSpec(name="fuel", fuel_cost_per_mile=0.95), **kw
    )

    assert pricey.effective["fuel_cost_per_mile"] == 0.95
    assert (
        baseline.effective["fuel_cost_per_mile"]
        == _container().config.cost_model.fuel_cost_per_mile
    )

    base_row = baseline.per_episode["profit_aware"][0]
    pricey_row = pricey.per_episode["profit_aware"][0]
    assert base_row["loads_completed"] > 0  # the world is non-degenerate
    # Same world seed, but a 0.55 -> 0.95 $/mi fuel jump rebuilds the cost stack
    # (realized evaluator + objective weights), so the outcome must move.
    assert pricey_row["total_profit"] != pytest.approx(
        base_row["total_profit"], abs=1.0
    )


def test_force_equipment_pins_the_trailer():
    base = _small_world()
    pinned = run_condition(
        ConditionSpec(name="hs", force_equipment="HS"),
        container=_container(),
        base=base,
        episode_count=3,
        base_seed=3000,
        solver_time_limit=0.1,
        model_path=None,
    )
    assert {r["ml_equipment"] for r in pinned.per_episode["profit_aware"]} == {"HS"}
    assert pinned.effective["force_equipment"] == "HS"

    free = run_condition(
        ConditionSpec(name="free"),
        container=_container(),
        base=base,
        episode_count=4,
        base_seed=3000,
        solver_time_limit=0.1,
        model_path=None,
    )
    sampled = {r["ml_equipment"] for r in free.per_episode["profit_aware"]}
    assert sampled  # something was sampled
    assert sampled <= set(ROUND_TRIP_ML_EQUIPMENT)


def test_classify_verdict_logic():
    def pp(lo, hi):
        return {"ci_low": lo, "ci_high": hi}

    # Paired profit CI entirely below zero -> regression.
    assert classify_verdict(pp(-50.0, -5.0), -3.0) == VERDICT_REGRESSION
    # Profit no worse (CI low >= 0) and deadhead no worse -> holds.
    assert classify_verdict(pp(5.0, 50.0), -3.0) == VERDICT_HOLDS
    assert classify_verdict(pp(0.0, 50.0), 0.0) == VERDICT_HOLDS  # boundary
    # Profit CI straddles zero -> neutral.
    assert classify_verdict(pp(-5.0, 50.0), -3.0) == VERDICT_NEUTRAL
    # Profit holds but deadhead rises -> neutral (not a clean win).
    assert classify_verdict(pp(5.0, 50.0), 4.0) == VERDICT_NEUTRAL


def test_artifact_missing_degrades_gracefully():
    base = _small_world()
    kw = dict(
        container=_container(),
        base=base,
        episode_count=1,
        base_seed=4000,
        solver_time_limit=0.1,
    )
    none_res = run_condition(ConditionSpec(name="baseline"), model_path=None, **kw)
    assert none_res.verdict == VERDICT_DEST_SKIPPED
    assert none_res.destination_aware is None
    assert none_res.paired_profit is None
    assert none_res.profit_aware is not None

    # A path that does not exist must behave exactly like ``None`` (the planner
    # is artifact-gated on ``.exists()``), not raise.
    missing = run_condition(
        ConditionSpec(name="baseline"),
        model_path=ROOT / "ml" / "artifacts" / "_does_not_exist.joblib",
        **kw,
    )
    assert missing.verdict == VERDICT_DEST_SKIPPED
    assert missing.destination_aware is None
    assert _profit_series(none_res) == _profit_series(missing)
