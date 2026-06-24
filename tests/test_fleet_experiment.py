"""Phase 8.3 - fleet dispatch A/B experiment harness tests."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from simulation.experiment import ConditionSpec, WorldDefaults
from simulation.fleet_experiment import (
    FLEET_KEY,
    GREEDY_KEY,
    build_fleet,
    run_fleet_condition,
)

BASE = WorldDefaults(
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    snapshots_per_day=8,
    loads_per_snapshot_mean=30,
    unposted_rate_fraction=0.15,
    max_post_age_hours=12.0,
    horizon_days=2,
    radius_mi=400.0,
    daily_drive_hours=11.0,
)


def _run(container, spec, *, fleet_size=3, episodes=2, base_seed=4000):
    return run_fleet_condition(
        spec,
        container=container,
        base=BASE,
        fleet_size=fleet_size,
        episode_count=episodes,
        base_seed=base_seed,
        solver_time_limit=0.3,
    )


def test_build_fleet_assigns_distinct_ids(container):
    import random

    trucks = build_fleet(
        random.Random(7),
        fleet_size=4,
        start_time=BASE.start_date,
        daily_drive_hours=11.0,
        force_equipment=None,
    )
    assert [t.truck_id for t in trucks] == [1, 2, 3, 4]
    # Heterogeneous: trailers are sampled per-truck (not all forced equal).
    assert len({t.truck_id for t in trucks}) == 4


def test_build_fleet_homogeneous_pins_equipment(container):
    import random

    trucks = build_fleet(
        random.Random(7),
        fleet_size=4,
        start_time=BASE.start_date,
        daily_drive_hours=11.0,
        force_equipment="HS",
    )
    # All trucks pinned to one trailer -> they contend for the same equipment market.
    assert len({t.trailer_type for t in trucks}) == 1


def test_condition_structure_and_verdict(container):
    spec = ConditionSpec(name="heterogeneous_baseline", rationale="headline")
    result = _run(container, spec)
    d = result.to_dict()

    assert d["name"] == "heterogeneous_baseline"
    assert d["fleet_size"] == 3
    assert d["homogeneous"] is False
    assert d["verdict"] in {"HOLDS", "NEUTRAL", "REGRESSION"}
    for arm in (GREEDY_KEY, FLEET_KEY):
        assert d[arm]["episode_count"] == 2
        assert "total_profit" in d[arm]["metrics"]
        assert len(d["per_episode"][arm]) == 2
    # Paired deltas are present and well-formed.
    for key in ("mean", "ci_low", "ci_high"):
        assert key in d["paired_profit"]
        assert key in d["paired_deadhead"]
    assert "profit_pct" in d["headline_delta"]


def test_condition_is_deterministic(container):
    spec = ConditionSpec(name="det", rationale="determinism")
    first = _run(container, spec).to_dict()
    second = _run(container, spec).to_dict()
    assert first == second


def test_homogeneous_flag_set_when_equipment_pinned(container):
    spec = ConditionSpec(
        name="homogeneous_contention", rationale="stress", force_equipment="HS"
    )
    result = _run(container, spec)
    assert result.homogeneous is True
    assert result.effective["force_equipment"] == "HS"


def test_k1_fleet_aware_equals_greedy(container):
    """With one truck there is no contention, so coordination is a no-op:
    the global assignment must reproduce the greedy trajectory exactly."""
    spec = ConditionSpec(name="k1", rationale="single-truck invariant")
    result = _run(container, spec, fleet_size=1, episodes=3, base_seed=5000)

    greedy_rows = result.per_episode[GREEDY_KEY]
    fleet_rows = result.per_episode[FLEET_KEY]
    assert len(greedy_rows) == len(fleet_rows) == 3
    for g, f in zip(greedy_rows, fleet_rows):
        assert g["contention_events"] == 0
        assert f["contention_events"] == 0
        assert g["total_profit"] == pytest.approx(f["total_profit"])
        assert g["total_deadhead_miles"] == pytest.approx(f["total_deadhead_miles"])
        assert g["loads_completed"] == f["loads_completed"]


def test_cost_override_changes_outcome(container):
    """A fuel-cost override must rebuild the scorer and move profit vs baseline."""
    baseline = _run(container, ConditionSpec(name="b", rationale="x"))
    expensive = _run(
        container,
        ConditionSpec(name="e", rationale="x", fuel_cost_per_mile=0.95),
        base_seed=4000,
    )
    base_profit = baseline.greedy["metrics"]["total_profit"]["mean"]
    exp_profit = expensive.greedy["metrics"]["total_profit"]["mean"]
    assert exp_profit < base_profit
