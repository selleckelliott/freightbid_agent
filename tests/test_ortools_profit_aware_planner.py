"""Deterministic micro-tests for ``ORToolsProfitAwarePlanner`` (Phase 2.2).

Loads use UT<->UT lanes (zero flat-rate tolls), so with the default cost
model a load's static profit is exactly::

    rate - 1.39 * miles - 49.50

(1.39 $/mi = fuel 0.55 + maintenance 0.18 + (28 + 5) driver/opportunity
$/h at 50 mph; 49.50 = 1.5 load/unload hours x 33 $/h.)
"""
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

from adapters.outbound.distance.haversine import HaversineDistanceProvider
from adapters.outbound.tolls.flat_rate import FlatRateTollEstimator
from application.config_loader import load_config
from application.evaluate_loads import EvaluateLoadsService
from application.ortools_profit_aware_planner import ORToolsProfitAwarePlanner

from .test_ortools_distance_planner import (
    BASE,
    BOISE,
    RENO,
    SALT_LAKE_CITY,
    _load,
    _make_planner,
    _truck,
)

ROOT = Path(__file__).resolve().parents[1]

# ~150 mi due south of Salt Lake City; ~100 mi further south again.
SOUTH_UT_ORIGIN = (38.59, -111.8910)
SOUTH_UT_DEST = (37.14, -111.8910)


@pytest.fixture(scope="module")
def config():
    return load_config(ROOT / "config")


def _make_profit_planner(config, time_limit=0.3, weights=None):
    evaluator = EvaluateLoadsService(
        distance_provider=HaversineDistanceProvider(),
        toll_estimator=FlatRateTollEstimator(),
        cost_model=config.cost_model,
        average_speed_mph=config.average_speed_mph,
        load_unload_hours=config.planning_constraints.average_load_unload_hours,
    )
    return ORToolsProfitAwarePlanner(
        distance_provider=HaversineDistanceProvider(),
        evaluate_loads_service=evaluator,
        constraints=config.planning_constraints,
        objective_weights=weights or config.ortools_objective_weights,
        solver_time_limit_seconds=time_limit,
        average_speed_mph=config.average_speed_mph,
        load_unload_hours=config.planning_constraints.average_load_unload_hours,
    )


def test_skips_load_not_worth_its_deadhead(config):
    """A feasible but marginal load is skipped when reaching it costs more
    than its profit; the at-the-truck high-profit load is served."""
    planner = _make_profit_planner(config)
    good = _load(1, SALT_LAKE_CITY, BOISE, miles=240, rate=850)  # profit ~$467
    # Profit ~$81 (>= $50 prefilter floor) but 150 deadhead miles cost
    # ~$208 to reach: serving it would destroy ~$127 of plan profit.
    marginal = _load(2, SOUTH_UT_ORIGIN, SOUTH_UT_DEST, miles=100, rate=270)

    plan = planner.build_plan([good, marginal], _truck(driver_hours_left=16.0))

    assert plan.feasible
    assert [s.load_id for s in plan.stops] == [1]
    assert plan.expected_profit > 0


def test_pickup_before_delivery_for_chained_loads(config):
    planner = _make_profit_planner(config)
    loads = [
        _load(1, SALT_LAKE_CITY, BOISE, miles=160, rate=620),
        _load(2, BOISE, RENO, miles=160, rate=620),
    ]

    plan = planner.build_plan(loads, _truck(driver_hours_left=14.0))

    assert plan.feasible
    assert [s.load_id for s in plan.stops] == [1, 2]
    for stop in plan.stops:
        assert stop.pickup_eta <= stop.delivery_eta


def test_excludes_load_outside_time_window(config):
    planner = _make_profit_planner(config)
    load = _load(1, SALT_LAKE_CITY, BOISE, miles=240, rate=850)
    expired = replace(
        load,
        pickup_window_start=BASE - timedelta(hours=4),
        pickup_window_end=BASE - timedelta(hours=1),
    )

    plan = planner.build_plan([expired], _truck())

    assert not plan.feasible
    assert plan.stops == []
    assert "Profit-Aware" in plan.rationale


def test_profit_aware_beats_distance_planner_when_hos_forces_a_choice(config):
    """Ablation: an 8h HOS budget fits only one load. The distance planner
    takes the shorter low-profit load; the profit-aware planner takes the
    longer high-profit one."""
    truck = _truck(driver_hours_left=8.0)
    short_low_profit = _load(1, SALT_LAKE_CITY, SOUTH_UT_DEST, miles=150, rate=318)  # ~$60
    long_high_profit = _load(2, SALT_LAKE_CITY, BOISE, miles=300, rate=970)  # ~$503
    loads = [short_low_profit, long_high_profit]

    distance_plan = _make_planner(config).build_plan(loads, truck)
    profit_plan = _make_profit_planner(config).build_plan(loads, truck)

    assert [s.load_id for s in distance_plan.stops] == [1]
    assert [s.load_id for s in profit_plan.stops] == [2]
    assert profit_plan.expected_profit > distance_plan.expected_profit
    assert "Profit-Aware" in profit_plan.rationale


def test_empty_load_list_returns_empty_valid_plan(config):
    planner = _make_profit_planner(config)

    plan = planner.build_plan([], _truck())

    assert not plan.feasible
    assert plan.stops == []
    assert plan.expected_profit == 0.0


def test_skip_profit_floor_below_business_floor_raises(config):
    """An objective floor under min_expected_profit would reward serving
    loads the replay rejects — the planner refuses the mis-calibration."""
    bad = replace(config.ortools_objective_weights, skip_profit_floor_dollars=25.0)

    with pytest.raises(ValueError, match="min_expected_profit"):
        _make_profit_planner(config, weights=bad)


def test_higher_skip_profit_floor_makes_solver_pickier(config):
    """Phase 2.3 pickiness knob: a load worth serving under the business
    floor ($50) becomes skippable when the objective floor is raised to
    $100, with business rules untouched."""
    # ~20 mi south of the truck (deadhead cost ~$28); static profit
    # = 308.50 - 1.39*100 - 49.50 = $120, replayed profit ~$92 (feasible).
    nearby_origin = (40.471, -111.8910)
    load = _load(1, nearby_origin, SOUTH_UT_DEST, miles=100, rate=308.50)
    truck = _truck()

    default_plan = _make_profit_planner(config).build_plan([load], truck)
    # Margin over $50 floor = $70 > $28 deadhead -> serve.
    assert [s.load_id for s in default_plan.stops] == [1]

    picky = replace(
        config.ortools_objective_weights, skip_profit_floor_dollars=100.0
    )
    picky_plan = _make_profit_planner(config, weights=picky).build_plan([load], truck)
    # Margin over $100 floor = $20 < $28 deadhead -> free to skip.
    assert picky_plan.stops == []
    assert not picky_plan.feasible
