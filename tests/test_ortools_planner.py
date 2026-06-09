from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from adapters.outbound.distance.haversine import HaversineDistanceProvider
from adapters.outbound.tolls.flat_rate import FlatRateTollEstimator
from application.config_loader import load_config
from application.evaluate_loads import EvaluateLoadsService
from application.ortools_planner import ORToolsPlanner
from domain.models.load import Load
from domain.models.truck_state import TruckState

ROOT = Path(__file__).resolve().parents[1]

BASE = datetime(2026, 5, 27, 16, 0, tzinfo=timezone.utc)

# Intermountain West coordinates (mirror the scenario generator catalog).
SALT_LAKE_CITY = (40.7608, -111.8910)
BOISE = (43.6150, -116.2023)
LAS_VEGAS = (36.1699, -115.1398)
RENO = (39.5296, -119.8138)


@pytest.fixture(scope="module")
def config():
    return load_config(ROOT / "config")


def _make_planner(config, time_limit=0.3):
    evaluator = EvaluateLoadsService(
        distance_provider=HaversineDistanceProvider(),
        toll_estimator=FlatRateTollEstimator(),
        cost_model=config.cost_model,
        average_speed_mph=config.average_speed_mph,
        load_unload_hours=config.planning_constraints.average_load_unload_hours,
    )
    return ORToolsPlanner(
        distance_provider=HaversineDistanceProvider(),
        evaluate_loads_service=evaluator,
        constraints=config.planning_constraints,
        solver_time_limit_seconds=time_limit,
        average_speed_mph=config.average_speed_mph,
        load_unload_hours=config.planning_constraints.average_load_unload_hours,
    )


def _truck(driver_hours_left=11.0, location=SALT_LAKE_CITY, trailer="Dry Van"):
    return TruckState(
        truck_id=1,
        current_city="Salt Lake City",
        current_state="UT",
        latitude=location[0],
        longitude=location[1],
        available_at=BASE,
        trailer_type=trailer,
        max_load_capacity=45000,
        current_load_id=None,
        home_city="Salt Lake City",
        home_state="UT",
        remaining_capacity=45000,
        driver_hours_left=driver_hours_left,
        speed=0.0,
        heading=0.0,
        timestamp=BASE,
    )


def _load(load_id, origin, dest, miles, rate, equipment="Dry Van", weight=20000):
    return Load(
        load_id=load_id,
        weight=weight,
        created_at=BASE - timedelta(hours=8),
        origin_city="O",
        origin_state="UT",
        origin_latitude=origin[0],
        origin_longitude=origin[1],
        destination_city="D",
        destination_state="UT",
        destination_latitude=dest[0],
        destination_longitude=dest[1],
        pickup_window_start=BASE,
        pickup_window_end=BASE + timedelta(hours=12),
        delivery_window_start=BASE,
        delivery_window_end=BASE + timedelta(hours=24),
        miles=miles,
        total_rate=rate,
        equipment_type=equipment,
    )


def test_single_feasible_load_is_served(config):
    planner = _make_planner(config)
    load = _load(1, SALT_LAKE_CITY, BOISE, miles=240, rate=850)

    plan = planner.build_plan([load], _truck())

    assert plan.feasible
    assert [s.load_id for s in plan.stops] == [1]
    stop = plan.stops[0]
    assert stop.pickup_eta <= stop.delivery_eta
    # Profit accounting mirrors the heuristic (revenue - cost).
    assert abs(stop.profit - (stop.revenue - stop.cost)) < 1e-6
    assert abs(plan.expected_profit - (plan.expected_revenue - plan.expected_cost)) < 1e-6


def test_equipment_mismatch_yields_empty_plan(config):
    planner = _make_planner(config)
    load = _load(1, SALT_LAKE_CITY, BOISE, miles=240, rate=850, equipment="Reefer")

    plan = planner.build_plan([load], _truck(trailer="Dry Van"))

    assert not plan.feasible
    assert plan.stops == []


def test_hos_budget_limits_loads_selected(config):
    # Each load is servable alone (~6.3h) but together exceed a 7h HOS budget.
    planner = _make_planner(config)
    truck = _truck(driver_hours_left=7.0)
    loads = [
        _load(1, SALT_LAKE_CITY, BOISE, miles=240, rate=850),
        _load(2, SALT_LAKE_CITY, LAS_VEGAS, miles=240, rate=850),
    ]

    plan = planner.build_plan(loads, truck)

    assert plan.feasible
    assert len(plan.stops) == 1
    total_driver_hours = sum(
        (s.deadhead_miles + s.load_miles) / config.average_speed_mph
        + config.planning_constraints.average_load_unload_hours
        for s in plan.stops
    )
    assert total_driver_hours <= 7.0 + 1e-6


def test_pickup_before_delivery_for_chained_loads(config):
    # Truck does load 1 (SLC->Boise) then load 2 (Boise->Reno) within a wide budget.
    planner = _make_planner(config)
    loads = [
        _load(1, SALT_LAKE_CITY, BOISE, miles=160, rate=620),
        _load(2, BOISE, RENO, miles=160, rate=620),
    ]

    plan = planner.build_plan(loads, _truck(driver_hours_left=14.0))

    assert plan.feasible
    assert len(plan.stops) == 2
    for stop in plan.stops:
        assert stop.pickup_eta <= stop.delivery_eta


def test_no_candidate_loads_returns_infeasible(config):
    planner = _make_planner(config)
    plan = planner.build_plan([], _truck())
    assert not plan.feasible
    assert plan.stops == []


def test_sample_data_plan_excludes_reefer_and_is_valid(container, sample_loads, sample_truck):
    plan = container.ortools_planner.build_plan(sample_loads, sample_truck)

    assert plan.feasible
    assert len(plan.stops) >= 1
    # Load 4 is a Reefer; the Dry Van truck can never carry it.
    assert 4 not in [s.load_id for s in plan.stops]
    assert abs(plan.expected_profit - (plan.expected_revenue - plan.expected_cost)) < 1e-6
