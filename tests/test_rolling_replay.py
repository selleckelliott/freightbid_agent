"""Tests for the rolling-replay loop: determinism, divergence, idle, and
one-shot reconciliation with the real profit-aware planner."""
from datetime import datetime, timedelta, timezone
from pathlib import Path

from adapters.inbound.api.container import build_container
from application.ortools_profit_aware_planner import ORToolsProfitAwarePlanner
from domain.models.plan import Plan, PlanStop
from domain.models.truck_state import TruckState
from ml.data.load_history_schema import LoadSnapshotRecord
from simulation.metrics import summarize_episodes
from simulation.rolling_replay import ReplayConfig, run_episode
from simulation.snapshot_board import SnapshotBoard

ROOT = Path(__file__).resolve().parents[1]
T0 = datetime(2026, 1, 1, 8, 0, tzinfo=timezone.utc)
DENVER = (39.74, -104.99)
DALLAS = (32.78, -96.80)
LA = (34.05, -118.24)


def make_truck(origin=DENVER, available_at=T0, trailer="Reefer"):
    lat, lon = origin
    return TruckState(
        truck_id=1, current_city="Denver", current_state="CO",
        latitude=lat, longitude=lon, available_at=available_at,
        trailer_type=trailer, max_load_capacity=50000.0, current_load_id=None,
        home_city="Denver", home_state="CO", remaining_capacity=50000.0,
        driver_hours_left=11.0, speed=50.0, heading=0.0, timestamp=available_at,
    )


def make_record(seq, snapshot_time, origin, dest, equipment="HS",
                rate_per_mile=3.0, miles=300.0, pickup_offset_h=4.0,
                pickup_len_h=6.0):
    o_lat, o_lon = origin
    d_lat, d_lon = dest
    pickup_start = snapshot_time + timedelta(hours=pickup_offset_h)
    pickup_end = pickup_start + timedelta(hours=pickup_len_h)
    dropoff_start = pickup_end + timedelta(hours=miles / 50.0)
    dropoff_end = dropoff_start + timedelta(hours=2.0)
    return LoadSnapshotRecord(
        snapshot_time=snapshot_time, load_id=f"L-{seq:06d}",
        origin_city="O", origin_state="CO", origin_lat=o_lat, origin_lon=o_lon,
        destination_city="D", destination_state="TX",
        destination_lat=d_lat, destination_lon=d_lon,
        pickup_start=pickup_start, pickup_end=pickup_end,
        dropoff_start=dropoff_start, dropoff_end=dropoff_end,
        equipment_type=equipment, loaded_miles=miles,
        posted_at=snapshot_time - timedelta(hours=1.0),
        total_rate=round(miles * rate_per_mile, 2),
    )


class StubPlanner:
    """Deterministic planner: picks loads[0] (or the max rate-per-mile load)."""

    def __init__(self, strategy="first"):
        self.strategy = strategy

    def build_plan(self, loads, truck_state, plan_id=1):
        if not loads:
            return Plan(plan_id=plan_id, truck_id=truck_state.truck_id,
                        horizon_hours=48.0)
        if self.strategy == "rpm":
            chosen = max(loads, key=lambda l: l.rate_per_mile)
        else:
            chosen = loads[0]
        stop = PlanStop(
            load_id=chosen.load_id, pickup_eta=chosen.pickup_window_start,
            delivery_eta=chosen.delivery_window_end, deadhead_miles=10.0,
            load_miles=chosen.miles, revenue=chosen.total_rate,
            cost=round(chosen.total_rate * 0.6, 2),
            profit=round(chosen.total_rate * 0.4, 2),
        )
        return Plan(plan_id=plan_id, truck_id=truck_state.truck_id,
                    horizon_hours=48.0, stops=[stop])


CFG = ReplayConfig(radius_mi=300.0)


class NeverPlanner:
    """Declines every board (always returns an empty plan)."""

    def build_plan(self, loads, truck_state, plan_id=1):
        return Plan(plan_id=plan_id, truck_id=truck_state.truck_id, horizon_hours=48.0)


def test_shared_world_determinism():
    records = [
        make_record(1, T0, DENVER, DALLAS),
        make_record(2, T0 + timedelta(hours=10), DALLAS, DENVER),
    ]
    end = T0 + timedelta(days=2)
    ep1 = run_episode(StubPlanner(), SnapshotBoard(records), make_truck(),
                      episode_end=end, ml_equipment="HS", config=CFG)
    ep2 = run_episode(StubPlanner(), SnapshotBoard(records), make_truck(),
                      episode_end=end, ml_equipment="HS", config=CFG)
    ids1 = [d.selected_load_id for d in ep1.decisions]
    ids2 = [d.selected_load_id for d in ep2.decisions]
    assert ids1 == ids2
    assert ep1.metrics.total_profit == ep2.metrics.total_profit


def test_realized_onward_chains_consecutive_loads():
    records = [
        make_record(1, T0, DENVER, DALLAS, miles=300.0),
        make_record(2, T0 + timedelta(hours=10), DALLAS, (29.76, -95.37),
                    miles=240.0, pickup_offset_h=16.0, pickup_len_h=10.0),
    ]
    ep = run_episode(StubPlanner(), SnapshotBoard(records), make_truck(),
                     episode_end=T0 + timedelta(days=2),
                     ml_equipment="HS", config=CFG)
    taken = [d for d in ep.decisions if d.selected_load_id is not None]
    assert [d.selected_load_id for d in taken] == [1, 2]
    # realized onward of load 1 == the deadhead actually driven to reach load 2
    assert taken[0].realized_onward == taken[1].deadhead_miles
    assert taken[1].realized_onward is None  # final load is censored


def test_idle_when_no_candidates_in_radius():
    records = [
        make_record(1, T0, LA, DALLAS),                       # far from Denver
        make_record(2, T0 + timedelta(hours=3), LA, DALLAS),  # still far
    ]
    ep = run_episode(StubPlanner(), SnapshotBoard(records), make_truck(),
                     episode_end=T0 + timedelta(days=1),
                     ml_equipment="HS", config=CFG)
    assert ep.metrics.loads_completed == 0
    assert ep.metrics.decision_count == 0
    assert ep.metrics.idle_hours == 3.0


def test_shadow_planner_records_divergence():
    # loads[0] has the LOWER rate-per-mile, so "first" and "rpm" disagree.
    records = [
        make_record(1, T0, DENVER, DALLAS, rate_per_mile=2.0),
        make_record(2, T0, (39.6, -104.8), DALLAS, rate_per_mile=4.0),
    ]
    ep = run_episode(
        StubPlanner("first"), SnapshotBoard(records), make_truck(),
        episode_end=T0 + timedelta(hours=6), ml_equipment="HS", config=CFG,
        shadow_planner=StubPlanner("rpm"),
    )
    first = ep.decisions[0]
    assert first.selected_load_id == 1
    assert first.shadow_selected_load_id == 2
    assert first.agreement is False
    assert ep.metrics.shadow_decision_count >= 1
    assert ep.metrics.decision_overlap_rate < 1.0

    summary = summarize_episodes([ep.metrics])
    assert summary["divergence"]["divergence_rate"] > 0.0


def test_both_skip_is_excluded_from_divergence():
    # A non-empty board both planners decline is a forced idle, not a policy
    # choice, so it must not count toward the overlap denominator.
    records = [make_record(1, T0, DENVER, DALLAS)]
    ep = run_episode(
        NeverPlanner(), SnapshotBoard(records), make_truck(),
        episode_end=T0 + timedelta(hours=6), ml_equipment="HS", config=CFG,
        shadow_planner=NeverPlanner(),
    )
    assert ep.decisions and ep.decisions[0].selected_load_id is None
    assert ep.decisions[0].agreement is None
    assert ep.metrics.shadow_decision_count == 0
    assert ep.metrics.decision_overlap_rate is None


def test_single_decision_reconciles_with_one_shot_planner():
    container = build_container(ROOT / "config")
    planner = ORToolsProfitAwarePlanner(
        distance_provider=container.evaluator.distance_provider,
        evaluate_loads_service=container.evaluator,
        constraints=container.config.planning_constraints,
        objective_weights=container.config.ortools_objective_weights,
        solver_time_limit_seconds=1.0,
        average_speed_mph=container.config.average_speed_mph,
        load_unload_hours=container.config.planning_constraints.average_load_unload_hours,
    )
    # one clearly-profitable load originating at the truck (≈0 deadhead)
    rec = make_record(1, T0, DENVER, DALLAS, rate_per_mile=3.0, miles=300.0)
    truck = make_truck(origin=DENVER, trailer="Reefer")

    one_shot_board = SnapshotBoard([rec])
    candidates = one_shot_board.visible_loads_at(T0, *DENVER, 300.0, "HS")
    one_shot_plan = planner.build_plan(candidates, truck)
    assert one_shot_plan.stops, "expected the profit-aware planner to take the load"
    stop = one_shot_plan.stops[0]

    ep = run_episode(planner, SnapshotBoard([rec]), truck,
                     episode_end=T0 + timedelta(days=2),
                     ml_equipment="HS", config=ReplayConfig(radius_mi=300.0))
    taken = [d for d in ep.decisions if d.selected_load_id is not None]
    assert len(taken) == 1
    assert taken[0].selected_load_id == stop.load_id
    assert taken[0].actual_profit == stop.profit
    assert taken[0].deadhead_miles == stop.deadhead_miles
    assert ep.metrics.total_profit == stop.profit
