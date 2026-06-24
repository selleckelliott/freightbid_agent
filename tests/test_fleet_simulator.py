"""Phase 8.2 - fleet rolling-horizon simulator.

Covers shared-board consumption (no double-booking), async truck availability,
per-truck HOS independence, idle accounting, determinism, the artifact-gated risk
hook, and - crucially - the K=1 invariant: with one truck the fleet loop reduces
exactly to the single-truck rolling replay.
"""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from adapters.inbound.api.container import build_container
from application.fleet.assignment_fleet_policy import AssignmentFleetPolicy
from application.fleet.greedy_fleet_policy import GreedyFleetPolicy
from application.fleet.pair_scorer import ProfitPairScorer
from domain.models.plan import Plan
from domain.models.truck_state import TruckState
from ml.data.load_history_schema import LoadSnapshotRecord
from simulation.fleet_simulator import FleetReplayConfig, run_fleet_episode
from simulation.rolling_replay import ReplayConfig, run_episode
from simulation.snapshot_board import SnapshotBoard

ROOT = Path(__file__).resolve().parents[1]
T0 = datetime(2026, 1, 1, 8, 0, tzinfo=timezone.utc)
DENVER = (39.74, -104.99)
DALLAS = (32.78, -96.80)
HOUSTON = (29.76, -95.37)
LA = (34.05, -118.24)

CFG = FleetReplayConfig(radius_mi=300.0)
SINGLE_CFG = ReplayConfig(radius_mi=300.0)


@pytest.fixture(scope="module")
def container():
    return build_container(ROOT / "config")


@pytest.fixture(scope="module")
def scorer(container):
    return ProfitPairScorer(container.evaluator, container.config.planning_constraints)


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def _truck(truck_id, origin=DENVER, trailer="Reefer", available_at=T0, hours=11.0):
    lat, lon = origin
    return TruckState(
        truck_id=truck_id, current_city="C", current_state="CO",
        latitude=lat, longitude=lon, available_at=available_at,
        trailer_type=trailer, max_load_capacity=50000.0, current_load_id=None,
        home_city="C", home_state="CO", remaining_capacity=50000.0,
        driver_hours_left=hours, speed=50.0, heading=0.0, timestamp=available_at,
    )


def _rec(seq, snapshot_time, origin, dest, equipment="HS", rate_per_mile=3.0,
         miles=300.0, pickup_offset_h=4.0, pickup_len_h=6.0, dest_state="TX"):
    o_lat, o_lon = origin
    d_lat, d_lon = dest
    pickup_start = snapshot_time + timedelta(hours=pickup_offset_h)
    pickup_end = pickup_start + timedelta(hours=pickup_len_h)
    dropoff_start = pickup_end + timedelta(hours=miles / 50.0)
    dropoff_end = dropoff_start + timedelta(hours=2.0)
    return LoadSnapshotRecord(
        snapshot_time=snapshot_time, load_id=f"L-{seq:06d}",
        origin_city="O", origin_state="CO", origin_lat=o_lat, origin_lon=o_lon,
        destination_city="D", destination_state=dest_state,
        destination_lat=d_lat, destination_lon=d_lon,
        pickup_start=pickup_start, pickup_end=pickup_end,
        dropoff_start=dropoff_start, dropoff_end=dropoff_end,
        equipment_type=equipment, loaded_miles=miles,
        posted_at=snapshot_time - timedelta(hours=1.0),
        total_rate=round(miles * rate_per_mile, 2),
    )


class _MyopicProfitPlanner:
    """Single-truck reference: pick the max expected-profit feasible load.

    Uses the exact same ``ProfitPairScorer`` and tie-break as ``GreedyFleetPolicy``
    so the single-truck ``run_episode`` trajectory must match the K=1 fleet loop.
    """

    def __init__(self, scorer):
        self._scorer = scorer

    def build_plan(self, loads, truck_state, plan_id=1):
        best = None
        for load in loads:
            sp = self._scorer.score(load, truck_state)
            if sp is None:
                continue
            if (best is None or sp.score > best.score
                    or (sp.score == best.score and sp.load.load_id < best.load.load_id)):
                best = sp
        stops = [best.stop] if best is not None else []
        return Plan(plan_id=plan_id, truck_id=truck_state.truck_id,
                    horizon_hours=48.0, stops=stops)


class _StubRisk:
    def __init__(self, p):
        self._p = p

    def default_probability(self, load):
        return self._p


# --------------------------------------------------------------------------- #
# K=1 invariant
# --------------------------------------------------------------------------- #
def test_k1_matches_single_truck_rolling_loop(scorer):
    """One truck, two chained loads + a forced idle gap: the fleet loop must
    reproduce the single-truck rolling replay exactly."""
    records = [
        _rec(1, T0, DENVER, DALLAS, miles=300.0),
        _rec(2, T0 + timedelta(hours=10), DALLAS, HOUSTON, miles=240.0,
             pickup_offset_h=1.0, pickup_len_h=12.0),
    ]
    truck = _truck(1, DENVER, hours=24.0)
    end = T0 + timedelta(days=2)

    ref = run_episode(
        _MyopicProfitPlanner(scorer), SnapshotBoard(records), truck,
        episode_end=end, ml_equipment="HS", config=SINGLE_CFG,
    )
    fleet = run_fleet_episode(
        GreedyFleetPolicy(scorer), SnapshotBoard(records), [truck],
        episode_end=end, config=CFG,
    )

    ref_ids = [d.selected_load_id for d in ref.decisions if d.selected_load_id]
    fleet_ids = [d.load_id for d in fleet.decisions]
    assert ref_ids == fleet_ids == [1, 2]
    assert fleet.metrics.total_profit == pytest.approx(ref.metrics.total_profit)
    assert fleet.metrics.total_deadhead_miles == pytest.approx(
        ref.metrics.total_deadhead_miles)
    assert fleet.metrics.loads_completed == ref.metrics.loads_completed
    assert fleet.metrics.total_idle_hours == pytest.approx(ref.metrics.idle_hours)
    assert fleet.metrics.truck_metrics[0].idle_hours == pytest.approx(
        ref.metrics.idle_hours)


def test_k1_greedy_equals_assignment(scorer):
    records = [
        _rec(1, T0, DENVER, DALLAS, miles=300.0),
        _rec(2, T0 + timedelta(hours=10), DALLAS, HOUSTON, miles=240.0,
             pickup_offset_h=1.0, pickup_len_h=12.0),
    ]
    truck = _truck(1, DENVER, hours=24.0)
    end = T0 + timedelta(days=2)
    g = run_fleet_episode(GreedyFleetPolicy(scorer), SnapshotBoard(records),
                          [truck], episode_end=end, config=CFG)
    a = run_fleet_episode(AssignmentFleetPolicy(scorer), SnapshotBoard(records),
                          [truck], episode_end=end, config=CFG)
    assert [d.load_id for d in g.decisions] == [d.load_id for d in a.decisions]
    assert g.metrics.total_profit == pytest.approx(a.metrics.total_profit)


# --------------------------------------------------------------------------- #
# Shared-board consumption
# --------------------------------------------------------------------------- #
def test_no_double_booking(scorer):
    """One load, two co-located trucks: it is executed exactly once."""
    records = [_rec(1, T0, DENVER, DALLAS)]
    trucks = [_truck(1, DENVER), _truck(2, DENVER)]
    ep = run_fleet_episode(GreedyFleetPolicy(scorer), SnapshotBoard(records),
                           trucks, episode_end=T0 + timedelta(hours=12), config=CFG)
    assert ep.metrics.loads_completed == 1
    assert [d.load_id for d in ep.decisions] == [1]
    assert ep.decisions[0].truck_id == 1  # lower id wins the greedy tie


def test_assignment_serves_two_trucks_two_loads(scorer):
    records = [
        _rec(1, T0, DENVER, DALLAS, dest_state="TX"),
        _rec(2, T0, DENVER, LA, dest_state="CA", miles=200.0),
    ]
    trucks = [_truck(1, DENVER), _truck(2, DENVER)]
    ep = run_fleet_episode(AssignmentFleetPolicy(scorer), SnapshotBoard(records),
                           trucks, episode_end=T0 + timedelta(hours=12), config=CFG)
    assert ep.metrics.loads_completed == 2
    assert {d.load_id for d in ep.decisions} == {1, 2}
    # two different drop-off markets -> diversified (HHI = 0.5)
    assert ep.metrics.destination_hhi == 0.5
    assert ep.metrics.contention_events >= 1  # both trucks saw both loads


# --------------------------------------------------------------------------- #
# Async availability
# --------------------------------------------------------------------------- #
def test_async_truck_availability(scorer):
    """A truck that is not free until later cannot be dispatched before then."""
    records = [
        _rec(1, T0, DENVER, DALLAS),
        _rec(2, T0, DENVER, HOUSTON, miles=240.0),
    ]
    trucks = [
        _truck(1, DENVER, available_at=T0),
        _truck(2, DENVER, available_at=T0 + timedelta(hours=6)),
    ]
    ep = run_fleet_episode(AssignmentFleetPolicy(scorer), SnapshotBoard(records),
                           trucks, episode_end=T0 + timedelta(hours=12), config=CFG)
    truck2_decisions = [d for d in ep.decisions if d.truck_id == 2]
    assert truck2_decisions  # truck 2 eventually works
    assert all(d.decision_time >= T0 + timedelta(hours=6) for d in truck2_decisions)


# --------------------------------------------------------------------------- #
# Per-truck HOS independence
# --------------------------------------------------------------------------- #
def test_per_truck_hos_is_independent(scorer):
    """Co-located trucks differ only in remaining hours: the depleted one cannot
    take even one 300mi load while its rested peer can."""
    records = [
        _rec(1, T0, DENVER, DALLAS),
        _rec(2, T0, DENVER, DALLAS),
    ]
    trucks = [
        _truck(1, DENVER, hours=11.0),  # enough for a 7.5h load
        _truck(2, DENVER, hours=1.0),   # cannot feasibly take any load
    ]
    ep = run_fleet_episode(AssignmentFleetPolicy(scorer), SnapshotBoard(records),
                           trucks, episode_end=T0 + timedelta(hours=12), config=CFG)
    by_truck = {m_idx: m for m_idx, m in enumerate(ep.metrics.truck_metrics)}
    assert by_truck[0].loads_completed == 1   # rested truck works
    assert by_truck[1].loads_completed == 0   # depleted truck cannot


# --------------------------------------------------------------------------- #
# Idle accounting
# --------------------------------------------------------------------------- #
def test_idle_when_no_candidates(scorer):
    records = [
        _rec(1, T0, LA, DALLAS),                        # far from Denver
        _rec(2, T0 + timedelta(hours=4), LA, DALLAS),   # still far
    ]
    trucks = [_truck(1, DENVER), _truck(2, DENVER)]
    ep = run_fleet_episode(GreedyFleetPolicy(scorer), SnapshotBoard(records),
                           trucks, episode_end=T0 + timedelta(hours=8), config=CFG)
    assert ep.metrics.loads_completed == 0
    assert ep.metrics.total_idle_hours > 0
    assert ep.metrics.mean_utilization_rate < 1.0
    assert ep.metrics.contention_events == 0  # nothing visible -> no contention


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def test_determinism(scorer):
    records = [
        _rec(1, T0, DENVER, DALLAS),
        _rec(2, T0, DENVER, HOUSTON, miles=260.0),
        _rec(3, T0, (39.6, -104.8), DALLAS, miles=280.0),
    ]
    trucks = [_truck(1, DENVER), _truck(2, DENVER)]
    end = T0 + timedelta(hours=12)

    def run():
        ep = run_fleet_episode(AssignmentFleetPolicy(scorer), SnapshotBoard(records),
                               trucks, episode_end=end, config=CFG)
        return [(d.truck_id, d.load_id, round(d.profit, 4)) for d in ep.decisions]

    runs = [run() for _ in range(4)]
    assert all(r == runs[0] for r in runs)


# --------------------------------------------------------------------------- #
# Artifact-gated risk hook
# --------------------------------------------------------------------------- #
def test_risk_hook_off_by_default(scorer):
    records = [_rec(1, T0, DENVER, DALLAS)]
    ep = run_fleet_episode(GreedyFleetPolicy(scorer), SnapshotBoard(records),
                           [_truck(1, DENVER)], episode_end=T0 + timedelta(hours=12),
                           config=CFG)
    assert ep.metrics.expected_collectible_profit is None
    assert ep.metrics.expected_default_exposure is None


def test_risk_hook_scores_collectible_and_exposure(scorer):
    records = [_rec(1, T0, DENVER, DALLAS)]
    ep = run_fleet_episode(
        GreedyFleetPolicy(scorer), SnapshotBoard(records), [_truck(1, DENVER)],
        episode_end=T0 + timedelta(hours=12), config=CFG, risk_scorer=_StubRisk(0.25),
    )
    d = ep.decisions[0]
    assert ep.metrics.expected_collectible_profit == pytest.approx(d.profit * 0.75)
    assert ep.metrics.expected_default_exposure == pytest.approx(d.revenue * 0.25)
