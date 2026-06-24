"""Phase 8.1 — fleet domain models + assignment policies.

Two layers of coverage:

* **Policy logic** via a hand-built ``StubScorer`` ({(truck_id, load_id): score})
  so greedy-vs-optimal behaviour, conflict-freedom, the at-most-one constraints,
  infeasible exclusion, and determinism are tested exactly, independent of the
  financial engine.
* **Engine reconciliation** via the real ``ProfitPairScorer`` over the shared
  ``EvaluateLoadsService`` — feasibility gating (equipment, deadhead cap, profit
  floor) and a real geometry where coordination provably beats greedy.
"""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from adapters.outbound.distance.haversine import HaversineDistanceProvider
from adapters.outbound.tolls.flat_rate import FlatRateTollEstimator
from application.config_loader import load_config
from application.evaluate_loads import EvaluateLoadsService
from application.fleet.assignment_fleet_policy import AssignmentFleetPolicy
from application.fleet.greedy_fleet_policy import GreedyFleetPolicy
from application.fleet.pair_scorer import ProfitPairScorer, ScoredPair
from domain.models.fleet import Assignment, Fleet
from domain.models.plan import PlanStop

from .test_ortools_distance_planner import (
    BOISE,
    SALT_LAKE_CITY,
    _load,
    _truck,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = datetime(2026, 5, 27, 16, 0, tzinfo=timezone.utc)

# Truck1 ~80mi NW of SLC, ~210mi SE of Boise: can reach both load origins under
# the 250mi deadhead cap, but is closer to SLC (so greedy grabs the SLC load).
MIDWEST_UT = (41.6, -113.0)


@pytest.fixture(scope="module")
def config():
    return load_config(ROOT / "config")


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def _fleet_truck(truck_id, location, trailer="Dry Van", hours=16.0):
    return replace(
        _truck(driver_hours_left=hours, location=location, trailer=trailer),
        truck_id=truck_id,
    )


def _make_scorer(config):
    evaluator = EvaluateLoadsService(
        distance_provider=HaversineDistanceProvider(),
        toll_estimator=FlatRateTollEstimator(),
        cost_model=config.cost_model,
        average_speed_mph=config.average_speed_mph,
        load_unload_hours=config.planning_constraints.average_load_unload_hours,
    )
    return ProfitPairScorer(evaluator, config.planning_constraints)


class StubScorer:
    """Scores from a ``{(truck_id, load_id): score}`` table; missing/None = infeasible.

    The built ``ScoredPair`` carries a ``PlanStop`` whose ``profit == score`` so a
    resulting ``Assignment.profit`` equals the table value, keeping assertions on
    policy behaviour purely about coordination, not financial computation.
    """

    def __init__(self, table):
        self._table = table

    def score(self, load, truck):
        score = self._table.get((truck.truck_id, load.load_id))
        if score is None:
            return None
        stop = PlanStop(
            load_id=load.load_id,
            pickup_eta=BASE,
            delivery_eta=BASE + timedelta(hours=8),
            deadhead_miles=0.0,
            load_miles=load.miles,
            revenue=score,
            cost=0.0,
            profit=score,
        )
        return ScoredPair(
            truck_id=truck.truck_id,
            load=load,
            evaluation=None,
            stop=stop,
            score=score,
            rationale="stub",
        )


def _two_loads():
    a = _load(1, SALT_LAKE_CITY, BOISE, miles=240, rate=850)
    b = _load(2, BOISE, SALT_LAKE_CITY, miles=240, rate=850)
    return a, b


def _both_see_both(trucks, loads):
    return {t.truck_id: list(loads) for t in trucks}


# --------------------------------------------------------------------------- #
# Fleet domain model
# --------------------------------------------------------------------------- #
def test_fleet_helpers():
    t1 = _fleet_truck(1, SALT_LAKE_CITY, trailer="Dry Van")
    t2 = _fleet_truck(2, BOISE, trailer="Reefer")
    t3 = replace(_fleet_truck(3, SALT_LAKE_CITY, trailer="Reefer"),
                 available_at=BASE + timedelta(hours=5))
    fleet = Fleet([t2, t3, t1])  # deliberately unsorted

    assert len(fleet) == 3
    assert [t.truck_id for t in fleet] == [2, 3, 1]
    assert fleet.truck_ids == [2, 3, 1]
    assert fleet.by_id(1) is t1
    assert fleet.by_id(99) is None
    # free_at returns only available trucks, in ascending id order.
    assert [t.truck_id for t in fleet.free_at(BASE)] == [1, 2]
    assert [t.truck_id for t in fleet.free_at(BASE + timedelta(hours=6))] == [1, 2, 3]
    assert fleet.equipment_mix() == {"Dry Van": 1, "Reefer": 2}


def test_assignment_convenience_properties():
    stop = PlanStop(
        load_id=7, pickup_eta=BASE, delivery_eta=BASE + timedelta(hours=8),
        deadhead_miles=12.0, load_miles=200.0, revenue=900.0, cost=600.0,
        profit=300.0,
    )
    a = Assignment(truck_id=3, load_id=7, stop=stop, score=320.0, rationale="r")
    assert a.profit == 300.0
    assert a.revenue == 900.0
    assert a.cost == 600.0
    assert a.deadhead_miles == 12.0
    assert a.loaded_miles == 200.0
    # score stays distinct from realized profit.
    assert a.score == 320.0


# --------------------------------------------------------------------------- #
# Policy logic (StubScorer)
# --------------------------------------------------------------------------- #
def test_assignment_beats_greedy_on_contention():
    """Classic greedy failure: truck1 grabs the load truck2 needed more."""
    t1, t2 = _fleet_truck(1, SALT_LAKE_CITY), _fleet_truck(2, SALT_LAKE_CITY)
    a, b = _two_loads()
    cands = _both_see_both([t1, t2], [a, b])
    table = {(1, 1): 500.0, (1, 2): 400.0, (2, 1): 480.0, (2, 2): 50.0}

    greedy = GreedyFleetPolicy(StubScorer(table)).assign([t1, t2], cands, BASE)
    coord = AssignmentFleetPolicy(StubScorer(table)).assign([t1, t2], cands, BASE)

    # Greedy: t1 takes its best (A), leaving t2 the scrap (B).
    assert {(x.truck_id, x.load_id) for x in greedy} == {(1, 1), (2, 2)}
    assert sum(x.score for x in greedy) == 550.0
    # Coordinated: swap so the totals are maximised.
    assert {(x.truck_id, x.load_id) for x in coord} == {(1, 2), (2, 1)}
    assert sum(x.score for x in coord) == 880.0
    assert sum(x.score for x in coord) > sum(x.score for x in greedy)


def test_no_double_booking_one_load_two_trucks():
    t1, t2 = _fleet_truck(1, SALT_LAKE_CITY), _fleet_truck(2, SALT_LAKE_CITY)
    (a, _b) = _two_loads()
    cands = {1: [a], 2: [a]}
    table = {(1, 1): 500.0, (2, 1): 450.0}

    coord = AssignmentFleetPolicy(StubScorer(table)).assign([t1, t2], cands, BASE)

    assert len(coord) == 1
    assert (coord[0].truck_id, coord[0].load_id) == (1, 1)  # higher scorer wins


def test_at_most_one_load_per_truck():
    t1 = _fleet_truck(1, SALT_LAKE_CITY)
    a, b = _two_loads()
    cands = {1: [a, b]}
    table = {(1, 1): 300.0, (1, 2): 400.0}

    coord = AssignmentFleetPolicy(StubScorer(table)).assign([t1], cands, BASE)

    assert len(coord) == 1
    assert coord[0].load_id == 2  # the truck's single best load


def test_infeasible_pairs_never_assigned():
    t1 = _fleet_truck(1, SALT_LAKE_CITY)
    a, b = _two_loads()
    cands = {1: [a, b]}
    table = {(1, 1): None, (1, 2): 200.0}  # A infeasible, B feasible

    greedy = GreedyFleetPolicy(StubScorer(table)).assign([t1], cands, BASE)
    coord = AssignmentFleetPolicy(StubScorer(table)).assign([t1], cands, BASE)

    assert [x.load_id for x in greedy] == [2]
    assert [x.load_id for x in coord] == [2]


def test_all_infeasible_yields_no_assignments():
    t1 = _fleet_truck(1, SALT_LAKE_CITY)
    a, b = _two_loads()
    cands = {1: [a, b]}
    table = {(1, 1): None, (1, 2): None}

    assert GreedyFleetPolicy(StubScorer(table)).assign([t1], cands, BASE) == []
    assert AssignmentFleetPolicy(StubScorer(table)).assign([t1], cands, BASE) == []


def test_empty_inputs():
    pol = AssignmentFleetPolicy(StubScorer({}))
    assert pol.assign([], {}, BASE) == []
    t1 = _fleet_truck(1, SALT_LAKE_CITY)
    assert pol.assign([t1], {}, BASE) == []


def test_no_contention_greedy_equals_assignment():
    """Disjoint candidate sets (no shared loads) -> identical results.

    A mini precursor to the K=1 simulator invariant in 8.2: with no contention,
    coordination has nothing to improve, so the two arms must agree exactly.
    """
    t1, t2 = _fleet_truck(1, SALT_LAKE_CITY), _fleet_truck(2, BOISE)
    a, b = _two_loads()
    cands = {1: [a], 2: [b]}  # each truck sees only its own load
    table = {(1, 1): 300.0, (2, 2): 400.0}

    greedy = GreedyFleetPolicy(StubScorer(table)).assign([t1, t2], cands, BASE)
    coord = AssignmentFleetPolicy(StubScorer(table)).assign([t1, t2], cands, BASE)

    norm = lambda res: sorted((x.truck_id, x.load_id, x.score) for x in res)
    assert norm(greedy) == norm(coord) == [(1, 1, 300.0), (2, 2, 400.0)]


def test_assignment_is_deterministic():
    t1, t2 = _fleet_truck(1, SALT_LAKE_CITY), _fleet_truck(2, SALT_LAKE_CITY)
    a, b = _two_loads()
    cands = _both_see_both([t1, t2], [a, b])
    table = {(1, 1): 500.0, (1, 2): 400.0, (2, 1): 480.0, (2, 2): 50.0}
    pol = AssignmentFleetPolicy(StubScorer(table))

    runs = [
        [(x.truck_id, x.load_id, x.score) for x in pol.assign([t1, t2], cands, BASE)]
        for _ in range(5)
    ]
    assert all(r == runs[0] for r in runs)
    # output is sorted by (truck_id, load_id)
    assert runs[0] == sorted(runs[0])


def test_candidate_cap_keeps_top_scores():
    t1 = _fleet_truck(1, SALT_LAKE_CITY)
    a, b = _two_loads()
    c = _load(3, SALT_LAKE_CITY, BOISE, miles=240, rate=850)
    cands = {1: [a, b, c]}
    table = {(1, 1): 100.0, (1, 2): 900.0, (1, 3): 500.0}
    pol = AssignmentFleetPolicy(StubScorer(table), max_candidates_per_truck=1)

    coord = pol.assign([t1], cands, BASE)
    assert [x.load_id for x in coord] == [2]  # only the top-scored survives the cap


# --------------------------------------------------------------------------- #
# Engine reconciliation (real ProfitPairScorer)
# --------------------------------------------------------------------------- #
def test_profit_scorer_scores_real_profit(config):
    scorer = _make_scorer(config)
    truck = _fleet_truck(1, SALT_LAKE_CITY)
    load = _load(1, SALT_LAKE_CITY, BOISE, miles=240, rate=850)

    scored = scorer.score(load, truck)

    assert scored is not None
    assert scored.score == scored.evaluation.expected_profit
    assert scored.score > 0
    # truck sits on the load origin -> negligible deadhead.
    assert scored.evaluation.deadhead_miles < 1.0
    assert scored.to_assignment().profit == pytest.approx(scored.score)


def test_profit_scorer_excludes_equipment_mismatch(config):
    scorer = _make_scorer(config)
    truck = _fleet_truck(1, SALT_LAKE_CITY, trailer="Dry Van")
    load = _load(1, SALT_LAKE_CITY, BOISE, miles=240, rate=850, equipment="Reefer")

    assert scorer.score(load, truck) is None


def test_profit_scorer_excludes_below_floor(config):
    scorer = _make_scorer(config)
    truck = _fleet_truck(1, SALT_LAKE_CITY)
    # Rate barely above cost -> profit under the $50 floor.
    load = _load(1, SALT_LAKE_CITY, BOISE, miles=240, rate=390)

    assert scorer.score(load, truck) is None


def test_coordination_beats_greedy_real_engine(config):
    """Real geometry: truck2 (at SLC) can only serve the SLC load (Boise is past
    its deadhead cap); truck1 can serve either but prefers the nearer SLC load.
    Greedy gives the SLC load to truck1 and strands truck2; coordination routes
    the SLC load to truck2 and the Boise load to truck1 for strictly more profit
    and full utilisation."""
    scorer = _make_scorer(config)
    t1 = _fleet_truck(1, MIDWEST_UT)
    t2 = _fleet_truck(2, SALT_LAKE_CITY)
    load_slc = _load(1, SALT_LAKE_CITY, BOISE, miles=240, rate=850)
    load_boise = _load(2, BOISE, SALT_LAKE_CITY, miles=240, rate=850)
    cands = _both_see_both([t1, t2], [load_slc, load_boise])

    greedy = GreedyFleetPolicy(scorer).assign([t1, t2], cands, BASE)
    coord = AssignmentFleetPolicy(scorer).assign([t1, t2], cands, BASE)

    # Greedy: t1 grabs the nearer SLC load; t2 is stranded (Boise > deadhead cap).
    assert {(x.truck_id, x.load_id) for x in greedy} == {(1, 1)}
    # Coordination: t2 -> SLC load, t1 -> Boise load; both trucks working.
    assert {(x.truck_id, x.load_id) for x in coord} == {(1, 2), (2, 1)}
    assert sum(x.profit for x in coord) > sum(x.profit for x in greedy)
    assert len(coord) == 2  # higher utilisation


def test_real_scorer_no_contention_greedy_equals_assignment(config):
    """K-disjoint reconciliation with the real engine: each truck parked on its
    own load origin, far from the other -> both arms agree."""
    scorer = _make_scorer(config)
    t1 = _fleet_truck(1, SALT_LAKE_CITY)
    t2 = _fleet_truck(2, BOISE)
    load_slc = _load(1, SALT_LAKE_CITY, BOISE, miles=240, rate=850)
    load_boise = _load(2, BOISE, SALT_LAKE_CITY, miles=240, rate=850)
    cands = {1: [load_slc], 2: [load_boise]}

    greedy = GreedyFleetPolicy(scorer).assign([t1, t2], cands, BASE)
    coord = AssignmentFleetPolicy(scorer).assign([t1, t2], cands, BASE)

    norm = lambda res: sorted((x.truck_id, x.load_id) for x in res)
    assert norm(greedy) == norm(coord) == [(1, 1), (2, 2)]
