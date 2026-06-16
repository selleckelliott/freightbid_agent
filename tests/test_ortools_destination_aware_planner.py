"""Deterministic micro-tests for ``ORToolsDestinationAwarePlanner`` (Phase 3.2).

The planner extends the profit-aware solver by discounting each load by the
*expected onward-deadhead cost of its destination* (predicted by the Phase 3.1
model). These tests inject a tiny stub service — no trained ``.joblib`` artifact
is needed — so the integration logic (penalty folding, board adapter, equipment
mapping, feature flag) is verified fast and deterministically.

Numbers reuse the proven profit-aware fixtures: a UT<->UT load ~20 mi south of
the truck has static profit ``308.50 - 1.39*100 - 49.50 = $120`` and costs
~$28 of deadhead to reach, so the profit-aware planner serves it (margin over
the $50 floor = $70 > $28).
"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from adapters.outbound.distance.haversine import HaversineDistanceProvider
from adapters.outbound.tolls.flat_rate import FlatRateTollEstimator
from application.config_loader import load_config
from application.destination_desirability_service import DestinationDesirabilityService
from application.evaluate_loads import EvaluateLoadsService
from application.ortools_destination_aware_planner import (
    ORToolsDestinationAwarePlanner,
    domain_equipment_to_ml,
)

from .test_ortools_distance_planner import (
    BOISE,
    RENO,
    SALT_LAKE_CITY,
    _load,
    _truck,
)
from .test_ortools_profit_aware_planner import SOUTH_UT_DEST, _make_profit_planner

ROOT = Path(__file__).resolve().parents[1]

# ~20 mi south of Salt Lake City: deadhead ~$28 to reach (mirrors the proven
# profit-aware pickiness fixture).
NEARBY_ORIGIN = (40.471, -111.8910)


@pytest.fixture(scope="module")
def config():
    return load_config(ROOT / "config")


def _make_dest_aware_planner(config, service, weight=1.0, time_limit=0.3):
    evaluator = EvaluateLoadsService(
        distance_provider=HaversineDistanceProvider(),
        toll_estimator=FlatRateTollEstimator(),
        cost_model=config.cost_model,
        average_speed_mph=config.average_speed_mph,
        load_unload_hours=config.planning_constraints.average_load_unload_hours,
    )
    return ORToolsDestinationAwarePlanner(
        distance_provider=HaversineDistanceProvider(),
        evaluate_loads_service=evaluator,
        constraints=config.planning_constraints,
        objective_weights=config.ortools_objective_weights,
        destination_service=service,
        destination_weight=weight,
        solver_time_limit_seconds=time_limit,
        average_speed_mph=config.average_speed_mph,
        load_unload_hours=config.planning_constraints.average_load_unload_hours,
    )


class _StubService:
    """Stand-in for ``DestinationDesirabilityService`` keyed on destination state."""

    def __init__(self, miles_by_state, default=0.0):
        self.miles_by_state = miles_by_state
        self.default = default
        self.calls = []

    def predict_next_deadhead(self, *, destination_state, visible_loads=(), **kwargs):
        self.calls.append(
            {
                "destination_state": destination_state,
                "n_board": len(list(visible_loads)),
                **kwargs,
            }
        )
        return self.miles_by_state.get(destination_state, self.default)


class _RecordingModel:
    """Records the feature frames it is asked to predict on."""

    def __init__(self):
        self.frames = []

    def predict(self, frame):
        self.frames.append(frame)
        return np.array([10.0])


def test_weak_destination_makes_marginal_load_skippable(config):
    """A load the profit-aware planner serves is dropped once its destination's
    expected onward deadhead is priced in."""
    load = replace(
        _load(1, NEARBY_ORIGIN, SOUTH_UT_DEST, miles=100, rate=308.50),
        destination_state="NM",
    )
    truck = _truck()

    # Baseline: profit-aware serves it ($120 static profit > $28 deadhead).
    base = _make_profit_planner(config).build_plan([load], truck)
    assert [s.load_id for s in base.stops] == [1]

    # 200 predicted onward miles -> ~$278 cost wipes the $120 profit below the
    # floor, so the destination-aware planner declines it.
    plan = _make_dest_aware_planner(config, _StubService({"NM": 200.0})).build_plan(
        [load], truck
    )
    assert plan.stops == []
    assert not plan.feasible


def test_strong_destination_keeps_load(config):
    """The signal discriminates: a low-onward-deadhead destination is served."""
    load = replace(
        _load(1, NEARBY_ORIGIN, SOUTH_UT_DEST, miles=100, rate=308.50),
        destination_state="ID",
    )

    plan = _make_dest_aware_planner(config, _StubService({"ID": 5.0})).build_plan(
        [load], _truck()
    )

    assert [s.load_id for s in plan.stops] == [1]
    assert "onward-deadhead" in plan.rationale


def test_service_none_matches_profit_aware(config):
    """Feature flag off (no service) ⇒ identical selection to the parent."""
    loads = [
        _load(1, SALT_LAKE_CITY, BOISE, miles=160, rate=620),
        _load(2, BOISE, RENO, miles=160, rate=620),
    ]
    truck = _truck(driver_hours_left=14.0)

    dest = _make_dest_aware_planner(config, None).build_plan(loads, truck)
    prof = _make_profit_planner(config).build_plan(loads, truck)

    assert [s.load_id for s in dest.stops] == [s.load_id for s in prof.stops]


def test_zero_weight_disables_destination_signal(config):
    """``destination_weight=0`` neutralises even an extreme prediction."""
    load = replace(
        _load(1, NEARBY_ORIGIN, SOUTH_UT_DEST, miles=100, rate=308.50),
        destination_state="NM",
    )

    plan = _make_dest_aware_planner(
        config, _StubService({"NM": 9999.0}), weight=0.0
    ).build_plan([load], _truck())

    assert [s.load_id for s in plan.stops] == [1]


def test_board_adapter_maps_equipment_and_excludes_self(config):
    """The real feature builder receives ML-vocab equipment and a board that
    excludes the candidate itself."""
    model = _RecordingModel()
    service = DestinationDesirabilityService(model)
    load = _load(
        1, SALT_LAKE_CITY, SOUTH_UT_DEST, miles=100, rate=308.50, equipment="Flatbed"
    )

    _make_dest_aware_planner(config, service).build_plan(
        [load], _truck(trailer="Flatbed")
    )

    assert model.frames, "the model should be queried for the candidate destination"
    frame = model.frames[0]
    assert frame["equipment_type"].iloc[0] == "F"  # domain Flatbed -> ML F
    assert frame["mode"].iloc[0] == "TL"
    # Single candidate, excluded from its own board -> no nearby loads.
    assert int(frame["loads_within_50"].iloc[0]) == 0


def test_board_excludes_only_self_not_peers(config):
    """With two candidates, each load's board sees the *other* one."""
    stub = _StubService({"UT": 10.0})
    near = _load(1, SALT_LAKE_CITY, SOUTH_UT_DEST, miles=100, rate=308.50)
    # A second load whose origin sits at the first load's destination, so it is
    # "near" load 1's destination on the board.
    peer = _load(2, SOUTH_UT_DEST, SALT_LAKE_CITY, miles=100, rate=308.50)

    _make_dest_aware_planner(config, stub).build_plan([near, peer], _truck())

    by_state = [c for c in stub.calls]
    assert by_state, "service should be consulted"
    # Every candidate's board excludes itself, so it can never exceed peers count.
    assert all(c["n_board"] == 1 for c in by_state)


def test_domain_equipment_to_ml_mapping():
    assert domain_equipment_to_ml("Flatbed") == "F"
    assert domain_equipment_to_ml("Dry Van") == "FSDV"
    assert domain_equipment_to_ml("Reefer") == "HS"
    # Unknown trailers fall back to the generic hot-shot bucket.
    assert domain_equipment_to_ml("Conestoga") == "HS"
