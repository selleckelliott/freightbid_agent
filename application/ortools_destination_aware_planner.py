"""Destination-aware OR-Tools planner (Phase 3.2).

The profit-aware planner (Phase 2.2) prices the deadhead needed to *reach* each
load, but is blind to the deadhead a load's **destination** will impose on the
*next* load. That forward cost is exactly what the Phase 3.1 destination-
desirability model predicts — and exactly what a load board never shows (the
board only computes deadhead to a point you type in, never the open-ended
"where will this leave me?").

This planner closes the loop. It subclasses ``ORToolsProfitAwarePlanner`` and
changes a single objective hook, ``_drop_penalty``: each load's skip penalty is
its static profit **minus the expected onward-deadhead cost of its
destination**, above the floor::

    skip_penalty = max(0, static_profit - dest_cost - floor) * profit_multiplier
    dest_cost    = predicted_next_deadhead_miles * deadhead_$_per_mile * weight

Folding ``dest_cost`` in *before* the floor keeps the solver's serve-vs-skip
break-even exact (the same shift the parent uses for the floor). A load
delivering into a strong market keeps its full value; one delivering into a weak
market loses skip-incentive, so the solver declines otherwise-profitable freight
that would strand the truck. The penalty is position-independent (it depends
only on the load's destination, arrival window and the visible board), so it
stays a per-load disjunction penalty — no path-dependent arc cost is needed.

Two small adapters bridge the domain ↔ ML vocabulary boundary; both are
deliberately coarse and documented (see below). With ``destination_service=None``
the planner is byte-for-byte the profit-aware planner — the ML signal is a
feature flag, not a fork.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from domain.models.load import Load
from domain.models.truck_state import TruckState
from domain.policies.constraints import PlanningConstraints
from domain.policies.ortools_objective_weights import (
    CENTS_PER_DOLLAR,
    ORToolsObjectiveWeights,
)
from ports.distance_provider import DistanceProviderPort

from .evaluate_loads import EvaluateLoadsService
from .ortools_profit_aware_planner import ORToolsProfitAwarePlanner

# Domain trailer vocabulary -> the hot-shot equipment codes the ML layer trained
# on (see ml/markets.py). Hot-shot boards describe truck *capability*, not
# freight type, so this is a deliberately coarse adapter: Flatbed -> the flatbed
# code, Dry Van -> the only van-capable hot-shot config, and everything else
# (incl. Reefer, which has no open-deck analogue) -> the generic Hot Shot
# bucket. equipment_type is one of ~28 model features and not the dominant one,
# so the coarseness is acceptable; it is recorded as a known boundary
# simplification rather than hidden.
_EQUIPMENT_TO_ML: Dict[str, str] = {
    "Flatbed": "F",
    "Dry Van": "FSDV",
    "Reefer": "HS",
}
_DEFAULT_ML_EQUIPMENT = "HS"


def domain_equipment_to_ml(equipment_type: str) -> str:
    """Map a domain trailer label to the ML hot-shot equipment vocabulary."""
    return _EQUIPMENT_TO_ML.get(equipment_type, _DEFAULT_ML_EQUIPMENT)


@dataclass(frozen=True)
class _BoardLoad:
    """A domain ``Load`` re-shaped as a decision-time board item.

    Exposes exactly the attributes the ML feature builder reads off each visible
    load (``ml/features/destination_features.py``): origin coordinates, ML-vocab
    equipment, loaded miles, rate-per-mile, plus age/views defaults. This is the
    same adapter contract a future real Truckstop board feed would satisfy.
    """

    load_id: str
    origin_lat: float
    origin_lon: float
    equipment_type: str
    rate_per_mile: float
    loaded_miles: float
    load_age_hours: float = 0.0
    load_views: str = "low"


def _to_board_load(load: Load) -> _BoardLoad:
    return _BoardLoad(
        load_id=str(load.load_id),
        origin_lat=load.origin_latitude,
        origin_lon=load.origin_longitude,
        equipment_type=domain_equipment_to_ml(load.equipment_type),
        rate_per_mile=load.rate_per_mile,
        loaded_miles=load.miles,
    )


class ORToolsDestinationAwarePlanner(ORToolsProfitAwarePlanner):
    """Profit-aware planner that discounts loads by destination desirability."""

    PLANNER_LABEL = "OR-Tools Destination-Aware"

    def __init__(
        self,
        distance_provider: DistanceProviderPort,
        evaluate_loads_service: EvaluateLoadsService,
        constraints: PlanningConstraints,
        objective_weights: ORToolsObjectiveWeights,
        destination_service: Optional[Any] = None,
        destination_weight: float = 1.0,
        solver_time_limit_seconds: float = 1.0,
        average_speed_mph: float = 50.0,
        load_unload_hours: float = 1.5,
    ):
        super().__init__(
            distance_provider=distance_provider,
            evaluate_loads_service=evaluate_loads_service,
            constraints=constraints,
            objective_weights=objective_weights,
            solver_time_limit_seconds=solver_time_limit_seconds,
            average_speed_mph=average_speed_mph,
            load_unload_hours=load_unload_hours,
        )
        # Duck-typed: anything exposing ``predict_next_deadhead`` works (the real
        # DestinationDesirabilityService, or a test stub). None disables the
        # signal entirely, leaving pure profit-aware behaviour.
        self._destination_service = destination_service
        self._destination_weight = destination_weight
        self._board: List[_BoardLoad] = []
        self._dest_cost_cache: Dict[int, float] = {}

    # ------------------------------------------------------------------ public
    def build_plan(
        self,
        loads: List[Load],
        truck_state: TruckState,
        plan_id: int = 1,
    ):
        if self._destination_service is not None:
            # The decision-time board is the prefiltered candidate set — the
            # loads we could actually take right now. Each candidate's onward
            # deadhead is predicted against the others posted near its
            # destination (the candidate is excluded from its own board below).
            self._board = [
                _to_board_load(load)
                for load in self._prefilter(loads, truck_state)
            ]
        self._dest_cost_cache = {}
        try:
            return super().build_plan(loads, truck_state, plan_id)
        finally:
            self._board = []
            self._dest_cost_cache = {}

    # --------------------------------------------------------- objective hooks
    def _drop_penalty(self, load: Load, truck_state: TruckState) -> int:
        """Skip penalty = static profit, *net of destination cost*, above floor.

        Identical to the profit-aware parent except the load's value is first
        reduced by the expected onward-deadhead cost of its destination, so
        loads that strand the truck become cheaper to skip.
        """
        if self._destination_service is None:
            return super()._drop_penalty(load, truck_state)
        margin = (
            self._static_profit(load, truck_state)
            - self._destination_cost_dollars(load)
            - self._skip_profit_floor
        )
        if margin <= 0:
            return 0
        return int(round(margin * self.objective_weights.profit_cents_multiplier))

    # ----------------------------------------------------------------- helpers
    def _destination_cost_dollars(self, load: Load) -> float:
        """Expected onward-deadhead cost (USD) if the truck delivers ``load``.

        ``predicted_miles`` comes from the Phase 3.1 model via the injected
        service; converting at the cost model's deadhead rate puts it on the
        same dollar scale as ``static_profit``. ``delivery_time`` (the load's
        scheduled delivery-window end) is used as the arrival timestamp the
        model needs for its daypart features — a deterministic, position-
        independent proxy for when the truck would free up at the destination.
        """
        cached = self._dest_cost_cache.get(load.load_id)
        if cached is not None:
            return cached
        board = [b for b in self._board if b.load_id != str(load.load_id)]
        predicted_miles = self._destination_service.predict_next_deadhead(
            destination_lat=load.destination_latitude,
            destination_lon=load.destination_longitude,
            destination_state=load.destination_state,
            arrival_time=load.delivery_time,
            equipment_type=domain_equipment_to_ml(load.equipment_type),
            visible_loads=board,
            load_age_hours=0.0,
            mode="TL",
        )
        dollars_per_mile = (
            self.objective_weights.deadhead_cost_cents_per_mile / CENTS_PER_DOLLAR
        )
        cost = max(0.0, predicted_miles) * dollars_per_mile * self._destination_weight
        self._dest_cost_cache[load.load_id] = cost
        return cost

    def _explain(self, plan) -> str:
        base = super()._explain(plan)
        if not plan.stops or self._destination_service is None:
            return base
        return (
            base
            + " It also discounted each load by the Phase 3.1 model's expected "
            "onward-deadhead cost, so freight delivering into weak markets was "
            "skipped in favour of loads that reposition the truck well."
        )
