"""Profit-aware OR-Tools planner (Phase 2.2).

Same pickup-and-delivery routing machinery as ``ORToolsDistancePlanner``
(time windows, HOS budget, paired pickup/delivery disjunctions, open route,
replay-through-evaluator financials) — only the *objective* differs.

Instead of minimising travel distance, the solver minimises **negative
expected plan profit, in integer cents**:

* Repositioning (deadhead) arcs cost their true business rate per mile —
  fuel x deadhead multiplier + maintenance + pro-rated driver/opportunity
  time — derived from ``CostModel`` via ``ORToolsObjectiveWeights``.
* A load's own pickup -> delivery arc costs **0** in the objective: the
  loaded leg's cost is already inside its static profit, so pricing the arc
  too would double-count it.
* Skipping a load incurs a penalty equal to its *static profit* (expected
  profit evaluated with zero deadhead, i.e. position-independent) in cents.
  High-profit freight is expensive to skip; break-even freight is free to
  skip.

Minimising ``deadhead cost + skipped static profit`` is equivalent to
maximising ``selected static profit - deadhead cost`` — the expected profit
of the plan. The solver therefore answers the real business question:
*which subset of loads is worth taking, and in what order?*
"""
from __future__ import annotations

from dataclasses import replace

from ortools.constraint_solver import routing_enums_pb2

from domain.models.load import Load
from domain.models.plan import Plan
from domain.models.truck_state import TruckState
from domain.policies.constraints import PlanningConstraints
from domain.policies.ortools_objective_weights import ORToolsObjectiveWeights
from ports.distance_provider import DistanceProviderPort

from .evaluate_loads import EvaluateLoadsService
from .ortools_distance_planner import ORToolsDistancePlanner, _RoutingNode


class ORToolsProfitAwarePlanner(ORToolsDistancePlanner):
    """OR-Tools planner whose objective is expected plan profit."""

    PLANNER_LABEL = "OR-Tools Profit-Aware"
    # PATH_CHEAPEST_ARC builds the first route from arc costs alone, which are
    # uninformative here (loaded arcs cost 0; the signal lives in the
    # profit-proportional disjunction penalties). Cheapest *insertion* weighs
    # those penalties, giving the metaheuristic a sane starting point.
    FIRST_SOLUTION_STRATEGY = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    )
    # Objective cost that dominates any achievable skip-penalty total, used to
    # veto repositioning arcs the replay's max-deadhead rule would reject.
    ARC_VETO_COST = 50_000_000
    # The replay pipeline (the financial source of truth shared with the
    # heuristic) serves loads strictly one at a time: pickup, deliver, then
    # reposition. Mirroring that here keeps solver plans replay-consistent
    # instead of being silently truncated after the fact.
    SEQUENTIAL_PICKUP_DELIVERY = True

    def __init__(
        self,
        distance_provider: DistanceProviderPort,
        evaluate_loads_service: EvaluateLoadsService,
        constraints: PlanningConstraints,
        objective_weights: ORToolsObjectiveWeights,
        solver_time_limit_seconds: float = 1.0,
        average_speed_mph: float = 50.0,
        load_unload_hours: float = 1.5,
    ):
        super().__init__(
            distance_provider=distance_provider,
            evaluate_loads_service=evaluate_loads_service,
            constraints=constraints,
            solver_time_limit_seconds=solver_time_limit_seconds,
            average_speed_mph=average_speed_mph,
            load_unload_hours=load_unload_hours,
        )
        self.objective_weights = objective_weights

    # --------------------------------------------------------- objective hooks
    def _arc_cost(
        self, from_node: _RoutingNode, to_node: _RoutingNode, miles: float
    ) -> int:
        if (
            from_node.node_type == "pickup"
            and to_node.node_type == "delivery"
            and from_node.load is to_node.load
        ):
            # Loaded leg: its cost already lives inside the load's static
            # profit (the skip penalty), so it must be free here.
            return 0
        if (
            to_node.node_type == "pickup"
            and miles > self.constraints.max_deadhead_miles
        ):
            # The authoritative replay rejects any load reached over more
            # deadhead than the business cap allows; make such transitions
            # prohibitively expensive so the solver never relies on them.
            return self.ARC_VETO_COST
        return int(
            round(miles * self.objective_weights.deadhead_cost_cents_per_mile)
        )

    def _drop_penalty(self, load: Load, truck_state: TruckState) -> int:
        """Skip penalty = static profit *above the business profit floor*.

        The replay's feasibility rule only accepts a load whose
        position-aware profit clears ``min_expected_profit``, so serving is
        only worth it when ``static_profit - deadhead_cost`` beats the floor.
        Shifting the penalty by the floor makes the solver's serve-vs-skip
        break-even point coincide with that rule exactly.
        """
        margin = (
            self._static_profit(load, truck_state)
            - self.constraints.min_expected_profit
        )
        if margin <= 0:
            return 0
        return int(round(margin * self.objective_weights.profit_cents_multiplier))

    # ----------------------------------------------------------------- helpers
    def _static_profit(self, load: Load, truck_state: TruckState) -> float:
        """Expected profit with the truck already at the load's origin.

        Zero deadhead makes the figure position-independent: revenue minus
        loaded-leg fuel/maintenance, tolls, and driver/opportunity time
        (including load/unload). The deadhead actually required to reach the
        load is priced separately by the repositioning arcs, so selection
        and routing stay consistent without double-counting.
        """
        at_origin = replace(
            truck_state,
            latitude=load.origin_latitude,
            longitude=load.origin_longitude,
            current_city=load.origin_city,
            current_state=load.origin_state,
        )
        evaluation = self.evaluate_loads_service.evaluate_one(load, at_origin)
        return evaluation.expected_profit

    def _explain(self, plan: Plan) -> str:
        base = super()._explain(plan)
        if not plan.stops:
            return base
        return (
            base
            + " The solver maximised expected profit (cents-scaled objective): "
            "profitable loads carry skip penalties equal to their static "
            "profit while empty miles are charged at the cost-model rate, so "
            "it selects and sequences only freight worth its deadhead."
        )
