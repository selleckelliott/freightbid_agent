"""OR-Tools single-truck pickup-and-delivery planner (Phase 2).

Each load is modelled as a linked pickup node and delivery node. OR-Tools
solves an open-route (no return-to-home) vehicle routing problem that:

* respects pickup-before-delivery precedence,
* respects pickup / delivery time windows,
* keeps cumulative working time within the driver's remaining HOS budget,
* may *drop* loads (via penalised disjunctions) when they cannot all be served.

The objective for this first version minimises **travel distance** (and thus
deadhead) while serving as many feasible loads as possible — it deliberately
does *not* optimise profit yet (see the Phase 2 design note).

Selection/sequencing is decided by OR-Tools; the resulting load order is then
*replayed* through the very same ``EvaluateLoadsService`` + ``feasibility_checker``
pipeline the heuristic ``PlanBuilderService`` uses. That replay is the source of
truth for the returned ``Plan`` financials and for final feasibility, which keeps
the head-to-head benchmark fair: both planners obey identical validity rules and
identical cost accounting; only their load selection/order differs.

The *objective* is isolated behind two small hooks (``_arc_cost`` and
``_drop_penalty``) so that planner variants — e.g. the Phase 2.2
``ORToolsProfitAwarePlanner`` — can swap the optimisation goal while sharing
all of the correctness-critical routing machinery above.
"""
from __future__ import annotations

import logging
import math
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from domain.models.load import Load
from domain.models.plan import Plan, PlanStop
from domain.models.truck_state import TruckState
from domain.policies.constraints import PlanningConstraints
from domain.policies.feasibility import feasibility_checker
from ports.distance_provider import DistanceProviderPort

from .evaluate_loads import EvaluateLoadsService

logger = logging.getLogger(__name__)

# Miles -> integer arc cost, preserving two decimal places for OR-Tools.
DISTANCE_SCALE = 100
# Penalty for dropping a load. Must dominate any plausible arc distance so the
# solver prefers serving feasible loads (pure distance-min would drop them all),
# then minimises distance among the loads it serves.
DROP_PENALTY = 10_000_000


@dataclass
class _RoutingNode:
    node_id: int
    load: Optional[Load]
    node_type: str  # "depot" | "pickup" | "delivery" | "end"
    latitude: float
    longitude: float
    tw_start_min: int = 0
    tw_end_min: int = 0


class ORToolsDistancePlanner:
    """OR-Tools pickup-and-delivery planner conforming to the ``Planner`` interface."""

    PLANNER_LABEL = "OR-Tools Distance"
    # Greedy cheapest-arc works well when the objective *is* arc distance;
    # objective variants may override (see ORToolsProfitAwarePlanner).
    FIRST_SOLUTION_STRATEGY = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    LOCAL_SEARCH_METAHEURISTIC = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    # When True, a delivery must immediately follow its pickup (strict
    # full-truckload semantics: serve one load completely, then the next).
    SEQUENTIAL_PICKUP_DELIVERY = False

    def __init__(
        self,
        distance_provider: DistanceProviderPort,
        evaluate_loads_service: EvaluateLoadsService,
        constraints: PlanningConstraints,
        solver_time_limit_seconds: float = 1.0,
        average_speed_mph: float = 50.0,
        load_unload_hours: float = 1.5,
    ):
        self.distance_provider = distance_provider
        self.evaluate_loads_service = evaluate_loads_service
        self.constraints = constraints
        self.solver_time_limit_seconds = solver_time_limit_seconds
        self.average_speed_mph = average_speed_mph
        self.load_unload_hours = load_unload_hours

    # ------------------------------------------------------------------ public
    def build_plan(
        self,
        loads: List[Load],
        truck_state: TruckState,
        plan_id: int = 1,
    ) -> Plan:
        horizon_min = int(self.constraints.planning_horizon_hours * 60)

        candidates = self._prefilter(loads, truck_state)
        if not candidates:
            return self._empty_plan(plan_id, truck_state)

        nodes, pd_pairs = self._build_nodes(candidates, truck_state, horizon_min)
        ordered_loads = self._solve(nodes, pd_pairs, truck_state, horizon_min)
        if not ordered_loads:
            return self._empty_plan(plan_id, truck_state)

        return self._replay_to_plan(ordered_loads, truck_state, plan_id)

    # ----------------------------------------------------------------- helpers
    def _prefilter(self, loads: List[Load], truck_state: TruckState) -> List[Load]:
        """Keep only loads that are feasible in the best case (zero deadhead).

        Each load is evaluated as if the truck were already sitting at its origin
        — the most favourable possible position. If it fails the shared
        ``feasibility_checker`` even then (wrong trailer, over capacity, over a
        miles/cost cap, below ``min_expected_profit``, outside its windows, or
        beyond the horizon), no route position can rescue it, so it is dropped.
        This stops the distance objective from "spending" its route on loads the
        authoritative replay would only discard, and it can never remove a load
        the heuristic could have used.
        """
        horizon_end = truck_state.available_at + timedelta(
            hours=self.constraints.planning_horizon_hours
        )
        out: List[Load] = []
        for load in loads:
            at_origin = replace(
                truck_state,
                latitude=load.origin_latitude,
                longitude=load.origin_longitude,
                current_city=load.origin_city,
                current_state=load.origin_state,
            )
            evaluation = self.evaluate_loads_service.evaluate_one(load, at_origin)
            feasible, _reason = feasibility_checker(
                evaluation,
                at_origin,
                self.constraints,
                self.evaluate_loads_service.cost_model,
            )
            if not feasible:
                continue
            if evaluation.delivery_eta is not None and evaluation.delivery_eta > horizon_end:
                continue
            out.append(load)
        return out

    def _build_nodes(
        self,
        loads: List[Load],
        truck_state: TruckState,
        horizon_min: int,
    ) -> Tuple[List[_RoutingNode], List[Tuple[int, int]]]:
        start = truck_state.available_at
        nodes: List[_RoutingNode] = [
            _RoutingNode(0, None, "depot", truck_state.latitude, truck_state.longitude, 0, 0)
        ]
        pd_pairs: List[Tuple[int, int]] = []

        for load in loads:
            pickup_id = len(nodes)
            nodes.append(
                _RoutingNode(
                    pickup_id,
                    load,
                    "pickup",
                    load.origin_latitude,
                    load.origin_longitude,
                    # Truck may wait for the window to open (matches heuristic
                    # pickup_eta = max(arrival, pickup_window_start)).
                    self._clamp(self._minutes_from_start(load.pickup_window_start, start), 0, horizon_min),
                    self._clamp(self._minutes_from_start(load.pickup_window_end, start), 0, horizon_min),
                )
            )
            delivery_id = len(nodes)
            nodes.append(
                _RoutingNode(
                    delivery_id,
                    load,
                    "delivery",
                    load.destination_latitude,
                    load.destination_longitude,
                    # No early-delivery waiting (the heuristic does not wait for
                    # delivery_window_start), only the upper bound is enforced.
                    0,
                    self._clamp(self._minutes_from_start(load.delivery_window_end, start), 0, horizon_min),
                )
            )
            pd_pairs.append((pickup_id, delivery_id))

        # Dummy end node makes the route OPEN: the truck does not drive home.
        end_id = len(nodes)
        nodes.append(
            _RoutingNode(end_id, None, "end", truck_state.latitude, truck_state.longitude, 0, horizon_min)
        )
        return nodes, pd_pairs

    def _solve(
        self,
        nodes: List[_RoutingNode],
        pd_pairs: List[Tuple[int, int]],
        truck_state: TruckState,
        horizon_min: int,
    ) -> List[Load]:
        n = len(nodes)
        end_id = nodes[-1].node_id

        distance = [[0] * n for _ in range(n)]
        travel_time = [[0] * n for _ in range(n)]
        for i in range(n):
            if nodes[i].node_type == "end":
                continue
            service = (
                math.ceil(self.load_unload_hours * 60.0)
                if nodes[i].node_type == "pickup"
                else 0
            )
            for j in range(n):
                if i == j or nodes[j].node_type == "end":
                    continue
                # The loaded leg (a load's own pickup -> delivery) is costed on the
                # load's stated miles, exactly like EvaluateLoadsService. Every other
                # arc is a deadhead/repositioning move measured by haversine, matching
                # how the evaluator computes deadhead. This keeps the OR-Tools model
                # consistent with the authoritative replay so its routes survive it.
                if (
                    nodes[i].node_type == "pickup"
                    and nodes[j].node_type == "delivery"
                    and nodes[i].load is nodes[j].load
                ):
                    miles = nodes[i].load.miles
                else:
                    miles = self.distance_provider.miles_between(
                        nodes[i].latitude, nodes[i].longitude,
                        nodes[j].latitude, nodes[j].longitude,
                    )
                distance[i][j] = self._arc_cost(nodes[i], nodes[j], miles)
                # ceil keeps the model conservative so a route OR-Tools deems
                # feasible survives the float-precision replay.
                travel_time[i][j] = math.ceil(miles / self.average_speed_mph * 60.0) + service

        manager = pywrapcp.RoutingIndexManager(n, 1, [0], [end_id])
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            return distance[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

        distance_idx = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(distance_idx)

        def time_callback(from_index, to_index):
            return travel_time[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

        time_idx = routing.RegisterTransitCallback(time_callback)

        # Wall-clock dimension: slack allows waiting for windows; capped at horizon.
        routing.AddDimension(time_idx, horizon_min, horizon_min, True, "Time")
        time_dim = routing.GetDimensionOrDie("Time")

        # Working-time dimension: zero slack => cumulative = travel+service only
        # (waiting excluded), capped at the driver's remaining HOS budget. This
        # mirrors how the heuristic decrements driver_hours_left and is what keeps
        # the comparison fair on the binding constraint.
        driver_cap = max(1, int(truck_state.driver_hours_left * 60))
        routing.AddDimension(time_idx, 0, driver_cap, True, "DriverHours")

        for node in nodes:
            if node.node_type in ("pickup", "delivery"):
                idx = manager.NodeToIndex(node.node_id)
                time_dim.CumulVar(idx).SetRange(node.tw_start_min, node.tw_end_min)

        solver = routing.solver()
        for pickup_id, delivery_id in pd_pairs:
            pickup_index = manager.NodeToIndex(pickup_id)
            delivery_index = manager.NodeToIndex(delivery_id)
            routing.AddPickupAndDelivery(pickup_index, delivery_index)
            solver.Add(routing.VehicleVar(pickup_index) == routing.VehicleVar(delivery_index))
            solver.Add(time_dim.CumulVar(pickup_index) <= time_dim.CumulVar(delivery_index))
            # Per-node disjunctions tied together so a load is dropped as a pair.
            routing.AddDisjunction(
                [pickup_index],
                self._drop_penalty(nodes[pickup_id].load, truck_state),
            )
            routing.AddDisjunction([delivery_index], 0)
            solver.Add(routing.ActiveVar(pickup_index) == routing.ActiveVar(delivery_index))
            if self.SEQUENTIAL_PICKUP_DELIVERY:
                # Served => the delivery is the very next stop after its
                # pickup; dropped => NextVar(pickup) self-loops as OR-Tools
                # requires for inactive nodes.
                solver.Add(
                    routing.NextVar(pickup_index)
                    == delivery_index * routing.ActiveVar(pickup_index)
                    + pickup_index * (1 - routing.ActiveVar(pickup_index))
                )

        params = pywrapcp.DefaultRoutingSearchParameters()
        params.first_solution_strategy = self.FIRST_SOLUTION_STRATEGY
        params.local_search_metaheuristic = self.LOCAL_SEARCH_METAHEURISTIC
        params.time_limit.FromMilliseconds(
            max(1, int(self.solver_time_limit_seconds * 1000))
        )

        solution = routing.SolveWithParameters(params)
        if solution is None:
            return []

        ordered: List[Load] = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            node = nodes[manager.IndexToNode(index)]
            if node.node_type == "pickup" and node.load is not None:
                ordered.append(node.load)
            index = solution.Value(routing.NextVar(index))
        return ordered

    # --------------------------------------------------------- objective hooks
    def _arc_cost(
        self, from_node: _RoutingNode, to_node: _RoutingNode, miles: float
    ) -> int:
        """Objective cost of one arc. v1 minimises pure travel distance."""
        return int(miles * DISTANCE_SCALE)

    def _drop_penalty(self, load: Load, truck_state: TruckState) -> int:
        """Objective penalty for dropping ``load``. v1 serves every feasible
        load (flat dominant penalty), then minimises distance among them."""
        return DROP_PENALTY

    def _replay_to_plan(
        self,
        ordered_loads: List[Load],
        truck_state: TruckState,
        plan_id: int,
    ) -> Plan:
        """Replay the OR-Tools-chosen sequence through the shared evaluation +
        feasibility pipeline (identical to ``PlanBuilderService``)."""
        plan = Plan(
            plan_id=plan_id,
            truck_id=truck_state.truck_id,
            horizon_hours=self.constraints.planning_horizon_hours,
        )
        horizon_end = truck_state.available_at + timedelta(
            hours=self.constraints.planning_horizon_hours
        )
        current = deepcopy(truck_state)

        for load in ordered_loads:
            evaluation = self.evaluate_loads_service.evaluate_one(load, current)

            feasible, _reason = feasibility_checker(
                evaluation,
                current,
                self.constraints,
                self.evaluate_loads_service.cost_model,
            )
            if not feasible:
                continue
            if evaluation.delivery_eta is not None and evaluation.delivery_eta > horizon_end:
                continue

            plan.stops.append(
                PlanStop(
                    load_id=load.load_id,
                    pickup_eta=evaluation.pickup_eta,
                    delivery_eta=evaluation.delivery_eta,
                    deadhead_miles=evaluation.deadhead_miles,
                    load_miles=load.miles,
                    revenue=evaluation.expected_revenue,
                    cost=evaluation.total_cost,
                    profit=evaluation.expected_profit,
                    rationale=f"{self.PLANNER_LABEL} optimized route stop",
                )
            )
            plan.expected_revenue += evaluation.expected_revenue
            plan.expected_cost += evaluation.total_cost
            plan.expected_profit += evaluation.expected_profit
            plan.expected_deadhead_miles += evaluation.deadhead_miles
            plan.expected_load_miles += load.miles
            plan.expected_deadhead_cost += evaluation.deadhead_cost
            plan.expected_load_cost += evaluation.load_cost
            plan.expected_toll_cost += evaluation.toll_cost
            plan.expected_time_cost += evaluation.time_cost

            current = replace(
                current,
                latitude=load.destination_latitude,
                longitude=load.destination_longitude,
                current_city=load.destination_city,
                current_state=load.destination_state,
                available_at=evaluation.delivery_eta,
                driver_hours_left=max(
                    0.0, current.driver_hours_left - evaluation.driver_hours
                ),
            )

        plan.feasible = len(plan.stops) > 0
        plan.rationale = self._explain(plan)
        return plan

    def _empty_plan(self, plan_id: int, truck_state: TruckState) -> Plan:
        plan = Plan(
            plan_id=plan_id,
            truck_id=truck_state.truck_id,
            horizon_hours=self.constraints.planning_horizon_hours,
        )
        plan.feasible = False
        plan.rationale = (
            f"{self.PLANNER_LABEL} found no feasible pickup-and-delivery route "
            "within the planning horizon."
        )
        return plan

    def _explain(self, plan: Plan) -> str:
        if not plan.stops:
            return (
                f"{self.PLANNER_LABEL} found no feasible pickup-and-delivery "
                "route within the planning horizon."
            )
        ids = ", ".join(str(s.load_id) for s in plan.stops)
        return (
            f"{self.PLANNER_LABEL} sequenced {len(plan.stops)} load(s) [{ids}] over "
            f"{plan.horizon_hours:.0f}h horizon. "
            f"Revenue=${plan.expected_revenue:,.2f}, "
            f"Cost=${plan.expected_cost:,.2f}, "
            f"Profit=${plan.expected_profit:,.2f}, "
            f"Deadhead={plan.expected_deadhead_miles:.0f}mi."
        )

    @staticmethod
    def _minutes_from_start(dt: datetime, start: datetime) -> float:
        return (dt - start).total_seconds() / 60.0

    @staticmethod
    def _clamp(value: float, lo: int, hi: int) -> int:
        return int(max(lo, min(hi, value)))
