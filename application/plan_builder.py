import logging
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from typing import List

from domain.models.load import Load
from domain.models.plan import Plan, PlanStop
from domain.models.truck_state import TruckState
from domain.policies.constraints import PlanningConstraints
from domain.policies.feasibility import feasibility_checker
from domain.scoring.scoring_strategy import ScoringStrategy

from .evaluate_loads import EvaluateLoadsService

logger = logging.getLogger(__name__)


class PlanBuilderService:
    """Greedy single-truck planner over a configurable horizon (default 48h).

    Picks the next-best feasible load by score, advances truck state to the
    delivery location/time, and repeats until no feasible load fits.
    """

    def __init__(
        self,
        scoring_strategy: ScoringStrategy,
        constraints: PlanningConstraints,
        evaluate_loads_service: EvaluateLoadsService,
    ):
        self.scoring_strategy = scoring_strategy
        self.constraints = constraints
        self.evaluate_loads_service = evaluate_loads_service

    def build_plan(
        self,
        loads: List[Load],
        truck_state: TruckState,
        plan_id: int = 1,
    ) -> Plan:
        horizon = self.constraints.planning_horizon_hours
        horizon_end = truck_state.available_at + timedelta(hours=horizon)

        plan = Plan(
            plan_id=plan_id,
            truck_id=truck_state.truck_id,
            horizon_hours=horizon,
        )

        remaining = {l.load_id: l for l in loads}
        current = deepcopy(truck_state)

        while remaining:
            best = None
            for load in remaining.values():
                evaluation = self.evaluate_loads_service.evaluate_one(load, current)

                feasible, reason = feasibility_checker(
                    evaluation,
                    current,
                    self.constraints,
                    self.evaluate_loads_service.cost_model,
                )
                if not feasible:
                    continue
                if evaluation.delivery_eta > horizon_end:
                    continue

                score = self.scoring_strategy.score_load(evaluation)
                if best is None or score.score > best[1].score:
                    best = (evaluation, score)

            if best is None:
                break

            evaluation, score = best
            load = evaluation.load

            stop = PlanStop(
                load_id=load.load_id,
                pickup_eta=evaluation.pickup_eta,
                delivery_eta=evaluation.delivery_eta,
                deadhead_miles=evaluation.deadhead_miles,
                load_miles=load.miles,
                revenue=evaluation.expected_revenue,
                cost=evaluation.total_cost,
                profit=evaluation.expected_profit,
                rationale=score.rationale or "",
            )
            plan.stops.append(stop)
            plan.expected_revenue += evaluation.expected_revenue
            plan.expected_cost += evaluation.total_cost
            plan.expected_profit += evaluation.expected_profit
            plan.expected_deadhead_miles += evaluation.deadhead_miles
            plan.expected_load_miles += load.miles
            plan.expected_deadhead_cost += evaluation.deadhead_cost
            plan.expected_load_cost += evaluation.load_cost
            plan.expected_toll_cost += evaluation.toll_cost
            plan.expected_time_cost += evaluation.time_cost
            plan.score += score.score

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
            remaining.pop(load.load_id)

        plan.feasible = len(plan.stops) > 0
        plan.rationale = self._explain(plan)
        return plan

    @staticmethod
    def _explain(plan: Plan) -> str:
        if not plan.stops:
            return "No feasible load could be sequenced within the planning horizon."
        ids = ", ".join(str(s.load_id) for s in plan.stops)
        return (
            f"Sequenced {len(plan.stops)} load(s) [{ids}] over {plan.horizon_hours:.0f}h "
            f"horizon. Revenue=${plan.expected_revenue:,.2f}, "
            f"Cost=${plan.expected_cost:,.2f}, "
            f"Profit=${plan.expected_profit:,.2f}, "
            f"Deadhead={plan.expected_deadhead_miles:.0f}mi."
        )
