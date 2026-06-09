import logging
from datetime import timedelta
from typing import List, Tuple

from domain.models.load import Load
from domain.models.load_evaluation import LoadEvaluation
from domain.models.score_result import ScoreResult
from domain.models.truck_state import TruckState
from domain.policies.constraints import PlanningConstraints
from domain.policies.feasibility import feasibility_checker
from domain.scoring.scoring_strategy import ScoringStrategy

from .evaluate_loads import EvaluateLoadsService

logger = logging.getLogger(__name__)

DEFAULT_TOP_N = 10


class RecommendLoadsService:
    def __init__(
        self,
        scoring_strategy: ScoringStrategy,
        constraints: PlanningConstraints,
        evaluate_loads_service: EvaluateLoadsService,
    ):
        self.scoring_strategy = scoring_strategy
        self.feasibility_checker = feasibility_checker
        self.constraints = constraints
        self.evaluate_loads_service = evaluate_loads_service

    def recommend_loads(
        self,
        loads: List[Load],
        truck_state: TruckState,
        top_n: int = DEFAULT_TOP_N,
    ) -> List[Tuple[LoadEvaluation, ScoreResult]]:
        evaluations = self.evaluate_loads_service.evaluate_loads(loads, truck_state)
        horizon_end = truck_state.available_at + timedelta(
            hours=self.constraints.planning_horizon_hours
        )

        ranked: List[Tuple[LoadEvaluation, ScoreResult]] = []
        for evaluation in evaluations:
            is_feasible, reason = self.feasibility_checker(
                evaluation,
                truck_state,
                self.constraints,
                self.evaluate_loads_service.cost_model,
            )
            if evaluation.delivery_eta and evaluation.delivery_eta > horizon_end:
                is_feasible, reason = False, "Delivery ETA beyond planning horizon"

            if not is_feasible:
                logger.info(
                    "Load %s infeasible: %s",
                    evaluation.load.load_id,
                    reason,
                )
                continue

            result = self.scoring_strategy.score_load(evaluation)
            ranked.append((evaluation, result))

        ranked.sort(key=lambda x: x[1].score, reverse=True)
        return ranked[:top_n]
