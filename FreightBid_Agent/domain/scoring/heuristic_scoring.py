from domain.policies.constraints import CostModel
from domain.policies.scoring_weights import ScoringWeights
from domain.scoring.scoring_strategy import ScoringStrategy
from domain.models.load_evaluation import LoadEvaluation
from services.evaluate_loads import EvaluateLoadsService


class HeuristicScoringStrategy(ScoringStrategy):

    def __init__(self, scoring_weights: ScoringWeights, cost_model: CostModel):
        self.scoring_weights = scoring_weights
        self.cost_model = cost_model

    def score_load(self, load: LoadEvaluation) -> float:
        profit = EvaluateLoadsService.calculate_profit(load, self.cost_model)

        score = (
            profit * self.scoring_weights.profit_weight
            - load.deadhead_miles * self.scoring_weights.deadhead_miles_penalty
            - load.driver_hours * self.scoring_weights.driver_hours_penalty
        )

        return score
    
