from domain.models.truck_state import TruckState
from domain.scoring.scoring_strategy import ScoringStrategy
from domain.models.load_evaluation import LoadEvaluation

def calculate_profit(load: LoadEvaluation, cost_model) -> float:
    load_cost = (
        load.load_miles * cost_model.fuel_cost_per_mile +
        load.load_miles * cost_model.maintenance_cost_per_mile +
        load.driver_hours * cost_model.driver_cost_per_hour
    )
    deadhead_cost = (
        load.deadhead_miles * cost_model.fuel_cost_per_mile +
        load.deadhead_miles * cost_model.maintenance_cost_per_mile +
        load.driver_hours * cost_model.driver_cost_per_hour
    )
    total_cost = load_cost + deadhead_cost
    profit = load.expected_revenue - total_cost
    return profit

class HeuristicScoringStrategy(ScoringStrategy):

    def __init__(self, scoring_weights, cost_model):
        self.scoring_weights = scoring_weights
        self.cost_model = cost_model

    def score_load(self, load: LoadEvaluation, truck_state: TruckState) -> float:
        profit = calculate_profit(load, self.cost_model)

        score = (
            profit * self.scoring_weights.profit_weight
            - load.deadhead_miles * self.scoring_weights.deadhead_miles_penalty
            - load.driver_hours * self.scoring_weights.driver_hours_penalty
        )

        return score
    
