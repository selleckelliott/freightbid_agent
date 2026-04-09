from domain.policies.feasibility import feasibility_checker
from services.evaluate_loads import EvaluateLoadsService
from domain.policies.constraints import PlanningConstraints

class RecommendLoadsService:
    def __init__(self, scoring_strategy, constraints: PlanningConstraints):
        self.scoring_strategy = scoring_strategy
        self.feasibility_checker = feasibility_checker
        self.constraints = constraints
        self.evaluate_loads_service = EvaluateLoadsService()

    def recommend_loads(self, loads: list, truck_state):
        feasible_loads = []

        for load in loads:
            load_evaluation = self.evaluate_loads_service.evaluate_loads([load], truck_state)[0]
            is_feasible, reason = self.feasibility_checker(load_evaluation, truck_state, self.constraints)

            if not is_feasible:
                print(f"Load {load_evaluation.load.load_id} is not feasible after evaluation: {reason}")
                continue

            score = self.scoring_strategy.score_load(load_evaluation)
            feasible_loads.append((load_evaluation, score))

        # Sort loads by score in descending order    
        feasible_loads.sort(key=lambda x: x[1], reverse=True)

        return feasible_loads