from domain.policies.feasibility import feasibility_checker

class RecommendLoadsService:
    def __init__(self, scoring_strategy):
        self.scoring_strategy = scoring_strategy
        self.feasibility_checker = feasibility_checker

    def recommend_loads(self, loads, truck_state):
        feasible_loads = []

        for load in loads:
            is_feasible, reason = self.feasibility_checker(load, truck_state, self.constraints)

            if not is_feasible:
                print(f"Load {load.id} is not feasible: {reason}")
                continue
            
            score = self.scoring_strategy.score_load(load, truck_state)
            feasible_loads.append((load, score))
            feasible_loads.sort(key=lambda x: x[1], reverse=True)
            
            return feasible_loads