from domain.models.load_evaluation import LoadEvaluation
from domain.models.score_result import ScoreResult
from domain.policies.constraints import CostModel
from domain.policies.scoring_weights import ScoringWeights
from domain.scoring.scoring_strategy import ScoringStrategy


class HeuristicScoringStrategy(ScoringStrategy):

    def __init__(self, scoring_weights: ScoringWeights, cost_model: CostModel):
        self.scoring_weights = scoring_weights
        self.cost_model = cost_model

    def score_load(self, load: LoadEvaluation) -> ScoreResult:
        rate_per_mile = (
            load.expected_revenue / load.total_miles if load.total_miles > 0 else 0.0
        )

        w = self.scoring_weights
        profit_component = load.expected_profit * w.profit_weight
        rpm_component = rate_per_mile * w.rate_per_mile_weight
        deadhead_penalty = load.deadhead_miles * w.deadhead_miles_penalty
        hours_penalty = load.driver_hours * w.driver_hours_penalty

        score = profit_component + rpm_component - deadhead_penalty - hours_penalty

        rationale = (
            f"profit=${load.expected_profit:,.2f} x {w.profit_weight} "
            f"+ rpm=${rate_per_mile:.2f} x {w.rate_per_mile_weight} "
            f"- deadhead={load.deadhead_miles:.0f}mi x {w.deadhead_miles_penalty} "
            f"- hours={load.driver_hours:.1f}h x {w.driver_hours_penalty} "
            f"=> score={score:.2f}"
        )

        return ScoreResult(
            load_id=load.load.load_id,
            score=score,
            expected_profit=load.expected_profit,
            expected_revenue=load.expected_revenue,
            deadhead_miles=load.deadhead_miles,
            driver_hours=load.driver_hours,
            rate_per_mile=rate_per_mile,
            feasible=True,
            rationale=rationale,
            components={
                "profit_component": profit_component,
                "rpm_component": rpm_component,
                "deadhead_penalty": deadhead_penalty,
                "hours_penalty": hours_penalty,
            },
        )