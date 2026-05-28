from dataclasses import dataclass

from domain.models.load_evaluation import LoadEvaluation
from domain.policies.constraints import BiddingConstraints
from domain.policies.scoring_weights import BidPolicy


@dataclass
class BidRange:
    load_id: int
    min_bid: float
    target_bid: float
    max_bid: float
    breakeven: float
    expected_profit_at_target: float
    rate_per_mile_at_target: float
    rationale: str


class BidRecommenderService:
    def __init__(self, bid_policy: BidPolicy, bidding_constraints: BiddingConstraints):
        self.bid_policy = bid_policy
        self.bidding_constraints = bidding_constraints

    def recommend(self, evaluation: LoadEvaluation) -> BidRange:
        cost = evaluation.total_cost
        bp = self.bid_policy
        bc = self.bidding_constraints

        def _clamp(amount: float) -> float:
            amt = max(amount, bc.min_bid_amount)
            amt = min(amt, bc.max_bid_amount)
            miles = evaluation.load.miles or 1.0
            rpm = amt / miles
            if rpm < bc.min_rate_per_mile:
                amt = bc.min_rate_per_mile * miles
            if rpm > bc.max_rate_per_mile:
                amt = bc.max_rate_per_mile * miles
            return amt

        min_bid = _clamp(cost * (1 + bp.min_margin))
        target = _clamp(cost * (1 + bp.target_margin))
        max_bid = _clamp(cost * (1 + bp.max_margin))

        miles = evaluation.load.miles or 1.0
        rpm_target = target / miles
        rationale = (
            f"Cost=${cost:,.2f}, target margin={bp.target_margin:.0%}, "
            f"target=${target:,.2f} (${rpm_target:.2f}/mi). "
            f"Range [${min_bid:,.2f}, ${max_bid:,.2f}] clamped to "
            f"[${bc.min_bid_amount:.0f}, ${bc.max_bid_amount:.0f}] and "
            f"[${bc.min_rate_per_mile:.2f}, ${bc.max_rate_per_mile:.2f}]/mi."
        )

        return BidRange(
            load_id=evaluation.load.load_id,
            min_bid=min_bid,
            target_bid=target,
            max_bid=max_bid,
            breakeven=cost,
            expected_profit_at_target=target - cost,
            rate_per_mile_at_target=rpm_target,
            rationale=rationale,
        )
