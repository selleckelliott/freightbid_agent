"""Cost-plus-margin bid recommender, with optional EV surfacing (Phase 4.3b).

``BidRecommenderService`` keeps its original, deterministic cost-plus-margin behavior.
When an :class:`~application.ev_bid_recommender.EVBidRecommender` is injected (the model
feature flag is on and the 4.2 artifact is present) it *additionally* annotates each
``BidRange`` with the EV ladder — win probability, expected value, and the recommended
EV ask — **without changing** the existing ``min``/``target``/``max`` bid. With no EV
recommender wired (the default), the output is byte-identical to pre-4.3.
"""
from dataclasses import dataclass
from datetime import datetime
from math import isnan
from typing import List, Optional

from domain.models.bid_recommendation import BidOption
from domain.models.load import Load
from domain.models.load_evaluation import LoadEvaluation
from domain.policies.constraints import BiddingConstraints
from domain.policies.scoring_weights import BidPolicy
from ml.features.winnability_features import BidQuery


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
    # -- Phase 4.3b: optional EV surfacing (None unless a model is wired) --------
    # ``winnability_available`` is None when no EV recommender is wired at all, True
    # when the model produced a usable ladder, and False when it was tried but no
    # signal was available (graceful fallback). The remaining fields are populated
    # only when a finite EV recommendation exists, so no NaN ever reaches the wire.
    winnability_available: Optional[bool] = None
    win_probability_at_target: Optional[float] = None
    expected_value_at_target: Optional[float] = None
    ev_recommended_label: Optional[str] = None
    ev_recommended_bid: Optional[float] = None
    ev_recommended_rate_per_mile: Optional[float] = None
    ladder: Optional[List[BidOption]] = None
    # -- Phase 5.1: optional risk-adjusted EV (None unless payment risk is wired) --
    # Populated only when the recommendation actually applied risk adjustment. The
    # ``*_at_target`` fields mirror the recommended rung; ``risk_adjusted_ev_positive``
    # / ``risk_adjusted_warning`` carry the honest "every ask loses money" signal.
    payment_risk_available: Optional[bool] = None
    risk_adjusted_ev_at_target: Optional[float] = None
    p_default_at_target: Optional[float] = None
    p_collect_at_target: Optional[float] = None
    expected_pay_days_at_target: Optional[float] = None
    delay_penalty_at_target: Optional[float] = None
    expected_collected_revenue_at_target: Optional[float] = None
    risk_adjusted_profit_at_target: Optional[float] = None
    risk_adjusted_ev_positive: Optional[bool] = None
    risk_adjusted_warning: Optional[str] = None


def bid_query_from_load(load: Load, decided_at: datetime) -> BidQuery:
    """Map a board :class:`~domain.models.load.Load` to a serving ``BidQuery``.

    Copies only the **observable**, decision-time load fields the winnability feature
    builder reads. The live ``Load`` model carries no broker board columns or
    competition signal, so those query fields stay at their ``unknown``/``NaN``
    defaults (the HistGradientBoosting model handles missing values natively) — live EV
    is therefore coarser than the full-snapshot offline benchmark, and plumbing
    broker/competition through the live board is future work.

    ``decided_at`` is the decision anchor (the requesting truck's ``available_at``); it
    feeds only the load-age and time-of-day features, and being request-derived it keeps
    rendered demos deterministic.
    """
    return BidQuery(
        snapshot_time=decided_at,
        origin_lat=load.origin_latitude,
        origin_lon=load.origin_longitude,
        equipment_type=load.equipment_type,
        loaded_miles=load.miles,
        posted_at=load.created_at,
        weight=load.weight,
        total_rate=load.total_rate,
    )


class BidRecommenderService:
    def __init__(
        self,
        bid_policy: BidPolicy,
        bidding_constraints: BiddingConstraints,
        ev_recommender=None,
    ):
        self.bid_policy = bid_policy
        self.bidding_constraints = bidding_constraints
        # Optional application/ev_bid_recommender.EVBidRecommender. ``None`` => the
        # service is byte-identical to the pre-4.3 cost-plus-margin recommender.
        self._ev = ev_recommender

    def recommend(
        self,
        evaluation: LoadEvaluation,
        *,
        decided_at: Optional[datetime] = None,
    ) -> BidRange:
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

        bid_range = BidRange(
            load_id=evaluation.load.load_id,
            min_bid=min_bid,
            target_bid=target,
            max_bid=max_bid,
            breakeven=cost,
            expected_profit_at_target=target - cost,
            rate_per_mile_at_target=rpm_target,
            rationale=rationale,
        )

        if self._ev is not None:
            self._attach_ev(bid_range, evaluation, decided_at)

        return bid_range

    # -- Phase 4.3b: additive EV annotation -----------------------------------
    def _attach_ev(
        self,
        bid_range: BidRange,
        evaluation: LoadEvaluation,
        decided_at: Optional[datetime],
    ) -> None:
        """Annotate ``bid_range`` with the EV ladder, never touching the margin bid."""
        load = evaluation.load
        decided = decided_at or load.created_at
        query = bid_query_from_load(load, decided)
        est_cost = evaluation.total_cost if evaluation.total_cost else None

        rec = self._ev.recommend(
            query,
            load_id=load.load_id,
            estimated_total_cost=est_cost,
        )
        bid_range.winnability_available = rec.winnability_available
        if not rec.winnability_available:
            return  # graceful fallback: leave EV fields None (no NaN on the wire)

        chosen = rec.option(rec.recommended_label)
        if chosen is None or isnan(chosen.win_probability):
            return  # model present but no in-support candidate; keep EV fields None

        miles = load.miles or 1.0
        bid_range.win_probability_at_target = chosen.win_probability
        bid_range.expected_value_at_target = chosen.expected_value
        bid_range.ev_recommended_label = rec.recommended_label
        bid_range.ev_recommended_bid = rec.recommended_ask
        bid_range.ev_recommended_rate_per_mile = round(rec.recommended_ask / miles, 4)
        bid_range.ladder = rec.options

        # Phase 5.1: fold the recommended rung's risk-adjusted EV breakdown through,
        # only when payment risk was actually applied (else these stay None — the 4.3b
        # output is unchanged).
        if rec.payment_risk_available:
            bid_range.payment_risk_available = True
            bid_range.risk_adjusted_ev_at_target = chosen.risk_adjusted_ev
            bid_range.p_default_at_target = chosen.p_default
            bid_range.p_collect_at_target = chosen.p_collect
            bid_range.expected_pay_days_at_target = chosen.expected_pay_days
            bid_range.delay_penalty_at_target = chosen.delay_penalty
            bid_range.expected_collected_revenue_at_target = (
                chosen.expected_collected_revenue
            )
            bid_range.risk_adjusted_profit_at_target = chosen.risk_adjusted_profit_if_won
            bid_range.risk_adjusted_ev_positive = rec.risk_adjusted_ev_positive
            bid_range.risk_adjusted_warning = rec.risk_adjusted_warning
