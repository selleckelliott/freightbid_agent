from dataclasses import asdict

from application.bid_recommender import BidRange
from domain.models.bid_draft import BidDraft
from domain.models.load import Load
from domain.models.truck_state import TruckState

from .schemas import (
    BidAuditEventDTO,
    BidDraftDTO,
    BidLadderRungDTO,
    BidRangeDTO,
    LoadDTO,
    TruckStateDTO,
)


def load_from_dto(dto: LoadDTO) -> Load:
    return Load(**dto.model_dump())


def truck_from_dto(dto: TruckStateDTO) -> TruckState:
    return TruckState(**dto.model_dump())


def load_to_dto(load: Load) -> LoadDTO:
    return LoadDTO(**asdict(load))


def bid_range_to_dto(bid: BidRange) -> BidRangeDTO:
    """Map a ``BidRange`` (with its optional EV ladder) to the API DTO.

    The ladder is a list of domain ``BidOption`` dataclasses; convert each rung
    explicitly so the EV fields stay null when no model is wired.
    """
    ladder = (
        [BidLadderRungDTO(**asdict(rung)) for rung in bid.ladder]
        if bid.ladder is not None
        else None
    )
    return BidRangeDTO(
        load_id=bid.load_id,
        min_bid=bid.min_bid,
        target_bid=bid.target_bid,
        max_bid=bid.max_bid,
        breakeven=bid.breakeven,
        expected_profit_at_target=bid.expected_profit_at_target,
        rate_per_mile_at_target=bid.rate_per_mile_at_target,
        rationale=bid.rationale,
        winnability_available=bid.winnability_available,
        win_probability_at_target=bid.win_probability_at_target,
        expected_value_at_target=bid.expected_value_at_target,
        ev_recommended_label=bid.ev_recommended_label,
        ev_recommended_bid=bid.ev_recommended_bid,
        ev_recommended_rate_per_mile=bid.ev_recommended_rate_per_mile,
        ladder=ladder,
        payment_risk_available=bid.payment_risk_available,
        risk_adjusted_ev_at_target=bid.risk_adjusted_ev_at_target,
        p_default_at_target=bid.p_default_at_target,
        p_collect_at_target=bid.p_collect_at_target,
        expected_pay_days_at_target=bid.expected_pay_days_at_target,
        delay_penalty_at_target=bid.delay_penalty_at_target,
        expected_collected_revenue_at_target=bid.expected_collected_revenue_at_target,
        risk_adjusted_profit_at_target=bid.risk_adjusted_profit_at_target,
        risk_adjusted_ev_positive=bid.risk_adjusted_ev_positive,
        risk_adjusted_warning=bid.risk_adjusted_warning,
    )


def bid_draft_to_dto(draft: BidDraft) -> BidDraftDTO:
    """Map a ``BidDraft`` aggregate (status, deltas, snapshot, audit trail) to its DTO."""
    return BidDraftDTO(
        bid_id=draft.bid_id,
        load_id=draft.load_id,
        truck_id=draft.truck_id,
        status=draft.status.value,
        recommended_amount=draft.recommended_amount,
        recommended_rate_per_mile=draft.recommended_rate_per_mile,
        current_amount=draft.current_amount,
        delta_from_recommended=draft.delta_from_recommended,
        delta_percent=draft.delta_percent,
        rationale=draft.rationale,
        created_at=draft.created_at,
        expires_at=draft.expires_at,
        updated_at=draft.updated_at,
        edit_reason=draft.edit_reason,
        submission_ref=draft.submission_ref,
        winnability_available=draft.winnability_available,
        win_probability=draft.win_probability,
        expected_value=draft.expected_value,
        ev_recommended_label=draft.ev_recommended_label,
        ev_recommended_bid=draft.ev_recommended_bid,
        audit=[
            BidAuditEventDTO(
                at=event.at,
                action=event.action,
                actor_id=event.actor_id,
                from_status=event.from_status.value if event.from_status else None,
                to_status=event.to_status.value,
                note=event.note,
                amount_before=event.amount_before,
                amount_after=event.amount_after,
            )
            for event in draft.audit
        ],
    )
