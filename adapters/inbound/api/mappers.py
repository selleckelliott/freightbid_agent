from dataclasses import asdict

from application.bid_recommender import BidRange
from domain.models.load import Load
from domain.models.truck_state import TruckState

from .schemas import BidLadderRungDTO, BidRangeDTO, LoadDTO, TruckStateDTO


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
    )
