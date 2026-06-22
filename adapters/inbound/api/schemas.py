from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class LoadDTO(BaseModel):
    load_id: int
    weight: float
    created_at: datetime
    origin_city: str
    origin_state: str
    origin_latitude: float
    origin_longitude: float
    destination_city: str
    destination_state: str
    destination_latitude: float
    destination_longitude: float
    pickup_window_start: datetime
    pickup_window_end: datetime
    delivery_window_start: datetime
    delivery_window_end: datetime
    miles: float
    total_rate: float
    equipment_type: str


class TruckStateDTO(BaseModel):
    truck_id: int
    current_city: str
    current_state: str
    latitude: float
    longitude: float
    available_at: datetime
    trailer_type: str
    max_load_capacity: float
    current_load_id: Optional[int] = None
    home_city: str
    home_state: str
    remaining_capacity: float
    driver_hours_left: float
    speed: float = 0.0
    heading: float = 0.0
    timestamp: datetime


class IngestRequest(BaseModel):
    loads: List[LoadDTO]


class IngestResponse(BaseModel):
    accepted: int


class RankRequest(BaseModel):
    truck: TruckStateDTO
    top_n: int = 10
    load_ids: Optional[List[int]] = None


class RankedLoad(BaseModel):
    load_id: int
    score: float
    expected_profit: float
    expected_revenue: float
    rate_per_mile: float
    deadhead_miles: float
    driver_hours: float
    pickup_eta: datetime
    delivery_eta: datetime
    rationale: str
    bid: "BidRangeDTO"


class RankResponse(BaseModel):
    truck_id: int
    ranked: List[RankedLoad]


class PlanStopDTO(BaseModel):
    load_id: int
    pickup_eta: datetime
    delivery_eta: datetime
    deadhead_miles: float
    load_miles: float
    revenue: float
    cost: float
    profit: float
    rationale: str


class PlanResponse(BaseModel):
    plan_id: int
    truck_id: int
    horizon_hours: float
    stops: List[PlanStopDTO]
    expected_revenue: float
    expected_cost: float
    expected_profit: float
    expected_deadhead_miles: float
    expected_load_miles: float
    expected_deadhead_cost: float
    expected_load_cost: float
    expected_toll_cost: float
    expected_time_cost: float
    feasible: bool
    score: float
    rationale: str


class BidLadderRungDTO(BaseModel):
    label: str
    ask_amount: float
    ask_rpm: float
    estimated_cost: float
    profit_if_won: float
    win_probability: float
    expected_value: float
    extrapolated: bool
    rationale: str
    # -- Phase 5.1: optional risk-adjusted EV (null unless payment risk is wired) --
    # Kept in sync with domain ``BidOption`` so ``BidLadderRungDTO(**asdict(rung))``
    # keeps working (the mapper expands every dataclass field as a keyword).
    risk_adjusted_ev: Optional[float] = None
    p_default: Optional[float] = None
    p_collect: Optional[float] = None
    expected_pay_days: Optional[float] = None
    delay_penalty: Optional[float] = None
    expected_collected_revenue: Optional[float] = None
    risk_adjusted_profit_if_won: Optional[float] = None


class BidRangeDTO(BaseModel):
    load_id: int
    min_bid: float
    target_bid: float
    max_bid: float
    breakeven: float
    expected_profit_at_target: float
    rate_per_mile_at_target: float
    rationale: str
    # -- Phase 4.3b: optional EV surfacing (null unless the model is wired) --------
    winnability_available: Optional[bool] = None
    win_probability_at_target: Optional[float] = None
    expected_value_at_target: Optional[float] = None
    ev_recommended_label: Optional[str] = None
    ev_recommended_bid: Optional[float] = None
    ev_recommended_rate_per_mile: Optional[float] = None
    ladder: Optional[List[BidLadderRungDTO]] = None
    # -- Phase 5.1: optional risk-adjusted EV (null unless payment risk is wired) --
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


RankedLoad.model_rebuild()


# -- Phase 4.4: human-in-the-loop bid approval workflow -----------------------


class CreateBidDraftRequest(BaseModel):
    truck: TruckStateDTO
    load_id: int
    actor_id: Optional[str] = None


class EditBidRequest(BaseModel):
    amount: float
    reason: Optional[str] = None
    actor_id: Optional[str] = None


class BidActionRequest(BaseModel):
    actor_id: Optional[str] = None
    note: Optional[str] = None


class BidAuditEventDTO(BaseModel):
    at: datetime
    action: str
    actor_id: str
    from_status: Optional[str] = None
    to_status: str
    note: Optional[str] = None
    amount_before: Optional[float] = None
    amount_after: Optional[float] = None


class BidDraftDTO(BaseModel):
    bid_id: int
    load_id: int
    truck_id: int
    status: str
    recommended_amount: float
    recommended_rate_per_mile: float
    current_amount: float
    delta_from_recommended: float
    delta_percent: float
    rationale: str
    created_at: datetime
    expires_at: datetime
    updated_at: datetime
    edit_reason: Optional[str] = None
    submission_ref: Optional[str] = None
    # Recommendation snapshot (EV surfacing, 4.3b) — null when the model is off.
    winnability_available: Optional[bool] = None
    win_probability: Optional[float] = None
    expected_value: Optional[float] = None
    ev_recommended_label: Optional[str] = None
    ev_recommended_bid: Optional[float] = None
    audit: List[BidAuditEventDTO]


class BidQueueResponse(BaseModel):
    bids: List[BidDraftDTO]
