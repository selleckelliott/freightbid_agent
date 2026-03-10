from dataclasses import dataclass
from datetime import datetime
from domain.enums.bid_status import BidStatus

@dataclass
class Bid:
    bid_id: int

    load_id: int
    truck_id: int
    plan_id: int

    bid_amount: float

    rate_per_mile: float

    created_at: datetime

    expected_profit: float

    acceptance_probability: float

    status: BidStatus  # e.g., 'pending', 'accepted', 'rejected'