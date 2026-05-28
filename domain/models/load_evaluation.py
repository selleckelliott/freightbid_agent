from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from domain.models.load import Load


@dataclass
class LoadEvaluation:
    load: Load
    deadhead_miles: float
    total_miles: float
    driver_hours: float
    expected_revenue: float
    pickup_eta: Optional[datetime] = None
    delivery_eta: Optional[datetime] = None
    deadhead_cost: float = 0.0
    load_cost: float = 0.0
    toll_cost: float = 0.0
    time_cost: float = 0.0
    total_cost: float = 0.0
    expected_profit: float = 0.0