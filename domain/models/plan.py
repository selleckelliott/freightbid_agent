from dataclasses import dataclass, field
from datetime import datetime
from typing import List


@dataclass
class PlanStop:
    load_id: int
    pickup_eta: datetime
    delivery_eta: datetime
    deadhead_miles: float
    load_miles: float
    revenue: float
    cost: float
    profit: float
    rationale: str = ""


@dataclass
class Plan:
    plan_id: int
    truck_id: int
    horizon_hours: float

    stops: List[PlanStop] = field(default_factory=list)

    expected_revenue: float = 0.0
    expected_cost: float = 0.0
    expected_profit: float = 0.0

    expected_deadhead_miles: float = 0.0
    expected_load_miles: float = 0.0
    expected_deadhead_cost: float = 0.0
    expected_load_cost: float = 0.0
    expected_toll_cost: float = 0.0
    expected_time_cost: float = 0.0

    feasible: bool = True
    score: float = 0.0
    rationale: str = ""

    @property
    def load_ids(self) -> List[int]:
        return [s.load_id for s in self.stops]