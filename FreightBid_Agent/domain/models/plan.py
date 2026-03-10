from dataclasses import dataclass
from datetime import datetime

@dataclass
class Plan:
    plan_id: int

    truck_id: int
    load_ids: list[int]

    expected_revenue: float
    expected_cost: float
    expected_profit: float

    expected_deadhead_miles: float
    expected_load_miles: float
    expected_deadhead_cost: float
    expected_load_cost: float

    feasible: bool

    score: float

    rationale: str