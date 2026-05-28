from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ScoreResult:
    load_id: int
    score: float
    expected_profit: float
    expected_revenue: float
    deadhead_miles: float
    driver_hours: float
    rate_per_mile: float
    feasible: bool
    rationale: Optional[str] = None
    components: dict = field(default_factory=dict)