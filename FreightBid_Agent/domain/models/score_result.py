from dataclasses import dataclass
from typing import Optional

@dataclass

class ScoreResult:
    load_id: int
    score: float
    rationale: Optional[str] = None
    expected_profit: float
    deadhead_miles: float
    driver_hours: float
    feasible: bool