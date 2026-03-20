from dataclasses import dataclass
from domain.models.load import Load

@dataclass
class LoadEvaluation:
    load: Load
    deadhead_miles: float
    total_miles: float
    driver_hours: float
    expected_revenue: float