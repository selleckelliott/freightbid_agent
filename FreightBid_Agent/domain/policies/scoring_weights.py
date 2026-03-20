from dataclasses import dataclass

@dataclass
class ScoringWeights:
    profit_weight: float
    rate_per_mile_weight: float
    deadhead_miles_penalty: float
    driver_hours_penalty: float