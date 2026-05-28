from dataclasses import dataclass


@dataclass
class ScoringWeights:
    profit_weight: float = 1.0
    rate_per_mile_weight: float = 0.0
    deadhead_miles_penalty: float = 0.0
    driver_hours_penalty: float = 0.0


@dataclass
class BidPolicy:
    target_margin: float = 0.20
    min_margin: float = 0.05
    max_margin: float = 0.35
    acceptance_curve_steepness: float = 6.0