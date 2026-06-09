from dataclasses import dataclass


@dataclass
class PlanningConstraints:
    max_deadhead_miles: float
    max_load_miles: float
    max_total_miles: float

    max_deadhead_cost: float
    max_load_cost: float
    max_total_cost: float

    min_expected_profit: float

    max_driver_hours: float

    planning_time_limit_seconds: int

    planning_horizon_hours: float = 48.0
    average_load_unload_hours: float = 1.5


@dataclass
class CostModel:
    fuel_cost_per_mile: float
    driver_cost_per_hour: float
    maintenance_cost_per_mile: float
    time_opportunity_cost_per_hour: float = 0.0
    deadhead_fuel_multiplier: float = 1.0


@dataclass
class BiddingConstraints:
    min_bid_amount: float
    max_bid_amount: float

    min_rate_per_mile: float
    max_rate_per_mile: float

    bidding_time_limit_seconds: int
