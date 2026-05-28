from datetime import timedelta
from typing import List

from domain.models.load import Load
from domain.models.load_evaluation import LoadEvaluation
from domain.models.truck_state import TruckState
from domain.policies.constraints import CostModel
from ports.distance_provider import DistanceProviderPort
from ports.toll_estimator import TollEstimatorPort

DEFAULT_AVERAGE_SPEED_MPH = 50.0


class EvaluateLoadsService:
    def __init__(
        self,
        distance_provider: DistanceProviderPort,
        toll_estimator: TollEstimatorPort,
        cost_model: CostModel,
        average_speed_mph: float = DEFAULT_AVERAGE_SPEED_MPH,
        load_unload_hours: float = 1.5,
    ):
        if average_speed_mph <= 0:
            raise ValueError("average_speed_mph must be positive")
        self.distance_provider = distance_provider
        self.toll_estimator = toll_estimator
        self.cost_model = cost_model
        self.average_speed_mph = average_speed_mph
        self.load_unload_hours = load_unload_hours

    def evaluate_loads(
        self, loads: List[Load], truck_state: TruckState
    ) -> List[LoadEvaluation]:
        return [self.evaluate_one(load, truck_state) for load in loads]

    def evaluate_one(self, load: Load, truck_state: TruckState) -> LoadEvaluation:
        deadhead_miles = self.distance_provider.miles_between(
            truck_state.latitude,
            truck_state.longitude,
            load.origin_latitude,
            load.origin_longitude,
        )
        total_miles = deadhead_miles + load.miles
        driver_hours = total_miles / self.average_speed_mph + self.load_unload_hours
        expected_revenue = load.total_rate

        cm = self.cost_model
        deadhead_cost = deadhead_miles * (
            cm.fuel_cost_per_mile * cm.deadhead_fuel_multiplier
            + cm.maintenance_cost_per_mile
        )
        load_cost = load.miles * (cm.fuel_cost_per_mile + cm.maintenance_cost_per_mile)
        toll_cost = self.toll_estimator.estimate(
            load.miles, load.origin_state, load.destination_state
        )
        time_cost = driver_hours * (
            cm.driver_cost_per_hour + cm.time_opportunity_cost_per_hour
        )
        total_cost = deadhead_cost + load_cost + toll_cost + time_cost
        expected_profit = expected_revenue - total_cost

        deadhead_hours = deadhead_miles / self.average_speed_mph
        pickup_eta = max(
            truck_state.available_at + timedelta(hours=deadhead_hours),
            load.pickup_window_start,
        )
        drive_hours = load.miles / self.average_speed_mph
        delivery_eta = pickup_eta + timedelta(
            hours=drive_hours + self.load_unload_hours
        )

        return LoadEvaluation(
            load=load,
            deadhead_miles=deadhead_miles,
            total_miles=total_miles,
            driver_hours=driver_hours,
            expected_revenue=expected_revenue,
            pickup_eta=pickup_eta,
            delivery_eta=delivery_eta,
            deadhead_cost=deadhead_cost,
            load_cost=load_cost,
            toll_cost=toll_cost,
            time_cost=time_cost,
            total_cost=total_cost,
            expected_profit=expected_profit,
        )
