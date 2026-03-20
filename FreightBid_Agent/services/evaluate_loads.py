from domain.models.load import Load
from domain.models.truck_state import TruckState
from domain.models.load_evaluation import LoadEvaluation

class EvaluateLoadsService:

    def evaluate_loads(self, loads: list[Load], truck_state: TruckState):
        evaluated_loads = []

        for load in loads:
            deadhead_miles = self.calculate_distance(
                truck_state.latitude, truck_state.longitude,
                load.origin_latitude, load.origin_longitude
            )
            total_miles = deadhead_miles + load.miles
            speed = truck_state.speed if truck_state.speed > 0 else 1  # Avoid division by zero
            driver_hours = total_miles / speed
            expected_revenue = load.rate * load.miles

            evaluation = LoadEvaluation(
                load=load,
                deadhead_miles=deadhead_miles,
                total_miles=total_miles,
                driver_hours=driver_hours,
                expected_revenue=expected_revenue
            )

            evaluated_loads.append(evaluation)

        return evaluated_loads
    
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        from math import radians, cos, sin, asin, sqrt

        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 3956  # Radius of Earth in miles
        return c * r