from domain.models.load import Load
from domain.models.truck_state import TruckState

class EvaluateLoadsService:

    def evaluate_loads(self, loads: list[Load], truck_state: TruckState):
        evaluated_loads = []

        for load in loads:
            deadhead_miles = self.calculate_distance(
                truck_state.latitude, truck_state.longitude,
                load.origin_latitude, load.origin_longitude
            )
            total_miles = deadhead_miles + load.miles
            driver_hours = total_miles / truck_state.speed
            expected_revenue = load.rate * load.miles

            evaluated_loads.append((load, deadhead_miles, total_miles, driver_hours, expected_revenue))

        evaluated_loads.sort(key=lambda x: x[1], reverse=True)
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