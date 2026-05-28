from math import asin, cos, radians, sin, sqrt

from ports.distance_provider import DistanceProviderPort

EARTH_RADIUS_MILES = 3956.0


class HaversineDistanceProvider(DistanceProviderPort):
    def miles_between(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        rlat1, rlon1, rlat2, rlon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlon = rlon2 - rlon1
        dlat = rlat2 - rlat1
        a = sin(dlat / 2) ** 2 + cos(rlat1) * cos(rlat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        return c * EARTH_RADIUS_MILES
