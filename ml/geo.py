"""Great-circle distance shared across the ML layer.

Mirrors ``adapters.outbound.distance.haversine`` so the model speaks the same
mileage units as the planner, without the ML package importing the adapter
layer. (Road-vs-straight-line is a Truckstop discovery question; v1 uses
straight-line miles consistently for both labels and features.)
"""
from __future__ import annotations

from math import asin, cos, radians, sin, sqrt

EARTH_RADIUS_MILES = 3956.0


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    rlat1, rlon1, rlat2, rlon2 = map(radians, (lat1, lon1, lat2, lon2))
    dlon, dlat = rlon2 - rlon1, rlat2 - rlat1
    a = sin(dlat / 2) ** 2 + cos(rlat1) * cos(rlat2) * sin(dlon / 2) ** 2
    return 2 * asin(sqrt(a)) * EARTH_RADIUS_MILES
