from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class TruckState:
    truck_id: int

    current_city: str
    current_state: str

    latitude: float
    longitude: float

    available_at: datetime

    trailer_type: str

    max_load_capacity: float

    current_load_id: Optional[int]

    home_city: str
    home_state: str

    remaining_capacity: float

    driver_hours_left: float

    speed: float
    heading: float

    timestamp: datetime