from dataclasses import dataclass
from datetime import datetime

@dataclass
class Load:
    load_id: int

    weight: float

    created_at: datetime

    origin_city: str
    origin_state: str

    origin_latitude: float
    origin_longitude: float

    destination_city: str
    destination_state: str

    destination_latitude: float
    destination_longitude: float

    pickup_time: datetime
    delivery_time: datetime

    miles: float
    rate: float

    equipment_type: str