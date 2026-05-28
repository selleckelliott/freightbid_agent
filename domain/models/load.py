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

    pickup_window_start: datetime
    pickup_window_end: datetime
    delivery_window_start: datetime
    delivery_window_end: datetime

    miles: float
    total_rate: float

    equipment_type: str

    @property
    def pickup_time(self) -> datetime:
        return self.pickup_window_start

    @property
    def delivery_time(self) -> datetime:
        return self.delivery_window_end

    @property
    def rate_per_mile(self) -> float:
        return self.total_rate / self.miles if self.miles > 0 else 0.0