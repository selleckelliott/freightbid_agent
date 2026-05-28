from datetime import datetime, timezone

from ports.clock import ClockPort


class SystemClock(ClockPort):
    def now(self) -> datetime:
        return datetime.now(timezone.utc)


class FixedClock(ClockPort):
    def __init__(self, value: datetime):
        self.value = value

    def now(self) -> datetime:
        return self.value
