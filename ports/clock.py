from abc import ABC, abstractmethod
from datetime import datetime


class ClockPort(ABC):
    @abstractmethod
    def now(self) -> datetime: ...
