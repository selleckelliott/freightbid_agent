from abc import ABC, abstractmethod


class DistanceProviderPort(ABC):
    @abstractmethod
    def miles_between(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float: ...
