from abc import ABC, abstractmethod


class TollEstimatorPort(ABC):
    @abstractmethod
    def estimate(
        self,
        miles: float,
        origin_state: str,
        destination_state: str,
    ) -> float: ...
