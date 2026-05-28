from abc import ABC, abstractmethod
from typing import List, Optional

from domain.models.truck_state import TruckState


class TruckRepositoryPort(ABC):
    @abstractmethod
    def upsert(self, truck: TruckState) -> TruckState: ...

    @abstractmethod
    def get(self, truck_id: int) -> Optional[TruckState]: ...

    @abstractmethod
    def list_all(self) -> List[TruckState]: ...
