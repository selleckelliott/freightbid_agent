from typing import Dict, List, Optional

from domain.models.truck_state import TruckState
from ports.truck_repository import TruckRepositoryPort


class InMemoryTruckRepository(TruckRepositoryPort):
    def __init__(self) -> None:
        self._trucks: Dict[int, TruckState] = {}

    def upsert(self, truck: TruckState) -> TruckState:
        self._trucks[truck.truck_id] = truck
        return truck

    def get(self, truck_id: int) -> Optional[TruckState]:
        return self._trucks.get(truck_id)

    def list_all(self) -> List[TruckState]:
        return list(self._trucks.values())
