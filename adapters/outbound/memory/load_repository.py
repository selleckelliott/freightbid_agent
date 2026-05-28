from typing import Dict, Iterable, List, Optional

from domain.models.load import Load
from ports.load_repository import LoadRepositoryPort


class InMemoryLoadRepository(LoadRepositoryPort):
    def __init__(self) -> None:
        self._loads: Dict[int, Load] = {}

    def add_many(self, loads: Iterable[Load]) -> List[Load]:
        stored = []
        for load in loads:
            self._loads[load.load_id] = load
            stored.append(load)
        return stored

    def get(self, load_id: int) -> Optional[Load]:
        return self._loads.get(load_id)

    def list_all(self) -> List[Load]:
        return list(self._loads.values())

    def clear(self) -> None:
        self._loads.clear()
