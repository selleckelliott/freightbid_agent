from abc import ABC, abstractmethod
from typing import Iterable, List, Optional

from domain.models.load import Load


class LoadRepositoryPort(ABC):
    @abstractmethod
    def add_many(self, loads: Iterable[Load]) -> List[Load]: ...

    @abstractmethod
    def get(self, load_id: int) -> Optional[Load]: ...

    @abstractmethod
    def list_all(self) -> List[Load]: ...

    @abstractmethod
    def clear(self) -> None: ...
