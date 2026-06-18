from abc import ABC, abstractmethod
from typing import List, Optional

from domain.models.bid_draft import BidDraft


class BidApprovalRepositoryPort(ABC):
    """Storage for human-in-the-loop bid drafts (Phase 4.4).

    Mirrors the load/truck repository idiom. The Phase 4.4 adapter is in-memory only
    (process lifetime); a durable Postgres adapter is deferred.
    """

    @abstractmethod
    def next_id(self) -> int:
        """Return a fresh, monotonic draft id."""
        ...

    @abstractmethod
    def add(self, draft: BidDraft) -> BidDraft: ...

    @abstractmethod
    def get(self, bid_id: int) -> Optional[BidDraft]: ...

    @abstractmethod
    def list_all(self) -> List[BidDraft]: ...

    @abstractmethod
    def update(self, draft: BidDraft) -> BidDraft: ...

    @abstractmethod
    def clear(self) -> None: ...
