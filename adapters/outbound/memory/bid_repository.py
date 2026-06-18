from itertools import count
from typing import Dict, List, Optional

from domain.models.bid_draft import BidDraft
from ports.bid_repository import BidApprovalRepositoryPort


class InMemoryBidApprovalRepository(BidApprovalRepositoryPort):
    """Process-lifetime store for bid drafts.

    Drafts are held by reference, so a domain mutation is visible immediately; ``update``
    is kept explicit to honor the port contract and keep a future Postgres swap clean.
    """

    def __init__(self) -> None:
        self._drafts: Dict[int, BidDraft] = {}
        self._counter = count(1)

    def next_id(self) -> int:
        return next(self._counter)

    def add(self, draft: BidDraft) -> BidDraft:
        self._drafts[draft.bid_id] = draft
        return draft

    def get(self, bid_id: int) -> Optional[BidDraft]:
        return self._drafts.get(bid_id)

    def list_all(self) -> List[BidDraft]:
        return list(self._drafts.values())

    def update(self, draft: BidDraft) -> BidDraft:
        self._drafts[draft.bid_id] = draft
        return draft

    def clear(self) -> None:
        self._drafts.clear()
