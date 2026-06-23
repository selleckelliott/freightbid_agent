"""Phase 7.2 — the thin live ``pull`` flow: board -> validate -> map -> ingest.

:class:`LoadBoardIngestService` is the only place a :class:`~ports.load_board.LoadBoardPort` meets the
load repository. It pulls raw external rows from the board, runs them through the Phase 7.1
:func:`~application.ingestion.import_contract.validate_loads` contract (reject-row-not-batch), and adds
the accepted domain :class:`~domain.models.load.Load` objects to the repository. Bad rows become a
structured error report; an unavailable board degrades to a no-op report (``available=False`` + reason),
never an exception in the live path. The synthetic ingestion path (``POST /loads``) is untouched — this
is an *alternate*, additive ingress.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from application.ingestion.import_contract import validate_loads
from ports.load_board import LoadBoardPort, LoadBoardUnavailable
from ports.load_repository import LoadRepositoryPort


@dataclass(frozen=True)
class BoardIngestReport:
    """The outcome of one ``pull``: what the board offered, what the contract accepted, what persisted."""

    source: str
    available: bool
    fetched: int
    accepted: int
    rejected: int
    load_ids: List[int] = field(default_factory=list)
    errors: List[dict] = field(default_factory=list)
    reason: Optional[str] = None  # set iff the board was unavailable
    replaced: bool = False
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "available": self.available,
            "fetched": self.fetched,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "load_ids": list(self.load_ids),
            "errors": list(self.errors),
            "reason": self.reason,
            "replaced": self.replaced,
            "fetched_at": self.fetched_at.isoformat(),
        }


class LoadBoardIngestService:
    """Pull raw rows from a load board, validate via the 7.1 contract, and ingest the accepted loads."""

    def __init__(self, board: LoadBoardPort, load_repo: LoadRepositoryPort):
        self._board = board
        self._load_repo = load_repo

    def pull(self, *, limit: Optional[int] = None, replace: bool = False) -> BoardIngestReport:
        avail = self._board.availability()
        if not avail.available:
            return BoardIngestReport(
                source=avail.source, available=False, fetched=0, accepted=0, rejected=0,
                reason=avail.reason, replaced=False,
            )
        try:
            batch = self._board.fetch_raw(limit=limit)
        except LoadBoardUnavailable as exc:
            return BoardIngestReport(
                source=self._board.source, available=False, fetched=0, accepted=0, rejected=0,
                reason=exc.reason, replaced=False,
            )

        result = validate_loads(batch.rows)
        if replace:
            self._load_repo.clear()
        added = self._load_repo.add_many(result.loads)
        return BoardIngestReport(
            source=batch.source,
            available=True,
            fetched=batch.count,
            accepted=result.accepted,
            rejected=result.rejected,
            load_ids=[load.load_id for load in added],
            errors=result.error_report(),
            replaced=replace,
            fetched_at=batch.fetched_at,
        )
