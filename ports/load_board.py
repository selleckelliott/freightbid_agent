"""Phase 7.2 — load-board ingress port.

A :class:`LoadBoardPort` is the integration boundary for an *external-style* load board. It is
deliberately **dumb transport**: it emits raw, feed-shaped dictionaries (the messy alias/unit keys the
Phase 7.1 :class:`~application.ingestion.real_load_schema.RawExternalLoad` contract accepts) and knows
nothing about the domain model. Validation and the map into :class:`~domain.models.load.Load` stay in
the 7.1 contract, so the anti-corruption layer has a single home and the board can never smuggle an
unvalidated object into the engine.

Two adapters implement it (mirroring ``ports/compiled_dispatcher.py``'s no-op/real split):

* :class:`~adapters.outbound.load_board.sandbox.SandboxLoadBoardAdapter` — a seeded, deterministic
  generator of external-style rows (the default; lets the whole pull flow run with no external data), and
* :class:`~adapters.outbound.load_board.replay.RecordedLoadBoardReplayAdapter` — replays a recorded feed
  file through the 7.1 readers.

There is **no real Truckstop API** here. This is the sandbox/replay boundary that proves FreightBid can
consume external-style data through an adapter; live integration is explicitly out of scope for Phase 7.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

__all__ = [
    "LoadBoardAvailability",
    "RawLoadBatch",
    "LoadBoardUnavailable",
    "LoadBoardPort",
    "REASON_DISABLED",
    "REASON_NO_FEED",
    "REASON_FETCH_ERROR",
]

# -- Fail-closed reason codes (stable strings — surfaced in availability + the pull report) ------
REASON_DISABLED = "disabled"
REASON_NO_FEED = "no_feed"
REASON_FETCH_ERROR = "fetch_error"


class LoadBoardUnavailable(RuntimeError):
    """Raised by ``fetch_raw`` when the board cannot serve. Carries a ``REASON_*`` code so the ingest
    service can record *why* the pull degraded instead of crashing the live path."""

    def __init__(self, reason: str, detail: Optional[str] = None):
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class LoadBoardAvailability:
    """Cheap, side-effect-free check: can this board serve right now, and if not, why."""

    available: bool
    source: str
    reason: Optional[str] = None  # None iff available; else a REASON_* code
    detail: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "source": self.source,
            "reason": self.reason,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class RawLoadBatch:
    """One batch of **raw**, feed-shaped rows pulled from a board.

    ``rows`` are plain dicts in the external dialect (NOT domain ``Load`` objects) — they are handed to
    the 7.1 contract for validation. ``cursor`` / ``exhausted`` let a replay board page through a
    recorded feed across successive pulls; a generator board leaves them ``None`` / ``True``.
    """

    source: str
    rows: List[Dict[str, Any]] = field(default_factory=list)
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    cursor: Optional[int] = None
    exhausted: bool = True

    @property
    def count(self) -> int:
        return len(self.rows)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "count": self.count,
            "fetched_at": self.fetched_at.isoformat(),
            "cursor": self.cursor,
            "exhausted": self.exhausted,
        }


class LoadBoardPort(ABC):
    """Outbound port: a source of external-style raw load rows.

    Implementations must **never** return domain objects or mutate any repository — a board only
    *transports* raw rows. Ingestion (validate -> map -> persist) is the ingest service's job.
    """

    #: Stable, human-readable source name (e.g. ``"sandbox"`` / ``"replay"``).
    source: str = "load_board"

    @abstractmethod
    def availability(self) -> LoadBoardAvailability:
        """Whether the board can serve right now, and if not, a ``REASON_*`` code."""

    @abstractmethod
    def fetch_raw(self, *, limit: Optional[int] = None) -> RawLoadBatch:
        """Return a batch of raw external rows, or raise :class:`LoadBoardUnavailable`.

        ``limit`` caps the number of rows returned. Implementations must be **pure** with respect to
        the rest of the system (no repository writes); a replay board may advance its own internal
        cursor so successive calls page through the recorded feed.
        """
