"""Phase 7.2 — a **replay** load board that re-emits a recorded external feed.

Reads a recorded feed file (JSON or CSV) through the Phase 7.1 readers and pages through it across
successive ``fetch_raw`` calls (cursor-based replay), so a recorded board session can be re-run
deterministically. Like the sandbox board it only *transports* raw rows; the 7.1 contract validates and
maps them. A missing/empty/unparseable feed fails closed via :meth:`availability` /
:class:`~ports.load_board.LoadBoardUnavailable` and never crashes app boot.
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from application.ingestion.import_contract import read_rows_from_file
from ports.load_board import (
    REASON_FETCH_ERROR,
    REASON_NO_FEED,
    LoadBoardAvailability,
    LoadBoardPort,
    LoadBoardUnavailable,
    RawLoadBatch,
)


class RecordedLoadBoardReplayAdapter(LoadBoardPort):
    """Replays a recorded external feed file through the 7.1 readers, paging via an internal cursor."""

    source = "replay"

    def __init__(self, feed_path: str | Path, fmt: Optional[str] = None):
        self._path = Path(feed_path)
        self._fmt = fmt
        self._rows: Optional[List[Dict[str, Any]]] = None
        self._load_error: Optional[str] = None
        self._cursor = 0
        self._ensure_loaded()

    # -- feed loading (eager, but failures are captured for availability) ---
    def _ensure_loaded(self) -> None:
        if self._rows is not None or self._load_error is not None:
            return
        if not self._path.exists():
            self._load_error = f"recorded feed not found: {self._path}"
            return
        try:
            self._rows = read_rows_from_file(self._path, self._fmt)
        except Exception as exc:  # noqa: BLE001 — a bad feed must never block boot
            self._load_error = f"unreadable feed {self._path}: {exc}"

    @property
    def total_rows(self) -> int:
        return len(self._rows) if self._rows else 0

    def reset(self) -> None:
        """Rewind the replay cursor to the start of the feed."""
        self._cursor = 0

    # -- port ---------------------------------------------------------------
    def availability(self) -> LoadBoardAvailability:
        if self._load_error is not None:
            return LoadBoardAvailability(
                available=False, source=self.source, reason=REASON_NO_FEED, detail=self._load_error
            )
        if not self._rows:
            return LoadBoardAvailability(
                available=False, source=self.source, reason=REASON_NO_FEED,
                detail=f"recorded feed is empty: {self._path}",
            )
        return LoadBoardAvailability(available=True, source=self.source, detail=str(self._path))

    def fetch_raw(self, *, limit: Optional[int] = None) -> RawLoadBatch:
        if self._load_error is not None:
            raise LoadBoardUnavailable(REASON_NO_FEED, detail=self._load_error)
        if self._rows is None:
            raise LoadBoardUnavailable(REASON_FETCH_ERROR, detail="feed not loaded")
        start = self._cursor
        end = len(self._rows) if limit is None else min(len(self._rows), start + max(0, int(limit)))
        rows = [dict(r) for r in self._rows[start:end]]
        self._cursor = end
        return RawLoadBatch(
            source=self.source,
            rows=rows,
            fetched_at=datetime.now(timezone.utc),
            cursor=end,
            exhausted=end >= len(self._rows),
        )
