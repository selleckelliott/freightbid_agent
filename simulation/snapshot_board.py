"""Decision-time load board over a Phase 3.1 synthetic snapshot stream.

A :class:`SnapshotBoard` wraps the time-stamped ``LoadSnapshotRecord`` stream
produced by ``ml.data.synthetic_history_generator.generate_history`` and answers
the only question the rolling replay asks of the world: *what loads can the truck
actually take right now?* It is the bridge between the ML data vocabulary
(hot-shot equipment codes, ``L-000042`` string ids) and the planner's domain
``Load`` model.

Visibility rules at decision time ``t`` for a truck at ``(lat, lon)`` carrying an
``ml_equipment`` trailer:

* use the most recent snapshot at or before ``t`` (the board the dispatcher would
  see — never a future snapshot, which would be leakage);
* keep only loads of the truck's exact equipment class;
* drop "call for rate" loads (``total_rate is None``) — non-viable, matching the
  Phase 3.1 labeling convention;
* drop loads whose pickup window has already closed (``pickup_end < t``);
* keep only loads whose origin is within ``radius_mi`` of the truck;
* drop loads already consumed earlier in the episode.

Surviving records are adapted to domain ``Load`` objects. The native record is
retained (keyed by the integer load id) so the simulator can recover the chosen
load's true destination when advancing the truck.
"""
from __future__ import annotations

from bisect import bisect_right
from datetime import datetime
from typing import Dict, List, Optional, Sequence

from domain.models.load import Load
from ml.data.load_history_schema import LoadSnapshotRecord
from ml.geo import haversine_miles

# Hot-shot equipment codes that round-trip losslessly through the domain trailer
# vocabulary: this map is the exact inverse of
# ``application.ortools_destination_aware_planner.domain_equipment_to_ml`` for
# these three codes, so a board load mapped ML -> domain -> ML is unchanged and
# the Phase 3.2 planner needs no modification. ``FSD`` has no clean domain
# pre-image (it would collapse onto the same label as ``F``), so v1 episodes draw
# the truck's equipment only from these three classes.
EQUIPMENT_ML_TO_DOMAIN: Dict[str, str] = {
    "F": "Flatbed",
    "FSDV": "Dry Van",
    "HS": "Reefer",
}
ROUND_TRIP_ML_EQUIPMENT = tuple(EQUIPMENT_ML_TO_DOMAIN.keys())


def record_int_id(native_id: str) -> int:
    """Map a generator load id (``L-000042``) to its stable integer sequence.

    The domain ``Load`` model keys on an ``int``; generator ids are unique
    sequential strings, so the trailing number is a lossless, deterministic key.
    Falls back to a hash for any unexpected id shape.
    """
    try:
        return int(native_id.rsplit("-", 1)[-1])
    except (ValueError, IndexError):
        return abs(hash(native_id)) % (10**9)


class SnapshotBoard:
    """Indexed, consumable view over a synthetic world's snapshot records."""

    def __init__(self, records: Sequence[LoadSnapshotRecord]):
        self._by_time: Dict[datetime, List[LoadSnapshotRecord]] = {}
        for rec in records:
            self._by_time.setdefault(rec.snapshot_time, []).append(rec)
        self._times: List[datetime] = sorted(self._by_time.keys())
        self._consumed: set[int] = set()
        self._record_by_int: Dict[int, LoadSnapshotRecord] = {}

    # ------------------------------------------------------------- snapshot clock
    @property
    def snapshot_times(self) -> List[datetime]:
        return list(self._times)

    def active_snapshot_time(self, t: datetime) -> Optional[datetime]:
        """Latest snapshot at or before ``t`` (``None`` if ``t`` predates all)."""
        idx = bisect_right(self._times, t) - 1
        if idx < 0:
            return None
        return self._times[idx]

    def next_snapshot_after(self, t: datetime) -> Optional[datetime]:
        """First snapshot strictly after ``t`` (``None`` if none remain)."""
        idx = bisect_right(self._times, t)
        if idx >= len(self._times):
            return None
        return self._times[idx]

    # ------------------------------------------------------------- visible board
    def visible_loads_at(
        self,
        t: datetime,
        lat: float,
        lon: float,
        radius_mi: float,
        ml_equipment: str,
    ) -> List[Load]:
        snap = self.active_snapshot_time(t)
        if snap is None:
            return []
        out: List[Load] = []
        for rec in self._by_time[snap]:
            if rec.equipment_type != ml_equipment:
                continue
            if rec.total_rate is None:  # "call for rate" -> non-viable
                continue
            if rec.pickup_end < t:  # pickup window already closed
                continue
            int_id = record_int_id(rec.load_id)
            if int_id in self._consumed:
                continue
            if haversine_miles(lat, lon, rec.origin_lat, rec.origin_lon) > radius_mi:
                continue
            out.append(self._to_domain_load(rec, int_id))
        return out

    # ------------------------------------------------------------------ adapter
    def _to_domain_load(self, rec: LoadSnapshotRecord, int_id: int) -> Load:
        self._record_by_int[int_id] = rec
        return Load(
            load_id=int_id,
            weight=rec.weight,
            created_at=rec.posted_at,
            origin_city=rec.origin_city,
            origin_state=rec.origin_state,
            origin_latitude=rec.origin_lat,
            origin_longitude=rec.origin_lon,
            destination_city=rec.destination_city,
            destination_state=rec.destination_state,
            destination_latitude=rec.destination_lat,
            destination_longitude=rec.destination_lon,
            pickup_window_start=rec.pickup_start,
            pickup_window_end=rec.pickup_end,
            delivery_window_start=rec.dropoff_start,
            delivery_window_end=rec.dropoff_end,
            miles=rec.loaded_miles,
            total_rate=rec.total_rate,
            equipment_type=EQUIPMENT_ML_TO_DOMAIN.get(rec.equipment_type, "Flatbed"),
        )

    # ----------------------------------------------------------------- registry
    def record_for(self, int_id: int) -> Optional[LoadSnapshotRecord]:
        """The native record behind a previously-adapted domain load id."""
        return self._record_by_int.get(int_id)

    def is_consumed(self, int_id: int) -> bool:
        return int_id in self._consumed

    def mark_consumed(self, int_id: int) -> None:
        self._consumed.add(int_id)
