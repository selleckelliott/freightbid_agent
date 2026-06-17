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
* drop loads already consumed earlier in the episode;
* drop loads "taken" by competitors (optional view-biased thinning, see below).

**Competition thinning (Phase 3.4).** To stress-test the planners under a busier
board, an optional ``competition_take_rate`` removes that fraction of the world's
loads up front — modelling rival trucks covering them before our truck can bid.
Selection is *view-biased* (high ``load_views`` loads are likelier to be taken,
since they are the contested ones) and fully deterministic given
``competition_seed`` (Efraimidis-Spirakis weighted sampling without replacement),
so the **same** loads vanish for every planner facing the same world — the A/B
stays fair. ``competition_take_rate = 0.0`` (the default) is a no-op, preserving
Phase 3.3 behaviour exactly.

Surviving records are adapted to domain ``Load`` objects. The native record is
retained (keyed by the integer load id) so the simulator can recover the chosen
load's true destination when advancing the truck.
"""
from __future__ import annotations

import random
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

# Relative likelihood that a competitor grabs a load before our truck bids, keyed
# by its ``load_views`` bucket: heavily-viewed loads are the contested ones and so
# disappear first. Monotonic high > med > low > be_the_first.
_VIEW_TAKE_WEIGHT: Dict[str, float] = {
    "high": 8.0,
    "med": 4.0,
    "low": 2.0,
    "be_the_first": 1.0,
}
_DEFAULT_TAKE_WEIGHT = 2.0


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

    def __init__(
        self,
        records: Sequence[LoadSnapshotRecord],
        *,
        competition_take_rate: float = 0.0,
        competition_seed: Optional[int] = None,
    ):
        self._by_time: Dict[datetime, List[LoadSnapshotRecord]] = {}
        for rec in records:
            self._by_time.setdefault(rec.snapshot_time, []).append(rec)
        self._times: List[datetime] = sorted(self._by_time.keys())
        self._consumed: set[int] = set()
        self._record_by_int: Dict[int, LoadSnapshotRecord] = {}
        self._competition_taken: set[int] = self._select_competition_taken(
            records, competition_take_rate, competition_seed
        )

    @staticmethod
    def _select_competition_taken(
        records: Sequence[LoadSnapshotRecord],
        take_rate: float,
        seed: Optional[int],
    ) -> set[int]:
        """Deterministically pick the loads competitors grab (view-biased).

        Aggregates each unique load to the highest ``load_views`` weight it ever
        reaches, then draws a ``take_rate`` fraction without replacement using
        Efraimidis-Spirakis weighted sampling (``key = u**(1/w)``, take the largest
        keys), so heavier-viewed loads are preferentially removed. The whole load
        is removed for the entire episode (a covered load does not reappear).
        """
        if not take_rate or take_rate <= 0.0:
            return set()
        weight_by_id: Dict[int, float] = {}
        for rec in records:
            iid = record_int_id(rec.load_id)
            w = _VIEW_TAKE_WEIGHT.get(rec.load_views, _DEFAULT_TAKE_WEIGHT)
            if w > weight_by_id.get(iid, 0.0):
                weight_by_id[iid] = w
        ids = sorted(weight_by_id)
        n_take = int(round(min(take_rate, 1.0) * len(ids)))
        if n_take <= 0:
            return set()
        rng = random.Random(seed)
        keyed = [
            (rng.random() ** (1.0 / weight_by_id[iid]), iid) for iid in ids
        ]
        keyed.sort(reverse=True)
        return {iid for _, iid in keyed[:n_take]}

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
            if int_id in self._consumed or int_id in self._competition_taken:
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

    # ------------------------------------------------------------- competition
    @property
    def competition_taken_ids(self) -> set[int]:
        """Loads removed up front by the view-biased competition thinning."""
        return set(self._competition_taken)

    def is_competition_taken(self, int_id: int) -> bool:
        return int_id in self._competition_taken
