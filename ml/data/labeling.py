"""Label construction: ``expected_next_deadhead_miles`` (Phase 3.1).

For a completed load, the label answers: *after I deliver here, within the
search window, how far must I deadhead to the nearest viable next load?* A
"viable" next load shares equipment, picks up within the window after arrival,
and clears a rate-per-mile bar (loads with no posted rate are not viable). When
nothing qualifies, the label censors at ``max_deadhead_cap_miles``.

Labels may use future information (that is what makes them ground truth); the
leakage discipline lives in *features* (decision-time only) and in the
train/test split (see ``ml.training.dataset``).
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass
from datetime import timedelta
from typing import Callable, List, Sequence

from ml.data.load_history_schema import LoadSnapshotRecord
from ml.geo import haversine_miles

DistanceFn = Callable[[float, float, float, float], float]


@dataclass(frozen=True)
class LabelConfig:
    search_window_hours: float = 8.0
    min_rate_per_mile: float = 1.75
    max_deadhead_cap_miles: float = 300.0


def is_viable_next(
    completed: LoadSnapshotRecord,
    candidate: LoadSnapshotRecord,
    cfg: LabelConfig,
) -> bool:
    if candidate.load_id == completed.load_id:
        return False
    if candidate.equipment_type != completed.equipment_type:
        return False
    rpm = candidate.rate_per_mile
    if rpm is None or rpm < cfg.min_rate_per_mile:
        return False
    return True


def build_label(
    completed: LoadSnapshotRecord,
    candidates: Sequence[LoadSnapshotRecord],
    cfg: LabelConfig,
    *,
    distance_fn: DistanceFn = haversine_miles,
) -> float:
    """Minimum deadhead from ``completed``'s destination to a viable next load.

    ``candidates`` may be the full history; this function applies the window,
    equipment, and rate filters itself.
    """
    arrival = completed.arrival_time
    window_end = arrival + timedelta(hours=cfg.search_window_hours)
    best: float | None = None
    for cand in candidates:
        if cand.pickup_time < arrival or cand.pickup_time > window_end:
            continue
        if not is_viable_next(completed, cand, cfg):
            continue
        dist = distance_fn(
            completed.destination_lat,
            completed.destination_lon,
            cand.origin_lat,
            cand.origin_lon,
        )
        if best is None or dist < best:
            best = dist
    if best is None:
        return cfg.max_deadhead_cap_miles
    return min(best, cfg.max_deadhead_cap_miles)


class FutureLoadIndex:
    """Loads sorted by pickup time for fast window slicing during labeling."""

    def __init__(self, records: Sequence[LoadSnapshotRecord]):
        self._sorted = sorted(records, key=lambda r: r.pickup_time)
        self._keys = [r.pickup_time for r in self._sorted]

    def in_window(self, start, end) -> List[LoadSnapshotRecord]:
        lo = bisect.bisect_left(self._keys, start)
        hi = bisect.bisect_right(self._keys, end)
        return self._sorted[lo:hi]


def label_records(
    completed: Sequence[LoadSnapshotRecord],
    future: Sequence[LoadSnapshotRecord],
    cfg: LabelConfig,
    *,
    distance_fn: DistanceFn = haversine_miles,
) -> List[float]:
    """Vectorized-ish labeling: build a pickup-time index over ``future`` once,
    then label each completed load against only the loads in its window."""
    index = FutureLoadIndex(future)
    labels: List[float] = []
    for rec in completed:
        arrival = rec.arrival_time
        window = index.in_window(
            arrival, arrival + timedelta(hours=cfg.search_window_hours)
        )
        labels.append(build_label(rec, window, cfg, distance_fn=distance_fn))
    return labels
