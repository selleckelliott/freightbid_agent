"""Decision-time feature builder for destination desirability (Phase 3.1).

Every feature here must be knowable *before accepting the load* — at the moment
the load appears on the board (its ``snapshot_time``). The market-density
features are computed against the loads visible on that same board whose origins
sit near the candidate's destination ("if I drop here, what is posted nearby
right now?"). Nothing reads market state after arrival; that would be leakage.

The exact same builder runs at training and at serving time (via
``DestinationQuery``), so there is no train/serve skew.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import cos, pi, sin
from statistics import median
from typing import Any, Dict, List, Protocol, Sequence

from ml.config import FeatureConfig
from ml.geo import haversine_miles
from ml.markets import nearest_zone


def daypart(hour: int) -> str:
    if 6 <= hour < 11:
        return "morning"
    if 11 <= hour < 17:
        return "midday"
    if 17 <= hour < 22:
        return "evening"
    return "overnight"


class _DestinationLike(Protocol):
    destination_lat: float
    destination_lon: float
    destination_state: str
    equipment_type: str
    mode: str
    load_id: str | None

    @property
    def arrival_time(self) -> datetime: ...

    @property
    def load_age_hours(self) -> float: ...


@dataclass(frozen=True)
class DestinationQuery:
    """Lightweight serving-time stand-in for a ``LoadSnapshotRecord``.

    Carries only what the feature builder needs to score a candidate delivery.
    """

    destination_lat: float
    destination_lon: float
    destination_state: str
    equipment_type: str
    arrival_dt: datetime
    mode: str = "TL"
    load_age_hours_value: float = 0.0
    load_id: str | None = None

    @property
    def arrival_time(self) -> datetime:
        return self.arrival_dt

    @property
    def load_age_hours(self) -> float:
        return self.load_age_hours_value


def build_features(
    load: _DestinationLike,
    board: Sequence[Any],
    cfg: FeatureConfig = FeatureConfig(),
) -> Dict[str, Any]:
    """Build the feature dict for one candidate delivery.

    ``board`` is the set of loads visible at decision time (same snapshot). The
    candidate itself, if present, is excluded by ``load_id``.
    """
    arrival = load.arrival_time
    hour = arrival.hour
    dow = arrival.weekday()

    # Distance from this destination to each currently-posted load's origin.
    near: List[tuple[float, Any]] = []
    for other in board:
        if load.load_id is not None and other.load_id == load.load_id:
            continue
        dist = haversine_miles(
            load.destination_lat,
            load.destination_lon,
            other.origin_lat,
            other.origin_lon,
        )
        near.append((dist, other))

    feats: Dict[str, Any] = {
        "destination_lat": load.destination_lat,
        "destination_lon": load.destination_lon,
        "destination_zone": nearest_zone(load.destination_lat, load.destination_lon),
        "destination_state": load.destination_state,
        "equipment_type": load.equipment_type,
        "mode": load.mode,
        "arrival_hour": hour,
        "arrival_dow": dow,
        "is_weekend": int(dow >= 5),
        "arrival_month": arrival.month,
        "sin_hour": sin(2 * pi * hour / 24.0),
        "cos_hour": cos(2 * pi * hour / 24.0),
        "sin_dow": sin(2 * pi * dow / 7.0),
        "cos_dow": cos(2 * pi * dow / 7.0),
        "load_age_hours": load.load_age_hours,
    }

    for radius in cfg.radius_miles:
        r = int(radius)
        feats[f"loads_within_{r}"] = sum(1 for d, _ in near if d <= radius)
        feats[f"equip_match_within_{r}"] = sum(
            1
            for d, other in near
            if d <= radius and other.equipment_type == load.equipment_type
        )
        # Equipment-matched onward loads that are still uncontested on the board
        # ("Load Views" = Be The First / Low) — an easy-to-grab onward supply
        # signal a dispatcher can read at decision time.
        feats[f"open_match_within_{r}"] = sum(
            1
            for d, other in near
            if d <= radius
            and other.equipment_type == load.equipment_type
            and getattr(other, "load_views", "low") in ("be_the_first", "low")
        )

    ring = [other for d, other in near if d <= cfg.rate_radius_miles]
    rpms = [
        other.rate_per_mile for other in ring if other.rate_per_mile is not None
    ]
    miles = [other.loaded_miles for other in ring]
    ages = [other.load_age_hours for other in ring]
    feats["posted_loads_near"] = len(ring)
    feats["median_rate_per_mile_near"] = float(median(rpms)) if rpms else 0.0
    feats["median_loaded_miles_near"] = float(median(miles)) if miles else 0.0
    feats["median_load_age_near"] = float(median(ages)) if ages else 0.0

    return feats


# Columns the model treats as categorical (native HistGBR handling).
CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "destination_zone",
    "destination_state",
    "equipment_type",
    "mode",
)

# Non-feature bookkeeping columns produced alongside features in the dataset.
NON_FEATURE_COLUMNS: tuple[str, ...] = ("label", "snapshot_time", "split")


def feature_columns(all_columns: Sequence[str]) -> List[str]:
    return [c for c in all_columns if c not in NON_FEATURE_COLUMNS]
