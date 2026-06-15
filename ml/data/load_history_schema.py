"""Historical load-snapshot schema + JSONL serialization.

A ``LoadSnapshotRecord`` is one load as it appeared on the board at a particular
``snapshot_time`` (the decision-time clock). Field names mirror the domain
``Load`` model so a future Truckstop adapter can map onto the same shape.

Discovery-driven additions (Phase 3.0.5): ``posted_at`` (load age is observable
at decision time and signals staleness) and a nullable ``total_rate`` ("call for
rate" loads carry no rate and are treated as non-viable when labeling).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

_ISO = "%Y-%m-%dT%H:%M:%SZ"


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime(_ISO)


def parse_dt(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass(frozen=True)
class LoadSnapshotRecord:
    snapshot_time: datetime
    load_id: str
    origin_city: str
    origin_state: str
    origin_lat: float
    origin_lon: float
    destination_city: str
    destination_state: str
    destination_lat: float
    destination_lon: float
    pickup_start: datetime
    pickup_end: datetime
    dropoff_start: datetime
    dropoff_end: datetime
    equipment_type: str
    loaded_miles: float
    posted_at: datetime
    total_rate: float | None = None

    # -- decision-time accessors -------------------------------------------
    @property
    def pickup_time(self) -> datetime:
        return self.pickup_start

    @property
    def arrival_time(self) -> datetime:
        """Estimated arrival at the destination (start of the dropoff window)."""
        return self.dropoff_start

    @property
    def rate_per_mile(self) -> float | None:
        if self.total_rate is None or self.loaded_miles <= 0:
            return None
        return self.total_rate / self.loaded_miles

    @property
    def load_age_hours(self) -> float:
        return (self.snapshot_time - self.posted_at).total_seconds() / 3600.0

    # -- (de)serialization --------------------------------------------------
    _DT_FIELDS = (
        "snapshot_time",
        "pickup_start",
        "pickup_end",
        "dropoff_start",
        "dropoff_end",
        "posted_at",
    )

    def to_json_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, value in self.__dict__.items():
            out[key] = iso(value) if key in self._DT_FIELDS else value
        return out

    @classmethod
    def from_json_dict(cls, data: Dict[str, Any]) -> "LoadSnapshotRecord":
        kwargs = dict(data)
        for field in cls._DT_FIELDS:
            kwargs[field] = parse_dt(kwargs[field])
        return cls(**kwargs)


def write_jsonl(records: Iterable[LoadSnapshotRecord], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec.to_json_dict()) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> List[LoadSnapshotRecord]:
    path = Path(path)
    records: List[LoadSnapshotRecord] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(LoadSnapshotRecord.from_json_dict(json.loads(line)))
    return records
