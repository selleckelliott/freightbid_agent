"""Phase 7.2 — a seeded, deterministic **sandbox** load board.

Emits external-style raw rows (messy money/weight strings, equipment codes, full or abbreviated state
names, single dates or explicit windows, per-mile *or* total rate) that exercise the Phase 7.1
:class:`~application.ingestion.real_load_schema.RawExternalLoad` contract. It generates *valid* rows by
construction so the live ``pull`` flow ingests cleanly with no external data — the contract's negative
paths are covered by the 7.1 fixtures/tests, not here. Same ``seed`` => byte-identical rows.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from random import Random
from typing import Any, Dict, List, Optional, Tuple

from ports.load_board import (
    REASON_NO_FEED,
    LoadBoardAvailability,
    LoadBoardPort,
    RawLoadBatch,
)

# (city, full state name, 2-letter code, lat, lon)
_CITIES: Tuple[Tuple[str, str, str, float, float], ...] = (
    ("Dallas", "Texas", "TX", 32.7767, -96.7970),
    ("Houston", "Texas", "TX", 29.7604, -95.3698),
    ("Atlanta", "Georgia", "GA", 33.7490, -84.3880),
    ("Memphis", "Tennessee", "TN", 35.1495, -90.0490),
    ("Oklahoma City", "Oklahoma", "OK", 35.4676, -97.5164),
    ("Little Rock", "Arkansas", "AR", 34.7465, -92.2896),
    ("Kansas City", "Missouri", "MO", 39.0997, -94.5786),
    ("Nashville", "Tennessee", "TN", 36.1627, -86.7816),
    ("Phoenix", "Arizona", "AZ", 33.4484, -112.0740),
    ("Denver", "Colorado", "CO", 39.7392, -104.9903),
)

_EQUIPMENT_CODES = ("V", "R", "FD", "HS", "SD", "Reefer", "Flatbed", "Hotshot")

# A fixed base date keeps generated pickup/delivery timestamps deterministic (not wall-clock).
_BASE_DATE = datetime(2024, 6, 3, tzinfo=timezone.utc)


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 3958.7613  # earth radius, miles
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


class SandboxLoadBoardAdapter(LoadBoardPort):
    """A deterministic generator of external-style load rows (the default board)."""

    source = "sandbox"

    def __init__(self, seed: int = 7, count: int = 12, *, base_date: Optional[datetime] = None):
        self._seed = int(seed)
        self._count = max(0, int(count))
        self._base_date = base_date or _BASE_DATE
        self._rows: List[Dict[str, Any]] = self._generate()

    # -- generation ---------------------------------------------------------
    def _generate(self) -> List[Dict[str, Any]]:
        rng = Random(self._seed)
        rows: List[Dict[str, Any]] = []
        for i in range(self._count):
            o = rng.randrange(len(_CITIES))
            d = rng.randrange(len(_CITIES))
            while d == o:
                d = rng.randrange(len(_CITIES))
            oc, os_full, os_code, olat, olon = _CITIES[o]
            dc, ds_full, ds_code, dlat, dlon = _CITIES[d]
            miles = round(_haversine_miles(olat, olon, dlat, dlon), 1)
            rpm = round(rng.uniform(2.10, 4.60), 2)
            weight = rng.randrange(8000, 26000, 500)
            equip = _EQUIPMENT_CODES[rng.randrange(len(_EQUIPMENT_CODES))]
            pickup = self._base_date + timedelta(days=i % 5, hours=rng.randrange(6, 12))
            transit_days = max(1, int(miles // 500) + 1)
            delivery = pickup + timedelta(days=transit_days, hours=rng.randrange(0, 6))

            row: Dict[str, Any] = {
                "posting_id": str(1000 + i),
                "reference": f"SB-{self._seed}-{i:03d}",
                "equipment": equip,
                "weight_lbs": f"{weight:,} lb",
                "origin_city": oc,
                # alternate full state name vs 2-letter code to exercise normalization
                "origin_state": os_full if i % 2 == 0 else os_code,
                "origin_lat": olat,
                "origin_lng": olon,
                "destination_city": dc,
                "destination_state": ds_code if i % 2 == 0 else ds_full,
                "dest_lat": dlat,
                "dest_lng": dlon,
                "trip_miles": miles,
                "posted_at": (pickup - timedelta(days=1)).isoformat(),
                "broker_id": f"MC{200000 + (i * 37) % 700000}",
                "broker_name": f"Sandbox Brokerage {chr(65 + (i % 26))}",
            }
            # alternate rate dialect: per-mile string vs total money string
            if i % 2 == 0:
                row["rate_per_mile"] = f"${rpm:.2f}"
            else:
                row["total_rate"] = f"${round(rpm * miles):,}"
            # alternate window dialect: explicit windows vs single dates
            if i % 3 == 0:
                row["pickup_window_start"] = pickup.isoformat()
                row["pickup_window_end"] = (pickup + timedelta(hours=8)).isoformat()
                row["delivery_window_start"] = delivery.isoformat()
                row["delivery_window_end"] = (delivery + timedelta(hours=8)).isoformat()
            else:
                row["pickup_date"] = pickup.isoformat()
                row["delivery_date"] = delivery.isoformat()
            rows.append(row)
        return rows

    # -- port ---------------------------------------------------------------
    def availability(self) -> LoadBoardAvailability:
        if self._count <= 0:
            return LoadBoardAvailability(
                available=False, source=self.source, reason=REASON_NO_FEED,
                detail="sandbox count is 0",
            )
        return LoadBoardAvailability(available=True, source=self.source)

    def fetch_raw(self, *, limit: Optional[int] = None) -> RawLoadBatch:
        rows = self._rows if limit is None else self._rows[: max(0, int(limit))]
        # deep-ish copy so a caller can never mutate the board's canonical rows
        rows = [dict(r) for r in rows]
        return RawLoadBatch(
            source=self.source,
            rows=rows,
            fetched_at=datetime.now(timezone.utc),
            cursor=len(rows),
            exhausted=True,
        )
