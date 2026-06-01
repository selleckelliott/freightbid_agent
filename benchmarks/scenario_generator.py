"""Scenario generator for benchmark/load testing.

Produces JSON scenario files shaped to match `IngestRequest` + `RankRequest`
(see ``adapters/inbound/api/schemas.py``) so each scenario can drive a full
``POST /loads`` → ``POST /rank`` / ``POST /plan`` flow without modification.

Origins (truck + load pickups) are sampled from a curated set of
Intermountain West and West Coast cities; destinations are sampled from a
broader western/central US pool so most lanes are realistic long-haul.

Usage
-----
As a library:

    from benchmarks.scenario_generator import (
        generate_random_load,
        generate_random_truck,
        generate_random_scenario,
        generate_scenarios,
    )

    scenarios = generate_scenarios(1000, seed=42)
    generate_scenarios(1000, seed=42, out_dir="benchmarks/scenarios/gen")

As a script:

    python -m benchmarks.scenario_generator --count 1000 --seed 42 \
        --out-dir benchmarks/scenarios/gen
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

EARTH_RADIUS_MILES = 3956.0

# ---------------------------------------------------------------------------
# City catalogs (name, state, lat, lon)
# ---------------------------------------------------------------------------

# Truck domiciles + load pickups: Intermountain West + West Coast.
ORIGIN_CITIES: List[Dict[str, Any]] = [
    # --- West Coast ---
    {"city": "Seattle",        "state": "WA", "lat": 47.6062, "lon": -122.3321},
    {"city": "Tacoma",         "state": "WA", "lat": 47.2529, "lon": -122.4443},
    {"city": "Spokane",        "state": "WA", "lat": 47.6588, "lon": -117.4260},
    {"city": "Portland",       "state": "OR", "lat": 45.5152, "lon": -122.6784},
    {"city": "Eugene",         "state": "OR", "lat": 44.0521, "lon": -123.0868},
    {"city": "Medford",        "state": "OR", "lat": 42.3265, "lon": -122.8756},
    {"city": "Sacramento",     "state": "CA", "lat": 38.5816, "lon": -121.4944},
    {"city": "Oakland",        "state": "CA", "lat": 37.8044, "lon": -122.2712},
    {"city": "San Francisco",  "state": "CA", "lat": 37.7749, "lon": -122.4194},
    {"city": "Fresno",         "state": "CA", "lat": 36.7378, "lon": -119.7871},
    {"city": "Los Angeles",    "state": "CA", "lat": 34.0522, "lon": -118.2437},
    {"city": "Long Beach",     "state": "CA", "lat": 33.7701, "lon": -118.1937},
    {"city": "San Diego",      "state": "CA", "lat": 32.7157, "lon": -117.1611},
    {"city": "Ontario",        "state": "CA", "lat": 34.0633, "lon": -117.6509},
    {"city": "Stockton",       "state": "CA", "lat": 37.9577, "lon": -121.2908},
    # --- Intermountain West ---
    {"city": "Boise",          "state": "ID", "lat": 43.6150, "lon": -116.2023},
    {"city": "Idaho Falls",    "state": "ID", "lat": 43.4917, "lon": -112.0339},
    {"city": "Billings",       "state": "MT", "lat": 45.7833, "lon": -108.5007},
    {"city": "Missoula",       "state": "MT", "lat": 46.8721, "lon": -113.9940},
    {"city": "Cheyenne",       "state": "WY", "lat": 41.1400, "lon": -104.8202},
    {"city": "Casper",         "state": "WY", "lat": 42.8666, "lon": -106.3131},
    {"city": "Salt Lake City", "state": "UT", "lat": 40.7608, "lon": -111.8910},
    {"city": "Ogden",          "state": "UT", "lat": 41.2230, "lon": -111.9738},
    {"city": "Provo",          "state": "UT", "lat": 40.2338, "lon": -111.6585},
    {"city": "Reno",           "state": "NV", "lat": 39.5296, "lon": -119.8138},
    {"city": "Las Vegas",      "state": "NV", "lat": 36.1699, "lon": -115.1398},
    {"city": "Denver",         "state": "CO", "lat": 39.7392, "lon": -104.9903},
    {"city": "Colorado Springs","state": "CO","lat": 38.8339, "lon": -104.8214},
    {"city": "Grand Junction", "state": "CO", "lat": 39.0639, "lon": -108.5506},
    {"city": "Phoenix",        "state": "AZ", "lat": 33.4484, "lon": -112.0740},
    {"city": "Tucson",         "state": "AZ", "lat": 32.2226, "lon": -110.9747},
    {"city": "Flagstaff",      "state": "AZ", "lat": 35.1983, "lon": -111.6513},
    {"city": "Albuquerque",    "state": "NM", "lat": 35.0844, "lon": -106.6504},
    {"city": "Santa Fe",       "state": "NM", "lat": 35.6870, "lon": -105.9378},
]

# Destinations: origin pool + broader CONUS so lanes can run east as well.
EXTRA_DESTINATION_CITIES: List[Dict[str, Any]] = [
    {"city": "Dallas",      "state": "TX", "lat": 32.7767, "lon": -96.7970},
    {"city": "Fort Worth",  "state": "TX", "lat": 32.7555, "lon": -97.3308},
    {"city": "Houston",     "state": "TX", "lat": 29.7604, "lon": -95.3698},
    {"city": "El Paso",     "state": "TX", "lat": 31.7619, "lon": -106.4850},
    {"city": "Oklahoma City","state": "OK","lat": 35.4676, "lon": -97.5164},
    {"city": "Kansas City", "state": "MO", "lat": 39.0997, "lon": -94.5786},
    {"city": "Omaha",       "state": "NE", "lat": 41.2565, "lon": -95.9345},
    {"city": "Minneapolis", "state": "MN", "lat": 44.9778, "lon": -93.2650},
    {"city": "Chicago",     "state": "IL", "lat": 41.8781, "lon": -87.6298},
    {"city": "St. Louis",   "state": "MO", "lat": 38.6270, "lon": -90.1994},
    {"city": "Memphis",     "state": "TN", "lat": 35.1495, "lon": -90.0490},
    {"city": "Nashville",   "state": "TN", "lat": 36.1627, "lon": -86.7816},
    {"city": "Atlanta",     "state": "GA", "lat": 33.7490, "lon": -84.3880},
]
DESTINATION_CITIES: List[Dict[str, Any]] = ORIGIN_CITIES + EXTRA_DESTINATION_CITIES

EQUIPMENT_TYPES: Sequence[str] = ("Dry Van", "Reefer", "Flatbed")

# Rate-per-mile distribution by equipment (USD/mi).
RPM_RANGE: Dict[str, tuple] = {
    "Dry Van": (1.75, 3.25),
    "Reefer":  (2.10, 3.90),
    "Flatbed": (2.00, 3.60),
}

DEFAULT_HORIZON_HOURS = 48.0
AVG_SPEED_MPH = 50.0  # rough planning speed for window sizing


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    rlat1, rlon1, rlat2, rlon2 = map(radians, (lat1, lon1, lat2, lon2))
    dlon, dlat = rlon2 - rlon1, rlat2 - rlat1
    a = sin(dlat / 2) ** 2 + cos(rlat1) * cos(rlat2) * sin(dlon / 2) ** 2
    return 2 * asin(sqrt(a)) * EARTH_RADIUS_MILES


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _resolve_rng(seed_or_rng: Optional[Any]) -> random.Random:
    if isinstance(seed_or_rng, random.Random):
        return seed_or_rng
    return random.Random(seed_or_rng)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GeneratorConfig:
    """Tunable knobs for scenario generation."""
    loads_min: int = 4
    loads_max: int = 25
    base_time: datetime = datetime(2026, 5, 27, 8, 0, tzinfo=timezone.utc)
    horizon_hours: float = DEFAULT_HORIZON_HOURS
    top_n_min: int = 5
    top_n_max: int = 25

    # Fraction of loads whose origin must sit within ``near_truck_miles`` of
    # the truck. Loads outside that radius are "discovery" loads -- they
    # exercise the feasibility checker but rarely score well.
    near_truck_ratio: float = 0.8
    near_truck_miles: float = 200.0

    # Fraction of loads whose equipment_type matches the truck's trailer.
    # Mismatches are filtered out by the feasibility checker.
    match_trailer_ratio: float = 0.85

    # Cap on haversine origin->destination distance (miles). Keeps lanes
    # short enough to fit inside the truck's `driver_hours_left` budget.
    max_lane_miles: float = 450.0

    # Pickup window opens this many hours after the truck is available, plus
    # the haversine drive time from truck -> origin. Keeps `pickup_eta`
    # inside the window for nearby loads.
    pickup_buffer_hours: float = 1.0
    pickup_window_hours_min: float = 4.0
    pickup_window_hours_max: float = 10.0


def generate_random_truck(
    truck_id: int = 1,
    seed: Optional[Any] = None,
    config: GeneratorConfig = GeneratorConfig(),
) -> Dict[str, Any]:
    """Return a dict matching ``TruckStateDTO``.

    Domicile is sampled from the Intermountain West / West Coast pool.
    """
    rng = _resolve_rng(seed)
    home = rng.choice(ORIGIN_CITIES)
    trailer = rng.choice(EQUIPMENT_TYPES)
    capacity = {"Dry Van": 45000, "Reefer": 44000, "Flatbed": 48000}[trailer]
    available_at = config.base_time + timedelta(hours=rng.uniform(0, 12))
    return {
        "truck_id": truck_id,
        "current_city": home["city"],
        "current_state": home["state"],
        "latitude": home["lat"],
        "longitude": home["lon"],
        "available_at": _iso(available_at),
        "trailer_type": trailer,
        "max_load_capacity": capacity,
        "current_load_id": None,
        "home_city": home["city"],
        "home_state": home["state"],
        "remaining_capacity": capacity,
        "driver_hours_left": round(rng.uniform(8.0, 11.0), 1),
        "speed": 0,
        "heading": 0,
        "timestamp": _iso(available_at),
    }


def generate_random_load(
    load_id: int = 1,
    seed: Optional[Any] = None,
    config: GeneratorConfig = GeneratorConfig(),
    truck_lat: Optional[float] = None,
    truck_lon: Optional[float] = None,
    truck_available_at: Optional[datetime] = None,
    truck_trailer: Optional[str] = None,
    near_truck: bool = False,
    match_trailer: bool = False,
) -> Dict[str, Any]:
    """Return a dict matching ``LoadDTO``.

    If ``truck_lat``/``truck_lon`` are provided and ``near_truck`` is True,
    the load origin is constrained to within ``config.near_truck_miles`` of
    the truck (falls back to the full pool if no city qualifies). When the
    truck position is given, pickup windows are shifted so they open after
    the truck can realistically reach the origin. Destination is constrained
    to within ``config.max_lane_miles`` of the origin so lanes fit a typical
    HOS budget. If ``match_trailer`` is True, ``equipment_type`` is forced
    to ``truck_trailer``.
    """
    rng = _resolve_rng(seed)

    if near_truck and truck_lat is not None and truck_lon is not None:
        nearby = [
            c for c in ORIGIN_CITIES
            if _haversine_miles(truck_lat, truck_lon, c["lat"], c["lon"])
            <= config.near_truck_miles
        ]
        origin = rng.choice(nearby) if nearby else rng.choice(ORIGIN_CITIES)
    else:
        origin = rng.choice(ORIGIN_CITIES)

    # Constrain destination to within max_lane_miles of origin.
    reachable = [
        c for c in DESTINATION_CITIES
        if (c["city"], c["state"]) != (origin["city"], origin["state"])
        and _haversine_miles(origin["lat"], origin["lon"], c["lat"], c["lon"])
        <= config.max_lane_miles
    ]
    if reachable:
        dest = rng.choice(reachable)
    else:
        # Fallback: any non-self city
        while True:
            dest = rng.choice(DESTINATION_CITIES)
            if (dest["city"], dest["state"]) != (origin["city"], origin["state"]):
                break

    miles = round(
        _haversine_miles(origin["lat"], origin["lon"], dest["lat"], dest["lon"]) * 1.15,
        1,
    )
    if match_trailer and truck_trailer:
        equipment = truck_trailer
    else:
        equipment = rng.choice(EQUIPMENT_TYPES)
    rpm_lo, rpm_hi = RPM_RANGE[equipment]
    rate_per_mile = rng.uniform(rpm_lo, rpm_hi)
    total_rate = round(miles * rate_per_mile, 2)

    created = config.base_time

    # Anchor the pickup window so a feasible truck can actually make it.
    if truck_lat is not None and truck_lon is not None and truck_available_at is not None:
        deadhead_mi = _haversine_miles(truck_lat, truck_lon, origin["lat"], origin["lon"]) * 1.15
        drive_hours = deadhead_mi / AVG_SPEED_MPH
        earliest = truck_available_at + timedelta(
            hours=drive_hours + config.pickup_buffer_hours
        )
        # Window opens at `earliest` plus a small jitter (0-4h slack).
        pickup_start = earliest + timedelta(hours=rng.uniform(0.0, 4.0))
    else:
        pickup_start = created + timedelta(hours=rng.uniform(8, 36))

    pickup_end = pickup_start + timedelta(
        hours=rng.uniform(config.pickup_window_hours_min, config.pickup_window_hours_max)
    )
    transit_hours = miles / AVG_SPEED_MPH
    delivery_start = pickup_start + timedelta(hours=transit_hours)
    delivery_end = delivery_start + timedelta(hours=rng.uniform(4, 12))

    return {
        "load_id": load_id,
        "weight": round(rng.uniform(5_000, 44_000), 0),
        "created_at": _iso(created),
        "origin_city": origin["city"],
        "origin_state": origin["state"],
        "origin_latitude": origin["lat"],
        "origin_longitude": origin["lon"],
        "destination_city": dest["city"],
        "destination_state": dest["state"],
        "destination_latitude": dest["lat"],
        "destination_longitude": dest["lon"],
        "pickup_window_start": _iso(pickup_start),
        "pickup_window_end": _iso(pickup_end),
        "delivery_window_start": _iso(delivery_start),
        "delivery_window_end": _iso(delivery_end),
        "miles": miles,
        "total_rate": total_rate,
        "equipment_type": equipment,
    }


def generate_random_scenario(
    scenario_id: int = 1,
    seed: Optional[Any] = None,
    config: GeneratorConfig = GeneratorConfig(),
) -> Dict[str, Any]:
    """Build one self-contained scenario document.

    Shape matches what the API consumes:
      - ``truck`` → TruckStateDTO
      - ``loads`` → list[LoadDTO]   (body for POST /loads is ``{"loads": ...}``)
      - ``top_n`` → int             (used by POST /rank)
    Plus ``name`` / ``description`` / ``scenario_id`` metadata.
    """
    rng = _resolve_rng(seed)
    n_loads = rng.randint(config.loads_min, config.loads_max)

    truck = generate_random_truck(truck_id=scenario_id, seed=rng, config=config)
    truck_lat = truck["latitude"]
    truck_lon = truck["longitude"]
    truck_available_at = datetime.strptime(
        truck["available_at"], "%Y-%m-%dT%H:%M:%SZ"
    ).replace(tzinfo=timezone.utc)

    loads = []
    for i in range(n_loads):
        near = rng.random() < config.near_truck_ratio
        match = rng.random() < config.match_trailer_ratio
        loads.append(
            generate_random_load(
                load_id=i + 1,
                seed=rng,
                config=config,
                truck_lat=truck_lat,
                truck_lon=truck_lon,
                truck_available_at=truck_available_at,
                truck_trailer=truck["trailer_type"],
                near_truck=near,
                match_trailer=match,
            )
        )
    top_n_lo = min(config.top_n_min, n_loads)
    top_n_hi = max(top_n_lo, min(config.top_n_max, n_loads))
    top_n = rng.randint(top_n_lo, top_n_hi)

    return {
        "scenario_id": scenario_id,
        "name": f"scenario_{scenario_id:04d}",
        "description": (
            f"Auto-generated scenario {scenario_id}: truck domiciled in "
            f"{truck['current_city']}, {truck['current_state']} with "
            f"{n_loads} candidate loads."
        ),
        "truck": truck,
        "top_n": top_n,
        "loads": loads,
    }


def generate_scenarios(
    count: int,
    seed: Optional[int] = 0,
    out_dir: Optional[str | Path] = None,
    config: GeneratorConfig = GeneratorConfig(),
) -> List[Dict[str, Any]]:
    """Generate ``count`` scenarios deterministically from ``seed``.

    If ``out_dir`` is provided, each scenario is also written to
    ``{out_dir}/scenario_{NNNN}.json``. Returns the list of scenario dicts
    either way.
    """
    if count <= 0:
        raise ValueError("count must be positive")

    master = _resolve_rng(seed)
    scenarios: List[Dict[str, Any]] = []
    out_path = Path(out_dir) if out_dir else None
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)

    for i in range(1, count + 1):
        # Derive a per-scenario seed so individual scenarios are reproducible
        # in isolation even when re-running a subset of the batch.
        per_seed = master.randrange(2**31)
        scenario = generate_random_scenario(scenario_id=i, seed=per_seed, config=config)
        scenarios.append(scenario)

        if out_path is not None:
            fname = out_path / f"scenario_{i:04d}.json"
            fname.write_text(json.dumps(scenario, indent=2))

    return scenarios


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> None:
    p = argparse.ArgumentParser(description="Generate benchmark scenarios.")
    p.add_argument("--count", type=int, required=True, help="Number of scenarios.")
    p.add_argument("--seed", type=int, default=0, help="Master RNG seed.")
    p.add_argument(
        "--out-dir",
        default="benchmarks/scenarios/gen",
        help="Directory to write scenario_NNNN.json files into.",
    )
    p.add_argument(
        "--loads-min", type=int, default=GeneratorConfig.loads_min,
    )
    p.add_argument(
        "--loads-max", type=int, default=GeneratorConfig.loads_max,
    )
    args = p.parse_args()

    cfg = GeneratorConfig(loads_min=args.loads_min, loads_max=args.loads_max)
    scenarios = generate_scenarios(
        count=args.count, seed=args.seed, out_dir=args.out_dir, config=cfg
    )
    total_loads = sum(len(s["loads"]) for s in scenarios)
    print(
        f"Generated {len(scenarios)} scenarios "
        f"({total_loads} loads) into {args.out_dir}"
    )


if __name__ == "__main__":
    _main()
