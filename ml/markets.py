"""Market taxonomy for the destination desirability model.

Two things live here, intentionally separated:

* ``MARKET_HUBS`` / ``nearest_zone`` — a *real-world* taxonomy of western/central
  freight metros used to bucket any lat/lon into a coarse market ``zone``. This
  is legitimate domain knowledge a dispatcher has; the feature builder may use
  it at decision time.
* ``MARKET_PROFILES`` — *synthetic* per-market knobs (outbound load density,
  average rate, volatility) used only by the synthetic history generator to
  manufacture learnable structure. The model never sees these directly; it must
  infer market strength from observable load density.

Strong hubs emit many loads (easy to leave); weak markets emit few (a truck
that delivers there tends to strand and deadhead farther for its next load).
"""
from __future__ import annotations

from dataclasses import dataclass

from ml.geo import haversine_miles


@dataclass(frozen=True)
class MarketProfile:
    name: str
    state: str
    lat: float
    lon: float
    outbound_density: float   # relative volume of loads originating here
    avg_rate_per_mile: float  # USD/mi center of the local rate distribution
    volatility: float         # std-dev fraction applied to rate + counts
    # Per-market equipment composition of the loads that originate here. This is
    # what makes equipment interact with geography: a Flatbed delivering into a
    # reefer-heavy metro finds few onward flatbed loads (high next-deadhead),
    # something a zone-only baseline cannot see but the model can.
    equipment_mix: tuple[tuple[str, float], ...] = (
        ("Dry Van", 0.50),
        ("Reefer", 0.25),
        ("Flatbed", 0.25),
    )


# name, state, lat, lon, outbound_density, avg_rate_per_mile, volatility, equipment_mix
MARKET_PROFILES: tuple[MarketProfile, ...] = (
    MarketProfile("Dallas",        "TX", 32.7767,  -96.7970, 1.00, 2.40, 0.15,
                  equipment_mix=(("Flatbed", 0.45), ("Dry Van", 0.40), ("Reefer", 0.15))),
    MarketProfile("Houston",       "TX", 29.7604,  -95.3698, 0.95, 2.35, 0.16,
                  equipment_mix=(("Flatbed", 0.50), ("Dry Van", 0.35), ("Reefer", 0.15))),
    MarketProfile("Los Angeles",   "CA", 34.0522, -118.2437, 0.90, 2.60, 0.20,
                  equipment_mix=(("Reefer", 0.45), ("Dry Van", 0.45), ("Flatbed", 0.10))),
    MarketProfile("Denver",        "CO", 39.7392, -104.9903, 0.80, 2.30, 0.18,
                  equipment_mix=(("Dry Van", 0.55), ("Reefer", 0.25), ("Flatbed", 0.20))),
    MarketProfile("Phoenix",       "AZ", 33.4484, -112.0740, 0.75, 2.40, 0.18,
                  equipment_mix=(("Dry Van", 0.45), ("Reefer", 0.40), ("Flatbed", 0.15))),
    MarketProfile("Kansas City",   "MO", 39.0997,  -94.5786, 0.70, 2.20, 0.20,
                  equipment_mix=(("Dry Van", 0.55), ("Reefer", 0.30), ("Flatbed", 0.15))),
    MarketProfile("Salt Lake City","UT", 40.7608, -111.8910, 0.50, 2.20, 0.25,
                  equipment_mix=(("Dry Van", 0.45), ("Flatbed", 0.30), ("Reefer", 0.25))),
    MarketProfile("Las Vegas",     "NV", 36.1699, -115.1398, 0.45, 2.30, 0.30,
                  equipment_mix=(("Dry Van", 0.60), ("Reefer", 0.30), ("Flatbed", 0.10))),
    MarketProfile("Albuquerque",   "NM", 35.0844, -106.6504, 0.30, 2.10, 0.35,
                  equipment_mix=(("Flatbed", 0.40), ("Dry Van", 0.40), ("Reefer", 0.20))),
    MarketProfile("Boise",         "ID", 43.6150, -116.2023, 0.25, 2.15, 0.40,
                  equipment_mix=(("Reefer", 0.45), ("Dry Van", 0.40), ("Flatbed", 0.15))),
)

MARKET_HUBS: tuple[tuple[str, float, float], ...] = tuple(
    (p.name, p.lat, p.lon) for p in MARKET_PROFILES
)

_BY_NAME = {p.name: p for p in MARKET_PROFILES}


def market_by_name(name: str) -> MarketProfile:
    return _BY_NAME[name]


def nearest_zone(lat: float, lon: float) -> str:
    """Coarse market bucket: the nearest known metro hub."""
    return min(
        MARKET_PROFILES,
        key=lambda p: haversine_miles(lat, lon, p.lat, p.lon),
    ).name
