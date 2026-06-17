"""Synthetic historical load-board generator (Phase 3.1).

Manufactures a JSONL stream of ``LoadSnapshotRecord`` rows with *learnable*
market structure so the destination-desirability model has something real to
learn:

* Origins are sampled in proportion to each market's ``outbound_density``, so
  strong hubs flood the board and weak markets barely appear.
* Destinations follow a gravity model (``dest_density / distance**alpha``):
  loads flow toward strong, nearby markets. A load delivering into a weak market
  therefore has few onward loads originating nearby -> larger next-deadhead.
* Weekday/daypart factors modulate how many loads exist per snapshot, so the
  *arrival time* of a delivery affects how easy the next load is to find.
* A configurable fraction of loads post no rate ("call for rate"), and every
  load carries a ``posted_at`` so load age is observable.
* Phase 3.1.1: each load also carries the real board fields discovered from
  Truckstop — hot-shot ``equipment_type`` (HS/F/FSD/FSDV), ``weight``/``length``
  (and usually-blank ``width``/``height``), ``mode`` (TL/PTL/LTL), and a
  ``load_views`` competition bucket that grows with time-on-board and rate.
* Phase 4.1: each load is posted by a broker sampled from ``ml/brokers.py`` (home
  markets make quality correlate with origin geography), and the load's
  *observable* broker columns + quality flags (``commodity``, ``tarp_required``,
  ``appointment_required``) are attached. Broker/quality randomness is drawn from a
  per-load auxiliary stream, so the original Phase 3.1 generation is byte-identical
  and the destination model is unaffected.

Everything is seeded and reproducible.

CLI::

    python -m ml.data.synthetic_history_generator --config config/ml_config.yaml
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import exp
from pathlib import Path
from typing import List, Sequence

from ml.config import SyntheticDataConfig, load_ml_config
from ml.brokers import (
    BrokerPoolParams,
    BrokerProfile,
    build_broker_pool,
    observable_broker_columns,
    sample_broker_for_origin,
)
from ml.data.load_history_schema import LoadSnapshotRecord, write_jsonl
from ml.geo import haversine_miles
from ml.markets import MARKET_PROFILES, MarketProfile

ROAD_FACTOR = 1.15          # straight-line -> road miles approximation
GRAVITY_ALPHA = 1.5         # distance decay for lane selection
COORD_JITTER_DEG = 0.18     # spread loads around a metro center (~12 mi std)
MIN_RATE_FLOOR = 1.10       # posted rates never fall below this USD/mi
AVG_SPEED_MPH = 50.0

# -- Phase 3.1.1 board-field synthesis --------------------------------------
# weight (lbs) and length (ft) ranges by hot-shot equipment class; hotshot loads
# stay under the board's 0-15,000 lb / 0-40 ft filters seen in discovery.
_WEIGHT_RANGES = {
    "HS":   (1500.0, 12000.0),
    "F":    (6000.0, 15000.0),
    "FSD":  (5000.0, 15000.0),
    "FSDV": (3000.0, 14000.0),
}
_LENGTH_RANGES = {
    "HS":   (8.0, 40.0),
    "F":    (24.0, 40.0),
    "FSD":  (24.0, 40.0),
    "FSDV": (16.0, 40.0),
}
# width/height are blank on the board most of the time -> usually None.
_DIM_PRESENT_FRACTION = 0.4
MODE_WEIGHTS = (("TL", 0.55), ("PTL", 0.33), ("LTL", 0.12))
# "Load Views" competition buckets, keyed by a synthetic view count (descending).
_VIEW_BUCKETS = ((30, "high"), (10, "med"), (1, "low"), (0, "be_the_first"))

# -- Phase 4.1 load-quality synthesis ---------------------------------------
# Commodity vocab by hot-shot equipment class (decision-time observable).
_COMMODITIES = {
    "HS":   ("general freight", "auto parts", "palletized goods", "tools", "equipment"),
    "F":    ("steel", "lumber", "building materials", "pipe", "machinery"),
    "FSD":  ("machinery", "steel coils", "construction equip", "pipe", "lumber"),
    "FSDV": ("machinery", "palletized goods", "building materials", "steel", "equipment"),
}
# Open-deck freight gets tarped far more often than enclosed hot-shot loads.
_TARP_PROB = {"HS": 0.08, "F": 0.45, "FSD": 0.45, "FSDV": 0.30}
_APPOINTMENT_PROB = 0.30


def _pick_quality(rng: random.Random, equipment: str) -> tuple[str, bool, bool]:
    commodity = rng.choice(_COMMODITIES.get(equipment, _COMMODITIES["HS"]))
    tarp_required = rng.random() < _TARP_PROB.get(equipment, 0.1)
    appointment_required = rng.random() < _APPOINTMENT_PROB
    return commodity, tarp_required, appointment_required


# ---------------------------------------------------------------------------
# Weighted sampling helpers
# ---------------------------------------------------------------------------

def _weighted_choice(rng: random.Random, items: Sequence, weights: Sequence[float]):
    total = sum(weights)
    r = rng.random() * total
    upto = 0.0
    for item, w in zip(items, weights):
        upto += w
        if upto >= r:
            return item
    return items[-1]


def _poisson(rng: random.Random, lam: float) -> int:
    """Knuth's algorithm; lam is modest (~40) so this is cheap."""
    if lam <= 0:
        return 0
    target = exp(-lam)
    k, p = 0, 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= target:
            return k - 1


def _weekday_factor(dt: datetime) -> float:
    # Mon-Fri busy, Sat moderate, Sun quiet.
    return (1.0, 1.0, 1.0, 1.0, 1.0, 0.65, 0.45)[dt.weekday()]


def _daypart_factor(hour: int) -> float:
    if 6 <= hour < 18:
        return 1.0          # business hours
    if 18 <= hour < 22:
        return 0.7          # evening
    return 0.4              # overnight


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GeneratorParams:
    start_date: datetime
    days: int
    snapshots_per_day: int
    loads_per_snapshot_mean: float
    unposted_rate_fraction: float
    max_post_age_hours: float
    seed: int

    @classmethod
    def from_config(cls, cfg: SyntheticDataConfig) -> "GeneratorParams":
        return cls(
            start_date=cfg.start_date,
            days=cfg.days,
            snapshots_per_day=cfg.snapshots_per_day,
            loads_per_snapshot_mean=cfg.loads_per_snapshot_mean,
            unposted_rate_fraction=cfg.unposted_rate_fraction,
            max_post_age_hours=cfg.max_post_age_hours,
            seed=cfg.seed,
        )


def _jitter(rng: random.Random, value: float) -> float:
    return value + rng.gauss(0.0, COORD_JITTER_DEG)


def _pick_destination(
    rng: random.Random, origin: MarketProfile
) -> MarketProfile:
    others = [m for m in MARKET_PROFILES if m.name != origin.name]
    weights = []
    for m in others:
        dist = max(haversine_miles(origin.lat, origin.lon, m.lat, m.lon), 1.0)
        weights.append(m.outbound_density / (dist ** GRAVITY_ALPHA))
    return _weighted_choice(rng, others, weights)


def _pick_dimensions(
    rng: random.Random, equipment: str
) -> tuple[float, float, float | None, float | None]:
    w_lo, w_hi = _WEIGHT_RANGES.get(equipment, (3000.0, 15000.0))
    l_lo, l_hi = _LENGTH_RANGES.get(equipment, (16.0, 40.0))
    weight = round(rng.uniform(w_lo, w_hi), -1)        # nearest 10 lb
    length = float(round(rng.uniform(l_lo, l_hi)))     # whole feet
    width = (
        round(rng.uniform(7.0, 8.5), 1)
        if rng.random() < _DIM_PRESENT_FRACTION
        else None
    )
    height = (
        round(rng.uniform(3.0, 10.5), 1)
        if rng.random() < _DIM_PRESENT_FRACTION
        else None
    )
    return weight, length, width, height


def _pick_mode(rng: random.Random, weight: float, length: float) -> str:
    # Heavy/long loads skew full truckload; light loads skew partial / LTL.
    if weight >= 12000.0 or length >= 36.0:
        weights = (("TL", 0.75), ("PTL", 0.20), ("LTL", 0.05))
    elif weight <= 5000.0:
        weights = (("TL", 0.35), ("PTL", 0.42), ("LTL", 0.23))
    else:
        weights = MODE_WEIGHTS
    return _weighted_choice(rng, [m for m, _ in weights], [w for _, w in weights])


def _load_views_bucket(
    rng: random.Random,
    origin: MarketProfile,
    load_age_hours: float,
    rpm: float | None,
) -> str:
    """Synthetic "Load Views" bucket. Views accumulate with time on the board,
    rate attractiveness, and mild market popularity, so fresh loads land in
    ``be_the_first``/``low`` (uncontested) while aging/cheap loads draw a crowd.
    """
    rpm_pull = 0.0 if rpm is None else max(0.0, rpm - 1.8)
    lam = 0.6 + 1.4 * load_age_hours + 6.0 * rpm_pull + 2.0 * origin.outbound_density
    views = _poisson(rng, max(0.0, lam))
    for threshold, bucket in _VIEW_BUCKETS:
        if views >= threshold:
            return bucket
    return "be_the_first"


def _generate_load(
    rng: random.Random,
    load_seq: int,
    snapshot_time: datetime,
    params: GeneratorParams,
    pool: Sequence[BrokerProfile],
) -> LoadSnapshotRecord:
    origin = _weighted_choice(
        rng, MARKET_PROFILES, [m.outbound_density for m in MARKET_PROFILES]
    )
    dest = _pick_destination(rng, origin)

    o_lat, o_lon = _jitter(rng, origin.lat), _jitter(rng, origin.lon)
    d_lat, d_lon = _jitter(rng, dest.lat), _jitter(rng, dest.lon)
    miles = round(haversine_miles(o_lat, o_lon, d_lat, d_lon) * ROAD_FACTOR, 1)

    equipment = _weighted_choice(
        rng,
        [e for e, _ in origin.equipment_mix],
        [w for _, w in origin.equipment_mix],
    )

    vol = (origin.volatility + dest.volatility) / 2.0
    base_rpm = (origin.avg_rate_per_mile + dest.avg_rate_per_mile) / 2.0
    rpm = max(MIN_RATE_FLOOR, base_rpm * (1.0 + rng.gauss(0.0, vol)))
    if rng.random() < params.unposted_rate_fraction:
        total_rate: float | None = None
    else:
        total_rate = round(miles * rpm, 2)

    posted_at = snapshot_time - timedelta(
        hours=rng.uniform(0.0, params.max_post_age_hours)
    )
    pickup_start = snapshot_time + timedelta(hours=rng.uniform(3.0, 18.0))
    pickup_end = pickup_start + timedelta(hours=rng.uniform(2.0, 6.0))
    drive_hours = miles / AVG_SPEED_MPH
    dropoff_start = pickup_end + timedelta(hours=drive_hours)
    dropoff_end = dropoff_start + timedelta(hours=rng.uniform(2.0, 6.0))

    weight, length, width, height = _pick_dimensions(rng, equipment)
    mode = _pick_mode(rng, weight, length)
    load_age_hours = (snapshot_time - posted_at).total_seconds() / 3600.0
    load_views = _load_views_bucket(rng, origin, load_age_hours, rpm)

    # Phase 4.1: attach an observable broker (market-correlated) + quality flags.
    # Draw from a per-load auxiliary stream so the original Phase 3.1 generation
    # draws above are byte-identical — the destination dataset/model is unchanged.
    aux = random.Random(f"{params.seed}:bq:{load_seq}")
    broker = sample_broker_for_origin(aux, pool, origin.name)
    broker_columns = observable_broker_columns(broker)
    commodity, tarp_required, appointment_required = _pick_quality(aux, equipment)

    return LoadSnapshotRecord(
        snapshot_time=snapshot_time,
        load_id=f"L-{load_seq:06d}",
        origin_city=origin.name,
        origin_state=origin.state,
        origin_lat=round(o_lat, 5),
        origin_lon=round(o_lon, 5),
        destination_city=dest.name,
        destination_state=dest.state,
        destination_lat=round(d_lat, 5),
        destination_lon=round(d_lon, 5),
        pickup_start=pickup_start,
        pickup_end=pickup_end,
        dropoff_start=dropoff_start,
        dropoff_end=dropoff_end,
        equipment_type=equipment,
        loaded_miles=miles,
        posted_at=posted_at,
        total_rate=total_rate,
        weight=weight,
        length=length,
        width=width,
        height=height,
        mode=mode,
        load_views=load_views,
        commodity=commodity,
        tarp_required=tarp_required,
        appointment_required=appointment_required,
        **broker_columns,
    )


def generate_history(
    params: GeneratorParams,
    pool: Sequence[BrokerProfile] | None = None,
) -> List[LoadSnapshotRecord]:
    rng = random.Random(params.seed)
    if pool is None:
        pool = build_broker_pool(BrokerPoolParams())
    records: List[LoadSnapshotRecord] = []
    interval = 24.0 / params.snapshots_per_day
    load_seq = 0
    for day in range(params.days):
        for slot in range(params.snapshots_per_day):
            snapshot_time = params.start_date + timedelta(
                days=day, hours=slot * interval
            )
            lam = (
                params.loads_per_snapshot_mean
                * _weekday_factor(snapshot_time)
                * _daypart_factor(snapshot_time.hour)
            )
            n_loads = _poisson(rng, lam)
            for _ in range(n_loads):
                records.append(
                    _generate_load(rng, load_seq, snapshot_time, params, pool)
                )
                load_seq += 1
    return records


def generate_to_file(
    params: GeneratorParams, output_path: str | Path
) -> tuple[int, Path]:
    records = generate_history(params)
    path = Path(output_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    write_jsonl(records, path)
    return len(records), path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic load history.")
    parser.add_argument("--config", default=None, help="Path to ml_config.yaml")
    parser.add_argument("--out", default=None, help="Override output JSONL path")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    cfg = load_ml_config(args.config) if args.config else load_ml_config()
    params = GeneratorParams.from_config(cfg.synthetic_data)
    if args.seed is not None:
        params = GeneratorParams(**{**params.__dict__, "seed": args.seed})
    out = args.out or cfg.synthetic_data.output_path
    count, path = generate_to_file(params, out)
    print(f"Wrote {count:,} load records to {path}")


if __name__ == "__main__":
    main()
