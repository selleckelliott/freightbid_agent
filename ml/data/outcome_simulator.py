"""Outcome simulator: the synthetic *outcome world* for Phase 4.1.

Given decision-time ``LoadSnapshotRecord``s and the hidden ``BrokerProfile`` pool,
this realizes the six outcome processes the user asked for and emits the
ground-truth labels 4.2 will learn from:

1. **brokers pay quickly** / 2. **brokers are risky** — payment is drawn from the
   broker's *latent* ``true_pay_days`` / ``true_default_prob`` → ``payment_outcome``
   + ``realized_pay_days``.
3. **loads highly contested** — a latent ``contention_intensity`` is built from the
   load's rate appeal, origin-market strength, freshness, and its observable
   ``load_views`` bucket (so the board's competition column is a noisy read of it).
4. **which bid prices win** — each load has a hidden ``reservation_rpm`` (the most a
   broker will pay). A carrier's ask **wins when it is at or below** that reserve,
   softened by a logistic. Win probability therefore **falls as the ask rises** —
   the economically correct direction, and the one that gives the EV bid optimizer
   in 4.3 a real "more margin vs. lower win-rate" tradeoff.
5. **loads disappear quickly** — time-to-cover is exponential with a half-life that
   shrinks as contention rises; loads that don't cover within the horizon are
   censored.
6. **no-rate loads require negotiation** — "call for rate" loads (``total_rate is
   None``) settle at a negotiated rate anchored on the broker's reserve.

Leakage contract (mirrors ``ml/data/labeling.py``): the *latents*
(``reservation_rpm``, ``contention_intensity``, ``true_pay_days``,
``true_default_prob``, ``rate_bias``) may appear in **labels** here because they are
ground truth. They must never reach a ``LoadSnapshotRecord``, the snapshot JSONL, or
a feature. Decision-time code only ever sees the observable board columns.

Determinism: every load's outcome is driven by an rng seeded from
``(seed, load_id, snapshot_time)``, so results are independent of iteration order
and byte-reproducible.
"""
from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, TYPE_CHECKING, Tuple

from ml.brokers import BrokerProfile, broker_index
from ml.data.load_history_schema import LoadSnapshotRecord, iso
from ml.data.outcome_schema import (
    PAYMENT_DEFAULT,
    PAYMENT_LATE,
    PAYMENT_PAID,
    BidTrialRecord,
    LoadOutcomeRecord,
)
from ml.markets import market_by_name, nearest_zone

if TYPE_CHECKING:
    from ml.config import OutcomesConfig

_VIEW_SCORE = {"be_the_first": 0.0, "low": 0.33, "med": 0.66, "high": 1.0}


@dataclass(frozen=True)
class OutcomeConfig:
    # -- reservation / win ---------------------------------------------------
    reservation_center_mult: float = 1.05     # reserve sits just above market rate
    reservation_contention_drop: float = 0.15  # high contention lowers the reserve
    reservation_noise: float = 0.06
    reservation_floor_rpm: float = 1.00
    win_logistic_scale_rpm: float = 0.06       # softness of ask ≤ reserve (rpm units)
    bid_trial_rpm_multipliers: Tuple[float, ...] = (0.85, 0.95, 1.0, 1.05, 1.15, 1.25)
    # -- contention ----------------------------------------------------------
    contention_views_weight: float = 0.50
    contention_density_weight: float = 0.25
    contention_rate_weight: float = 0.25
    contention_noise: float = 0.05
    # -- coverage / disappearance -------------------------------------------
    base_cover_halflife_hours: float = 18.0    # low-contention loads can linger
    contention_cover_factor: float = 4.0       # higher → contested loads vanish faster
    cover_horizon_hours: float = 24.0          # censor cap
    # -- negotiation ---------------------------------------------------------
    negotiated_premium: float = 1.0            # settle ≈ reserve × this
    negotiated_noise: float = 0.05
    # -- payment -------------------------------------------------------------
    pay_days_noise: float = 5.0
    late_pay_threshold_days: float = 45.0
    seed: int = 44

    @classmethod
    def from_config(cls, cfg: "OutcomesConfig") -> "OutcomeConfig":
        return cls(
            reservation_center_mult=cfg.reservation_center_mult,
            reservation_contention_drop=cfg.reservation_contention_drop,
            reservation_noise=cfg.reservation_noise,
            win_logistic_scale_rpm=cfg.win_logistic_scale_rpm,
            bid_trial_rpm_multipliers=tuple(cfg.bid_trial_rpm_multipliers),
            base_cover_halflife_hours=cfg.base_cover_halflife_hours,
            contention_cover_factor=cfg.contention_cover_factor,
            cover_horizon_hours=cfg.cover_horizon_hours,
            negotiated_premium=cfg.negotiated_premium,
            late_pay_threshold_days=cfg.late_pay_threshold_days,
            seed=cfg.seed,
        )


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def win_prob(reservation_rpm: float, bid_rpm: float, scale_rpm: float) -> float:
    """P(broker accepts an ask of ``bid_rpm``) = sigmoid((reserve − ask)/scale).

    Monotonically **decreasing** in ``bid_rpm``: a higher ask is less likely to be
    accepted. Pure function — reused by the trial sampler here and by 4.2.
    """
    scale = max(scale_rpm, 1e-6)
    return _sigmoid((reservation_rpm - bid_rpm) / scale)


def _rng_for(seed: int, *parts: object) -> random.Random:
    key = "|".join([str(seed), *(str(p) for p in parts)])
    digest = hashlib.sha1(key.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _market_rate(record: LoadSnapshotRecord) -> float:
    zone = nearest_zone(record.origin_lat, record.origin_lon)
    return market_by_name(zone).avg_rate_per_mile


def _origin_density(record: LoadSnapshotRecord) -> float:
    zone = nearest_zone(record.origin_lat, record.origin_lon)
    return market_by_name(zone).outbound_density


def contention_intensity(
    record: LoadSnapshotRecord, market_rate: float, cfg: OutcomeConfig, rng: random.Random
) -> float:
    """Latent demand pressure on this load, in ~[0, 1].

    Built from the observable ``load_views`` bucket (so the board column is a noisy
    read of contention) plus hidden structure (origin-market strength and how
    attractive the posted rate is relative to market)."""
    views_score = _VIEW_SCORE.get(record.load_views or "low", 0.33)
    density = min(max(_origin_density(record), 0.0), 1.0)
    posted_rpm = record.rate_per_mile
    if posted_rpm is None or market_rate <= 0:
        rate_term = 0.5  # "call for rate": no observable rate appeal
    else:
        rate_term = min(max(0.5 + (posted_rpm / market_rate - 1.0) * 1.5, 0.0), 1.0)
    raw = (
        cfg.contention_views_weight * views_score
        + cfg.contention_density_weight * density
        + cfg.contention_rate_weight * rate_term
        + rng.gauss(0.0, cfg.contention_noise)
    )
    return min(max(raw, 0.0), 1.0)


def reservation_rpm(
    broker: Optional[BrokerProfile],
    market_rate: float,
    contention: float,
    cfg: OutcomeConfig,
    rng: random.Random,
) -> float:
    """Hidden max rpm the broker will pay for this load."""
    rate_bias = broker.rate_bias if broker is not None else 1.0
    center = market_rate * rate_bias * cfg.reservation_center_mult
    center *= 1.0 - cfg.reservation_contention_drop * contention
    center *= 1.0 + rng.gauss(0.0, cfg.reservation_noise)
    return max(cfg.reservation_floor_rpm, center)


def _time_to_cover(
    contention: float, cfg: OutcomeConfig, rng: random.Random
) -> Tuple[float, bool, bool]:
    halflife = cfg.base_cover_halflife_hours / (1.0 + cfg.contention_cover_factor * contention)
    halflife = max(halflife, 1e-3)
    lam = math.log(2.0) / halflife
    t = rng.expovariate(lam)
    if t <= cfg.cover_horizon_hours:
        return t, True, False
    return cfg.cover_horizon_hours, False, True


def _payment(
    broker: Optional[BrokerProfile], cfg: OutcomeConfig, rng: random.Random
) -> Tuple[str, Optional[float]]:
    default_prob = broker.true_default_prob if broker is not None else 0.05
    true_pay_days = broker.true_pay_days if broker is not None else 35.0
    if rng.random() < default_prob:
        return PAYMENT_DEFAULT, None
    pay_days = max(1.0, rng.gauss(true_pay_days, cfg.pay_days_noise))
    outcome = PAYMENT_LATE if pay_days > cfg.late_pay_threshold_days else PAYMENT_PAID
    return outcome, pay_days


def simulate_outcomes(
    records: Sequence[LoadSnapshotRecord],
    pool: Sequence[BrokerProfile],
    cfg: OutcomeConfig,
) -> List[LoadOutcomeRecord]:
    """Realize one ``LoadOutcomeRecord`` per snapshot record (seeded per-load)."""
    by_id: Dict[str, BrokerProfile] = broker_index(pool)
    outcomes: List[LoadOutcomeRecord] = []
    for rec in records:
        broker = by_id.get(rec.broker_id) if rec.broker_id else None
        rng = _rng_for(cfg.seed, rec.load_id, iso(rec.snapshot_time))
        market_rate = _market_rate(rec)
        contention = contention_intensity(rec, market_rate, cfg, rng)
        reserve = reservation_rpm(broker, market_rate, contention, cfg, rng)
        tt_cover, covered, censored = _time_to_cover(contention, cfg, rng)
        payment_outcome, pay_days = _payment(broker, cfg, rng)

        negotiation_required = rec.total_rate is None
        if negotiation_required:
            settled_rpm = reserve * cfg.negotiated_premium * (
                1.0 + rng.gauss(0.0, cfg.negotiated_noise)
            )
            negotiated_rate = round(max(0.0, settled_rpm) * rec.loaded_miles, 2)
        else:
            negotiated_rate = None

        outcomes.append(
            LoadOutcomeRecord(
                snapshot_time=rec.snapshot_time,
                load_id=rec.load_id,
                broker_id=rec.broker_id,
                covered=covered,
                time_to_cover_hours=round(tt_cover, 3),
                cover_censored=censored,
                reservation_rpm=round(reserve, 4),
                contention_intensity=round(contention, 4),
                negotiation_required=negotiation_required,
                negotiated_rate=negotiated_rate,
                payment_outcome=payment_outcome,
                realized_pay_days=round(pay_days, 2) if pay_days is not None else None,
            )
        )
    return outcomes


def sample_bid_trials(
    records: Sequence[LoadSnapshotRecord],
    outcomes: Sequence[LoadOutcomeRecord],
    cfg: OutcomeConfig,
) -> List[BidTrialRecord]:
    """Materialize ``(bid_rpm, won)`` rows over a neutral rpm grid per load.

    Bids are placed relative to the load's **market** rate (the observable anchor a
    carrier actually reasons about), not the hidden reserve, and scored with
    ``win_prob`` + a seeded draw. This is the directly-trainable winnability table.
    """
    reserve_by_key: Dict[Tuple[str, str], float] = {
        (o.load_id, iso(o.snapshot_time)): o.reservation_rpm for o in outcomes
    }
    trials: List[BidTrialRecord] = []
    for rec in records:
        key = (rec.load_id, iso(rec.snapshot_time))
        reserve = reserve_by_key.get(key)
        if reserve is None:
            continue
        market_rate = _market_rate(rec)
        rng = _rng_for(cfg.seed + 1, rec.load_id, iso(rec.snapshot_time))
        for mult in cfg.bid_trial_rpm_multipliers:
            bid_rpm = round(market_rate * mult, 4)
            p = win_prob(reserve, bid_rpm, cfg.win_logistic_scale_rpm)
            won = rng.random() < p
            trials.append(
                BidTrialRecord(
                    snapshot_time=rec.snapshot_time,
                    load_id=rec.load_id,
                    broker_id=rec.broker_id,
                    bid_rpm=bid_rpm,
                    won=won,
                )
            )
    return trials
