"""Broker taxonomy for the load-quality / winnability outcome world (Phase 4.1).

Mirrors ``ml/markets.py``: a *synthetic* pool of brokers whose **hidden latent**
knobs drive the outcome simulator, paired with **noisy observable** signals that a
dispatcher actually sees on the board (and that a model may use at decision time).

The split is the whole point:

* **Observable** (``credit_bucket``, ``days_to_pay``, ``bonded``, ``age_days``,
  ``quick_pay_available``) — what Truckstop shows. These get copied onto each
  ``LoadSnapshotRecord`` and are legitimate decision-time features. A configurable
  slice of brokers has ``credit_bucket == "unknown"`` (paywalled / unrated): the
  *missingness is itself a signal*, never silently imputed.
* **Hidden latent** (``true_pay_days``, ``true_default_prob``, ``rate_bias``) —
  ground truth the simulator uses to realize payment and bid-win outcomes. These
  must **never** appear on a load record, in the snapshot JSONL, or as a feature;
  the observable columns are only a noisy reflection of them, which is exactly what
  makes the winnability problem non-trivial.

Quality is *correlated but noisy*: better brokers tend to be bonded, pay sooner,
default less, and pay a little above market — but the board only shows a noisy,
sometimes-missing view of that, so the model must infer it.

The pool is seeded and reproducible. ``home_market`` + ``posting_weight`` are board
structure (which brokers flood which markets); they are neither a record field nor
a feature.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, TYPE_CHECKING, Tuple

from ml.markets import MARKET_PROFILES

if TYPE_CHECKING:
    from ml.config import BrokersConfig

CREDIT_BUCKETS: Tuple[str, ...] = ("A", "B", "C", "unknown")

# Neutral, obviously-synthetic name parts — no real broker PII ever ships.
_NAME_PREFIX: Tuple[str, ...] = (
    "Summit", "Cardinal", "Frontier", "Granite", "Vector", "Meridian", "Harbor",
    "Cascade", "Ironwood", "Sterling", "Beacon", "Redwood", "Anchor", "Pioneer",
    "Crossroads", "Highline", "Cobalt", "Sequoia", "Birchwood", "Lone Star",
)
_NAME_SUFFIX: Tuple[str, ...] = (
    "Logistics", "Freight", "Transport Brokers", "Dispatch", "Carriers Group",
    "Brokerage", "Shipping", "Trucking", "Cargo Partners", "Freightways",
)


@dataclass(frozen=True)
class BrokerProfile:
    broker_id: str
    name: str
    # -- decision-time observable signals (board-visible; allowed as features) --
    credit_bucket: str            # "A" / "B" / "C" / "unknown"
    days_to_pay: Optional[int]    # shown on the board; None when credit is unknown
    bonded: bool
    age_days: int                 # broker tenure on the board
    quick_pay_available: bool
    # -- hidden latent ground truth (NEVER exposed at decision time) ------------
    true_pay_days: float          # actual mean days-to-pay the simulator realizes
    true_default_prob: float      # actual probability a load goes unpaid
    rate_bias: float              # multiplier on a load's market reservation rpm
    # -- board structure (assignment only; not a record field, not a feature) --
    home_market: str
    posting_weight: float

    @property
    def is_credit_known(self) -> bool:
        return self.credit_bucket != "unknown"


def _synth_name(rng: random.Random, broker_id: str) -> str:
    prefix = rng.choice(_NAME_PREFIX)
    suffix = rng.choice(_NAME_SUFFIX)
    return f"{prefix} {suffix}"


def _hashed_handle(broker_id: str) -> str:
    """Stable short hash, handy if a caller wants a PII-free join key."""
    return hashlib.sha1(broker_id.encode("utf-8")).hexdigest()[:10]


def _credit_bucket_from_quality(rng: random.Random, quality: float) -> str:
    """Noisy map from latent quality to a shown A/B/C bucket (pre-``unknown``)."""
    jitter = rng.gauss(0.0, 0.12)
    q = min(max(quality + jitter, 0.0), 1.0)
    if q >= 0.66:
        return "A"
    if q >= 0.33:
        return "B"
    return "C"


@dataclass(frozen=True)
class BrokerPoolParams:
    pool_size: int = 40
    bonded_fraction: float = 0.45
    unknown_credit_fraction: float = 0.18
    quick_pay_fraction: float = 0.30
    pay_days_fast: float = 18.0      # latent pay-days for the best brokers
    pay_days_slow: float = 55.0      # latent pay-days for the worst brokers
    default_prob_best: float = 0.01
    default_prob_worst: float = 0.16
    seed: int = 43

    @classmethod
    def from_config(cls, cfg: "BrokersConfig") -> "BrokerPoolParams":
        return cls(
            pool_size=cfg.pool_size,
            bonded_fraction=cfg.bonded_fraction,
            unknown_credit_fraction=cfg.unknown_credit_fraction,
            quick_pay_fraction=cfg.quick_pay_fraction,
            pay_days_fast=cfg.pay_days_fast,
            pay_days_slow=cfg.pay_days_slow,
            default_prob_best=cfg.default_prob_best,
            default_prob_worst=cfg.default_prob_worst,
            seed=cfg.seed,
        )

def build_broker_pool(params: BrokerPoolParams) -> Tuple[BrokerProfile, ...]:
    """Seeded, reproducible pool of brokers with correlated-but-noisy quality."""
    rng = random.Random(params.seed)
    markets = MARKET_PROFILES
    market_weights = [m.outbound_density for m in markets]

    brokers: List[BrokerProfile] = []
    for i in range(params.pool_size):
        broker_id = f"BRK{i + 1:04d}"
        quality = rng.random()  # latent 0..1, higher = better to work with

        true_pay_days = (
            params.pay_days_slow
            - quality * (params.pay_days_slow - params.pay_days_fast)
            + rng.gauss(0.0, 3.0)
        )
        true_pay_days = float(max(7.0, true_pay_days))
        true_default_prob = (
            params.default_prob_worst
            - quality * (params.default_prob_worst - params.default_prob_best)
        )
        true_default_prob = float(min(0.40, max(0.0, true_default_prob + rng.gauss(0.0, 0.01))))
        # Good brokers pay a touch above market; weak ones lowball. Weakly tied to
        # quality so rate and payment quality are correlated but not identical.
        rate_bias = float(0.93 + 0.16 * quality + rng.gauss(0.0, 0.04))

        bonded = rng.random() < (0.20 + 0.55 * quality)
        quick_pay_available = rng.random() < (
            params.quick_pay_fraction * (0.6 + 0.8 * quality)
        )
        age_days = int(max(30, rng.gauss(300 + 900 * quality, 250)))

        credit_bucket = _credit_bucket_from_quality(rng, quality)
        # Observed days-to-pay is a noisy read of the latent (brokers self-report /
        # board estimates lag reality).
        days_to_pay: Optional[int] = int(max(5, round(true_pay_days + rng.gauss(0.0, 4.0))))

        home_market = rng.choices(markets, weights=market_weights, k=1)[0].name
        posting_weight = float(max(0.1, rng.gauss(1.0, 0.6)))

        brokers.append(
            BrokerProfile(
                broker_id=broker_id,
                name=_synth_name(rng, broker_id),
                credit_bucket=credit_bucket,
                days_to_pay=days_to_pay,
                bonded=bonded,
                age_days=age_days,
                quick_pay_available=quick_pay_available,
                true_pay_days=true_pay_days,
                true_default_prob=true_default_prob,
                rate_bias=rate_bias,
                home_market=home_market,
                posting_weight=posting_weight,
            )
        )

    # Overwrite a random slice with "unknown" credit (paywalled / unrated): the
    # board shows neither a bucket nor days-to-pay. Latents are untouched — the
    # truth still exists, the dispatcher just can't see it.
    n_unknown = int(round(params.unknown_credit_fraction * params.pool_size))
    for idx in rng.sample(range(params.pool_size), min(n_unknown, params.pool_size)):
        b = brokers[idx]
        brokers[idx] = BrokerProfile(
            broker_id=b.broker_id,
            name=b.name,
            credit_bucket="unknown",
            days_to_pay=None,
            bonded=b.bonded,
            age_days=b.age_days,
            quick_pay_available=b.quick_pay_available,
            true_pay_days=b.true_pay_days,
            true_default_prob=b.true_default_prob,
            rate_bias=b.rate_bias,
            home_market=b.home_market,
            posting_weight=b.posting_weight,
        )

    return tuple(brokers)


def broker_index(pool: Sequence[BrokerProfile]) -> Dict[str, BrokerProfile]:
    return {b.broker_id: b for b in pool}


def sample_broker_for_origin(
    rng: random.Random,
    pool: Sequence[BrokerProfile],
    origin_market: str,
    *,
    home_bias: float = 3.0,
) -> BrokerProfile:
    """Pick a broker for a load originating in ``origin_market``.

    Brokers post ``posting_weight`` loads on average, and are ``home_bias``× more
    likely to post from their home market — so broker quality correlates with
    origin geography (learnable) without being deterministic.
    """
    weights = [
        b.posting_weight * (home_bias if b.home_market == origin_market else 1.0)
        for b in pool
    ]
    return rng.choices(list(pool), weights=weights, k=1)[0]


# Single source of truth for the leakage boundary (used by the generator to copy
# only safe columns onto a load, and by the leakage-guard test). The board shows
# the OBSERVABLE columns; the HIDDEN ones are ground truth the simulator may use
# for labels but that must never reach a load record, the snapshot JSONL, or a
# feature.
OBSERVABLE_BROKER_COLUMNS: Tuple[str, ...] = (
    "broker_id",
    "broker_name",
    "broker_credit_bucket",
    "broker_days_to_pay",
    "broker_bonded",
    "broker_quick_pay_available",
    "broker_age_days",
)
HIDDEN_BROKER_FIELDS: Tuple[str, ...] = (
    "true_pay_days",
    "true_default_prob",
    "rate_bias",
    "posting_weight",
    "home_market",
)


def observable_broker_columns(broker: BrokerProfile) -> Dict[str, object]:
    """Decision-time-safe broker columns to copy onto a ``LoadSnapshotRecord``."""
    return {
        "broker_id": broker.broker_id,
        "broker_name": broker.name,
        "broker_credit_bucket": broker.credit_bucket,
        "broker_days_to_pay": broker.days_to_pay,
        "broker_bonded": broker.bonded,
        "broker_quick_pay_available": broker.quick_pay_available,
        "broker_age_days": broker.age_days,
    }
