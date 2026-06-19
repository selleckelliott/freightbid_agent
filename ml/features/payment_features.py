"""Decision-time feature builder for payment risk (Phase 5.2).

Whether a broker pays — and how slowly — is a property of **the broker**, not of the
ask. So this builder is the payment analogue of ``ml/features/winnability_features.py``
with one deliberate amputation: **every ask-derived column is removed**. There is no
``bid_rpm``, no ``ask_to_market_ratio``, no ``ask_to_posted_ratio`` here, because the
rate a carrier offers has no bearing on whether the broker's check clears. What remains
is the observable board picture of the load and, above all, the broker: credit bucket,
posted days-to-pay, bonded/quick-pay flags, account age — the columns a dispatcher
eyeballs when deciding whether a load is worth the collection risk.

``has_posted_rate`` is kept: it is an observable *load* attribute (a "call for rate"
load is a different kind of posting), not an ask ratio. ``market_rate`` is kept too — it
is a coarse market read derived from the load's origin, knowable before any ask exists.

Leakage discipline matches Phase 4.2: nothing reads the hidden outcome world (the
broker's true default probability, true pay days, rate bias, or the load's reservation
rpm / contention). Unknown-credit brokers legitimately carry ``broker_days_to_pay=None``
— that missingness is itself signal, so it is left as ``NaN`` for the gradient booster
to branch on rather than imputed away.

The same builder runs at training and at serving time and accepts either a
``LoadSnapshotRecord`` or a serving-time :class:`~ml.features.winnability_features.BidQuery`
(payment simply ignores the ask that ``BidQuery`` also carries), so there is no
train/serve skew.
"""
from __future__ import annotations

from math import cos, nan, pi, sin
from typing import Any, Dict, List, Sequence

from ml.features.winnability_features import (
    _SnapshotLike,
    market_rate_for,
)
from ml.markets import market_by_name, nearest_zone


def _num_bool(value) -> float:
    """Boolean → float with ``None`` preserved as ``NaN`` (a real "unknown")."""
    if value is None:
        return nan
    return 1.0 if value else 0.0


def build_payment_features(load: _SnapshotLike) -> Dict[str, Any]:
    """Build the observable, ask-free feature dict for one load's payment risk.

    ``load`` is any decision-time snapshot (a ``LoadSnapshotRecord`` or a ``BidQuery``).
    Every value is knowable on the board before a single dollar is bid.
    """
    market_rate = market_rate_for(load.origin_lat, load.origin_lon)
    has_posted = load.rate_per_mile is not None

    decided = load.snapshot_time
    hour = decided.hour
    dow = decided.weekday()
    density = market_by_name(nearest_zone(load.origin_lat, load.origin_lon)).outbound_density

    return {
        # -- market read (observable, ask-free) ----------------------------
        "market_rate": float(market_rate),
        "has_posted_rate": 1.0 if has_posted else 0.0,
        # -- load attributes -----------------------------------------------
        "loaded_miles": float(load.loaded_miles),
        "weight": float(load.weight),
        "length": float(load.length),
        "equipment_type": load.equipment_type,
        "mode": load.mode,
        "commodity": load.commodity or "unknown",
        "tarp_required": _num_bool(load.tarp_required),
        "appointment_required": _num_bool(load.appointment_required),
        # -- broker observable board columns (the payment signal) ----------
        "broker_credit_bucket": load.broker_credit_bucket or "unknown",
        "broker_days_to_pay": (
            float(load.broker_days_to_pay)
            if load.broker_days_to_pay is not None
            else nan
        ),
        "broker_bonded": _num_bool(load.broker_bonded),
        "broker_quick_pay_available": _num_bool(load.broker_quick_pay_available),
        "broker_age_days": (
            float(load.broker_age_days) if load.broker_age_days is not None else nan
        ),
        # -- market / snapshot context -------------------------------------
        "origin_zone": nearest_zone(load.origin_lat, load.origin_lon),
        "origin_outbound_density": float(density),
        "decision_hour": hour,
        "decision_dow": dow,
        "is_weekend": int(dow >= 5),
        "sin_hour": sin(2 * pi * hour / 24.0),
        "cos_hour": cos(2 * pi * hour / 24.0),
        "sin_dow": sin(2 * pi * dow / 7.0),
        "cos_dow": cos(2 * pi * dow / 7.0),
        # -- competition + age ---------------------------------------------
        "load_views": load.load_views or "low",
        "load_age_hours": float(load.load_age_hours),
    }


# Columns the model treats as categorical (native HistGBM handling, no one-hot).
PAYMENT_CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "equipment_type",
    "mode",
    "commodity",
    "broker_credit_bucket",
    "origin_zone",
    "load_views",
)

# Bookkeeping / label columns produced alongside features (never fed to a model).
# This includes the primary label (``default``), the pay-days regression target and
# its helper flag, the raw outcome columns the labels are derived from, the split tag,
# the snapshot clock, and the join/identity ids.
PAYMENT_NON_FEATURE_COLUMNS: tuple[str, ...] = (
    "default",
    "is_default",
    "pay_days",
    "payment_outcome",
    "realized_pay_days",
    "snapshot_time",
    "split",
    "load_id",
    "broker_id",
)


def payment_feature_columns(all_columns: Sequence[str]) -> List[str]:
    return [c for c in all_columns if c not in PAYMENT_NON_FEATURE_COLUMNS]
