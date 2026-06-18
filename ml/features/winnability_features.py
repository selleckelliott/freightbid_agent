"""Decision-time feature builder for bid winnability (Phase 4.2).

Every feature here is knowable **at the moment a carrier decides what to bid** on a
load it can see on the board: the ask itself, the load's observable attributes, the
broker's observable board columns, a coarse market read, the competition signal, and
the load's age. Nothing reads the hidden outcome world — the broker's reservation
rate, true pay days, default probability, rate bias, or the load's latent contention
never appear (they are ground truth for the Phase 4.1 simulator only). A leakage-guard
test asserts that none of those names reach the feature matrix.

The ask is expressed three ways, because a dispatcher reasons relative to *anchors*,
not in absolute rpm:

* ``bid_rpm`` — the raw ask.
* ``ask_to_market_ratio`` — ask ÷ the load's origin-market average rpm (the same
  market anchor the trial sampler bids against; this is the dominant win driver).
* ``ask_to_posted_ratio`` — ask ÷ the load's posted rpm, or ``NaN`` when the load is
  "call for rate" (no posted rate). Missing is left as ``NaN`` (HGB handles it
  natively) and paired with the explicit ``has_posted_rate`` flag, so the model can
  branch on "no posted rate" cleanly rather than reading a fabricated sentinel.

The same builder runs at training and at serving time (via ``BidQuery``), so there is
no train/serve skew — Phase 4.3 will score candidate asks through this exact function.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import cos, isnan, nan, pi, sin
from typing import Any, Dict, List, Optional, Protocol, Sequence

from ml.markets import market_by_name, nearest_zone


def _num_bool(value: Optional[bool]) -> float:
    """Boolean → float with ``None`` preserved as ``NaN`` (a real "unknown")."""
    if value is None:
        return nan
    return 1.0 if value else 0.0


def market_rate_for(origin_lat: float, origin_lon: float) -> float:
    """Origin-market average rpm — the observable anchor the carrier bids against.

    Identical to the simulator's ``_market_rate`` (origin metro's
    ``avg_rate_per_mile``): a coarse, domain-knowledge market read, not a latent.
    """
    return market_by_name(nearest_zone(origin_lat, origin_lon)).avg_rate_per_mile


class _SnapshotLike(Protocol):
    snapshot_time: datetime
    origin_lat: float
    origin_lon: float
    equipment_type: str
    mode: str
    loaded_miles: float
    weight: float
    length: float
    load_views: str
    commodity: Optional[str]
    tarp_required: Optional[bool]
    appointment_required: Optional[bool]
    broker_credit_bucket: Optional[str]
    broker_days_to_pay: Optional[int]
    broker_bonded: Optional[bool]
    broker_quick_pay_available: Optional[bool]
    broker_age_days: Optional[int]

    @property
    def rate_per_mile(self) -> Optional[float]: ...

    @property
    def load_age_hours(self) -> float: ...


@dataclass(frozen=True)
class BidQuery:
    """Serving-time stand-in for a board load + a candidate ask (Phase 4.3 reuse).

    Carries exactly what the winnability feature builder needs, so a recommender can
    score a grid of asks for one load without materializing a full
    ``LoadSnapshotRecord``.
    """

    snapshot_time: datetime
    origin_lat: float
    origin_lon: float
    equipment_type: str
    loaded_miles: float
    posted_at: datetime
    mode: str = "TL"
    weight: float = 0.0
    length: float = 0.0
    load_views: str = "low"
    total_rate: Optional[float] = None
    commodity: Optional[str] = None
    tarp_required: Optional[bool] = None
    appointment_required: Optional[bool] = None
    broker_credit_bucket: Optional[str] = None
    broker_days_to_pay: Optional[int] = None
    broker_bonded: Optional[bool] = None
    broker_quick_pay_available: Optional[bool] = None
    broker_age_days: Optional[int] = None

    @property
    def rate_per_mile(self) -> Optional[float]:
        if self.total_rate is None or self.loaded_miles <= 0:
            return None
        return self.total_rate / self.loaded_miles

    @property
    def load_age_hours(self) -> float:
        return (self.snapshot_time - self.posted_at).total_seconds() / 3600.0

    @classmethod
    def from_snapshot(cls, record: "_SnapshotLike") -> "BidQuery":
        """Build a serving query from a decision-time snapshot record.

        Copies only the **observable** columns the feature builder reads — the same
        ones a live Truckstop adapter would expose — so Phase 4.3 scores held-out
        snapshots through the exact training-time feature path (no skew, no leakage).
        """
        return cls(
            snapshot_time=record.snapshot_time,
            origin_lat=record.origin_lat,
            origin_lon=record.origin_lon,
            equipment_type=record.equipment_type,
            loaded_miles=record.loaded_miles,
            posted_at=record.posted_at,
            mode=getattr(record, "mode", "TL"),
            weight=getattr(record, "weight", 0.0),
            length=getattr(record, "length", 0.0),
            load_views=getattr(record, "load_views", "low"),
            total_rate=record.total_rate,
            commodity=getattr(record, "commodity", None),
            tarp_required=getattr(record, "tarp_required", None),
            appointment_required=getattr(record, "appointment_required", None),
            broker_credit_bucket=getattr(record, "broker_credit_bucket", None),
            broker_days_to_pay=getattr(record, "broker_days_to_pay", None),
            broker_bonded=getattr(record, "broker_bonded", None),
            broker_quick_pay_available=getattr(record, "broker_quick_pay_available", None),
            broker_age_days=getattr(record, "broker_age_days", None),
        )


def build_winnability_features(load: _SnapshotLike, bid_rpm: float) -> Dict[str, Any]:
    """Build the feature dict for one ``(load, ask)`` pair.

    ``load`` is any decision-time snapshot (a ``LoadSnapshotRecord`` or a ``BidQuery``);
    ``bid_rpm`` is the candidate ask. All values are decision-time observable.
    """
    market_rate = market_rate_for(load.origin_lat, load.origin_lon)
    posted_rpm = load.rate_per_mile
    has_posted = posted_rpm is not None

    ask_to_market = bid_rpm / market_rate if market_rate > 0 else nan
    ask_to_posted = bid_rpm / posted_rpm if (has_posted and posted_rpm) else nan

    decided = load.snapshot_time
    hour = decided.hour
    dow = decided.weekday()
    density = market_by_name(nearest_zone(load.origin_lat, load.origin_lon)).outbound_density

    return {
        # -- the ask (relative to its anchors) -----------------------------
        "bid_rpm": float(bid_rpm),
        "market_rate": float(market_rate),
        "ask_to_market_ratio": float(ask_to_market),
        "ask_to_posted_ratio": float(ask_to_posted),
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
        # -- broker observable board columns -------------------------------
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
CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "equipment_type",
    "mode",
    "commodity",
    "broker_credit_bucket",
    "origin_zone",
    "load_views",
)

# Bookkeeping columns produced alongside features in the dataset (never fed to a model).
NON_FEATURE_COLUMNS: tuple[str, ...] = (
    "label",
    "snapshot_time",
    "split",
    "load_id",
    "broker_id",
)


def feature_columns(all_columns: Sequence[str]) -> List[str]:
    return [c for c in all_columns if c not in NON_FEATURE_COLUMNS]


def ask_ratio_bucket(ratio: float, edges: Sequence[float]) -> int:
    """Index of the half-open ``[edges[i], edges[i+1])`` bin containing ``ratio``.

    Clamped to the first/last bin; ``NaN`` (e.g. a missing posted ratio) maps to the
    final bin so the heuristic baseline degrades gracefully instead of erroring.
    """
    if ratio is None or (isinstance(ratio, float) and isnan(ratio)):
        return len(edges) - 2
    for i in range(len(edges) - 1):
        if ratio < edges[i + 1]:
            return max(i, 0)
    return len(edges) - 2
