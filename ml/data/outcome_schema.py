"""Outcome / label schema for the winnability dataset (Phase 4.1).

Two artifacts, both keyed back to a ``LoadSnapshotRecord`` by
``(load_id, snapshot_time)``:

* ``LoadOutcomeRecord`` — one realized outcome per load: did it get covered and how
  fast, was it paid, did a no-rate load require negotiation. It also carries the
  **hidden ground-truth** ``reservation_rpm`` and ``contention_intensity`` that the
  simulator used. These are *labels*, not decision-time signals — exactly like the
  next-deadhead labels in ``ml/data/labeling.py``, they may encode the latent world
  because that is what makes them ground truth. They live here, never on the
  snapshot record / snapshot JSONL / features.
* ``BidTrialRecord`` — materialized ``(bid_rpm, won)`` rows: for each covered load,
  a seeded sweep of candidate bids over a neutral rpm grid scored against the load's
  reservation. This is the directly-trainable winnability table 4.2 consumes.

Datetime (de)serialization reuses ``iso``/``parse_dt`` from the snapshot schema so
the two files round-trip identically.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ml.data.load_history_schema import iso, parse_dt

# Payment outcome categories realized by the simulator.
PAYMENT_PAID = "paid"
PAYMENT_LATE = "late"
PAYMENT_DEFAULT = "default"
PAYMENT_OUTCOMES = (PAYMENT_PAID, PAYMENT_LATE, PAYMENT_DEFAULT)


@dataclass(frozen=True)
class LoadOutcomeRecord:
    snapshot_time: datetime
    load_id: str
    broker_id: Optional[str]
    # -- coverage / "disappears quickly" -----------------------------------
    covered: bool
    time_to_cover_hours: float        # censored at the simulator horizon
    cover_censored: bool              # True when no cover occurred within horizon
    # -- "which bid prices win" (hidden ground truth) ----------------------
    reservation_rpm: float            # min rpm the broker would accept on this load
    contention_intensity: float       # latent 0..~1 demand pressure on this load
    # -- "no-rate loads require negotiation" -------------------------------
    negotiation_required: bool
    negotiated_rate: Optional[float]  # settled total rate after negotiation (or None)
    # -- "brokers pay quickly / are risky" ---------------------------------
    payment_outcome: str              # paid / late / default
    realized_pay_days: Optional[float]  # None when defaulted

    _DT_FIELDS = ("snapshot_time",)

    def to_json_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, value in self.__dict__.items():
            out[key] = iso(value) if key in self._DT_FIELDS else value
        return out

    @classmethod
    def from_json_dict(cls, data: Dict[str, Any]) -> "LoadOutcomeRecord":
        kwargs = dict(data)
        for field in cls._DT_FIELDS:
            kwargs[field] = parse_dt(kwargs[field])
        return cls(**kwargs)


@dataclass(frozen=True)
class BidTrialRecord:
    """One ``(bid_rpm, won)`` example, joinable to a snapshot by id + time."""
    snapshot_time: datetime
    load_id: str
    broker_id: Optional[str]
    bid_rpm: float
    won: bool

    _DT_FIELDS = ("snapshot_time",)

    def to_json_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, value in self.__dict__.items():
            out[key] = iso(value) if key in self._DT_FIELDS else value
        return out

    @classmethod
    def from_json_dict(cls, data: Dict[str, Any]) -> "BidTrialRecord":
        kwargs = dict(data)
        for field in cls._DT_FIELDS:
            kwargs[field] = parse_dt(kwargs[field])
        return cls(**kwargs)


def _write_jsonl(records: Iterable[Any], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec.to_json_dict()) + "\n")
            count += 1
    return count


def write_outcomes(records: Iterable[LoadOutcomeRecord], path: str | Path) -> int:
    return _write_jsonl(records, path)


def write_bid_trials(records: Iterable[BidTrialRecord], path: str | Path) -> int:
    return _write_jsonl(records, path)


def read_outcomes(path: str | Path) -> List[LoadOutcomeRecord]:
    path = Path(path)
    out: List[LoadOutcomeRecord] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(LoadOutcomeRecord.from_json_dict(json.loads(line)))
    return out


def read_bid_trials(path: str | Path) -> List[BidTrialRecord]:
    path = Path(path)
    out: List[BidTrialRecord] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(BidTrialRecord.from_json_dict(json.loads(line)))
    return out
