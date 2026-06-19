"""Payment-risk dataset assembly + three-way grouped time split (Phase 5.2).

The payment analogue of ``ml/training/winnability_dataset.py``. Where the winnability
builder joins *bid trials* to their snapshot, this one joins the **realized payment
outcome** of each load to the snapshot the carrier saw at decision time, producing one
feature row per load labeled ``default`` (the broker never paid).

Join key is ``(load_id, iso(snapshot_time))`` — the same identity the Phase 4.1 build
stamps on both files. Each row carries:

* ``default`` — the primary target, ``int(payment_outcome == "default")`` (positive
  class = catastrophic total-loss non-payment, the minority class).
* ``pay_days`` / ``is_default`` — the secondary regression target (realized days to
  pay) and a convenience flag; the pay-days regressor trains on non-default rows only,
  where ``realized_pay_days`` is populated.
* ``payment_outcome`` / ``realized_pay_days`` — the raw outcome columns the labels are
  derived from (bookkeeping; excluded from the feature matrix by
  ``payment_feature_columns``).

The split is the **three-way time split** used in 4.2: the first ``train_fraction`` of
the snapshot timeline trains every model, the next ``validation_fraction`` drives the
calibration decision (so the test set tunes nothing), and the remaining tail is scored
once. One load = one snapshot_time = one outcome, so a load never straddles a boundary.

Features come only from the snapshot's **observable** columns via
``build_payment_features`` (ask-free); the simulator's latents never enter.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from ml.config import MLConfig
from ml.data.build_winnability_dataset import build_winnability_dataset
from ml.data.load_history_schema import LoadSnapshotRecord, iso, read_jsonl
from ml.data.outcome_schema import (
    PAYMENT_DEFAULT,
    LoadOutcomeRecord,
    read_outcomes,
)
from ml.features.payment_features import build_payment_features

ROOT = Path(__file__).resolve().parents[2]

LABEL = "default"
PAY_DAYS = "pay_days"
TARGET = "broker_default"


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def ensure_payment_data(cfg: MLConfig) -> Tuple[Path, Path]:
    """Ensure the Phase 4.1 snapshot + outcome artifacts exist; build them if not.

    Returns ``(snapshot_path, outcomes_path)``. Reuses the 4.1 builder — it emits the
    payment outcomes alongside the snapshots and trials — so no new labels are made.
    """
    snapshot_path = resolve_path(cfg.outcomes.snapshot_path)
    outcomes_path = resolve_path(cfg.outcomes.outcomes_path)
    if not (snapshot_path.exists() and outcomes_path.exists()):
        build_winnability_dataset(cfg)
    return snapshot_path, outcomes_path


def load_snapshots_and_outcomes(
    cfg: MLConfig,
) -> Tuple[List[LoadSnapshotRecord], List[LoadOutcomeRecord]]:
    snapshot_path, outcomes_path = ensure_payment_data(cfg)
    return read_jsonl(snapshot_path), read_outcomes(outcomes_path)


def _split_boundaries(
    snapshot_times: List, train_fraction: float, validation_fraction: float
):
    times = sorted(set(snapshot_times))
    n = len(times)
    i_train = min(int(n * train_fraction), n - 1)
    i_val = min(int(n * (train_fraction + validation_fraction)), n - 1)
    return times[i_train], times[i_val]


def build_payment_frame(
    snapshots: List[LoadSnapshotRecord],
    outcomes: List[LoadOutcomeRecord],
    cfg: MLConfig,
) -> pd.DataFrame:
    """Join each outcome to its snapshot, build ask-free features, label ``default``."""
    by_key: Dict[Tuple[str, str], LoadSnapshotRecord] = {
        (s.load_id, iso(s.snapshot_time)): s for s in snapshots
    }
    boundary_train, boundary_val = _split_boundaries(
        [o.snapshot_time for o in outcomes],
        cfg.payment_risk.train_fraction,
        cfg.payment_risk.validation_fraction,
    )

    rows: List[dict] = []
    for outcome in outcomes:
        snap = by_key.get((outcome.load_id, iso(outcome.snapshot_time)))
        if snap is None:
            continue
        if outcome.snapshot_time < boundary_train:
            split_name = "train"
        elif outcome.snapshot_time < boundary_val:
            split_name = "validation"
        else:
            split_name = "test"
        is_default = int(outcome.payment_outcome == PAYMENT_DEFAULT)
        feats = build_payment_features(snap)
        feats[LABEL] = is_default
        feats["is_default"] = is_default
        # The pay-days target is only defined when the broker actually paid; defaulted
        # loads carry None and are excluded from the regressor downstream.
        feats[PAY_DAYS] = (
            float(outcome.realized_pay_days)
            if outcome.realized_pay_days is not None
            else float("nan")
        )
        feats["payment_outcome"] = outcome.payment_outcome
        feats["realized_pay_days"] = outcome.realized_pay_days
        feats["snapshot_time"] = outcome.snapshot_time.isoformat()
        feats["split"] = split_name
        feats["load_id"] = outcome.load_id
        feats["broker_id"] = outcome.broker_id
        rows.append(feats)
    return pd.DataFrame(rows)


def build_dataset_from_config(cfg: MLConfig) -> pd.DataFrame:
    snapshots, outcomes = load_snapshots_and_outcomes(cfg)
    return build_payment_frame(snapshots, outcomes, cfg)
