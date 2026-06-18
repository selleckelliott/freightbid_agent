"""Winnability dataset assembly + three-way grouped time split (Phase 4.2).

Joins the Phase 4.1 bid-trial table to the decision-time snapshots it came from and
turns each ``(load, ask)`` trial into one feature row labeled ``won``. The split is a
**three-way time split** on ``snapshot_time``:

* ``train`` — first ``train_fraction`` of the snapshot timeline; fits every model.
* ``validation`` — the next ``validation_fraction``; drives the calibration decision
  (whether + how to calibrate) so the test set is never used to tune anything.
* ``test`` — the remaining tail; scored exactly once for the reported metrics.

Because all of a load's bid trials share a single ``snapshot_time``, the time
boundaries keep a load's trials wholly inside one split — there is no same-load
leakage across train/validation/test. A test asserts this invariant.

Leakage discipline mirrors Phase 3.1/4.1: features are built only from the snapshot's
**observable** columns (see ``ml/features/winnability_features.py``); the simulator's
latents never enter.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from ml.config import MLConfig
from ml.data.build_winnability_dataset import build_winnability_dataset
from ml.data.load_history_schema import LoadSnapshotRecord, iso, read_jsonl
from ml.data.outcome_schema import BidTrialRecord, read_bid_trials
from ml.features.winnability_features import build_winnability_features

ROOT = Path(__file__).resolve().parents[2]

LABEL = "label"
TARGET = "won_bid"


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def ensure_winnability_data(cfg: MLConfig) -> Tuple[Path, Path]:
    """Ensure the Phase 4.1 snapshot + trial artifacts exist; build them if not.

    Returns ``(snapshot_path, trials_path)``. The build is seeded and reproducible.
    """
    snapshot_path = resolve_path(cfg.outcomes.snapshot_path)
    trials_path = resolve_path(cfg.outcomes.trials_path)
    if not (snapshot_path.exists() and trials_path.exists()):
        build_winnability_dataset(cfg)
    return snapshot_path, trials_path


def load_snapshots_and_trials(
    cfg: MLConfig,
) -> Tuple[List[LoadSnapshotRecord], List[BidTrialRecord]]:
    snapshot_path, trials_path = ensure_winnability_data(cfg)
    return read_jsonl(snapshot_path), read_bid_trials(trials_path)


def _split_boundaries(
    snapshot_times: List, train_fraction: float, validation_fraction: float
):
    times = sorted(set(snapshot_times))
    n = len(times)
    i_train = min(int(n * train_fraction), n - 1)
    i_val = min(int(n * (train_fraction + validation_fraction)), n - 1)
    return times[i_train], times[i_val]


def build_winnability_frame(
    snapshots: List[LoadSnapshotRecord],
    trials: List[BidTrialRecord],
    cfg: MLConfig,
) -> pd.DataFrame:
    """Join trials to their snapshot, build features, label ``won``, tag the split."""
    by_key: Dict[Tuple[str, str], LoadSnapshotRecord] = {
        (s.load_id, iso(s.snapshot_time)): s for s in snapshots
    }
    boundary_train, boundary_val = _split_boundaries(
        [t.snapshot_time for t in trials],
        cfg.winnability.train_fraction,
        cfg.winnability.validation_fraction,
    )

    rows: List[dict] = []
    for trial in trials:
        snap = by_key.get((trial.load_id, iso(trial.snapshot_time)))
        if snap is None:
            continue
        if trial.snapshot_time < boundary_train:
            split_name = "train"
        elif trial.snapshot_time < boundary_val:
            split_name = "validation"
        else:
            split_name = "test"
        feats = build_winnability_features(snap, trial.bid_rpm)
        feats[LABEL] = int(bool(trial.won))
        feats["snapshot_time"] = trial.snapshot_time.isoformat()
        feats["split"] = split_name
        feats["load_id"] = trial.load_id
        feats["broker_id"] = trial.broker_id
        rows.append(feats)
    return pd.DataFrame(rows)


def build_dataset_from_config(cfg: MLConfig) -> pd.DataFrame:
    snapshots, trials = load_snapshots_and_trials(cfg)
    return build_winnability_frame(snapshots, trials, cfg)
