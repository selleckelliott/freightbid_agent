"""Dataset assembly + split-respecting labels (Phase 3.1).

Turns a list of ``LoadSnapshotRecord`` into a feature/label DataFrame and tags
each row ``train``/``test`` by a time-based boundary.

Leakage discipline:
* Features come only from the row's own decision-time snapshot board.
* Labels are computed against the *full* history so every row's next-load
  search sees the same, untruncated future — no artificial censoring at a pool
  edge. Two guards keep the split honest:
  - **Observability:** a row whose label window extends past the last snapshot
    cannot be truthfully labeled, so it is dropped (rather than mislabeled as
    "no load found").
  - **Embargo:** a train row whose label window reaches into the test period is
    dropped, so no training target is informed by a test-period load.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import pandas as pd

from ml.config import MLConfig
from ml.data.labeling import LabelConfig, label_records
from ml.data.load_history_schema import LoadSnapshotRecord, read_jsonl
from ml.data.synthetic_history_generator import GeneratorParams, generate_to_file
from ml.features.destination_features import build_features

ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def ensure_history(cfg: MLConfig) -> Path:
    path = resolve_path(cfg.synthetic_data.output_path)
    if not path.exists():
        generate_to_file(GeneratorParams.from_config(cfg.synthetic_data), path)
    return path


def load_history(cfg: MLConfig) -> List[LoadSnapshotRecord]:
    return read_jsonl(ensure_history(cfg))


def _split_boundary(records: List[LoadSnapshotRecord], fraction: float) -> datetime:
    times = sorted(r.snapshot_time for r in records)
    idx = min(int(len(times) * (1.0 - fraction)), len(times) - 1)
    return times[idx]


def build_dataset(records: List[LoadSnapshotRecord], cfg: MLConfig) -> pd.DataFrame:
    label_cfg = LabelConfig(
        search_window_hours=cfg.labeling.search_window_hours,
        min_rate_per_mile=cfg.labeling.min_rate_per_mile,
        max_deadhead_cap_miles=cfg.labeling.max_deadhead_cap_miles,
    )
    window = timedelta(hours=cfg.labeling.search_window_hours)
    boundary = _split_boundary(records, cfg.training.test_size_time_fraction)
    max_time = max(r.snapshot_time for r in records)

    boards: dict[datetime, List[LoadSnapshotRecord]] = defaultdict(list)
    for rec in records:
        boards[rec.snapshot_time].append(rec)

    # Only rows whose full label window is observable within the data.
    observable = [r for r in records if r.arrival_time + window <= max_time]
    # One labeling pass over the full history -> consistent, untruncated labels.
    labels = label_records(observable, records, label_cfg)

    rows: List[dict] = []
    for rec, label in zip(observable, labels):
        if rec.snapshot_time < boundary:
            # Embargo: skip train rows whose window peeks into the test period.
            if rec.arrival_time + window >= boundary:
                continue
            split_name = "train"
        else:
            split_name = "test"
        feats = build_features(rec, boards[rec.snapshot_time], cfg.features)
        feats["label"] = label
        feats["snapshot_time"] = rec.snapshot_time.isoformat()
        feats["split"] = split_name
        rows.append(feats)
    return pd.DataFrame(rows)


def build_dataset_from_config(cfg: MLConfig) -> pd.DataFrame:
    return build_dataset(load_history(cfg), cfg)
