"""Typed loader for ``config/ml_config.yaml`` (Phase 3.1)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "config" / "ml_config.yaml"


@dataclass(frozen=True)
class SyntheticDataConfig:
    start_date: datetime
    days: int
    snapshots_per_day: int
    loads_per_snapshot_mean: float
    unposted_rate_fraction: float
    max_post_age_hours: float
    seed: int
    output_path: str


@dataclass(frozen=True)
class LabelingConfig:
    search_window_hours: float
    min_rate_per_mile: float
    max_deadhead_cap_miles: float


@dataclass(frozen=True)
class FeatureConfig:
    radius_miles: List[float] = field(default_factory=lambda: [50.0, 100.0, 150.0])
    rate_radius_miles: float = 100.0


@dataclass(frozen=True)
class TrainingConfig:
    test_size_time_fraction: float
    model_type: str
    min_bucket_count: int
    top_k: int
    min_candidates_for_ranking: int
    random_seed: int


@dataclass(frozen=True)
class ArtifactConfig:
    model_path: str
    metadata_path: str


@dataclass(frozen=True)
class MLConfig:
    synthetic_data: SyntheticDataConfig
    labeling: LabelingConfig
    features: FeatureConfig
    training: TrainingConfig
    artifacts: ArtifactConfig


def _as_utc(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def load_ml_config(path: str | Path = DEFAULT_CONFIG_PATH) -> MLConfig:
    with Path(path).open("r", encoding="utf-8") as fh:
        doc: Dict[str, Any] = yaml.safe_load(fh) or {}

    sd = doc["synthetic_data"]
    synthetic = SyntheticDataConfig(
        start_date=_as_utc(sd["start_date"]),
        days=int(sd["days"]),
        snapshots_per_day=int(sd["snapshots_per_day"]),
        loads_per_snapshot_mean=float(sd["loads_per_snapshot_mean"]),
        unposted_rate_fraction=float(sd["unposted_rate_fraction"]),
        max_post_age_hours=float(sd["max_post_age_hours"]),
        seed=int(sd["seed"]),
        output_path=str(sd["output_path"]),
    )
    lb = doc["labeling"]
    labeling = LabelingConfig(
        search_window_hours=float(lb["search_window_hours"]),
        min_rate_per_mile=float(lb["min_rate_per_mile"]),
        max_deadhead_cap_miles=float(lb["max_deadhead_cap_miles"]),
    )
    ft = doc.get("features", {})
    features = FeatureConfig(
        radius_miles=[float(r) for r in ft.get("radius_miles", [50, 100, 150])],
        rate_radius_miles=float(ft.get("rate_radius_miles", 100)),
    )
    tr = doc["training"]
    training = TrainingConfig(
        test_size_time_fraction=float(tr["test_size_time_fraction"]),
        model_type=str(tr["model_type"]),
        min_bucket_count=int(tr["min_bucket_count"]),
        top_k=int(tr["top_k"]),
        min_candidates_for_ranking=int(tr["min_candidates_for_ranking"]),
        random_seed=int(tr["random_seed"]),
    )
    ar = doc["artifacts"]
    artifacts = ArtifactConfig(
        model_path=str(ar["model_path"]),
        metadata_path=str(ar["metadata_path"]),
    )
    return MLConfig(
        synthetic_data=synthetic,
        labeling=labeling,
        features=features,
        training=training,
        artifacts=artifacts,
    )
