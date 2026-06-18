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
class BrokersConfig:
    """Phase 4.1 broker-pool knobs (see ``ml/brokers.py``)."""
    pool_size: int = 40
    bonded_fraction: float = 0.45
    unknown_credit_fraction: float = 0.18
    quick_pay_fraction: float = 0.30
    pay_days_fast: float = 18.0
    pay_days_slow: float = 55.0
    default_prob_best: float = 0.01
    default_prob_worst: float = 0.16
    seed: int = 43


@dataclass(frozen=True)
class OutcomesConfig:
    """Phase 4.1 outcome-world knobs (see ``ml/data/outcome_simulator.py``)."""
    reservation_center_mult: float = 1.05
    reservation_contention_drop: float = 0.15
    reservation_noise: float = 0.06
    win_logistic_scale_rpm: float = 0.06
    bid_trial_rpm_multipliers: List[float] = field(
        default_factory=lambda: [0.85, 0.95, 1.0, 1.05, 1.15, 1.25]
    )
    base_cover_halflife_hours: float = 18.0
    contention_cover_factor: float = 4.0
    cover_horizon_hours: float = 24.0
    negotiated_premium: float = 1.0
    late_pay_threshold_days: float = 45.0
    seed: int = 44
    snapshot_path: str = "data/winnability_snapshots.jsonl"
    outcomes_path: str = "data/winnability_outcomes.jsonl"
    trials_path: str = "data/winnability_trials.jsonl"


@dataclass(frozen=True)
class WinnabilityConfig:
    """Phase 4.2 bid-winnability model knobs (see ``ml/training``).

    Split is a **three-way grouped time split** on ``snapshot_time``: the first
    ``train_fraction`` of snapshots train the model, the next
    ``validation_fraction`` drive the *calibration decision* (whether + how to
    calibrate), and the remaining tail is the held-out test set scored once. Because
    all of a load's bid trials share one ``snapshot_time`` they never straddle a
    boundary, so a load's trials stay wholly inside one slice.
    """
    train_fraction: float = 0.70
    validation_fraction: float = 0.10  # test = 1 - train - validation (0.20)
    random_seed: int = 42
    ece_calibration_threshold: float = 0.03  # calibrate only if validation ECE exceeds this
    n_reliability_bins: int = 10
    # Bin edges for the ask-ratio heuristic baseline; the six trial multipliers
    # (0.85/0.95/1.0/1.05/1.15/1.25) each fall in their own bin.
    ask_ratio_bin_edges: List[float] = field(
        default_factory=lambda: [0.0, 0.90, 0.975, 1.025, 1.10, 1.20, 100.0]
    )
    min_bucket_count: int = 25  # min rows for a grouped-baseline bucket to be trusted
    model_path: str = "ml/artifacts/winnability_model.joblib"
    metadata_path: str = "ml/artifacts/winnability_model_metadata.json"
    reliability_chart_path: str = "ml/artifacts/winnability_reliability.png"


@dataclass(frozen=True)
class MLConfig:
    synthetic_data: SyntheticDataConfig
    labeling: LabelingConfig
    features: FeatureConfig
    training: TrainingConfig
    artifacts: ArtifactConfig
    brokers: BrokersConfig = field(default_factory=BrokersConfig)
    outcomes: OutcomesConfig = field(default_factory=OutcomesConfig)
    winnability: WinnabilityConfig = field(default_factory=WinnabilityConfig)


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
    bk = doc.get("brokers", {})
    brokers = BrokersConfig(
        pool_size=int(bk.get("pool_size", 40)),
        bonded_fraction=float(bk.get("bonded_fraction", 0.45)),
        unknown_credit_fraction=float(bk.get("unknown_credit_fraction", 0.18)),
        quick_pay_fraction=float(bk.get("quick_pay_fraction", 0.30)),
        pay_days_fast=float(bk.get("pay_days_fast", 18.0)),
        pay_days_slow=float(bk.get("pay_days_slow", 55.0)),
        default_prob_best=float(bk.get("default_prob_best", 0.01)),
        default_prob_worst=float(bk.get("default_prob_worst", 0.16)),
        seed=int(bk.get("seed", 43)),
    )
    oc = doc.get("outcomes", {})
    outcomes = OutcomesConfig(
        reservation_center_mult=float(oc.get("reservation_center_mult", 1.05)),
        reservation_contention_drop=float(oc.get("reservation_contention_drop", 0.15)),
        reservation_noise=float(oc.get("reservation_noise", 0.06)),
        win_logistic_scale_rpm=float(oc.get("win_logistic_scale_rpm", 0.06)),
        bid_trial_rpm_multipliers=[
            float(m) for m in oc.get(
                "bid_trial_rpm_multipliers", [0.85, 0.95, 1.0, 1.05, 1.15, 1.25]
            )
        ],
        base_cover_halflife_hours=float(oc.get("base_cover_halflife_hours", 18.0)),
        contention_cover_factor=float(oc.get("contention_cover_factor", 4.0)),
        cover_horizon_hours=float(oc.get("cover_horizon_hours", 24.0)),
        negotiated_premium=float(oc.get("negotiated_premium", 1.0)),
        late_pay_threshold_days=float(oc.get("late_pay_threshold_days", 45.0)),
        seed=int(oc.get("seed", 44)),
        snapshot_path=str(oc.get("snapshot_path", "data/winnability_snapshots.jsonl")),
        outcomes_path=str(oc.get("outcomes_path", "data/winnability_outcomes.jsonl")),
        trials_path=str(oc.get("trials_path", "data/winnability_trials.jsonl")),
    )
    wn = doc.get("winnability", {})
    winnability = WinnabilityConfig(
        train_fraction=float(wn.get("train_fraction", 0.70)),
        validation_fraction=float(wn.get("validation_fraction", 0.10)),
        random_seed=int(wn.get("random_seed", 42)),
        ece_calibration_threshold=float(wn.get("ece_calibration_threshold", 0.03)),
        n_reliability_bins=int(wn.get("n_reliability_bins", 10)),
        ask_ratio_bin_edges=[
            float(e) for e in wn.get(
                "ask_ratio_bin_edges", [0.0, 0.90, 0.975, 1.025, 1.10, 1.20, 100.0]
            )
        ],
        min_bucket_count=int(wn.get("min_bucket_count", 25)),
        model_path=str(wn.get("model_path", "ml/artifacts/winnability_model.joblib")),
        metadata_path=str(
            wn.get("metadata_path", "ml/artifacts/winnability_model_metadata.json")
        ),
        reliability_chart_path=str(
            wn.get("reliability_chart_path", "ml/artifacts/winnability_reliability.png")
        ),
    )
    return MLConfig(
        synthetic_data=synthetic,
        labeling=labeling,
        features=features,
        training=training,
        artifacts=artifacts,
        brokers=brokers,
        outcomes=outcomes,
        winnability=winnability,
    )
