"""Tests for dataset assembly + leakage-safe split (Phase 3.1)."""
from datetime import datetime, timedelta, timezone

from ml.config import (
    ArtifactConfig,
    FeatureConfig,
    LabelingConfig,
    MLConfig,
    SyntheticDataConfig,
    TrainingConfig,
)
from ml.data.load_history_schema import LoadSnapshotRecord
from ml.data.synthetic_history_generator import GeneratorParams, generate_history
from ml.training.dataset import build_dataset
from ml.training.metrics import censoring_rate

CAP = 300.0
T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
ARRIVAL_OFFSET_H = 2.0
WINDOW_H = 8.0
DENVER = (39.7392, -104.9903)


def _config(**training_overrides) -> MLConfig:
    training = dict(
        test_size_time_fraction=0.2,
        model_type="hist_gradient_boosting",
        min_bucket_count=5,
        top_k=3,
        min_candidates_for_ranking=4,
        random_seed=42,
    )
    training.update(training_overrides)
    return MLConfig(
        synthetic_data=SyntheticDataConfig(
            start_date=T0, days=14, snapshots_per_day=8, loads_per_snapshot_mean=30.0,
            unposted_rate_fraction=0.15, max_post_age_hours=12.0, seed=5,
            output_path="data/_unused.jsonl",
        ),
        labeling=LabelingConfig(
            search_window_hours=WINDOW_H, min_rate_per_mile=1.75, max_deadhead_cap_miles=CAP
        ),
        features=FeatureConfig(),
        training=TrainingConfig(**training),
        artifacts=ArtifactConfig(model_path="m.joblib", metadata_path="m.json"),
    )


def _record(i: int) -> LoadSnapshotRecord:
    snap = T0 + timedelta(hours=i)
    return LoadSnapshotRecord(
        snapshot_time=snap,
        load_id=f"L-{i:04d}",
        origin_city="Denver", origin_state="CO", origin_lat=DENVER[0], origin_lon=DENVER[1],
        destination_city="Denver", destination_state="CO",
        destination_lat=DENVER[0], destination_lon=DENVER[1],
        pickup_start=snap + timedelta(hours=1.0),
        pickup_end=snap + timedelta(hours=1.5),
        dropoff_start=snap + timedelta(hours=ARRIVAL_OFFSET_H),
        dropoff_end=snap + timedelta(hours=ARRIVAL_OFFSET_H + 0.5),
        equipment_type="Dry Van",
        loaded_miles=100.0,
        posted_at=snap - timedelta(hours=1.0),
        total_rate=250.0,
    )


def _parse(series):
    return [datetime.fromisoformat(s) for s in series]


def test_split_respects_embargo_and_observability():
    records = [_record(i) for i in range(60)]
    cfg = _config()
    df = build_dataset(records, cfg)

    assert set(df["split"]) == {"train", "test"}

    # n=60, fraction=0.2 -> boundary index 48; horizon = arrival(2h)+window(8h)=10h.
    boundary = T0 + timedelta(hours=48)
    max_time = T0 + timedelta(hours=59)
    horizon = timedelta(hours=ARRIVAL_OFFSET_H + WINDOW_H)

    train_snaps = _parse(df[df["split"] == "train"]["snapshot_time"])
    test_snaps = _parse(df[df["split"] == "test"]["snapshot_time"])

    # Observability: no row's label window runs past the last snapshot.
    assert max(train_snaps + test_snaps) + horizon <= max_time
    # Embargo: every train row's window closes before the test period begins.
    assert max(train_snaps) + horizon < boundary
    # Split is time-ordered (no shuffle leakage).
    assert max(train_snaps) < min(test_snaps)


def test_labels_present_and_bounded_on_synthetic_history():
    cfg = _config()
    df = build_dataset(generate_history(GeneratorParams.from_config(cfg.synthetic_data)), cfg)

    assert {"train", "test"} <= set(df["split"])
    assert df["label"].notna().all()
    assert df["label"].min() >= 0.0
    assert df["label"].max() <= CAP
    # Guard against a regression reintroducing boundary-truncation censoring.
    assert censoring_rate(df["label"], CAP) < 0.5
