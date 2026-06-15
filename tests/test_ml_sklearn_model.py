"""Tests for the hurdle destination model (Phase 3.1)."""
from datetime import datetime, timezone

import numpy as np

from ml.config import (
    ArtifactConfig,
    FeatureConfig,
    LabelingConfig,
    MLConfig,
    SyntheticDataConfig,
    TrainingConfig,
)
from ml.data.synthetic_history_generator import GeneratorParams, generate_history
from ml.features.destination_features import CATEGORICAL_COLUMNS, feature_columns
from ml.models.baseline_destination_model import GlobalMeanModel
from ml.models.sklearn_destination_model import SklearnDestinationModel
from ml.training.dataset import build_dataset

CAP = 300.0


def _config() -> MLConfig:
    return MLConfig(
        synthetic_data=SyntheticDataConfig(
            start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
            days=14,
            snapshots_per_day=8,
            loads_per_snapshot_mean=35.0,
            unposted_rate_fraction=0.15,
            max_post_age_hours=12.0,
            seed=11,
            output_path="data/_unused.jsonl",
        ),
        labeling=LabelingConfig(
            search_window_hours=8.0, min_rate_per_mile=1.75, max_deadhead_cap_miles=CAP
        ),
        features=FeatureConfig(),
        training=TrainingConfig(
            test_size_time_fraction=0.2,
            model_type="hist_gradient_boosting",
            min_bucket_count=5,
            top_k=3,
            min_candidates_for_ranking=4,
            random_seed=42,
        ),
        artifacts=ArtifactConfig(model_path="m.joblib", metadata_path="m.json"),
    )


def _dataset():
    cfg = _config()
    df = build_dataset(generate_history(GeneratorParams.from_config(cfg.synthetic_data)), cfg)
    tr = df[df["split"] == "train"].reset_index(drop=True)
    te = df[df["split"] == "test"].reset_index(drop=True)
    cols = feature_columns(df.columns)
    cats = [c for c in CATEGORICAL_COLUMNS if c in cols]
    return tr, te, cols, cats


def test_predictions_bounded_and_beat_global_mean():
    tr, te, cols, cats = _dataset()
    ytr, yte = tr["label"].to_numpy(), te["label"].to_numpy()

    model = SklearnDestinationModel(cols, cats, 42, CAP).fit(tr, ytr)
    pred = model.predict(te)

    assert np.all(pred >= 0.0) and np.all(pred <= CAP)

    base = GlobalMeanModel().fit(tr, ytr)
    hurdle_mae = np.mean(np.abs(yte - pred))
    base_mae = np.mean(np.abs(yte - base.predict(te)))
    assert hurdle_mae < base_mae


def test_save_load_roundtrip_is_identical(tmp_path):
    tr, te, cols, cats = _dataset()
    model = SklearnDestinationModel(cols, cats, 42, CAP).fit(tr, tr["label"].to_numpy())
    before = model.predict(te)

    path = model.save(tmp_path / "model.joblib")
    restored = SklearnDestinationModel.load(path)
    after = restored.predict(te)

    assert np.allclose(before, after)
    assert restored.cap == CAP
    assert restored.bulk_loss == "absolute_error"


def test_all_censored_labels_predict_cap():
    tr, te, cols, cats = _dataset()
    y_all_cap = np.full(len(tr), CAP)
    model = SklearnDestinationModel(cols, cats, 42, CAP).fit(tr, y_all_cap)
    pred = model.predict(te)
    assert np.allclose(pred, CAP)


def test_no_censored_labels_stay_below_cap():
    tr, te, cols, cats = _dataset()
    rng = np.random.default_rng(0)
    y_bulk = rng.uniform(10.0, 60.0, size=len(tr))
    model = SklearnDestinationModel(cols, cats, 42, CAP).fit(tr, y_bulk)
    pred = model.predict(te)
    assert np.all(pred < CAP)
    assert model.classifier is None  # no positive class -> classifier skipped
