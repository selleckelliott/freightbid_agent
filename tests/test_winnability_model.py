"""Integration tests for the Phase 4.2 winnability model + dataset invariants."""
import dataclasses

import numpy as np
import pytest

from ml.brokers import HIDDEN_BROKER_FIELDS
from ml.config import load_ml_config
from ml.features.winnability_features import CATEGORICAL_COLUMNS, feature_columns
from ml.models.baseline_winnability_model import GlobalWinRateModel
from ml.models.sklearn_winnability_model import SklearnWinnabilityModel
from ml.training.winnability_dataset import (
    LABEL,
    build_winnability_frame,
    load_snapshots_and_trials,
)
from ml.training.winnability_metrics import expected_calibration_error

_LATENT_NAMES = set(HIDDEN_BROKER_FIELDS) | {"reservation_rpm", "contention_intensity"}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("winn")
    cfg = load_ml_config()
    outcomes = dataclasses.replace(
        cfg.outcomes,
        snapshot_path=str(tmp / "snap.jsonl"),
        outcomes_path=str(tmp / "out.jsonl"),
        trials_path=str(tmp / "trials.jsonl"),
    )
    cfg = dataclasses.replace(cfg, outcomes=outcomes)
    # Small but seeded build; ensure_winnability_data writes into the tmp paths.
    from ml.data.build_winnability_dataset import build_winnability_dataset

    build_winnability_dataset(cfg, days=8)
    snaps, trials = load_snapshots_and_trials(cfg)
    df = build_winnability_frame(snaps, trials, cfg)
    cols = feature_columns(df.columns)
    cats = [c for c in CATEGORICAL_COLUMNS if c in cols]
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    model = SklearnWinnabilityModel(cols, cats, 42).fit(train_df, train_df[LABEL].to_numpy())
    return {"cfg": cfg, "df": df, "cols": cols, "cats": cats, "model": model}


def test_three_way_split_present(built):
    splits = set(built["df"]["split"].unique())
    assert splits == {"train", "validation", "test"}


def test_same_load_trials_stay_in_one_split(built):
    # A load's six bid trials must never straddle a train/validation/test boundary.
    per_load = built["df"].groupby("load_id")["split"].nunique()
    assert int(per_load.max()) == 1


def test_no_latent_feature_columns(built):
    assert not (set(built["cols"]) & _LATENT_NAMES)
    # Bookkeeping id columns are not features either.
    assert "load_id" not in built["cols"]
    assert "broker_id" not in built["cols"]


def test_predict_proba_is_bounded_and_beats_global_baseline(built):
    df = built["df"]
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    y_test = test_df[LABEL].to_numpy()

    proba = built["model"].predict_proba(test_df)
    assert proba.shape == (len(test_df),)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)

    base = GlobalWinRateModel().fit(train_df, train_df[LABEL].to_numpy())
    eps = 1e-12
    model_ll = -np.mean(
        y_test * np.log(np.clip(proba, eps, 1)) + (1 - y_test) * np.log(np.clip(1 - proba, eps, 1))
    )
    base_p = np.clip(base.predict_proba(test_df), eps, 1 - eps)
    base_ll = -np.mean(
        y_test * np.log(base_p) + (1 - y_test) * np.log(1 - base_p)
    )
    assert model_ll < base_ll


def test_save_load_roundtrip_is_identical(built, tmp_path):
    df = built["df"]
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    before = built["model"].predict_proba(test_df)

    path = built["model"].save(tmp_path / "winn.joblib")
    restored = SklearnWinnabilityModel.load(path)
    after = restored.predict_proba(test_df)

    assert np.allclose(before, after)
    assert restored.feature_columns == built["model"].feature_columns


def test_calibration_machinery_attaches_and_predicts(built):
    # Exercise the prefit-calibration path regardless of whether it's needed in
    # production: a calibrated sibling shares the base model and still emits valid
    # probabilities, and the calibrator object is attached.
    df = built["df"]
    val_df = df[df["split"] == "validation"].reset_index(drop=True)
    y_val = val_df[LABEL].to_numpy()

    calibrated = built["model"].make_calibrated(val_df, y_val, method="isotonic")
    assert calibrated.is_calibrated
    assert calibrated.calibration_method == "isotonic"
    proba = calibrated.predict_proba(val_df)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
    # Isotonic fit on the slice should not be wildly miscalibrated on it.
    assert expected_calibration_error(y_val, proba, n_bins=10) < 0.1
