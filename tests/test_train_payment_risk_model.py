"""End-to-end smoke test for the Phase 5.2 payment-risk trainer (tiny world)."""
import dataclasses
import json

import pytest

from ml.config import load_ml_config
from ml.training.train_payment_risk_model import train


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("train_pay")
    cfg = load_ml_config()
    synthetic = dataclasses.replace(cfg.synthetic_data, days=6, loads_per_snapshot_mean=18)
    outcomes = dataclasses.replace(
        cfg.outcomes,
        snapshot_path=str(tmp / "snap.jsonl"),
        outcomes_path=str(tmp / "out.jsonl"),
        trials_path=str(tmp / "trials.jsonl"),
    )
    payment_risk = dataclasses.replace(
        cfg.payment_risk,
        model_path=str(tmp / "model.joblib"),
        metadata_path=str(tmp / "meta.json"),
        reliability_chart_path=str(tmp / "rel.png"),
    )
    cfg = dataclasses.replace(
        cfg, synthetic_data=synthetic, outcomes=outcomes, payment_risk=payment_risk
    )
    metadata = train(cfg)
    return {"tmp": tmp, "metadata": metadata}


def test_train_writes_three_artifacts(trained):
    tmp = trained["tmp"]
    assert (tmp / "model.joblib").exists()
    assert (tmp / "meta.json").exists()
    assert (tmp / "rel.png").exists()


def test_metadata_has_expected_keys(trained):
    meta = trained["metadata"]
    for key in (
        "model_name",
        "target",
        "rows",
        "base_default_rate",
        "served_model",
        "calibration_decision",
        "metrics_test",
        "pay_days_test",
        "features",
        "categorical_columns",
        "hyperparameters",
    ):
        assert key in meta
    # The reported metrics include the served GBM and the three baselines.
    assert "gbm_served" in meta["metrics_test"]
    assert "global_default_rate" in meta["metrics_test"]
    # Persisted JSON matches the returned metadata.
    on_disk = json.loads((trained["tmp"] / "meta.json").read_text(encoding="utf-8"))
    assert on_disk["model_name"] == meta["model_name"]


def test_features_are_ask_free(trained):
    cols = set(trained["metadata"]["features"])
    assert not (cols & {"bid_rpm", "ask_to_market_ratio", "ask_to_posted_ratio"})
