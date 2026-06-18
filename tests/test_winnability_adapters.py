"""Phase 4.3 — winnability adapter tests.

The model boundary: ``ModelWinnabilityAdapter`` loads a trained Phase 4.2 artifact and
scores a grid of asks (building features with the exact training-time builder, so no
train/serve skew and no latent leakage); ``NoopWinnabilityAdapter`` returns ``None`` so
the recommender degrades to cost-plus-margin.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from adapters.outbound.winnability.model_adapter import ModelWinnabilityAdapter
from adapters.outbound.winnability.noop_adapter import NoopWinnabilityAdapter
from ml.features.winnability_features import (
    CATEGORICAL_COLUMNS,
    BidQuery,
    build_winnability_features,
    feature_columns,
)
from ml.models.sklearn_winnability_model import SklearnWinnabilityModel

_LATENT_NAMES = {
    "reservation_rpm",
    "contention_intensity",
    "true_pay_days",
    "true_default_prob",
    "rate_bias",
    "won",
}


def _bid_query(miles: float, total_rate, equipment: str) -> BidQuery:
    t = datetime(2026, 1, 5, 12, 0, 0)
    return BidQuery(
        snapshot_time=t,
        origin_lat=32.78,
        origin_lon=-96.80,
        equipment_type=equipment,
        loaded_miles=miles,
        posted_at=t - timedelta(hours=3),
        total_rate=total_rate,
    )


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    """Train a tiny, seeded winnability model into tmp (no canonical artifact needed)."""
    rng = np.random.default_rng(0)
    rows, labels = [], []
    for _ in range(300):
        miles = float(rng.integers(150, 600))
        equip = rng.choice(["F", "HS", "FSD", "FSDV"])
        total = float(miles * rng.uniform(2.0, 3.0)) if rng.random() > 0.3 else None
        ask = float(rng.uniform(2.0, 3.0))
        rows.append(build_winnability_features(_bid_query(miles, total, equip), ask))
        # Lower asks win more often -> a learnable, decreasing relationship.
        labels.append(int(rng.random() < max(0.05, 1.0 - (ask - 2.0))))
    frame = pd.DataFrame(rows)
    cols = feature_columns(frame.columns)
    cats = [c for c in CATEGORICAL_COLUMNS if c in cols]
    model = SklearnWinnabilityModel(cols, cats, 42).fit(frame, np.array(labels))
    return model.save(tmp_path_factory.mktemp("artifacts") / "winn.joblib")


def test_model_adapter_scores_each_ask(model_path):
    adapter = ModelWinnabilityAdapter.from_artifact(model_path)
    rpms = [2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
    probs = adapter.win_probabilities(_bid_query(300.0, 720.0, "F"), rpms)
    assert probs is not None
    assert len(probs) == len(rpms)
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_model_adapter_empty_grid_returns_empty(model_path):
    adapter = ModelWinnabilityAdapter.from_artifact(model_path)
    assert adapter.win_probabilities(_bid_query(300.0, 720.0, "F"), []) == []


def test_noop_adapter_returns_none():
    adapter = NoopWinnabilityAdapter()
    assert adapter.win_probabilities(_bid_query(300.0, 720.0, "F"), [2.0, 2.5]) is None


def test_adapter_features_carry_no_latent_columns():
    feats = build_winnability_features(_bid_query(300.0, 720.0, "F"), 2.4)
    assert not (set(feats.keys()) & _LATENT_NAMES)


def test_model_adapter_is_callable_through_the_port_type(model_path):
    from ports.winnability import WinnabilityPort

    adapter = ModelWinnabilityAdapter.from_artifact(model_path)
    assert isinstance(adapter, WinnabilityPort)
    assert isinstance(NoopWinnabilityAdapter(), WinnabilityPort)
