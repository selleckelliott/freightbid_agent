"""Phase 5.2 — payment-risk adapter tests.

The model boundary: ``ModelPaymentRiskAdapter`` loads a trained Phase 5.2 artifact and
returns a ``PaymentEstimate`` (building features with the exact training-time, ask-free
builder, so no train/serve skew and no latent leakage); ``NoopPaymentRiskAdapter``
returns ``None`` so the recommender degrades to risk-blind behavior. ``from_config``
returns ``None`` when the artifact is missing — the no-op parity contract.
"""
from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from adapters.outbound.payment_risk.model_adapter import ModelPaymentRiskAdapter
from adapters.outbound.payment_risk.noop_adapter import NoopPaymentRiskAdapter
from ml.config import load_ml_config
from ml.features.payment_features import (
    PAYMENT_CATEGORICAL_COLUMNS,
    build_payment_features,
    payment_feature_columns,
)
from ml.features.winnability_features import BidQuery
from ml.models.sklearn_payment_risk_model import SklearnPaymentRiskModel
from ports.payment_risk import PaymentEstimate, PaymentRiskPort


def _bid_query(credit="A", bonded=True, quick=True) -> BidQuery:
    t = datetime(2026, 1, 5, 12, 0, 0)
    return BidQuery(
        snapshot_time=t,
        origin_lat=32.78,
        origin_lon=-96.80,
        equipment_type="F",
        loaded_miles=300.0,
        posted_at=t - timedelta(hours=3),
        total_rate=720.0,
        broker_credit_bucket=credit,
        broker_bonded=bonded,
        broker_quick_pay_available=quick,
        broker_days_to_pay=20,
        broker_age_days=400,
    )


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    """Train a tiny, seeded payment-risk model (with pay-days head) into tmp."""
    rng = np.random.default_rng(0)
    rows, labels, days = [], [], []
    for _ in range(400):
        credit = rng.choice(["A", "B", "C", "unknown"])
        bonded = bool(rng.integers(0, 2))
        quick = bool(rng.integers(0, 2))
        rows.append(build_payment_features(_bid_query(credit, bonded, quick)))
        labels.append(int(rng.random() < (0.30 if credit == "C" else 0.05)))
        days.append(float(rng.uniform(15, 50)))
    frame = pd.DataFrame(rows)
    frame["is_default"] = labels
    frame["pay_days"] = days
    cols = payment_feature_columns(
        [c for c in frame.columns if c not in ("is_default", "pay_days")]
    )
    cats = [c for c in PAYMENT_CATEGORICAL_COLUMNS if c in cols]
    model = SklearnPaymentRiskModel(cols, cats, 45).fit(frame, np.array(labels))
    paid = frame[frame["is_default"] == 0].reset_index(drop=True)
    model.fit_pay_days(paid, paid["pay_days"].to_numpy())
    return model.save(tmp_path_factory.mktemp("artifacts") / "pay.joblib")


def test_noop_adapter_returns_none():
    assert NoopPaymentRiskAdapter().estimate(_bid_query()) is None


def test_from_config_returns_none_when_artifact_missing(tmp_path):
    cfg = load_ml_config()
    pr = dataclasses.replace(cfg.payment_risk, model_path=str(tmp_path / "missing.joblib"))
    cfg = dataclasses.replace(cfg, payment_risk=pr)
    assert ModelPaymentRiskAdapter.from_config(cfg) is None


def test_from_artifact_returns_none_when_missing(tmp_path):
    assert ModelPaymentRiskAdapter.from_artifact(tmp_path / "nope.joblib") is None


def test_model_adapter_returns_payment_estimate(model_path):
    adapter = ModelPaymentRiskAdapter.from_artifact(model_path)
    est = adapter.estimate(_bid_query("C", False, False))
    assert isinstance(est, PaymentEstimate)
    assert 0.0 <= est.p_default <= 1.0
    assert abs(est.p_collect - (1.0 - est.p_default)) < 1e-12
    assert est.expected_pay_days is not None and est.expected_pay_days >= 0.0


def test_model_adapter_is_callable_through_the_port_type(model_path):
    adapter = ModelPaymentRiskAdapter.from_artifact(model_path)
    assert isinstance(adapter, PaymentRiskPort)
    assert isinstance(NoopPaymentRiskAdapter(), PaymentRiskPort)
