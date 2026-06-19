"""Tests for the Phase 5.2 ``SklearnPaymentRiskModel`` (P(default) + E[pay_days])."""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from ml.features.payment_features import (
    PAYMENT_CATEGORICAL_COLUMNS,
    build_payment_features,
    payment_feature_columns,
)
from ml.features.winnability_features import BidQuery
from ml.models.sklearn_payment_risk_model import SklearnPaymentRiskModel


def _bid_query(miles, total, equipment, bonded, quick, credit) -> BidQuery:
    t = datetime(2026, 1, 5, 12, 0, 0)
    return BidQuery(
        snapshot_time=t,
        origin_lat=32.78,
        origin_lon=-96.80,
        equipment_type=equipment,
        loaded_miles=miles,
        posted_at=t - timedelta(hours=3),
        total_rate=total,
        broker_credit_bucket=credit,
        broker_bonded=bonded,
        broker_quick_pay_available=quick,
        broker_days_to_pay=20 if credit != "unknown" else None,
        broker_age_days=400,
    )


@pytest.fixture(scope="module")
def trained():
    rng = np.random.default_rng(0)
    rows, labels, pay_days = [], [], []
    for _ in range(500):
        miles = float(rng.integers(150, 600))
        equip = rng.choice(["F", "HS", "FSD", "FSDV"])
        total = float(miles * rng.uniform(2.0, 3.0)) if rng.random() > 0.3 else None
        bonded = float(rng.integers(0, 2))
        quick = float(rng.integers(0, 2))
        credit = rng.choice(["A", "B", "C", "unknown"])
        rows.append(
            build_payment_features(_bid_query(miles, total, equip, bonded, quick, credit))
        )
        # Risky brokers (low credit, not bonded) default more; pay-days scale with risk.
        risk = (0.30 if credit == "C" else 0.05) + (0.05 if bonded < 0.5 else 0.0)
        is_default = int(rng.random() < risk)
        labels.append(is_default)
        pay_days.append(float(rng.uniform(15, 50)))
    frame = pd.DataFrame(rows)
    frame["label"] = labels
    frame["is_default"] = labels
    frame["pay_days"] = pay_days
    cols = payment_feature_columns(
        [c for c in frame.columns if c not in ("label", "is_default", "pay_days")]
    )
    cats = [c for c in PAYMENT_CATEGORICAL_COLUMNS if c in cols]
    model = SklearnPaymentRiskModel(cols, cats, 45).fit(frame, frame["label"].to_numpy())
    paid = frame[frame["is_default"] == 0].reset_index(drop=True)
    model.fit_pay_days(paid, paid["pay_days"].to_numpy())
    return {"frame": frame, "cols": cols, "cats": cats, "model": model}


def test_predict_proba_is_bounded_1d(trained):
    frame = trained["frame"]
    proba = trained["model"].predict_proba(frame)
    assert proba.shape == (len(frame),)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)


def test_deterministic_same_seed(trained):
    frame = trained["frame"]
    twin = SklearnPaymentRiskModel(trained["cols"], trained["cats"], 45).fit(
        frame, frame["label"].to_numpy()
    )
    assert np.allclose(twin.predict_proba(frame), trained["model"].predict_proba(frame))


def test_make_calibrated_sibling_differs(trained):
    frame = trained["frame"]
    base = trained["model"].predict_proba(frame)
    calibrated = trained["model"].make_calibrated(frame, frame["label"].to_numpy(), "isotonic")
    assert calibrated.is_calibrated
    assert calibrated.calibration_method == "isotonic"
    after = calibrated.predict_proba(frame)
    assert np.all(after >= 0.0) and np.all(after <= 1.0)
    assert not np.allclose(after, base)
    # The calibrated sibling carries the pay-days head by reference.
    assert calibrated.has_pay_days


def test_save_load_roundtrip_preserves_preds_and_regressor(trained, tmp_path):
    frame = trained["frame"]
    before = trained["model"].predict_proba(frame)
    before_days = trained["model"].predict_pay_days(frame)

    path = trained["model"].save(tmp_path / "pay.joblib")
    restored = SklearnPaymentRiskModel.load(path)

    assert np.allclose(before, restored.predict_proba(frame))
    assert restored.has_pay_days
    assert np.allclose(before_days, restored.predict_pay_days(frame))
    assert restored.feature_columns == trained["model"].feature_columns


def test_pay_days_predictions_are_positive(trained):
    frame = trained["frame"]
    preds = trained["model"].predict_pay_days(frame)
    assert preds.shape == (len(frame),)
    assert np.all(preds > 0.0)
