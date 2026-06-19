"""Tests for the Phase 5.2 payment-risk baselines (all ask-free)."""
import numpy as np
import pandas as pd

from ml.models.baseline_payment_model import (
    BondedQuickPayBaseline,
    CreditBucketBaseline,
    GlobalDefaultRateModel,
)


def _frame(buckets, bonded, quick, labels):
    return pd.DataFrame(
        {
            "broker_credit_bucket": buckets,
            "broker_bonded": bonded,
            "broker_quick_pay_available": quick,
            "label": labels,
        }
    )


def _grouped_frame():
    """Bucket A pays well (low default), bucket C is risky; bonded/quick mirrors it."""
    rows = []
    for i in range(200):
        rows.append({
            "broker_credit_bucket": "A", "broker_bonded": 1.0,
            "broker_quick_pay_available": 1.0, "label": 1 if i < 10 else 0,  # 5%
        })
    for i in range(200):
        rows.append({
            "broker_credit_bucket": "C", "broker_bonded": 0.0,
            "broker_quick_pay_available": 0.0, "label": 1 if i < 60 else 0,  # 30%
        })
    return pd.DataFrame(rows)


def _brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def _log_loss(p, y):
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def test_global_predicts_constant_default_rate():
    df = _grouped_frame()
    model = GlobalDefaultRateModel().fit(df, df["label"])
    preds = model.predict_proba(df)
    assert preds.shape == (len(df),)
    assert np.allclose(preds, df["label"].mean())


def test_each_baseline_returns_bounded_1d():
    df = _grouped_frame()
    for cls in (GlobalDefaultRateModel, CreditBucketBaseline, BondedQuickPayBaseline):
        model = cls().fit(df, df["label"])
        preds = model.predict_proba(df)
        assert preds.ndim == 1 and len(preds) == len(df)
        assert np.all(preds >= 0.0) and np.all(preds <= 1.0)


def test_credit_bucket_falls_back_to_global_for_unseen_bucket():
    df = _grouped_frame()
    model = CreditBucketBaseline(min_count=25).fit(df, df["label"])
    unseen = _frame(["Z"], [1.0], [1.0], [0])
    assert abs(model.predict_proba(unseen)[0] - df["label"].mean()) < 1e-9


def test_credit_bucket_falls_back_for_thin_bucket():
    # Bucket B has only a handful of rows -> below min_count -> global fallback.
    rows = _grouped_frame().to_dict("records")
    for i in range(5):
        rows.append({
            "broker_credit_bucket": "B", "broker_bonded": 1.0,
            "broker_quick_pay_available": 0.0, "label": 1,
        })
    df = pd.DataFrame(rows)
    model = CreditBucketBaseline(min_count=25).fit(df, df["label"])
    probe = _frame(["B"], [1.0], [0.0], [0])
    assert abs(model.predict_proba(probe)[0] - df["label"].mean()) < 1e-9


def test_bonded_quick_pay_falls_back_for_unseen_cell():
    df = _grouped_frame()
    model = BondedQuickPayBaseline(min_count=25).fit(df, df["label"])
    # An (unknown-flag) cell never seen in training -> global fallback.
    unseen = _frame(["A"], [float("nan")], [float("nan")], [0])
    assert abs(model.predict_proba(unseen)[0] - df["label"].mean()) < 1e-9


def test_grouped_baselines_beat_or_match_global_on_held_out():
    df = _grouped_frame()
    # Held-out slice with the same structure (defaults clearly separate by group).
    held = _grouped_frame()
    y = held["label"].to_numpy()
    g = GlobalDefaultRateModel().fit(df, df["label"]).predict_proba(held)
    c = CreditBucketBaseline(min_count=25).fit(df, df["label"]).predict_proba(held)
    b = BondedQuickPayBaseline(min_count=25).fit(df, df["label"]).predict_proba(held)
    assert _brier(c, y) <= _brier(g, y) + 1e-9
    assert _log_loss(c, y) <= _log_loss(g, y) + 1e-9
    assert _brier(b, y) <= _brier(g, y) + 1e-9
    assert _log_loss(b, y) <= _log_loss(g, y) + 1e-9
