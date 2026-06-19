"""Tests for the Phase 5.2 payment-risk dataset builder + leakage guard."""
import dataclasses

import pytest

from ml.brokers import HIDDEN_BROKER_FIELDS
from ml.config import load_ml_config
from ml.features.payment_features import payment_feature_columns
from ml.training.payment_risk_dataset import (
    LABEL,
    PAY_DAYS,
    build_payment_frame,
    load_snapshots_and_outcomes,
)

# Names that must never reach the feature matrix: latent broker ground truth, the raw
# outcome/label columns, the pay-days target, and the join/identity ids.
_FORBIDDEN = set(HIDDEN_BROKER_FIELDS) | {
    "payment_outcome",
    "realized_pay_days",
    "reservation_rpm",
    "contention_intensity",
    "default",
    "pay_days",
    "is_default",
    "broker_id",
    "load_id",
}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pay")
    cfg = load_ml_config()
    # Tiny, seeded world routed entirely into tmp so we never touch committed data.
    synthetic = dataclasses.replace(cfg.synthetic_data, days=6, loads_per_snapshot_mean=18)
    outcomes = dataclasses.replace(
        cfg.outcomes,
        snapshot_path=str(tmp / "snap.jsonl"),
        outcomes_path=str(tmp / "out.jsonl"),
        trials_path=str(tmp / "trials.jsonl"),
    )
    cfg = dataclasses.replace(cfg, synthetic_data=synthetic, outcomes=outcomes)
    snaps, outs = load_snapshots_and_outcomes(cfg)
    df = build_payment_frame(snaps, outs, cfg)
    return {"cfg": cfg, "df": df}


def test_frame_builds_and_joins_one_row_per_outcome(built):
    df = built["df"]
    assert len(df) > 0
    # The join is keyed on (load_id, snapshot_time); each load appears once.
    assert int(df.groupby("load_id").size().max()) == 1


def test_label_balance_sane(built):
    df = built["df"]
    rate = df[LABEL].mean()
    # Defaults are a minority but must be present and not degenerate.
    assert 0.0 < rate < 0.5
    assert set(df[LABEL].unique()) == {0, 1}


def test_three_way_split_disjoint_and_time_ordered(built):
    df = built["df"]
    assert set(df["split"].unique()) == {"train", "validation", "test"}
    # Time ordering: max train time <= min validation time <= min test time.
    t_max = {s: df[df["split"] == s]["snapshot_time"].max() for s in ("train", "validation")}
    t_min = {s: df[df["split"] == s]["snapshot_time"].min() for s in ("validation", "test")}
    assert t_max["train"] <= t_min["validation"]
    assert t_max["validation"] <= t_min["test"]


def test_default_label_matches_payment_outcome(built):
    df = built["df"]
    derived = (df["payment_outcome"] == "default").astype(int)
    assert (df[LABEL] == derived).all()
    assert (df["is_default"] == derived).all()


def test_pay_days_present_for_non_default_rows(built):
    df = built["df"]
    non_default = df[df["is_default"] == 0]
    assert non_default[PAY_DAYS].notna().all()
    # Defaulted loads carry no realized pay-days.
    defaulted = df[df["is_default"] == 1]
    assert defaulted[PAY_DAYS].isna().all()


def test_no_leaky_feature_columns(built):
    cols = set(payment_feature_columns(built["df"].columns))
    assert not (cols & _FORBIDDEN)
    # And no ask-derived columns slipped in (payment is ask-free).
    assert not (cols & {"bid_rpm", "ask_to_market_ratio", "ask_to_posted_ratio"})
