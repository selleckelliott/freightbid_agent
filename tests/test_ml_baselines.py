"""Tests for the non-ML baselines (Phase 3.1)."""
import numpy as np
import pandas as pd

from ml.models.baseline_destination_model import GlobalMeanModel, ZoneDaypartBaseline


def _frame():
    # zone A: 3 midday samples -> a populated (zone, daypart) bucket.
    # zone B: one sample per daypart -> no daypart bucket clears min_count=3,
    #         but the zone overall does (falls back to the zone mean).
    rows = [
        ("A", 12, 10.0),
        ("A", 12, 12.0),
        ("A", 12, 14.0),
        ("B", 12, 20.0),
        ("B", 8, 40.0),
        ("B", 20, 60.0),
    ]
    df = pd.DataFrame(rows, columns=["destination_zone", "arrival_hour", "label"])
    return df


def test_global_mean_predicts_constant():
    df = _frame()
    model = GlobalMeanModel().fit(df, df["label"])
    preds = model.predict(df)
    assert np.allclose(preds, df["label"].mean())
    assert len(preds) == len(df)


def test_zone_daypart_uses_bucket_when_available():
    df = _frame()
    model = ZoneDaypartBaseline(min_count=3).fit(df, df["label"])
    query = pd.DataFrame({"destination_zone": ["A"], "arrival_hour": [12]})
    # (A, midday) has 3 samples -> mean of 10/12/14.
    assert model.predict(query)[0] == 12.0


def test_zone_daypart_falls_back_to_zone_mean():
    df = _frame()
    model = ZoneDaypartBaseline(min_count=3).fit(df, df["label"])
    query = pd.DataFrame({"destination_zone": ["B"], "arrival_hour": [12]})
    # (B, midday) has only 1 sample -> falls back to zone B mean (20/40/60).
    assert model.predict(query)[0] == 40.0


def test_zone_daypart_falls_back_to_global_for_unknown_zone():
    df = _frame()
    model = ZoneDaypartBaseline(min_count=3).fit(df, df["label"])
    query = pd.DataFrame({"destination_zone": ["Z"], "arrival_hour": [12]})
    assert model.predict(query)[0] == df["label"].mean()
