"""Tests for the Phase 4.2 winnability baselines + simulator ask-monotonicity."""
import numpy as np
import pandas as pd

from ml.data.outcome_simulator import win_prob
from ml.models.baseline_winnability_model import (
    AskRatioHeuristicModel,
    BrokerMarketGroupedBaseline,
    GlobalWinRateModel,
    effective_ask_ratio,
)

_EDGES = [0.0, 0.90, 0.975, 1.025, 1.10, 1.20, 100.0]
_MULTS = [0.85, 0.95, 1.0, 1.05, 1.15, 1.25]


def _grid_frame(win_rates):
    """One bin per multiplier, each with 100 rows at the given win rate (no posted rate)."""
    rows = []
    for mult, rate in zip(_MULTS, win_rates):
        n_win = int(round(rate * 100))
        for i in range(100):
            rows.append(
                {
                    "ask_to_market_ratio": mult,
                    "ask_to_posted_ratio": float("nan"),
                    "has_posted_rate": 0.0,
                    "origin_zone": "Dallas",
                    "broker_credit_bucket": "A",
                    "label": 1 if i < n_win else 0,
                }
            )
    return pd.DataFrame(rows)


def test_global_win_rate_predicts_constant_base_rate():
    df = _grid_frame([0.9, 0.8, 0.6, 0.4, 0.2, 0.1])
    model = GlobalWinRateModel().fit(df, df["label"])
    preds = model.predict_proba(df)
    assert np.allclose(preds, df["label"].mean())
    assert len(preds) == len(df)


def test_effective_ratio_prefers_posted_then_market():
    df = pd.DataFrame(
        {
            "ask_to_posted_ratio": [1.1, float("nan")],
            "ask_to_market_ratio": [0.9, 1.2],
            "has_posted_rate": [1.0, 0.0],
        }
    )
    eff = effective_ask_ratio(df)
    assert eff[0] == 1.1  # posted used when present
    assert eff[1] == 1.2  # falls back to market when no posted rate


def test_ask_ratio_heuristic_is_monotone_decreasing_in_ask():
    # True win rate falls as the ask rises.
    df = _grid_frame([0.9, 0.8, 0.6, 0.4, 0.2, 0.1])
    model = AskRatioHeuristicModel(_EDGES, min_count=10).fit(df, df["label"])
    # Predict one representative row per ascending bin.
    probe = pd.DataFrame(
        {
            "ask_to_market_ratio": _MULTS,
            "ask_to_posted_ratio": [float("nan")] * len(_MULTS),
            "has_posted_rate": [0.0] * len(_MULTS),
        }
    )
    preds = model.predict_proba(probe)
    assert np.all(np.diff(preds) <= 1e-9)  # non-increasing across rising asks


def test_ask_ratio_heuristic_falls_back_to_global_for_sparse_bin():
    # Train only the two lowest-ask bins, so the higher-ask bins are never seen.
    rows = []
    for mult in (0.85, 0.95):
        for i in range(100):
            rows.append({
                "ask_to_market_ratio": mult, "ask_to_posted_ratio": float("nan"),
                "has_posted_rate": 0.0, "origin_zone": "Dallas",
                "broker_credit_bucket": "A", "label": 1 if i < 70 else 0,
            })
    df = pd.DataFrame(rows)
    model = AskRatioHeuristicModel(_EDGES, min_count=10).fit(df, df["label"])
    # A high-ask ratio (bin never populated in training) -> global rate fallback.
    probe = pd.DataFrame(
        {
            "ask_to_market_ratio": [1.25],
            "ask_to_posted_ratio": [float("nan")],
            "has_posted_rate": [0.0],
        }
    )
    assert abs(model.predict_proba(probe)[0] - df["label"].mean()) < 1e-9


def test_broker_market_grouped_uses_bucket_then_falls_back():
    # zone A / credit A populated; zone Z unseen -> coarser fallbacks.
    rows = []
    for i in range(60):
        rows.append({
            "ask_to_market_ratio": 1.0, "ask_to_posted_ratio": float("nan"),
            "has_posted_rate": 0.0, "origin_zone": "Dallas",
            "broker_credit_bucket": "A", "label": 1 if i < 45 else 0,
        })
    df = pd.DataFrame(rows)
    model = BrokerMarketGroupedBaseline(_EDGES, min_count=10).fit(df, df["label"])

    hit = pd.DataFrame({
        "ask_to_market_ratio": [1.0], "ask_to_posted_ratio": [float("nan")],
        "has_posted_rate": [0.0], "origin_zone": ["Dallas"],
        "broker_credit_bucket": ["A"],
    })
    assert abs(model.predict_proba(hit)[0] - 0.75) < 1e-9  # exact (zone,credit,bin) cell

    miss = pd.DataFrame({
        "ask_to_market_ratio": [1.0], "ask_to_posted_ratio": [float("nan")],
        "has_posted_rate": [0.0], "origin_zone": ["Nowhere"],
        "broker_credit_bucket": ["Z"],
    })
    # Unknown zone+credit -> falls through to the bin (or global) rate, still valid prob.
    p = model.predict_proba(miss)[0]
    assert 0.0 < p < 1.0


def test_simulator_win_prob_is_non_increasing_in_ask():
    """The generative ground truth: a higher ask never raises the true win probability."""
    reserve, scale = 2.5, 0.06
    asks = [1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
    probs = [win_prob(reserve, a, scale) for a in asks]
    assert np.all(np.diff(probs) <= 0.0)
    # And it spans a real range (not a degenerate flat line).
    assert probs[0] - probs[-1] > 0.5
