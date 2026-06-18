"""Tests for the Phase 4.2 calibration-first metrics."""
import numpy as np

from ml.training.winnability_metrics import (
    classification_metrics,
    evaluate_probabilities,
    expected_calibration_error,
    reliability_table,
)


def test_perfect_ranking_gives_auc_one():
    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.2, 0.8, 0.9]
    m = classification_metrics(y_true, y_prob)
    assert m["roc_auc"] == 1.0
    assert m["pr_auc"] == 1.0
    assert m["brier"] < 0.05


def test_auc_is_nan_on_single_class_slice():
    m = classification_metrics([1, 1, 1], [0.6, 0.7, 0.8])
    assert np.isnan(m["roc_auc"])
    assert np.isnan(m["pr_auc"])
    # Proper scoring rules are still defined.
    assert m["brier"] > 0.0


def test_ece_near_zero_for_well_calibrated_predictions():
    # In each probability bin the observed rate equals the predicted probability.
    y_prob = np.concatenate([np.full(100, 0.2), np.full(100, 0.8)])
    y_true = np.concatenate(
        [
            np.array([1] * 20 + [0] * 80),  # bin centered at 0.2 -> 20% positive
            np.array([1] * 80 + [0] * 20),  # bin centered at 0.8 -> 80% positive
        ]
    )
    ece = expected_calibration_error(y_true, y_prob, n_bins=10)
    assert ece < 1e-9


def test_ece_equals_gap_for_uniformly_miscalibrated_predictions():
    # All predictions 0.9 but only half actually win -> ECE == |0.9 - 0.5| == 0.4.
    y_prob = np.full(200, 0.9)
    y_true = np.array([1] * 100 + [0] * 100)
    ece = expected_calibration_error(y_true, y_prob, n_bins=10)
    assert abs(ece - 0.4) < 1e-9


def test_reliability_table_reports_bin_counts_and_rates():
    y_prob = np.full(200, 0.9)
    y_true = np.array([1] * 100 + [0] * 100)
    table = reliability_table(y_true, y_prob, n_bins=10)
    assert len(table) == 1  # all predictions land in one bin
    row = table[0]
    assert row["count"] == 200
    assert abs(row["mean_predicted"] - 0.9) < 1e-9
    assert abs(row["observed_rate"] - 0.5) < 1e-9


def test_evaluate_probabilities_bundles_metrics_and_table():
    y_true = [0, 1, 0, 1, 1, 0]
    y_prob = [0.2, 0.7, 0.3, 0.9, 0.6, 0.1]
    report = evaluate_probabilities(y_true, y_prob, n_bins=10)
    assert {"roc_auc", "pr_auc", "brier", "log_loss", "ece"} <= set(report["metrics"])
    assert abs(report["metrics"]["positive_rate"] - 0.5) < 1e-9
    assert isinstance(report["reliability_table"], list)
