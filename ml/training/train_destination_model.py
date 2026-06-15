"""Train the destination desirability model (Phase 3.1).

Builds the dataset, fits the two baselines and the gradient-boosting model on a
time-based train split, evaluates on the held-out tail, prints a comparison
table, and saves the model binary + a metadata JSON.

CLI::

    python -m ml.training.train_destination_model --config config/ml_config.yaml
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import pandas as pd

from ml.config import MLConfig, load_ml_config
from ml.features.destination_features import CATEGORICAL_COLUMNS, feature_columns
from ml.models.baseline_destination_model import GlobalMeanModel, ZoneDaypartBaseline
from ml.models.sklearn_destination_model import MODEL_NAME, SklearnDestinationModel
from ml.training.dataset import build_dataset, load_history, resolve_path
from ml.training.metrics import (
    bucket_metrics,
    censoring_rate,
    regression_metrics,
    top_k_hit_rate,
)

LABEL = "label"
TARGET = "expected_next_deadhead_miles"

_MODEL_LABELS = {
    "global_mean": "Global Mean",
    "zone_daypart": "Zone/Daypart",
    "gradient_boosting": "Hurdle GBM",
}


def _evaluate(test_df: pd.DataFrame, y_pred, cfg: MLConfig) -> Dict[str, float]:
    y_true = test_df[LABEL].to_numpy()
    metrics = regression_metrics(y_true, y_pred)
    metrics.update(bucket_metrics(y_true, y_pred))
    ranking = top_k_hit_rate(
        test_df["snapshot_time"],
        test_df["equipment_type"],
        y_true,
        y_pred,
        k=cfg.training.top_k,
        min_candidates=cfg.training.min_candidates_for_ranking,
    )
    metrics["top_k_hit_rate"] = ranking["top_k_hit_rate"]
    metrics["ranking_groups"] = ranking["ranking_groups"]
    return metrics


def format_table(results: Dict[str, Dict[str, float]]) -> str:
    header = (
        f"{'Model':<18}{'MAE':>8}{'RMSE':>9}{'MedAE':>8}"
        f"{'R2':>7}{'<=25mi':>9}{'<=50mi':>9}{'top3':>8}"
    )
    lines = [header, "-" * len(header)]
    for key in ("global_mean", "zone_daypart", "gradient_boosting"):
        if key not in results:
            continue
        m = results[key]
        lines.append(
            f"{_MODEL_LABELS[key]:<18}"
            f"{m['mae']:>8.1f}{m['rmse']:>9.1f}{m['median_ae']:>8.1f}"
            f"{m['r2']:>7.2f}{m['within_25mi'] * 100:>8.0f}%"
            f"{m['within_50mi'] * 100:>8.0f}%{m['top_k_hit_rate'] * 100:>7.0f}%"
        )
    return "\n".join(lines)


def train(cfg: MLConfig) -> Dict:
    records = load_history(cfg)
    df = build_dataset(records, cfg)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    cols = feature_columns(df.columns)
    cats = [c for c in CATEGORICAL_COLUMNS if c in cols]
    y_train = train_df[LABEL].to_numpy()

    models = {
        "global_mean": GlobalMeanModel().fit(train_df, y_train),
        "zone_daypart": ZoneDaypartBaseline(cfg.training.min_bucket_count).fit(
            train_df, y_train
        ),
        "gradient_boosting": SklearnDestinationModel(
            cols, cats, cfg.training.random_seed, cfg.labeling.max_deadhead_cap_miles
        ).fit(train_df, y_train),
    }

    results = {
        name: _evaluate(test_df, model.predict(test_df), cfg)
        for name, model in models.items()
    }

    cap = cfg.labeling.max_deadhead_cap_miles
    censoring = {
        "train": censoring_rate(train_df[LABEL], cap),
        "test": censoring_rate(test_df[LABEL], cap),
    }

    ml_model: SklearnDestinationModel = models["gradient_boosting"]
    model_path = resolve_path(cfg.artifacts.model_path)
    ml_model.save(model_path)

    metadata = {
        "model_name": MODEL_NAME,
        "target": TARGET,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "label_cap_miles": cap,
        "censoring_rate": censoring,
        "ranking_groups": results["gradient_boosting"]["ranking_groups"],
        "config": {
            "search_window_hours": cfg.labeling.search_window_hours,
            "min_rate_per_mile": cfg.labeling.min_rate_per_mile,
            "max_deadhead_cap_miles": cap,
            "radius_miles": cfg.features.radius_miles,
            "test_size_time_fraction": cfg.training.test_size_time_fraction,
            "top_k": cfg.training.top_k,
            "random_seed": cfg.training.random_seed,
        },
        "metrics": results,
        "features": cols,
    }
    metadata_path = resolve_path(cfg.artifacts.metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Train rows: {len(train_df):,}   Test rows: {len(test_df):,}")
    print(
        f"Label censoring (== {cap:.0f} mi cap): "
        f"train {censoring['train'] * 100:.1f}%  test {censoring['test'] * 100:.1f}%"
    )
    print(f"Ranking groups scored: {int(results['gradient_boosting']['ranking_groups'])}")
    print()
    print(format_table(results))
    print()
    print(f"Saved model    -> {model_path}")
    print(f"Saved metadata -> {metadata_path}")
    return metadata


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train the destination model.")
    parser.add_argument("--config", default=None, help="Path to ml_config.yaml")
    args = parser.parse_args(argv)
    cfg = load_ml_config(args.config) if args.config else load_ml_config()
    train(cfg)


if __name__ == "__main__":
    main()
