"""Evaluate a saved destination model on the held-out test split (Phase 3.1).

Rebuilds the dataset deterministically, loads the trained model artifact, and
prints its test metrics alongside the baseline numbers recorded in the metadata
JSON at training time.

CLI::

    python -m ml.training.evaluate_destination_model --config config/ml_config.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.config import MLConfig, load_ml_config
from ml.models.sklearn_destination_model import SklearnDestinationModel
from ml.training.dataset import build_dataset, load_history, resolve_path
from ml.training.metrics import bucket_metrics, regression_metrics, top_k_hit_rate
from ml.training.train_destination_model import LABEL, format_table


def evaluate(cfg: MLConfig) -> dict:
    model_path = resolve_path(cfg.artifacts.model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"No model at {model_path}. Run train_destination_model first."
        )
    model = SklearnDestinationModel.load(model_path)

    df = build_dataset(load_history(cfg), cfg)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    y_true = test_df[LABEL].to_numpy()
    y_pred = model.predict(test_df)

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

    results = {"gradient_boosting": metrics}
    metadata_path = resolve_path(cfg.artifacts.metadata_path)
    if metadata_path.exists():
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key in ("global_mean", "zone_daypart"):
            if key in meta.get("metrics", {}):
                results[key] = meta["metrics"][key]

    print(format_table(results))
    return metrics


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate the destination model.")
    parser.add_argument("--config", default=None, help="Path to ml_config.yaml")
    args = parser.parse_args(argv)
    cfg = load_ml_config(args.config) if args.config else load_ml_config()
    evaluate(cfg)


if __name__ == "__main__":
    main()
