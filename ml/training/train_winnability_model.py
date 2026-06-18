"""Train + evaluate the calibrated bid-winnability model (Phase 4.2).

Pipeline (calibration-first, test set touched once):

1. Build the trial-level dataset and the three-way grouped time split.
2. Fit three baselines (global win rate, ask-ratio heuristic, broker/market grouped)
   and a ``HistGradientBoostingClassifier`` on the **train** slice.
3. Decide calibration on the **validation** slice only: if the base model's validation
   ECE exceeds the configured threshold, fit prefit isotonic + sigmoid calibrators on
   validation, and serve whichever lowers validation ECE the most (and actually beats
   the uncalibrated model). Otherwise serve the uncalibrated model.
4. Evaluate every model **once** on the held-out **test** slice and report ROC AUC,
   PR AUC, Brier, log loss, and ECE; render the served model's reliability diagram.
5. Save the served model (joblib), a metadata JSON (committed result artifact), and the
   reliability PNG.

CLI::

    python -m ml.training.train_winnability_model --config config/ml_config.yaml
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ml.config import MLConfig, load_ml_config  # noqa: E402
from ml.features.winnability_features import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    feature_columns,
)
from ml.models.baseline_winnability_model import (  # noqa: E402
    AskRatioHeuristicModel,
    BrokerMarketGroupedBaseline,
    GlobalWinRateModel,
)
from ml.models.sklearn_winnability_model import MODEL_NAME, SklearnWinnabilityModel  # noqa: E402
from ml.training.winnability_dataset import (  # noqa: E402
    LABEL,
    TARGET,
    build_winnability_frame,
    load_snapshots_and_trials,
    resolve_path,
)
from ml.training.winnability_metrics import (  # noqa: E402
    expected_calibration_error,
    reliability_table,
    evaluate_probabilities,
)

_MODEL_LABELS = {
    "global_win_rate": "Global win rate",
    "ask_ratio": "Ask-ratio heuristic",
    "broker_market": "Broker/market group",
    "gbm_uncalibrated": "GBM (uncalibrated)",
    "gbm_served": "GBM (served)",
}


def _eval(y_true, y_prob, n_bins: int) -> Dict[str, float]:
    return evaluate_probabilities(y_true, y_prob, n_bins)["metrics"]


def format_table(results: Dict[str, Dict[str, float]]) -> str:
    header = (
        f"{'Model':<22}{'ROC AUC':>9}{'PR AUC':>9}{'Brier':>9}"
        f"{'LogLoss':>9}{'ECE':>8}"
    )
    lines = [header, "-" * len(header)]
    for key in (
        "global_win_rate",
        "ask_ratio",
        "broker_market",
        "gbm_uncalibrated",
        "gbm_served",
    ):
        if key not in results:
            continue
        m = results[key]
        lines.append(
            f"{_MODEL_LABELS[key]:<22}"
            f"{m['roc_auc']:>9.3f}{m['pr_auc']:>9.3f}{m['brier']:>9.4f}"
            f"{m['log_loss']:>9.4f}{m['ece']:>8.4f}"
        )
    return "\n".join(lines)


def _decide_calibration(
    model: SklearnWinnabilityModel, val_df, y_val, cfg: MLConfig
):
    """Validation-only calibration decision. Returns ``(served_model, decision)``."""
    n_bins = cfg.winnability.n_reliability_bins
    threshold = cfg.winnability.ece_calibration_threshold
    base_val_ece = expected_calibration_error(y_val, model.predict_proba(val_df), n_bins)
    decision: Dict[str, object] = {
        "threshold": threshold,
        "validation_ece_uncalibrated": base_val_ece,
        "calibrated": False,
    }
    if base_val_ece <= threshold:
        decision["note"] = "validation ECE within threshold; calibration unnecessary"
        return model, decision

    candidates = {
        method: model.make_calibrated(val_df, y_val, method)
        for method in ("isotonic", "sigmoid")
    }
    cand_ece = {
        method: expected_calibration_error(y_val, cand.predict_proba(val_df), n_bins)
        for method, cand in candidates.items()
    }
    decision["validation_ece_isotonic"] = cand_ece["isotonic"]
    decision["validation_ece_sigmoid"] = cand_ece["sigmoid"]
    best_method = min(cand_ece, key=cand_ece.get)
    if cand_ece[best_method] < base_val_ece:
        decision["calibrated"] = True
        decision["method"] = best_method
        decision["validation_ece_calibrated"] = cand_ece[best_method]
        return candidates[best_method], decision

    decision["note"] = (
        "calibration did not improve validation ECE; serving uncalibrated"
    )
    return model, decision


def _render_reliability_chart(table, served_metrics, out_png: Path, calibrated: bool) -> None:
    matplotlib.rcParams.update({
        "figure.facecolor": "#0d1117",
        "axes.facecolor": "#161b22",
        "axes.edgecolor": "#30363d",
        "axes.labelcolor": "#c9d1d9",
        "axes.titlecolor": "#e6edf3",
        "xtick.color": "#8b949e",
        "ytick.color": "#c9d1d9",
        "grid.color": "#21262d",
        "text.color": "#c9d1d9",
    })
    fig, ax = plt.subplots(figsize=(7.0, 6.4))
    ax.plot([0, 1], [0, 1], ls="--", color="#8b949e", lw=1.3, label="perfect calibration")
    if table:
        xs = [r["mean_predicted"] for r in table]
        ys = [r["observed_rate"] for r in table]
        counts = [r["count"] for r in table]
        cmax = max(counts) or 1
        sizes = [40 + 360 * (c / cmax) for c in counts]
        ax.plot(xs, ys, color="#58a6ff", lw=1.6, zorder=2)
        ax.scatter(xs, ys, s=sizes, color="#3fb950", edgecolor="#0d1117",
                   linewidth=0.8, zorder=3, label="test bins (size ∝ count)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("mean predicted P(win)")
    ax.set_ylabel("observed win rate")
    kind = "calibrated" if calibrated else "uncalibrated"
    ax.set_title(
        f"FreightBid Agent — Bid Winnability Reliability ({kind})\n"
        f"test ECE {served_metrics['ece']:.3f} · Brier {served_metrics['brier']:.3f} · "
        f"ROC AUC {served_metrics['roc_auc']:.3f}"
    )
    ax.grid(alpha=0.4)
    ax.legend(loc="upper left", facecolor="#161b22", edgecolor="#30363d", fontsize=9)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def train(cfg: MLConfig) -> Dict:
    snapshots, trials = load_snapshots_and_trials(cfg)
    df = build_winnability_frame(snapshots, trials, cfg)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "validation"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    cols = feature_columns(df.columns)
    cats = [c for c in CATEGORICAL_COLUMNS if c in cols]
    y_train = train_df[LABEL].to_numpy()
    y_val = val_df[LABEL].to_numpy()
    y_test = test_df[LABEL].to_numpy()

    edges = cfg.winnability.ask_ratio_bin_edges
    min_count = cfg.winnability.min_bucket_count
    n_bins = cfg.winnability.n_reliability_bins

    baselines = {
        "global_win_rate": GlobalWinRateModel().fit(train_df, y_train),
        "ask_ratio": AskRatioHeuristicModel(edges, min_count).fit(train_df, y_train),
        "broker_market": BrokerMarketGroupedBaseline(edges, min_count).fit(
            train_df, y_train
        ),
    }
    gbm = SklearnWinnabilityModel(cols, cats, cfg.winnability.random_seed).fit(
        train_df, y_train
    )

    served, calibration = _decide_calibration(gbm, val_df, y_val, cfg)

    # Final, one-time evaluation on the held-out test slice.
    results: Dict[str, Dict[str, float]] = {
        name: _eval(y_test, model.predict_proba(test_df), n_bins)
        for name, model in baselines.items()
    }
    results["gbm_uncalibrated"] = _eval(y_test, gbm.predict_proba(test_df), n_bins)
    served_test_prob = served.predict_proba(test_df)
    results["gbm_served"] = _eval(y_test, served_test_prob, n_bins)
    served_reliability = reliability_table(y_test, served_test_prob, n_bins)

    model_path = resolve_path(cfg.winnability.model_path)
    served.save(model_path)
    chart_path = resolve_path(cfg.winnability.reliability_chart_path)
    _render_reliability_chart(
        served_reliability, results["gbm_served"], chart_path, served.is_calibrated
    )

    base_rates = {
        "train": float(y_train.mean()),
        "validation": float(y_val.mean()),
        "test": float(y_test.mean()),
    }
    metadata = {
        "model_name": MODEL_NAME,
        "target": TARGET,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "rows": {
            "train": int(len(train_df)),
            "validation": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "base_win_rate": base_rates,
        "served_model": "calibrated" if served.is_calibrated else "uncalibrated",
        "calibration_method": served.calibration_method,
        "calibration_decision": calibration,
        "metrics_test": results,
        "reliability_table_test": served_reliability,
        "config": {
            "train_fraction": cfg.winnability.train_fraction,
            "validation_fraction": cfg.winnability.validation_fraction,
            "ece_calibration_threshold": cfg.winnability.ece_calibration_threshold,
            "n_reliability_bins": n_bins,
            "ask_ratio_bin_edges": list(edges),
            "min_bucket_count": min_count,
            "random_seed": cfg.winnability.random_seed,
        },
        "features": cols,
    }
    metadata_path = resolve_path(cfg.winnability.metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(
        f"Rows  train {len(train_df):,}  validation {len(val_df):,}  test {len(test_df):,}"
    )
    print(
        f"Base win rate  train {base_rates['train']:.3f}  "
        f"val {base_rates['validation']:.3f}  test {base_rates['test']:.3f}"
    )
    note = calibration.get("note") or (
        f"calibrated via {calibration.get('method')} "
        f"(val ECE {calibration['validation_ece_uncalibrated']:.3f} "
        f"-> {calibration.get('validation_ece_calibrated', float('nan')):.3f})"
    )
    print(f"Calibration: {note}")
    print()
    print(format_table(results))
    print()
    print(f"Saved model     -> {model_path}")
    print(f"Saved metadata  -> {metadata_path}")
    print(f"Saved chart     -> {chart_path}")
    return metadata


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train the bid-winnability model.")
    parser.add_argument("--config", default=None, help="Path to ml_config.yaml")
    args = parser.parse_args(argv)
    cfg = load_ml_config(args.config) if args.config else load_ml_config()
    train(cfg)


if __name__ == "__main__":
    main()
