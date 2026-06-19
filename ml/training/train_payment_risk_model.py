"""Train + evaluate the calibrated payment-risk model (Phase 5.2).

Pipeline (calibration-first, test set touched once) — the payment analogue of
``ml/training/train_winnability_model.py``:

1. Build the load-level payment dataset and the three-way grouped time split.
2. Fit three baselines (global default rate, credit-bucket, bonded/quick-pay) and a
   ``HistGradientBoostingClassifier`` for ``P(default)`` on the **train** slice, then
   fit the optional ``E[pay_days]`` regressor on the **non-default** train rows only.
3. Decide calibration on the **validation** slice only: if the base model's validation
   ECE exceeds the configured threshold, fit prefit isotonic + sigmoid calibrators and
   serve whichever lowers validation ECE the most (and beats the uncalibrated model).
   Otherwise serve the uncalibrated model. Default is a minority class, so a low ECE
   here is the headline result — ranking gains are honestly modest.
4. Evaluate every model **once** on the held-out **test** slice: ROC AUC, PR AUC,
   Brier, log loss, ECE for ``P(default)``; MAE + RMSE for ``E[pay_days]`` over the
   non-default test rows. Render the served model's reliability diagram.
5. Save the served model (joblib), a metadata JSON (committed result artifact), and the
   reliability PNG.

CLI::

    python -m ml.training.train_payment_risk_model --config config/ml_config.yaml
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error  # noqa: E402

from ml.config import MLConfig, load_ml_config  # noqa: E402
from ml.features.payment_features import (  # noqa: E402
    PAYMENT_CATEGORICAL_COLUMNS,
    payment_feature_columns,
)
from ml.models.baseline_payment_model import (  # noqa: E402
    BondedQuickPayBaseline,
    CreditBucketBaseline,
    GlobalDefaultRateModel,
)
from ml.models.sklearn_payment_risk_model import (  # noqa: E402
    MODEL_NAME,
    SklearnPaymentRiskModel,
)
from ml.training.payment_risk_dataset import (  # noqa: E402
    LABEL,
    PAY_DAYS,
    TARGET,
    build_payment_frame,
    load_snapshots_and_outcomes,
    resolve_path,
)
from ml.training.winnability_metrics import (  # noqa: E402
    evaluate_probabilities,
    expected_calibration_error,
    reliability_table,
)

_MODEL_LABELS = {
    "global_default_rate": "Global default rate",
    "credit_bucket": "Credit bucket",
    "bonded_quick_pay": "Bonded/quick-pay",
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
        "global_default_rate",
        "credit_bucket",
        "bonded_quick_pay",
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
    model: SklearnPaymentRiskModel, val_df, y_val, cfg: MLConfig
):
    """Validation-only calibration decision. Returns ``(served_model, decision)``."""
    n_bins = cfg.payment_risk.n_reliability_bins
    threshold = cfg.payment_risk.ece_calibration_threshold
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
    ax.set_xlabel("mean predicted P(default)")
    ax.set_ylabel("observed default rate")
    kind = "calibrated" if calibrated else "uncalibrated"
    ax.set_title(
        f"FreightBid Agent — Payment-Risk Reliability ({kind})\n"
        f"test ECE {served_metrics['ece']:.3f} · Brier {served_metrics['brier']:.3f} · "
        f"ROC AUC {served_metrics['roc_auc']:.3f}"
    )
    ax.grid(alpha=0.4)
    ax.legend(loc="upper left", facecolor="#161b22", edgecolor="#30363d", fontsize=9)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def _pay_days_metrics(model: SklearnPaymentRiskModel, test_df) -> Dict[str, object]:
    """MAE / RMSE for ``E[pay_days]`` over the **non-default** test rows.

    Defaulted loads have no realized pay-days, so they are excluded — the regressor is
    only ever asked about loads that actually paid.
    """
    paid = test_df[test_df["is_default"] == 0]
    paid = paid[paid[PAY_DAYS].notna()].reset_index(drop=True)
    n = int(len(paid))
    if n == 0 or model.pay_days_regressor is None:
        return {"n": n, "mae": float("nan"), "rmse": float("nan")}
    y_true = paid[PAY_DAYS].to_numpy(dtype=float)
    y_pred = model.predict_pay_days(paid)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(sqrt(mean_squared_error(y_true, y_pred)))
    return {"n": n, "mae": mae, "rmse": rmse}


def train(cfg: MLConfig) -> Dict:
    snapshots, outcomes = load_snapshots_and_outcomes(cfg)
    df = build_payment_frame(snapshots, outcomes, cfg)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "validation"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    cols = payment_feature_columns(df.columns)
    cats = [c for c in PAYMENT_CATEGORICAL_COLUMNS if c in cols]
    y_train = train_df[LABEL].to_numpy()
    y_val = val_df[LABEL].to_numpy()
    y_test = test_df[LABEL].to_numpy()

    min_count = cfg.payment_risk.min_bucket_count
    n_bins = cfg.payment_risk.n_reliability_bins

    baselines = {
        "global_default_rate": GlobalDefaultRateModel().fit(train_df, y_train),
        "credit_bucket": CreditBucketBaseline(min_count).fit(train_df, y_train),
        "bonded_quick_pay": BondedQuickPayBaseline(min_count).fit(train_df, y_train),
    }
    gbm = SklearnPaymentRiskModel(cols, cats, cfg.payment_risk.random_seed).fit(
        train_df, y_train
    )

    # Secondary head: E[pay_days] on the non-default train rows (they carry pay-days).
    pay_train = train_df[train_df["is_default"] == 0]
    pay_train = pay_train[pay_train[PAY_DAYS].notna()].reset_index(drop=True)
    if len(pay_train) > 0:
        gbm.fit_pay_days(pay_train, pay_train[PAY_DAYS].to_numpy())

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
    pay_days = _pay_days_metrics(served, test_df)

    model_path = resolve_path(cfg.payment_risk.model_path)
    served.save(model_path)
    chart_path = resolve_path(cfg.payment_risk.reliability_chart_path)
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
        "base_default_rate": base_rates,
        "served_model": "calibrated" if served.is_calibrated else "uncalibrated",
        "calibration_method": served.calibration_method,
        "calibration_decision": calibration,
        "metrics_test": results,
        "pay_days_test": pay_days,
        "reliability_table_test": served_reliability,
        "config": {
            "train_fraction": cfg.payment_risk.train_fraction,
            "validation_fraction": cfg.payment_risk.validation_fraction,
            "ece_calibration_threshold": cfg.payment_risk.ece_calibration_threshold,
            "n_reliability_bins": n_bins,
            "min_bucket_count": min_count,
            "random_seed": cfg.payment_risk.random_seed,
        },
        "hyperparameters": {
            "learning_rate": 0.08,
            "max_iter": 400,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 40,
            "l2_regularization": 0.1,
            "early_stopping": True,
            "validation_fraction": 0.15,
        },
        "features": cols,
        "categorical_columns": cats,
    }
    metadata_path = resolve_path(cfg.payment_risk.metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(
        f"Rows  train {len(train_df):,}  validation {len(val_df):,}  test {len(test_df):,}"
    )
    print(
        f"Base default rate  train {base_rates['train']:.3f}  "
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
    print(
        f"E[pay_days] (non-default test, n={pay_days['n']:,})  "
        f"MAE {pay_days['mae']:.2f}  RMSE {pay_days['rmse']:.2f} days"
    )
    print()
    print(f"Saved model     -> {model_path}")
    print(f"Saved metadata  -> {metadata_path}")
    print(f"Saved chart     -> {chart_path}")
    return metadata


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train the payment-risk model.")
    parser.add_argument("--config", default=None, help="Path to ml_config.yaml")
    args = parser.parse_args(argv)
    cfg = load_ml_config(args.config) if args.config else load_ml_config()
    train(cfg)


if __name__ == "__main__":
    main()
