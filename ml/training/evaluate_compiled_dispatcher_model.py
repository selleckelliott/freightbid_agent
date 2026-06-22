"""Per-head evaluation for the compiled dispatcher (Phase 6.3).

Scores each head against the right metric for its target — imbalance-aware classification for the
action / approval / warning heads (macro-F1, balanced accuracy, **per-class recall with support**,
never plain accuracy), and regression error for the bid-ratio and risk-adjusted-EV heads (plus the
reconstructed bid in dollars). Every head is scored on the rows where its target is *applicable*
(bid-ratio on biddable rows, EV on feasible rows), through each model's raw per-head surface so the
learned model and the majority baseline are compared apples-to-apples. The headline is action
macro-F1 model-vs-baseline.

CLI::

    python -m ml.training.evaluate_compiled_dispatcher_model \
        --model ml/artifacts/compiled_dispatcher_model.joblib \
        --dataset data/compiled_dispatcher_dataset.jsonl
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

from ml.models.compiled_dispatcher_model import (
    WARNING_HEADS,
    CompiledDispatcherModel,
    derive_action,
    derive_approval_required,
    derive_bid_ratio,
    derive_ev,
    derive_warnings,
    row_features,
)
from ml.training.compiled_dispatcher_dataset import load_rows, split_rows


# --------------------------------------------------------------------------- #
# Metric primitives
# --------------------------------------------------------------------------- #
def action_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict[str, Any]:
    labels = sorted(set(y_true) | set(y_pred))
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_class = {
        lab: {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, lab in enumerate(labels)
    }
    return {
        "n": int(len(y_true)),
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class": per_class,
    }


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.size == 0:
        return {"n": 0, "mae": None, "rmse": None, "r2": None}
    err = yp - yt
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    return {
        "n": int(yt.size),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": (1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0,
    }


def binary_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, Any]:
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    p, r, f1, _ = precision_recall_fscore_support(
        yt, yp, labels=[0, 1], average=None, zero_division=0
    )
    single_class = yt.sum() in (0, yt.size)
    return {
        "n": int(yt.size),
        "positives": int(yt.sum()),
        "precision": float(p[1]),
        "recall": float(r[1]),
        "f1": float(f1[1]),
        # balanced accuracy is undefined when the slice has a single true class
        "balanced_accuracy": None if single_class else float(balanced_accuracy_score(yt, yp)),
    }


# --------------------------------------------------------------------------- #
# Whole-model evaluation (works for the model and the baseline)
# --------------------------------------------------------------------------- #
def evaluate_model(model: Any, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    feats = [row_features(r) for r in rows]
    raw = model.predict_raw(feats)

    action = action_metrics([derive_action(r) for r in rows], list(raw["action"]))

    # bid-ratio head: biddable rows only (reconstruct dollar error too)
    ratios_true, ratios_pred, bid_true, bid_pred = [], [], [], []
    for i, r in enumerate(rows):
        true_ratio = derive_bid_ratio(r)
        if true_ratio is None:
            continue
        ratios_true.append(true_ratio)
        ratios_pred.append(float(raw["bid_ratio"][i]))
        miles = float(r["features"]["loaded_miles"])
        market = float(r["features"]["market_rate"])
        bid_true.append(float(r["targets"]["recommended_bid_amount"]))
        bid_pred.append(float(raw["bid_ratio"][i]) * market * miles)
    bid_ratio = regression_metrics(ratios_true, ratios_pred)
    bid_ratio["bid_dollar_mae"] = regression_metrics(bid_true, bid_pred)["mae"]

    # EV head: feasible rows only
    ev_true, ev_pred = [], []
    for i, r in enumerate(rows):
        ev = derive_ev(r)
        if ev is None:
            continue
        ev_true.append(float(ev))
        ev_pred.append(float(raw["risk_adjusted_ev"][i]))
    ev = regression_metrics(ev_true, ev_pred)

    warnings = {
        w: binary_metrics([derive_warnings(r)[w] for r in rows], list(raw["warnings"][w]))
        for w in WARNING_HEADS
    }
    warning_macro_f1 = float(np.mean([warnings[w]["f1"] for w in WARNING_HEADS]))
    approval = binary_metrics(
        [derive_approval_required(r) for r in rows], list(raw["approval_required"])
    )

    return {
        "action": action,
        "bid_ratio": bid_ratio,
        "risk_adjusted_ev": ev,
        "warnings": warnings,
        "warning_macro_f1": warning_macro_f1,
        "approval_required": approval,
    }


def compare_action_macro_f1(model_eval: Dict[str, Any], baseline_eval: Dict[str, Any]) -> Dict[str, Any]:
    m = model_eval["action"]["macro_f1"]
    b = baseline_eval["action"]["macro_f1"]
    return {
        "model_action_macro_f1": m,
        "baseline_action_macro_f1": b,
        "uplift": m - b,
        "model_beats_baseline": m > b,
    }


def format_report(model_eval: Dict[str, Any], comparison: Dict[str, Any]) -> str:
    def num(value: Any, spec: str) -> str:
        return format(value, spec) if value is not None else "n/a"

    a = model_eval["action"]
    lines = [
        f"Action       acc {a['accuracy']:.3f}  balanced-acc {a['balanced_accuracy']:.3f}  "
        f"macro-F1 {a['macro_f1']:.3f}  (baseline {comparison['baseline_action_macro_f1']:.3f}, "
        f"uplift {comparison['uplift']:+.3f})",
    ]
    for lab, m in a["per_class"].items():
        lines.append(
            f"  {lab:<18} recall {m['recall']:.3f}  precision {m['precision']:.3f}  "
            f"f1 {m['f1']:.3f}  (n={m['support']})"
        )
    br = model_eval["bid_ratio"]
    ev = model_eval["risk_adjusted_ev"]
    lines.append(
        f"Bid ratio    MAE {num(br['mae'], '.4f')}  RMSE {num(br['rmse'], '.4f')}  "
        f"R2 {num(br['r2'], '.3f')}  (bid $ MAE {num(br['bid_dollar_mae'], '.2f')}, n={br['n']})"
    )
    lines.append(
        f"Risk-adj EV  MAE {num(ev['mae'], '.2f')}  RMSE {num(ev['rmse'], '.2f')}  "
        f"R2 {num(ev['r2'], '.3f')}  (n={ev['n']})"
    )
    for w, m in model_eval["warnings"].items():
        lines.append(
            f"Warn {w:<16} recall {m['recall']:.3f}  precision {m['precision']:.3f}  "
            f"f1 {m['f1']:.3f}  (pos={m['positives']}/{m['n']})"
        )
    ap = model_eval["approval_required"]
    lines.append(
        f"Approval     recall {ap['recall']:.3f}  precision {ap['precision']:.3f}  "
        f"f1 {ap['f1']:.3f}  (pos={ap['positives']}/{ap['n']})"
    )
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved compiled dispatcher model.")
    parser.add_argument("--model", default="ml/artifacts/compiled_dispatcher_model.joblib")
    parser.add_argument("--dataset", default="data/compiled_dispatcher_dataset.jsonl")
    parser.add_argument("--seed", type=int, default=63)
    args = parser.parse_args(argv)

    from ml.models.baseline_compiled_dispatcher import MajorityCompiledDispatcherBaseline

    model = CompiledDispatcherModel.load(args.model)
    rows = load_rows(args.dataset)
    train, _val, test = split_rows(rows, seed=args.seed)
    baseline = MajorityCompiledDispatcherBaseline().fit(train)

    model_eval = evaluate_model(model, test)
    baseline_eval = evaluate_model(baseline, test)
    comparison = compare_action_macro_f1(model_eval, baseline_eval)
    print(format_report(model_eval, comparison))
    print()
    print(
        f"Model beats majority baseline on action macro-F1: {comparison['model_beats_baseline']}"
    )


if __name__ == "__main__":
    main()
