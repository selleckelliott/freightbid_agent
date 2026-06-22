"""Train + persist the compiled multi-head dispatcher model (Phase 6.3).

Pipeline (mirrors the winnability / payment trainers, test slice touched once):

1. Load the Phase 6.2 structured rows and make a deterministic, action-stratified
   train / validation / test split.
2. Fit the :class:`CompiledDispatcherModel` (five purpose-built heads, class-balanced) on **train**,
   and a majority baseline on the same rows.
3. Evaluate per head on **validation** and once on **test**; compare action macro-F1 to the baseline.
4. Persist the model (joblib, gitignored), a committed **metadata** descriptor (manifest hash,
   provenance, per-head test metrics, estimator types) and a committed **summary** (val+test+baseline).

The artifact carries the feature-manifest hash and refuses to serve on mismatch, so the
``inference_context``-only boundary survives into deployment.

CLI::

    python -m ml.training.train_compiled_dispatcher_model
    python -m ml.training.train_compiled_dispatcher_model --dataset data/compiled_dispatcher_dataset.jsonl
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from ml.data.compiled_agent_trace_schema import TEACHER_TRACE_SCHEMA_VERSION
from ml.models.baseline_compiled_dispatcher import MajorityCompiledDispatcherBaseline
from ml.models.compiled_dispatcher_model import (
    COMPILED_DISPATCHER_MODEL_VERSION,
    MODEL_NAME,
    TARGET_NAMES,
    CompiledDispatcherModel,
)
from ml.training.compiled_dispatcher_dataset import load_rows, split_rows
from ml.training.evaluate_compiled_dispatcher_model import (
    compare_action_macro_f1,
    evaluate_model,
    format_report,
)
from ml.workflows.freightbid_workflow_graph import WORKFLOW_GRAPH_VERSION

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "data" / "compiled_dispatcher_dataset.jsonl"
DEFAULT_DATASET_SUMMARY = ROOT / "artifacts" / "compiled_dispatcher_dataset_summary.json"
DEFAULT_MODEL = ROOT / "ml" / "artifacts" / "compiled_dispatcher_model.joblib"
DEFAULT_METADATA = ROOT / "ml" / "artifacts" / "compiled_dispatcher_model_metadata.json"
DEFAULT_SUMMARY = ROOT / "ml" / "artifacts" / "compiled_dispatcher_model_summary.json"
DEFAULT_SEED = 63


def estimator_types(model: CompiledDispatcherModel) -> Dict[str, str]:
    def clf(head) -> str:
        return type(head.estimator).__name__ if head.estimator is not None else "constant"

    def reg(head) -> str:
        return type(head.estimator).__name__ if head.estimator is not None else "constant"

    types = {
        "action": clf(model.action_head),
        "bid_ratio": reg(model.bid_ratio_head),
        "risk_adjusted_ev": reg(model.ev_head),
        "approval_required": clf(model.approval_head),
    }
    for w, head in model.warning_heads.items():
        types[f"warning::{w}"] = clf(head)
    return types


def train(
    rows: Sequence[Dict[str, Any]],
    *,
    seed: int = DEFAULT_SEED,
    dataset_provenance: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Train + evaluate; returns the model and a structured report (no disk writes)."""
    train_rows, val_rows, test_rows = split_rows(rows, seed=seed)
    model = CompiledDispatcherModel(random_state=seed).fit(train_rows)
    baseline = MajorityCompiledDispatcherBaseline().fit(train_rows)

    val_eval = evaluate_model(model, val_rows)
    test_eval = evaluate_model(model, test_rows)
    baseline_test_eval = evaluate_model(baseline, test_rows)
    comparison = compare_action_macro_f1(test_eval, baseline_test_eval)

    prov = dataset_provenance or {}
    model.set_provenance(
        teacher_trace_schema_version=prov.get(
            "teacher_trace_schema_version", TEACHER_TRACE_SCHEMA_VERSION
        ),
        workflow_graph_version=prov.get("workflow_graph_version", WORKFLOW_GRAPH_VERSION),
        source_policy_version=prov.get("source_policy_version"),
        dataset_version=prov.get("dataset_version"),
        dataset_determinism_hash=prov.get("determinism_hash"),
        dataset_git_commit=(prov.get("provenance") or {}).get("git_commit"),
        random_seed=seed,
        train_rows=len(train_rows),
        validation_rows=len(val_rows),
        test_rows=len(test_rows),
        target_names=list(TARGET_NAMES),
        estimator_types=estimator_types(model),
    )
    return {
        "model": model,
        "baseline": baseline,
        "splits": {"train": len(train_rows), "validation": len(val_rows), "test": len(test_rows)},
        "val_eval": val_eval,
        "test_eval": test_eval,
        "baseline_test_eval": baseline_test_eval,
        "comparison": comparison,
    }


def build_metadata(report: Dict[str, Any]) -> Dict[str, Any]:
    model: CompiledDispatcherModel = report["model"]
    prov = model.provenance
    return {
        "model_name": MODEL_NAME,
        "model_version": COMPILED_DISPATCHER_MODEL_VERSION,
        "trained_at": prov.get("trained_at"),
        "feature_manifest_hash": model.feature_manifest_hash,
        "feature_manifest": model.feature_manifest,
        "categorical_features": model.categorical_features,
        "teacher_trace_schema_version": prov.get("teacher_trace_schema_version"),
        "workflow_graph_version": prov.get("workflow_graph_version"),
        "source_policy_version": prov.get("source_policy_version"),
        "dataset_version": prov.get("dataset_version"),
        "dataset_determinism_hash": prov.get("dataset_determinism_hash"),
        "dataset_git_commit": prov.get("dataset_git_commit"),
        "random_seed": prov.get("random_seed"),
        "rows": report["splits"],
        "target_names": prov.get("target_names"),
        "estimator_types": prov.get("estimator_types"),
        "test_metrics": report["test_eval"],
        "action_macro_f1_vs_baseline": report["comparison"],
    }


def build_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model_name": MODEL_NAME,
        "model_version": COMPILED_DISPATCHER_MODEL_VERSION,
        "rows": report["splits"],
        "comparison": report["comparison"],
        "validation_metrics": report["val_eval"],
        "test_metrics": report["test_eval"],
        "baseline_test_metrics": report["baseline_test_eval"],
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train the compiled dispatcher model.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--dataset-summary", default=str(DEFAULT_DATASET_SUMMARY))
    parser.add_argument("--model-out", default=str(DEFAULT_MODEL))
    parser.add_argument("--metadata-out", default=str(DEFAULT_METADATA))
    parser.add_argument("--summary-out", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    rows = load_rows(args.dataset)
    if not rows:
        raise SystemExit(f"no rows in {args.dataset} (build the 6.2 dataset first)")

    dataset_provenance = None
    summary_path = Path(args.dataset_summary)
    if summary_path.exists():
        dataset_provenance = json.loads(summary_path.read_text(encoding="utf-8"))

    report = train(rows, seed=args.seed, dataset_provenance=dataset_provenance)
    model: CompiledDispatcherModel = report["model"]

    model_path = model.save(args.model_out)
    metadata = build_metadata(report)
    summary = build_summary(report)
    Path(args.metadata_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metadata_out).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    Path(args.summary_out).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    s = report["splits"]
    print(f"Rows  train {s['train']}  validation {s['validation']}  test {s['test']}")
    print()
    print(format_report(report["test_eval"], report["comparison"]))
    print()
    print(f"Saved model     -> {model_path}")
    print(f"Saved metadata  -> {args.metadata_out}")
    print(f"Saved summary   -> {args.summary_out}")


if __name__ == "__main__":
    main()
