"""Build the compiled-dispatcher training dataset from teacher traces (Phase 6.2).

Reads the Phase 6.1 teacher traces (``data/teacher_traces.jsonl``) and emits, deterministically,
the two dataset forms the compiled dispatcher (6.3) consumes:

* **structured rows** -> ``data/compiled_dispatcher_dataset.jsonl`` (gitignored) — inference-only
  features + output-side targets + coverage labels, the committed CI path's training matrix; and
* **conversations** -> ``data/compiled_dispatcher_conversations.jsonl`` (gitignored) — templated
  prompt -> JSON completion (+ a deterministic human-in-the-loop continuation), the optional LLM path.

The committed result is the lean ``artifacts/compiled_dispatcher_dataset_summary.json``: coverage
histograms, the feature/target schema, a determinism hash, provenance carried from the traces, and
the **leakage check** result. Every row's inputs pass through :func:`build_features`, so the
``inference_context``-only boundary is enforced at construction, not merely asserted after the fact.

Examples
--------
    # build from the canonical teacher traces -> committed summary + gitignored datasets
    python -m ml.data.build_compiled_dispatcher_dataset

    # quick smoke over the first N traces
    python -m ml.data.build_compiled_dispatcher_dataset --limit 200
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

from ml.data.compiled_agent_trace_schema import (
    AgentTrace,
    inference_field_names,
)
from ml.data.compiled_dispatcher_formatters import (
    DISPATCHER_DATASET_VERSION,
    assert_features_inference_only,
    build_targets,
    coverage_flags,
    render_conversation,
    scenario_category,
    to_structured_row,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACES = ROOT / "data" / "teacher_traces.jsonl"
DEFAULT_STRUCTURED = ROOT / "data" / "compiled_dispatcher_dataset.jsonl"
DEFAULT_CONVERSATIONS = ROOT / "data" / "compiled_dispatcher_conversations.jsonl"
DEFAULT_SUMMARY = ROOT / "artifacts" / "compiled_dispatcher_dataset_summary.json"


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def load_traces(path: str | Path, *, limit: Optional[int] = None) -> List[AgentTrace]:
    traces: List[AgentTrace] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            traces.append(AgentTrace.from_json_dict(json.loads(line)))
            if limit is not None and len(traces) >= limit:
                break
    return traces


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
class DispatcherDataset:
    """The two dataset forms plus a deterministic, order-independent fingerprint."""

    def __init__(self, rows: List["OrderedDict[str, Any]"],
                 conversations: List["OrderedDict[str, Any]"]):
        self.rows = rows
        self.conversations = conversations

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for row in self.rows:
            h.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        for conv in self.conversations:
            h.update(json.dumps(conv, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        return h.hexdigest()


def build_dataset(traces: List[AgentTrace]) -> DispatcherDataset:
    """Render both forms in a stable (scenario_id-sorted) order, enforcing the input boundary."""
    ordered = sorted(traces, key=lambda t: t.scenario_id)
    rows = [to_structured_row(t) for t in ordered]
    for row in rows:  # belt-and-suspenders: inputs never carry teacher-only fields
        assert_features_inference_only(row["features"])
    conversations = [render_conversation(t) for t in ordered]
    return DispatcherDataset(rows, conversations)


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def build_summary(traces: List[AgentTrace], dataset: DispatcherDataset) -> Dict[str, Any]:
    rows, convs = dataset.rows, dataset.conversations
    categories = Counter(r["scenario_category"] for r in rows)
    flags = Counter(f for r in rows for f in r["coverage_flags"])
    decisions = Counter(r["targets"]["decision"] for r in rows)
    branches = Counter(r["targets"]["hub_branch"] for r in rows)
    warnings = Counter(w for r in rows for w in r["targets"]["warnings"])
    human_actions = Counter(c["human_action"] for c in convs if c["human_action"] is not None)

    per_world: "OrderedDict[str, Counter]" = OrderedDict()
    for r in rows:
        per_world.setdefault(r["world"], Counter())[r["scenario_category"]] += 1
    world_rows = [
        {"world": w, "n": sum(cats.values()), "categories": dict(cats)}
        for w, cats in per_world.items()
    ]

    first = traces[0].metadata if traces else None
    target_keys = list(build_targets(traces[0]).keys()) if traces else []
    return {
        "dataset_version": DISPATCHER_DATASET_VERSION,
        "teacher_trace_schema_version": (first.teacher_trace_schema_version if first else None),
        "workflow_graph_version": (first.workflow_graph_version if first else None),
        "source_policy_version": (first.source_policy_version if first else None),
        "generated": {
            "n_examples": len(rows),
            "n_conversations": len(convs),
            "n_worlds": len(world_rows),
            "n_with_human_action": sum(human_actions.values()),
        },
        "determinism_hash": dataset.fingerprint(),
        "category_histogram": dict(categories),
        "coverage_flag_histogram": dict(flags),
        "decision_histogram": dict(decisions),
        "hub_branch_histogram": dict(branches),
        "warning_histogram": dict(warnings),
        "human_action_histogram": dict(human_actions),
        "per_world": world_rows,
        "train_eligibility": {
            "features_source": "inference_context",
            "feature_fields": sorted(inference_field_names()),
            "target_fields": target_keys,
            "note": (
                "Inputs (features/prompt) are inference_context only; targets/completion may "
                "include node_outputs (e.g. risk_adjusted_ev) as predicted outputs, never inputs."
            ),
            "leakage_check": "pass",
        },
        "provenance": {
            "git_commit": (first.git_commit if first else None),
            "model_artifact_ids": (first.model_artifact_ids if first else None),
            "random_seed": (first.random_seed if first else None),
        },
    }


# --------------------------------------------------------------------------- #
# Write
# --------------------------------------------------------------------------- #
def _write_jsonl(records: List["OrderedDict[str, Any]"], path: str | Path) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
    return len(records)


def _print_summary(summary: Dict[str, Any]) -> None:
    g = summary["generated"]
    print(
        f"compiled-dispatcher dataset: {g['n_examples']} rows / {g['n_conversations']} convs "
        f"over {g['n_worlds']} worlds ({g['n_with_human_action']} with a human action)"
    )
    print(f"  categories : {summary['category_histogram']}")
    print(f"  decisions  : {summary['decision_histogram']}")
    print(f"  human acts : {summary['human_action_histogram']}")
    print(f"  hash       : {summary['determinism_hash'][:16]}...")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--traces", default=str(DEFAULT_TRACES))
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N traces.")
    parser.add_argument("--out-structured", default=str(DEFAULT_STRUCTURED))
    parser.add_argument("--out-conversations", default=str(DEFAULT_CONVERSATIONS))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    args = parser.parse_args()

    traces = load_traces(args.traces, limit=args.limit)
    if not traces:
        raise SystemExit(f"no traces found in {args.traces} (run the teacher generator first)")

    dataset = build_dataset(traces)
    n_rows = _write_jsonl(dataset.rows, args.out_structured)
    n_convs = _write_jsonl(dataset.conversations, args.out_conversations)
    summary = build_summary(traces, dataset)
    out = Path(args.summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    _print_summary(summary)
    print(f"wrote {n_rows} rows -> {args.out_structured}")
    print(f"wrote {n_convs} conversations -> {args.out_conversations}")
    print(f"wrote summary -> {args.summary}")


if __name__ == "__main__":
    main()
