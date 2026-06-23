"""Phase 7.5 — Production-Readiness Capstone Demo (the Phase 7 capstone).

Runs the **whole** integration-ready operator workflow end to end, on external-style data,
through the *running* FastAPI app (driven in-process with a ``TestClient`` — no server, no
network, no live Truckstop). It stitches every Phase 7 capability into one honest narrative:

    1. Preflight config validation            (7.4 ``validate_config``)
    2. Liveness & readiness                    (7.4 ``GET /health`` + ``GET /ready``)
    3. External board ingress                  (7.2 board pull -> 7.1 contract validation)
    4. Broker contract + PII redaction         (7.1 ``RawExternalBroker`` -> redacted ``BrokerReference``)
    5. Operating-snapshot ingest               (the truck's feasibility-aligned candidate set)
    6. Source-engine recommendation            (the authoritative engine ranks + prices a bid)
    7. Human-in-the-loop approval              (4.4 draft -> approve -> submit-mock; draft -> reject)
    8. Durable audit export                    (7.3 ``GET /decisions`` -> JSONL/CSV/manifest bundle)
    9. Final readiness recheck                 (7.4 ``GET /ready`` — workflow complete)

Guardrails honored throughout: the **source engine remains authoritative**, ``submit-mock`` is a
*simulated* terminal state (never a real submission), no auto-bidding, and **raw broker PII never
survives** the redaction boundary (asserted on the exported payload).

Determinism
-----------
:func:`run_production_readiness_demo` returns **only deterministic content** (no timestamps, git
SHAs, or runtimes) — the same input yields a byte-identical ``facts`` block and ``stages`` list, so
a committed ``artifacts/production_readiness_summary.json`` is stable and the transcript is
reproducible. The volatile ``generated_at_utc`` / ``runtime_seconds`` are added only by
:func:`write_summary` when the CLI writes the file. The exported audit *bundle* carries volatile
provenance (git/timestamps), so it is written to a gitignored directory and is **not** committed —
only its structural shape (decision count, status counts, file names) is recorded.

Examples
--------
    # write the committed summary + transcript (and a gitignored export bundle)
    python -m benchmarks.run_production_readiness_demo

    # dry run — print the transcript, write nothing
    python -m benchmarks.run_production_readiness_demo --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fastapi.testclient import TestClient

from adapters.inbound.api.app import create_app
from adapters.inbound.api.container import build_container
from application.ingestion.real_broker_schema import broker_reference_from_mapping
from application.ingestion.redaction import assert_no_raw_pii, contains_pii
from application.services import ops_checks
from application.services.decision_exporter import DecisionExporter

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = ROOT / "config"
DEFAULT_TRUCK = ROOT / "sample_data" / "truck.json"
DEFAULT_LOADS = ROOT / "sample_data" / "loads.json"
DEFAULT_BROKERS = ROOT / "sample_data" / "external" / "brokers.csv"
DEFAULT_OUT = ROOT / "artifacts" / "production_readiness_summary.json"
DEFAULT_TRANSCRIPT = ROOT / "artifacts" / "production_readiness_transcript.md"
# Gitignored: the exported bundle carries volatile provenance, so it is regenerable, not committed.
DEFAULT_EXPORT_DIR = ROOT / "artifacts" / "production_readiness_export"

VERDICT_PASS = "PASS"
VERDICT_FAIL = "FAIL"

# Stage keys (stable identifiers surfaced in the summary + asserted by tests).
STAGE_PREFLIGHT = "preflight"
STAGE_READINESS = "readiness"
STAGE_BOARD = "board_ingress"
STAGE_BROKER = "broker_contract"
STAGE_INGEST = "operating_ingest"
STAGE_RECOMMEND = "recommendation"
STAGE_APPROVAL = "approval"
STAGE_EXPORT = "audit_export"
STAGE_FINAL = "final_readiness"


# --------------------------------------------------------------------------- #
# Stage recorder
# --------------------------------------------------------------------------- #
class _Stages:
    """Collects ordered stage results; a failing stage is captured, never raised."""

    def __init__(self) -> None:
        self._items: List[Dict[str, Any]] = []

    def run(self, key: str, title: str, fn: Callable[[], tuple[bool, str]]) -> bool:
        try:
            ok, summary = fn()
        except Exception as exc:  # noqa: BLE001 — a failed stage is a result, not a crash
            ok, summary = False, f"{type(exc).__name__}: {exc}"
        self._items.append(
            {"index": len(self._items) + 1, "key": key, "title": title, "ok": bool(ok), "summary": summary}
        )
        return bool(ok)

    def as_list(self) -> List[Dict[str, Any]]:
        return list(self._items)


def _round(value: Optional[float], ndigits: int) -> Optional[float]:
    return round(value, ndigits) if isinstance(value, (int, float)) else value


# --------------------------------------------------------------------------- #
# The demo
# --------------------------------------------------------------------------- #
def run_production_readiness_demo(
    client: Any,
    *,
    config_dir: Path | str,
    truck: Dict[str, Any],
    loads: Dict[str, Any],
    broker_rows: List[Dict[str, Any]],
    export_dir: Path | str,
    top_n: int = 5,
) -> Dict[str, Any]:
    """Drive the full Phase 7 operator workflow and return a deterministic result.

    ``client`` is anything with httpx-style ``.get``/``.post``/``.delete`` (the CLI/main path passes a
    FastAPI ``TestClient``; a real ``httpx.Client`` against a running server works too). The return
    value contains **no volatile fields** (see module docstring) so it is safe to commit/diff.
    """
    stages = _Stages()
    facts: Dict[str, Any] = {}
    # Origin/destination lookup (from the operating snapshot) for human-readable narration.
    lane = {
        l["load_id"]: f"{l.get('origin_city')} -> {l.get('destination_city')}"
        for l in loads.get("loads", [])
    }
    state: Dict[str, Any] = {}

    # 1. Preflight config validation (local, no server). ----------------------
    def _preflight() -> tuple[bool, str]:
        report = ops_checks.validate_config(config_dir)
        n = len(report["checks"])
        facts["preflight_ok"] = bool(report["ok"])
        facts["config_files_checked"] = n
        return report["ok"], f"all {n} config files load cleanly"

    stages.run(STAGE_PREFLIGHT, "Preflight - config validation", _preflight)

    # 2. Liveness & readiness. ------------------------------------------------
    def _readiness() -> tuple[bool, str]:
        health = client.get("/health").json()
        ready = client.get("/ready").json()
        board = ready["checks"]["load_board"]
        facts["health_ok"] = health.get("status") == "ok"
        facts["readiness_status"] = ready.get("status")
        ok = facts["health_ok"] and ready.get("status") in (
            ops_checks.STATUS_READY,
            ops_checks.STATUS_DEGRADED,
        )
        return ok, (
            f"/health -> {health.get('status')}; /ready -> {ready.get('status')} "
            f"(board: {board.get('source')} {'available' if board.get('available') else 'unavailable'})"
        )

    stages.run(STAGE_READINESS, "Liveness & readiness", _readiness)

    # 3. External board ingress: pull -> validate via the 7.1 contract. -------
    def _board() -> tuple[bool, str]:
        body = client.post("/loads/pull", json={"replace": True}).json()
        facts["board"] = {
            "source": body.get("source"),
            "available": bool(body.get("available")),
            "fetched": body.get("fetched"),
            "accepted": body.get("accepted"),
            "rejected": body.get("rejected"),
        }
        ok = bool(body.get("available")) and (body.get("fetched", 0) or 0) >= 1
        return ok, (
            f"pulled {body.get('fetched')} external-style loads from '{body.get('source')}'; "
            f"{body.get('accepted')} validated + accepted, {body.get('rejected')} rejected"
        )

    stages.run(STAGE_BOARD, "External board ingress (7.2 -> 7.1)", _board)

    # 4. Broker contract + PII redaction (7.1). -------------------------------
    def _broker() -> tuple[bool, str]:
        refs = [broker_reference_from_mapping(row, i) for i, row in enumerate(broker_rows)]
        ref_dicts = [r.as_dict() for r in refs]
        raw_had_pii = any(contains_pii(row) for row in broker_rows)
        # The redaction invariant: no raw email/phone survives onto any reference.
        assert_no_raw_pii(ref_dicts)
        leak = contains_pii(ref_dicts)
        sample = ref_dicts[0] if ref_dicts else {}
        contact = sample.get("contact", {})
        facts["broker_contract"] = {
            "brokers_validated": len(refs),
            "raw_had_pii": raw_had_pii,
            "pii_leak": leak,
            "sample_broker_id": sample.get("broker_id"),
            "sample_credit_bucket": sample.get("credit_bucket"),
            "sample_has_email": contact.get("has_email"),
            "sample_has_phone": contact.get("has_phone"),
            "sample_has_address": contact.get("has_address"),
            "sample_has_contact_name": contact.get("has_contact_name"),
            "sample_email_token_present": contact.get("email_token") is not None,
            "sample_phone_token_present": contact.get("phone_token") is not None,
        }
        ok = (not leak) and raw_had_pii and len(refs) >= 1
        return ok, (
            f"validated {len(refs)} broker(s); contact PII redacted at the chokepoint "
            f"(raw rows carried PII: {raw_had_pii}; reference carried PII: {leak})"
        )

    stages.run(STAGE_BROKER, "Broker contract + PII redaction (7.1)", _broker)

    # 5. Operating-snapshot ingest (the truck's feasible candidate set). ------
    def _ingest() -> tuple[bool, str]:
        # The board pull proved external-ingress connectivity; the recommendation runs on the
        # truck's actual operating snapshot (a feasibility-aligned truck + loads), so swap it in.
        client.delete("/loads")
        body = client.post("/loads", json=loads).json()
        accepted = body.get("accepted", 0)
        facts["operating_ingest_accepted"] = accepted
        return accepted >= 1, f"ingested {accepted} operating load(s) for truck {truck.get('truck_id')}"

    stages.run(STAGE_INGEST, "Operating-snapshot ingest", _ingest)

    # 6. Source-engine recommendation (authoritative). ------------------------
    def _recommend() -> tuple[bool, str]:
        body = client.post("/rank", json={"truck": truck, "top_n": top_n}).json()
        ranked = body.get("ranked", [])
        if not ranked:
            facts["recommendation"] = {"ranked_count": 0}
            return False, "no feasible loads to recommend"
        top = ranked[0]
        bid = top.get("bid") or {}
        state["top_load_id"] = top["load_id"]
        state["second_load_id"] = ranked[1]["load_id"] if len(ranked) > 1 else top["load_id"]
        facts["recommendation"] = {
            "ranked_count": len(ranked),
            "top_load_id": top["load_id"],
            "top_score": _round(top.get("score"), 3),
            "recommended_bid": _round(bid.get("target_bid"), 2),
            "rate_per_mile_at_target": _round(bid.get("rate_per_mile_at_target"), 4),
        }
        return True, (
            f"ranked {len(ranked)} feasible load(s); top = load {top['load_id']} "
            f"({lane.get(top['load_id'], 'lane n/a')}), recommended bid "
            f"${_round(bid.get('target_bid'), 2)} (source engine decides)"
        )

    stages.run(STAGE_RECOMMEND, "Source-engine recommendation", _recommend)

    # 7. Human-in-the-loop approval (4.4). ------------------------------------
    def _approval() -> tuple[bool, str]:
        if "top_load_id" not in state:
            return False, "no recommendation to act on"
        # Approve + simulate-submit the top load.
        d1 = client.post(
            "/bids", json={"truck": truck, "load_id": state["top_load_id"], "actor_id": "demo"}
        ).json()
        bid1 = d1["bid_id"]
        client.post(f"/bids/{bid1}/approve", json={"actor_id": "demo", "note": "within target margin"})
        s1 = client.post(f"/bids/{bid1}/submit-mock", json={"actor_id": "demo"}).json()
        # Reject a second candidate (demonstrates the other terminal path).
        d2 = client.post(
            "/bids", json={"truck": truck, "load_id": state["second_load_id"], "actor_id": "demo"}
        ).json()
        bid2 = d2["bid_id"]
        s2 = client.post(
            f"/bids/{bid2}/reject", json={"actor_id": "demo", "note": "holding out for a better lane"}
        ).json()
        facts["approval"] = {
            "approved_bid_id": bid1,
            "approved_load_id": state["top_load_id"],
            "approved_final_status": s1.get("status"),
            "rejected_bid_id": bid2,
            "rejected_load_id": state["second_load_id"],
            "rejected_final_status": s2.get("status"),
        }
        ok = s1.get("status") == "submitted_mock" and s2.get("status") == "rejected"
        return ok, (
            f"bid {bid1} drafted -> approved -> submit-mock (simulated); "
            f"bid {bid2} drafted -> rejected"
        )

    stages.run(STAGE_APPROVAL, "Human-in-the-loop approval (4.4)", _approval)

    # 8. Durable audit export (7.3). ------------------------------------------
    def _export() -> tuple[bool, str]:
        payload = client.get("/decisions").json()
        records = payload.get("decisions", [])
        provenance = payload.get("provenance", {})
        # Contact PII can only live on a record's (already-redacted) ``broker_reference`` — that is
        # the PII surface to guard, not the timestamps / synthetic refs that legitimately fill an
        # audit record. Assert no raw email/phone survives onto any carried broker reference.
        carried_brokers = [r.get("broker_reference") for r in records if r.get("broker_reference")]
        assert_no_raw_pii(carried_brokers)
        leak = contains_pii(carried_brokers)
        report = DecisionExporter.write_bundle(records, export_dir, provenance=provenance)
        files = sorted(
            Path(p).name for p in (report.jsonl_path, report.csv_path, report.manifest_path)
        )
        facts["audit_export"] = {
            "decision_count": report.decision_count,
            "status_counts": dict(report.status_counts),
            "files": files,
            "provenance_keys": sorted(provenance.keys()),
            "model_artifact_ids": provenance.get("model_artifact_ids"),
            "source_policy_version": provenance.get("source_policy_version"),
            "records_carrying_broker": len(carried_brokers),
            "pii_leak_in_export": leak,
        }
        ok = report.decision_count >= 1 and not leak
        return ok, (
            f"exported {report.decision_count} decision record(s) to an audit bundle "
            f"({', '.join(files)}) with model/config provenance; status counts "
            f"{dict(report.status_counts)}; no contact PII in export"
        )

    stages.run(STAGE_EXPORT, "Durable audit export (7.3)", _export)

    # 9. Final readiness recheck. ---------------------------------------------
    def _final() -> tuple[bool, str]:
        ready = client.get("/ready").json()
        facts["final_readiness_status"] = ready.get("status")
        ok = ready.get("status") in (ops_checks.STATUS_READY, ops_checks.STATUS_DEGRADED)
        return ok, f"/ready -> {ready.get('status')}; workflow complete"

    stages.run(STAGE_FINAL, "Final readiness recheck", _final)

    items = stages.as_list()
    passed = sum(1 for s in items if s["ok"])
    ok = passed == len(items)
    return {
        "phase": "7.5",
        "title": "Production-Readiness Capstone Demo",
        "ok": ok,
        "verdict": VERDICT_PASS if ok else VERDICT_FAIL,
        "stages_passed": passed,
        "stages_total": len(items),
        "stages": items,
        "facts": facts,
    }


# --------------------------------------------------------------------------- #
# Wiring + rendering
# --------------------------------------------------------------------------- #
def build_app_client(config_dir: Path | str = DEFAULT_CONFIG_DIR):
    """Build a fresh container + in-process ``TestClient`` (no server, no network)."""
    container = build_container(Path(config_dir))
    return TestClient(create_app(container)), container


def load_broker_rows(path: Path | str = DEFAULT_BROKERS) -> List[Dict[str, Any]]:
    """Read the committed external broker fixture (CSV) into raw row mappings (carry PII)."""
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def render_transcript(result: Dict[str, Any]) -> str:
    """Render a short, deterministic markdown demo transcript from a demo result."""
    lines: List[str] = []
    lines.append("# FreightBid - Production-Readiness Capstone Demo (Phase 7.5)")
    lines.append("")
    lines.append(
        "End-to-end operator workflow on external-style data, driven through the running FastAPI app "
        "in-process. The source engine stays authoritative; no live Truckstop, no auto-bidding, and "
        "raw broker PII never crosses the redaction boundary."
    )
    lines.append("")
    for stage in result["stages"]:
        mark = "OK  " if stage["ok"] else "FAIL"
        lines.append(f"## {stage['index']}. {stage['title']}")
        lines.append(f"{mark} {stage['summary']}")
        lines.append("")
    lines.append(
        f"VERDICT: {result['verdict']} - {result['stages_passed']}/{result['stages_total']} stages green."
    )
    lines.append("")
    return "\n".join(lines)


def write_summary(
    result: Dict[str, Any],
    path: Path | str,
    *,
    runtime_seconds: float,
    generated_at: Optional[datetime] = None,
) -> Path:
    """Write the committed summary JSON, wrapping the deterministic result with volatile run metadata."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": (generated_at or datetime.now(timezone.utc)).isoformat(),
        "runtime_seconds": round(runtime_seconds, 3),
        **result,
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return out


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 7.5 production-readiness capstone demo.")
    p.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    p.add_argument("--truck", type=Path, default=DEFAULT_TRUCK)
    p.add_argument("--loads", type=Path, default=DEFAULT_LOADS)
    p.add_argument("--brokers", type=Path, default=DEFAULT_BROKERS)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Committed summary JSON path.")
    p.add_argument("--transcript", type=Path, default=DEFAULT_TRANSCRIPT, help="Committed transcript path.")
    p.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR, help="Audit bundle dir (gitignored).")
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--dry-run", action="store_true", help="Print the transcript; write nothing.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    truck = json.loads(Path(args.truck).read_text(encoding="utf-8"))
    loads = json.loads(Path(args.loads).read_text(encoding="utf-8"))
    broker_rows = load_broker_rows(args.brokers)

    client, _container = build_app_client(args.config_dir)

    # The audit-bundle write is part of the demonstrated workflow (stage 8). Under --dry-run, route
    # it to a throwaway temp dir so a dry run persists *nothing* on disk.
    tmp_export = tempfile.TemporaryDirectory() if args.dry_run else None
    export_dir = Path(tmp_export.name) if tmp_export is not None else args.export_dir
    try:
        start = time.perf_counter()
        result = run_production_readiness_demo(
            client,
            config_dir=args.config_dir,
            truck=truck,
            loads=loads,
            broker_rows=broker_rows,
            export_dir=export_dir,
            top_n=args.top_n,
        )
        runtime = time.perf_counter() - start

        transcript = render_transcript(result)
        print(transcript)

        if not args.dry_run:
            summary_path = write_summary(result, args.out, runtime_seconds=runtime)
            Path(args.transcript).parent.mkdir(parents=True, exist_ok=True)
            Path(args.transcript).write_text(transcript, encoding="utf-8")
            print(f"Wrote {summary_path}")
            print(f"Wrote {args.transcript}")
            print(f"Audit bundle: {export_dir} (gitignored)")
    finally:
        if tmp_export is not None:
            tmp_export.cleanup()

    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
