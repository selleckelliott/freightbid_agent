"""Phase 7.4 — operations hardening: readiness, config validation, artifact availability, smoke test.

Side-effect-free inspection helpers shared by the inbound API (``GET /ready``) and the CLI
(``validate-config``, ``ready``, ``smoke-test``). They never mutate engine state and never make a
recommendation themselves — they report *what is wired*, *which model artifacts are present*, and
*whether the config files load cleanly*, so an operator (or a container orchestrator) can decide
whether FreightBid is ready to serve before sending real traffic.

Design notes
------------
* **Liveness vs readiness.** ``GET /health`` (Phase 1) stays the liveness probe — "the process is
  up". Readiness (here) is richer: it reports config + load-board + model-artifact status. Because
  every model dependency is *optional* (a fresh clone runs on rule-based fallbacks + the sandbox
  board by design), readiness reports ``status: "degraded"`` rather than failing — the engine still
  serves. ``"degraded"`` means "running, but an enabled dependency is missing or a board is down".
* **No side effects.** ``readiness_report`` only calls the boards' cheap
  :meth:`~ports.load_board.LoadBoardPort.availability` and stats the model-artifact paths; it never
  pulls, ranks, or drafts. ``run_smoke`` *does* exercise the write paths (that is the point of a
  smoke test) but is driven explicitly by an operator command, never by readiness.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from application.config_loader import (
    BidRecommenderConfig,
    CompiledDispatcherConfig,
    load_bid_approval_config,
    load_bid_recommender_config,
    load_compiled_dispatcher_config,
    load_config,
    load_load_board_config,
    load_objective_profiles,
)
from application.services.decision_exporter import (
    SOURCE_POLICY_VERSION,
    git_commit,
    git_describe,
)
from ports.compiled_dispatcher import CompiledDispatcherAvailability

ROOT = Path(__file__).resolve().parents[2]

# Stable readiness status strings (surfaced in the API body + CLI output).
STATUS_READY = "ready"
STATUS_DEGRADED = "degraded"


# --------------------------------------------------------------------------- #
# Config validation (local — no running server required)
# --------------------------------------------------------------------------- #
def validate_config(config_dir: str | Path) -> Dict[str, Any]:
    """Attempt to load every config file and report per-file ``ok``/``error``.

    A preflight an operator can run *before* booting the API (the loaders are pure disk reads). Never
    raises — a malformed/missing file is captured as a failed check with a typed message, so the
    command can exit non-zero with a precise reason instead of a stack trace.
    """
    cdir = Path(config_dir)
    checks: List[Dict[str, Any]] = []

    def _try(name: str, fn: Callable[[], Any]) -> None:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 — surface, don't crash the preflight
            checks.append({"name": name, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
        else:
            checks.append({"name": name, "ok": True, "error": None})

    _try("app_config", lambda: load_config(cdir))
    _try("bid_recommender", lambda: load_bid_recommender_config(cdir))
    _try("bid_approval", lambda: load_bid_approval_config(cdir))
    _try("compiled_dispatcher", lambda: load_compiled_dispatcher_config(cdir))
    _try("load_board", lambda: load_load_board_config(cdir))
    _try("objective_profiles", lambda: load_objective_profiles(cdir))

    return {
        "ok": all(c["ok"] for c in checks),
        "config_dir": str(cdir),
        "checks": checks,
    }


# --------------------------------------------------------------------------- #
# Artifact availability
# --------------------------------------------------------------------------- #
def _resolve(path_str: str, root: Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else root / p


def artifact_availability(
    bid_cfg: BidRecommenderConfig,
    compiled_cfg: CompiledDispatcherConfig,
    compiled_avail: CompiledDispatcherAvailability,
    root: Path = ROOT,
) -> Dict[str, Dict[str, Any]]:
    """Report each optional model: is it *enabled* in config, and is its artifact *present* on disk.

    The gitignored model artifacts are absent in a fresh clone, so the honest default is
    ``enabled=False, present=False`` for every head — the engine runs on rule-based fallbacks. The
    compiled dispatcher additionally carries its shadow ``available``/``reason`` (it never owns a
    decision; its presence is recorded for audit/ops only).
    """
    winnability = _resolve(bid_cfg.model_path, root)
    payment = _resolve(bid_cfg.payment_model_path, root)
    return {
        "winnability": {
            "enabled": bid_cfg.enabled,
            "path": bid_cfg.model_path,
            "present": winnability.exists(),
        },
        "payment_risk": {
            "enabled": bid_cfg.risk_adjusted_ev_enabled,
            "path": bid_cfg.payment_model_path,
            "present": payment.exists(),
        },
        "compiled_dispatcher": {
            "enabled": compiled_cfg.enabled,
            "path": compiled_cfg.artifact_path,
            "present": _resolve(compiled_cfg.artifact_path, root).exists(),
            "shadow_available": compiled_avail.available,
            "reason": compiled_avail.reason,
        },
    }


# --------------------------------------------------------------------------- #
# Readiness report
# --------------------------------------------------------------------------- #
def readiness_report(container: Any, root: Path = ROOT) -> Dict[str, Any]:
    """Aggregate config + load-board + artifact status into a single readiness payload.

    ``status`` is :data:`STATUS_READY` unless something is *enabled but unusable* — an enabled model
    whose artifact is missing, or a configured board that cannot serve — in which case it is
    :data:`STATUS_DEGRADED` with a human-readable ``warnings`` list. The engine still serves while
    degraded (rule-based fallbacks + the sandbox board), so this is informational, not a hard gate.
    """
    board = container.load_board.availability().to_dict()
    compiled_avail = container.compiled_dispatcher_shadow.availability()
    artifacts = artifact_availability(
        container.bid_recommender_config,
        container.compiled_dispatcher_config,
        compiled_avail,
        root,
    )

    warnings: List[str] = []
    for name in ("winnability", "payment_risk"):
        a = artifacts[name]
        if a["enabled"] and not a["present"]:
            warnings.append(f"{name} enabled but artifact missing: {a['path']}")
    compiled = artifacts["compiled_dispatcher"]
    if compiled["enabled"] and not compiled["shadow_available"]:
        warnings.append(f"compiled dispatcher enabled but unavailable: {compiled['reason']}")
    if not board["available"]:
        warnings.append(f"load board '{board['source']}' unavailable: {board['reason']}")

    return {
        "status": STATUS_READY if not warnings else STATUS_DEGRADED,
        "source_policy_version": SOURCE_POLICY_VERSION,
        "git_commit": git_commit(),
        "git_describe": git_describe(),
        "checks": {
            # the container built, so the source engine can serve; this is the liveness-of-engine bit
            "engine": {"ok": True},
            "load_board": board,
            "artifacts": artifacts,
        },
        "warnings": warnings,
    }


# --------------------------------------------------------------------------- #
# End-to-end smoke test (drives a running API)
# --------------------------------------------------------------------------- #
def run_smoke(
    client: Any, *, truck: Dict[str, Any], loads: Dict[str, Any], top_n: int = 5
) -> Dict[str, Any]:
    """Exercise the full operator workflow against a running API and report per-step pass/fail.

    ``client`` is anything with httpx-style ``.get``/``.post`` (the CLI passes a real ``httpx.Client``;
    tests pass a FastAPI ``TestClient``). Unlike the readiness probe this **mutates** state — it pulls
    sandbox loads, ingests the canonical sample loads, and drafts a bid — which is the point: it proves
    the write paths work end to end. It still never approves/submits and never bids for real.

    The board ``pull`` step proves Phase 7.2 *connectivity* (the sandbox board is deterministic but its
    rows are not feasibility-aligned to any one truck — they exercise the ingestion contract). The
    ``rank``/``bid`` steps then use the committed sample truck+loads pair, which is designed to be
    feasible together, so the source engine has something to rank.
    """
    steps: List[Dict[str, Any]] = []
    state: Dict[str, Any] = {}

    def _step(name: str, fn: Callable[[], tuple[bool, Any]]) -> None:
        try:
            ok, detail = fn()
        except Exception as exc:  # noqa: BLE001 — a failed step is a result, not a crash
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        steps.append({"name": name, "ok": bool(ok), "detail": detail})

    def _health() -> tuple[bool, Any]:
        body = client.get("/health").json()
        return body.get("status") == "ok", body

    def _ready() -> tuple[bool, Any]:
        body = client.get("/ready").json()
        return body.get("status") in (STATUS_READY, STATUS_DEGRADED), body.get("status")

    def _pull() -> tuple[bool, Any]:
        body = client.post("/loads/pull", json={"replace": True}).json()
        return (
            bool(body.get("available")) and body.get("fetched", 0) >= 1,
            f"fetched {body.get('fetched')}, accepted {body.get('accepted')}",
        )

    def _ingest() -> tuple[bool, Any]:
        body = client.post("/loads", json=loads).json()
        return body.get("accepted", 0) >= 1, f"ingested {body.get('accepted')} sample load(s)"

    def _rank() -> tuple[bool, Any]:
        body = client.post("/rank", json={"truck": truck, "top_n": top_n}).json()
        ranked = body.get("ranked", [])
        if ranked:
            state["load_id"] = ranked[0]["load_id"]
        return bool(ranked), f"{len(ranked)} ranked"

    def _bid() -> tuple[bool, Any]:
        if "load_id" not in state:
            return False, "no load available to bid"
        body = client.post(
            "/bids",
            json={"truck": truck, "load_id": state["load_id"], "actor_id": "smoke"},
        ).json()
        state["bid_id"] = body.get("bid_id")
        return state["bid_id"] is not None, f"bid_id {state['bid_id']}"

    def _decisions() -> tuple[bool, Any]:
        body = client.get("/decisions").json()
        records = body.get("decisions", [])
        return len(records) >= 1, f"{len(records)} record(s)"

    _step("health", _health)
    _step("ready", _ready)
    _step("pull", _pull)
    _step("ingest", _ingest)
    _step("rank", _rank)
    _step("bid_draft", _bid)
    _step("decisions", _decisions)

    return {"ok": all(s["ok"] for s in steps), "steps": steps}
