"""Render the FreightBid CLI `rank`/`plan` output to SVG demo assets.

Drives the real FastAPI app **in-process** (no uvicorn, no network port) via
Starlette's `TestClient`, feeds it the committed `sample_data/`, and renders the
exact same `rich` tables the CLI prints through the shared helpers in
`adapters/inbound/cli/render.py`. The SVGs are therefore faithful CLI output, not
a mock.

Non-destructive by default: writes to ``benchmarks/reproduced/`` (gitignored).
Pass ``--update-artifacts`` to (re)generate the committed
``benchmarks/demo_rank.svg`` / ``benchmarks/demo_plan.svg``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fastapi.testclient import TestClient
from rich.console import Console

from adapters.inbound.api.app import create_app
from adapters.inbound.api.container import build_container
from adapters.inbound.cli.render import render_plan, render_rank

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "sample_data"
COMMITTED_DIR = ROOT / "benchmarks"
REPRODUCED_DIR = ROOT / "benchmarks" / "reproduced"
SVG_WIDTH = 118


def _demo_payloads(client: TestClient) -> tuple[dict, dict]:
    loads = json.loads((SAMPLE / "loads.json").read_text(encoding="utf-8"))
    rank_req = json.loads((SAMPLE / "rank_request.json").read_text(encoding="utf-8"))
    client.delete("/loads")
    client.post("/loads", json=loads).raise_for_status()
    rank = client.post("/rank", json=rank_req)
    rank.raise_for_status()
    plan = client.post("/plan", json=rank_req)
    plan.raise_for_status()
    return rank.json(), plan.json()


def render_demo(out_dir: Path) -> list[Path]:
    """Render the rank + plan demo SVGs into ``out_dir`` and return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    container = build_container(ROOT / "config")
    client = TestClient(create_app(container))
    rank_data, plan_data = _demo_payloads(client)

    rank_console = Console(record=True, width=SVG_WIDTH)
    render_rank(rank_console, rank_data)
    rank_path = out_dir / "demo_rank.svg"
    # Fixed unique_id => deterministic, diffable SVG (no random element ids per run).
    rank_console.save_svg(
        str(rank_path),
        title="freightbid rank sample_data/truck.json",
        unique_id="freightbid-rank-demo",
    )

    plan_console = Console(record=True, width=SVG_WIDTH)
    render_plan(plan_console, plan_data)
    plan_path = out_dir / "demo_plan.svg"
    plan_console.save_svg(
        str(plan_path),
        title="freightbid plan sample_data/truck.json",
        unique_id="freightbid-plan-demo",
    )
    return [rank_path, plan_path]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render FreightBid CLI demo SVGs.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPRODUCED_DIR,
        help="output directory (default: benchmarks/reproduced/, gitignored)",
    )
    parser.add_argument(
        "--update-artifacts",
        action="store_true",
        help="write the committed benchmarks/demo_*.svg assets instead",
    )
    args = parser.parse_args(argv)
    out_dir = COMMITTED_DIR if args.update_artifacts else args.out_dir
    for path in render_demo(out_dir):
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
