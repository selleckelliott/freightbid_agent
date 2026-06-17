"""Shared `rich` rendering for the FreightBid CLI.

Both the Typer CLI (`adapters/inbound/cli/main.py`) and the demo SVG renderer
(`benchmarks/render_demo.py`) import these helpers, so the committed demo assets
are byte-for-byte the same tables the CLI prints — not a separate mock.
"""
from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.table import Table

RANK_COLUMNS = ["#", "Load", "Score", "Profit", "$/mi", "Deadhead", "Target Bid", "Pickup ETA"]
PLAN_COLUMNS = ["Seq", "Load", "Pickup ETA", "Delivery ETA", "Deadhead", "Load mi", "Revenue", "Cost", "Profit"]


def build_rank_table(data: dict[str, Any]) -> Table:
    """Build the ranked-loads table from a `/rank` response dict."""
    ranked = data["ranked"]
    table = Table(title=f"Top {len(ranked)} loads for truck {data['truck_id']}")
    for col in RANK_COLUMNS:
        table.add_column(col)
    for i, row in enumerate(ranked, 1):
        table.add_row(
            str(i),
            str(row["load_id"]),
            f"{row['score']:.2f}",
            f"${row['expected_profit']:,.0f}",
            f"${row['rate_per_mile']:.2f}",
            f"{row['deadhead_miles']:.0f}mi",
            f"${row['bid']['target_bid']:,.0f}",
            row["pickup_eta"],
        )
    return table


def render_rank(console: Console, data: dict[str, Any]) -> None:
    """Print the ranked-loads table plus per-load rationale lines."""
    console.print(build_rank_table(data))
    for row in data["ranked"]:
        console.print(f"[dim]Load {row['load_id']}: {row['rationale']}[/dim]")
        console.print(f"[dim]  Bid: {row['bid']['rationale']}[/dim]")


def build_plan_table(data: dict[str, Any]) -> Table:
    """Build the plan-stops table from a `/plan` response dict."""
    table = Table(title="Stops")
    for col in PLAN_COLUMNS:
        table.add_column(col)
    for i, s in enumerate(data["stops"], 1):
        table.add_row(
            str(i),
            str(s["load_id"]),
            s["pickup_eta"],
            s["delivery_eta"],
            f"{s['deadhead_miles']:.0f}",
            f"{s['load_miles']:.0f}",
            f"${s['revenue']:,.0f}",
            f"${s['cost']:,.0f}",
            f"${s['profit']:,.0f}",
        )
    return table


def render_plan(console: Console, data: dict[str, Any]) -> None:
    """Print the plan header, stops table, totals, and rationale."""
    console.print(
        f"[bold]Plan #{data['plan_id']}[/bold] truck={data['truck_id']} "
        f"horizon={data['horizon_hours']}h feasible={data['feasible']}"
    )
    console.print(build_plan_table(data))
    console.print(
        f"[bold]Totals[/bold] revenue=${data['expected_revenue']:,.2f} "
        f"cost=${data['expected_cost']:,.2f} profit=${data['expected_profit']:,.2f} "
        f"deadhead={data['expected_deadhead_miles']:.0f}mi"
    )
    console.print(f"[dim]{data['rationale']}[/dim]")
