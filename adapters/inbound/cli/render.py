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
    """Print the ranked-loads table plus per-load rationale lines.

    When a winnability model is wired (Phase 4.3b), each load also gets a dim EV line:
    the recommended EV ask + ladder when available, or an explicit cost-plus-margin
    fallback note when the model was unavailable. With no model wired the bid carries no
    EV keys and these lines are skipped — output is identical to the pre-4.3 CLI.
    """
    console.print(build_rank_table(data))
    for row in data["ranked"]:
        console.print(f"[dim]Load {row['load_id']}: {row['rationale']}[/dim]")
        bid = row["bid"]
        console.print(f"[dim]  Bid: {bid['rationale']}[/dim]")
        _render_bid_ev(console, bid)


def _render_bid_ev(console: Console, bid: dict[str, Any]) -> None:
    """Render the optional EV pick + ladder lines for one bid (no-op when absent)."""
    ev_bid = bid.get("ev_recommended_bid")
    if ev_bid is not None:
        label = bid.get("ev_recommended_label", "ev")
        win = bid.get("win_probability_at_target")
        ev = bid.get("expected_value_at_target")
        win_str = f"{win:.0%}" if win is not None else "n/a"
        ev_str = f"${ev:,.0f}" if ev is not None else "n/a"
        console.print(
            f"[dim]  EV pick: {label} ${ev_bid:,.0f} (win {win_str}, EV {ev_str})[/dim]"
        )
        ladder = bid.get("ladder")
        if ladder:
            rungs = "  ".join(
                f"{r['label']} ${r['ask_amount']:,.0f}@{r['win_probability']:.0%}"
                for r in ladder
            )
            console.print(f"[dim]  Ladder: {rungs}[/dim]")
    elif bid.get("winnability_available") is False:
        console.print(
            "[dim]  EV: winnability model unavailable — cost-plus-margin bid[/dim]"
        )


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
