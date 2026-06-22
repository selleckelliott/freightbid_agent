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
        _render_bid_risk(console, bid)
    elif bid.get("winnability_available") is False:
        console.print(
            "[dim]  EV: winnability model unavailable — cost-plus-margin bid[/dim]"
        )


def _render_bid_risk(console: Console, bid: dict[str, Any]) -> None:
    """Render the optional Phase 5.1 risk-adjusted EV line (no-op when risk is off)."""
    if not bid.get("payment_risk_available"):
        return
    pd = bid.get("p_default_at_target")
    days = bid.get("expected_pay_days_at_target")
    pen = bid.get("delay_penalty_at_target")
    ra = bid.get("risk_adjusted_ev_at_target")
    pd_str = f"{pd:.0%}" if pd is not None else "n/a"
    days_str = f"~{days:.0f}d" if days is not None else "n/a"
    pen_str = f"${pen:,.0f}" if pen is not None else "$0"
    ra_str = f"${ra:,.0f}" if ra is not None else "n/a"
    console.print(
        f"[dim]  Risk-adj: default {pd_str}, pay {days_str}, delay {pen_str}, "
        f"risk-adj EV {ra_str}[/dim]"
    )
    if bid.get("risk_adjusted_ev_positive") is False:
        warn = bid.get("risk_adjusted_warning") or (
            "All candidate asks have negative risk-adjusted EV."
        )
        console.print(f"[yellow]  ! {warn}[/yellow]")


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


# -- Phase 4.4: human-in-the-loop bid approval workflow -----------------------

BID_QUEUE_COLUMNS = ["Bid", "Status", "Load", "Truck", "Recommended", "Current", "Delta", "Expires"]
BID_AUDIT_COLUMNS = ["When", "Action", "Actor", "Transition", "Amount", "Note"]


def build_bid_queue_table(data: dict[str, Any]) -> Table:
    """Build the reviewable bid-draft queue table from a `/bids` response dict."""
    bids = data["bids"]
    table = Table(title=f"Bid drafts ({len(bids)})")
    for col in BID_QUEUE_COLUMNS:
        table.add_column(col)
    for b in bids:
        table.add_row(
            str(b["bid_id"]),
            b["status"],
            str(b["load_id"]),
            str(b["truck_id"]),
            f"${b['recommended_amount']:,.0f}",
            f"${b['current_amount']:,.0f}",
            f"${b['delta_from_recommended']:,.0f} ({b['delta_percent']:+.1f}%)",
            b["expires_at"],
        )
    return table


def render_bid_queue(console: Console, data: dict[str, Any]) -> None:
    """Print the bid-draft queue table."""
    console.print(build_bid_queue_table(data))


def build_bid_audit_table(data: dict[str, Any]) -> Table:
    """Build the audit-trail timeline table for a single bid draft."""
    table = Table(title="Audit trail")
    for col in BID_AUDIT_COLUMNS:
        table.add_column(col)
    for e in data["audit"]:
        transition = f"{e['from_status'] or '-'} -> {e['to_status']}"
        amount = f"${e['amount_after']:,.0f}" if e.get("amount_after") is not None else "-"
        table.add_row(
            e["at"],
            e["action"],
            e["actor_id"],
            transition,
            amount,
            e.get("note") or "",
        )
    return table


def render_bid_draft(console: Console, data: dict[str, Any]) -> None:
    """Print a single bid draft: header, amounts/delta, EV snapshot, audit timeline.

    A ``submitted_mock`` draft prints an explicit note that the submission is *simulated*
    for workflow validation only — no real broker/Truckstop bid is ever placed.
    """
    console.print(
        f"[bold]Bid #{data['bid_id']}[/bold] [{data['status']}] "
        f"load={data['load_id']} truck={data['truck_id']}"
    )
    console.print(
        f"Recommended ${data['recommended_amount']:,.0f} "
        f"(${data['recommended_rate_per_mile']:.2f}/mi)  |  "
        f"Current ${data['current_amount']:,.0f}  |  "
        f"Delta ${data['delta_from_recommended']:,.0f} ({data['delta_percent']:+.1f}%)"
    )
    if data.get("edit_reason"):
        console.print(f"[dim]Edit reason: {data['edit_reason']}[/dim]")
    console.print(f"[dim]Expires {data['expires_at']}[/dim]")
    console.print(f"[dim]{data['rationale']}[/dim]")
    _render_bid_draft_ev(console, data)
    if data.get("submission_ref"):
        console.print(
            f"[yellow]Submission (SIMULATED): {data['submission_ref']}[/yellow]"
        )
    if data["status"] == "submitted_mock":
        console.print(
            "[yellow]Note: 'submitted_mock' is a simulated submission for workflow "
            "validation only — no real broker/Truckstop bid was placed.[/yellow]"
        )
    console.print(build_bid_audit_table(data))


def _render_bid_draft_ev(console: Console, data: dict[str, Any]) -> None:
    """Render the optional EV snapshot line for a draft (no-op when the model is off)."""
    if not data.get("winnability_available"):
        return
    win = data.get("win_probability")
    ev = data.get("expected_value")
    win_str = f"{win:.0%}" if win is not None else "n/a"
    ev_str = f"${ev:,.0f}" if ev is not None else "n/a"
    label = data.get("ev_recommended_label", "ev")
    ev_bid = data.get("ev_recommended_bid")
    bid_str = f" pick {label} ${ev_bid:,.0f}" if ev_bid is not None else ""
    console.print(f"[dim]EV: win {win_str}, EV {ev_str}{bid_str}[/dim]")
