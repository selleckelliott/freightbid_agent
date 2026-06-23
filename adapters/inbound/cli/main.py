"""Typer CLI that talks to the FreightBid API."""
import json
from pathlib import Path

import httpx
import typer
from rich.console import Console

from adapters.inbound.cli.render import (
    render_bid_draft,
    render_bid_queue,
    render_plan,
    render_rank,
)
from application.services.decision_exporter import DecisionExporter

app = typer.Typer(help="FreightBid Dispatch Brain CLI")
console = Console()

DEFAULT_API = "http://localhost:8000"


def _client(api: str) -> httpx.Client:
    return httpx.Client(base_url=api, timeout=30.0)


@app.command()
def health(api: str = typer.Option(DEFAULT_API, "--api")):
    """Ping the API."""
    with _client(api) as c:
        r = c.get("/health")
        r.raise_for_status()
        console.print(r.json())


@app.command()
def ingest(file: Path, api: str = typer.Option(DEFAULT_API, "--api")):
    """Ingest loads from a JSON file containing {"loads": [...]}."""
    payload = json.loads(file.read_text(encoding="utf-8"))
    with _client(api) as c:
        r = c.post("/loads", json=payload)
        r.raise_for_status()
        console.print(r.json())


@app.command()
def clear(api: str = typer.Option(DEFAULT_API, "--api")):
    """Delete all ingested loads."""
    with _client(api) as c:
        r = c.delete("/loads")
        r.raise_for_status()
        console.print(r.json())


@app.command()
def pull(
    limit: int = typer.Option(None, "--limit", help="Cap the number of loads pulled."),
    replace: bool = typer.Option(False, "--replace", help="Clear existing loads before ingesting."),
    api: str = typer.Option(DEFAULT_API, "--api"),
):
    """Pull external-style loads from the configured sandbox/replay board (Phase 7.2).

    Validates each row through the real-world data contract and ingests the accepted loads. The board
    source is configured in config/load_board.yaml - there is no live Truckstop integration.
    """
    body = {"limit": limit, "replace": replace}
    with _client(api) as c:
        data = _check(c.post("/loads/pull", json=body))
    if not data.get("available", False):
        console.print(
            f"[yellow]Load board '{data.get('source')}' unavailable: {data.get('reason')}[/yellow]"
        )
        return
    console.print(
        f"[green]Pulled from {data['source']}[/green]: fetched {data['fetched']}, "
        f"accepted {data['accepted']}, rejected {data['rejected']}"
        + (" (replaced existing)" if data.get("replaced") else "")
    )
    for err in data.get("errors", []):
        console.print(f"  [red]row {err.get('row_index')}[/red] ({err.get('identifier')}): {err.get('errors')}")


@app.command()
def rank(
    truck_file: Path,
    top_n: int = typer.Option(10, "--top-n"),
    api: str = typer.Option(DEFAULT_API, "--api"),
):
    """Rank loads for a truck (truck JSON file)."""
    truck = json.loads(truck_file.read_text(encoding="utf-8"))
    body = {"truck": truck, "top_n": top_n}
    with _client(api) as c:
        r = c.post("/rank", json=body)
        r.raise_for_status()
        data = r.json()

    render_rank(console, data)


@app.command()
def plan(truck_file: Path, api: str = typer.Option(DEFAULT_API, "--api")):
    """Produce a single-truck plan for the next planning horizon."""
    truck = json.loads(truck_file.read_text(encoding="utf-8"))
    body = {"truck": truck}
    with _client(api) as c:
        r = c.post("/plan", json=body)
        r.raise_for_status()
        data = r.json()

    render_plan(console, data)


@app.command()
def export(
    out: Path,
    fmt: str = typer.Option("bundle", "--format", help="bundle | jsonl | csv"),
    status: str = typer.Option(None, "--status", help="Filter by bid status."),
    api: str = typer.Option(DEFAULT_API, "--api"),
):
    """Export decision records for offline audit (Phase 7.3).

    Fetches /decisions (recommendation snapshot + audit trail + model/config provenance) and writes
    locally - an audit bundle folder (decisions.jsonl + decisions.csv + manifest.json), or a single
    JSONL/CSV file. No server-side files are written; OUT is a local path on this machine.
    """
    params = {"status": status} if status else None
    with _client(api) as c:
        data = _check(c.get("/decisions", params=params))
    records = data.get("decisions", [])
    provenance = data.get("provenance", {})
    fmt_norm = fmt.lower()
    if fmt_norm == "jsonl":
        path = out if out.suffix else out / "decisions.jsonl"
        count = DecisionExporter.write_jsonl(records, path)
        console.print(f"[green]Wrote {count} decision(s)[/green] to {path}")
    elif fmt_norm == "csv":
        path = out if out.suffix else out / "decisions.csv"
        count = DecisionExporter.write_csv(records, path)
        console.print(f"[green]Wrote {count} decision(s)[/green] to {path}")
    elif fmt_norm == "bundle":
        report = DecisionExporter.write_bundle(records, out, provenance=provenance)
        console.print(
            f"[green]Exported {report.decision_count} decision(s)[/green] to {report.out_dir} "
            f"(jsonl + csv + manifest); status counts: {report.status_counts}"
        )
    else:
        console.print(f"[red]Unknown format '{fmt}'. Use bundle | jsonl | csv.[/red]")
        raise typer.Exit(code=1)


# -- Phase 4.4: human-in-the-loop bid approval workflow -----------------------

bids_app = typer.Typer(help="Human-in-the-loop bid approval workflow.")
app.add_typer(bids_app, name="bids")


def _check(r: httpx.Response) -> dict:
    """Return the JSON body, or print the API error detail and exit non-zero."""
    if r.is_error:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        console.print(f"[red]Error {r.status_code}: {detail}[/red]")
        raise typer.Exit(code=1)
    return r.json()


@bids_app.command("create")
def bids_create(
    truck_file: Path,
    load_id: int,
    actor: str = typer.Option(None, "--actor"),
    api: str = typer.Option(DEFAULT_API, "--api"),
):
    """Draft a bid for a (truck, load): re-runs the recommender and stores the draft."""
    truck = json.loads(truck_file.read_text(encoding="utf-8"))
    body = {"truck": truck, "load_id": load_id, "actor_id": actor}
    with _client(api) as c:
        data = _check(c.post("/bids", json=body))
    render_bid_draft(console, data)


@bids_app.command("list")
def bids_list(
    status: str = typer.Option(None, "--status"),
    api: str = typer.Option(DEFAULT_API, "--api"),
):
    """List bid drafts, optionally filtered by status."""
    params = {"status": status} if status else None
    with _client(api) as c:
        data = _check(c.get("/bids", params=params))
    render_bid_queue(console, data)


@bids_app.command("show")
def bids_show(bid_id: int, api: str = typer.Option(DEFAULT_API, "--api")):
    """Show a single bid draft with its full audit trail."""
    with _client(api) as c:
        data = _check(c.get(f"/bids/{bid_id}"))
    render_bid_draft(console, data)


@bids_app.command("edit")
def bids_edit(
    bid_id: int,
    amount: float,
    reason: str = typer.Option(None, "--reason"),
    actor: str = typer.Option(None, "--actor"),
    api: str = typer.Option(DEFAULT_API, "--api"),
):
    """Adjust a draft's bid amount (records the recommended-vs-adjusted delta)."""
    body = {"amount": amount, "reason": reason, "actor_id": actor}
    with _client(api) as c:
        data = _check(c.patch(f"/bids/{bid_id}", json=body))
    render_bid_draft(console, data)


@bids_app.command("approve")
def bids_approve(
    bid_id: int,
    note: str = typer.Option(None, "--note"),
    actor: str = typer.Option(None, "--actor"),
    api: str = typer.Option(DEFAULT_API, "--api"),
):
    """Approve a drafted/edited bid."""
    body = {"actor_id": actor, "note": note}
    with _client(api) as c:
        data = _check(c.post(f"/bids/{bid_id}/approve", json=body))
    render_bid_draft(console, data)


@bids_app.command("reject")
def bids_reject(
    bid_id: int,
    note: str = typer.Option(None, "--note"),
    actor: str = typer.Option(None, "--actor"),
    api: str = typer.Option(DEFAULT_API, "--api"),
):
    """Reject a bid draft (terminal)."""
    body = {"actor_id": actor, "note": note}
    with _client(api) as c:
        data = _check(c.post(f"/bids/{bid_id}/reject", json=body))
    render_bid_draft(console, data)


@bids_app.command("submit-mock")
def bids_submit_mock(
    bid_id: int,
    note: str = typer.Option(None, "--note"),
    actor: str = typer.Option(None, "--actor"),
    api: str = typer.Option(DEFAULT_API, "--api"),
):
    """Simulate submitting an approved bid (workflow validation only — no real bidding)."""
    body = {"actor_id": actor, "note": note}
    with _client(api) as c:
        data = _check(c.post(f"/bids/{bid_id}/submit-mock", json=body))
    render_bid_draft(console, data)


if __name__ == "__main__":
    app()
