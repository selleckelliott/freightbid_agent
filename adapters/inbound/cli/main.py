"""Typer CLI that talks to the FreightBid API."""
import json
from pathlib import Path

import httpx
import typer
from rich.console import Console
from rich.table import Table

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

    table = Table(title=f"Top {len(data['ranked'])} loads for truck {data['truck_id']}")
    for col in ["#", "Load", "Score", "Profit", "$/mi", "Deadhead", "Target Bid", "Pickup ETA"]:
        table.add_column(col)
    for i, row in enumerate(data["ranked"], 1):
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
    console.print(table)
    for row in data["ranked"]:
        console.print(
            f"[dim]Load {row['load_id']}: {row['rationale']}[/dim]"
        )
        console.print(
            f"[dim]  Bid: {row['bid']['rationale']}[/dim]"
        )


@app.command()
def plan(truck_file: Path, api: str = typer.Option(DEFAULT_API, "--api")):
    """Produce a single-truck plan for the next planning horizon."""
    truck = json.loads(truck_file.read_text(encoding="utf-8"))
    body = {"truck": truck}
    with _client(api) as c:
        r = c.post("/plan", json=body)
        r.raise_for_status()
        data = r.json()

    console.print(
        f"[bold]Plan #{data['plan_id']}[/bold] truck={data['truck_id']} "
        f"horizon={data['horizon_hours']}h feasible={data['feasible']}"
    )
    table = Table(title="Stops")
    for col in ["Seq", "Load", "Pickup ETA", "Delivery ETA", "Deadhead", "Load mi", "Revenue", "Cost", "Profit"]:
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
    console.print(table)
    console.print(
        f"[bold]Totals[/bold] revenue=${data['expected_revenue']:,.2f} "
        f"cost=${data['expected_cost']:,.2f} profit=${data['expected_profit']:,.2f} "
        f"deadhead={data['expected_deadhead_miles']:.0f}mi"
    )
    console.print(f"[dim]{data['rationale']}[/dim]")


if __name__ == "__main__":
    app()
