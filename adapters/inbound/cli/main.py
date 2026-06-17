"""Typer CLI that talks to the FreightBid API."""
import json
from pathlib import Path

import httpx
import typer
from rich.console import Console

from adapters.inbound.cli.render import render_plan, render_rank

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


if __name__ == "__main__":
    app()
