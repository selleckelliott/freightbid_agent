# FreightBid Agent — Phase 1

Deterministic **Dispatch Brain**: ranks loads and proposes a single-truck 48-hour plan with explanations.

## Architecture (Hexagonal / Ports & Adapters)

```
domain/          Pure business types (Load, TruckState, Plan, Bid, ScoreResult,
                 LoadEvaluation), policies (constraints, feasibility, weights),
                 and the ScoringStrategy interface (Strategy Pattern).
ports/           Outbound interfaces: LoadRepositoryPort, TruckRepositoryPort,
                 DistanceProviderPort, TollEstimatorPort, ClockPort.
adapters/
  inbound/api/   FastAPI app + Pydantic schemas + composition root.
  inbound/cli/   Typer CLI that calls the API (rich tables).
  outbound/memory/    In-memory repositories  (TEST adapter for the port).
  outbound/postgres/  SQLAlchemy + Postgres repositories (REAL adapter).
  outbound/distance/  Haversine distance provider.
  outbound/tolls/     Flat-rate per-state toll estimator.
application/     Use cases: EvaluateLoadsService, RecommendLoadsService,
                 PlanBuilderService, BidRecommenderService, ConfigLoader.
config/          Editable YAML for cost model, weights, constraints.
```

> "One port, two adapters": every outbound port has at least an **in-memory test
> adapter** and a **real adapter** (Postgres, Haversine, FlatRate). The
> `ScoringStrategy` is the swappable Strategy interface (heuristic today;
> ML/LP-based variants later).

## Cost Model
Fuel, tolls, time (driver + opportunity cost), and deadhead are tracked
**separately** on `LoadEvaluation` and rolled up into `Plan` totals.

## API
- `POST /loads` — ingest a batch of loads
- `GET  /loads` — list ingested loads
- `DELETE /loads` — clear ingested loads
- `POST /rank` — top-N ranked loads for a truck (with recommended bid range + rationale)
- `POST /plan` — propose a single-truck plan over the planning horizon (default 48h)
- `GET  /health` — health probe

## Quick start (Docker Compose)

```bash
docker compose up --build
# in another shell:
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/loads -H 'content-type: application/json' \
  -d @sample_data/loads.json
curl -s -X POST http://localhost:8000/rank -H 'content-type: application/json' \
  -d @sample_data/rank_request.json | jq
curl -s -X POST http://localhost:8000/plan -H 'content-type: application/json' \
  -d @sample_data/rank_request.json | jq
```

## Quick start (local)

```bash
pip install -r requirements.txt
uvicorn adapters.inbound.api.app:app --reload
# in another shell:
python -m adapters.inbound.cli.main ingest sample_data/loads.json
python -m adapters.inbound.cli.main rank sample_data/truck.json --top-n 10
python -m adapters.inbound.cli.main plan sample_data/truck.json
```

## Demo output

A quick end-to-end run against the sample data (truck `101`, 4 loads):

**`GET /health`**
```json
{"status": "ok"}
```

**`POST /loads`** (body: `sample_data/loads.json`)
```json
{"accepted": 4}
```

**`POST /rank`** (body: `sample_data/rank_request.json`) — top ranked loads:
```json
{
  "truck_id": 101,
  "ranked": [
    {
      "load_id": 1,
      "score": 612.48,
      "expected_profit": 466.90,
      "expected_revenue": 850.00,
      "rate_per_mile": 3.54,
      "deadhead_miles": 0.0,
      "driver_hours": 6.3,
      "pickup_eta": "2026-05-27T18:00:00Z",
      "delivery_eta": "2026-05-28T00:18:00Z",
      "rationale": "profit=$466.90 x 1.0 + rpm=$3.54 x 50.0 - deadhead=0mi x 0.5 - hours=6.3h x 5.0 => score=612.48",
      "bid": {
        "min_bid": 402.26,
        "target_bid": 459.72,
        "max_bid": 517.19,
        "breakeven": 383.10,
        "rationale": "Cost=$383.10, target margin=20%, target=$459.72 ($1.92/mi). Range [$402.26, $517.19] clamped to [$100, $25000] and [$1.00, $6.00]/mi."
      }
    },
    {
      "load_id": 3,
      "score": 513.13,
      "expected_profit": 370.20,
      "expected_revenue": 720.00,
      "rate_per_mile": 3.43,
      "deadhead_miles": 0.0,
      "driver_hours": 5.7,
      "pickup_eta": "2026-05-27T20:00:00Z",
      "delivery_eta": "2026-05-28T01:42:00Z",
      "bid": {
        "min_bid": 367.29,
        "target_bid": 419.76,
        "max_bid": 472.23,
        "breakeven": 349.80
      }
    }
  ]
}
```

**`POST /plan`** (body: `sample_data/rank_request.json`) — proposed 48h plan:
```json
{
  "plan_id": 1,
  "truck_id": 101,
  "horizon_hours": 48.0,
  "stops": [
    {
      "load_id": 1,
      "pickup_eta": "2026-05-27T18:00:00Z",
      "delivery_eta": "2026-05-28T00:18:00Z",
      "deadhead_miles": 0.0,
      "load_miles": 240.0,
      "revenue": 850.00,
      "cost": 383.10,
      "profit": 466.90
    }
  ],
  "expected_revenue": 850.00,
  "expected_cost": 383.10,
  "expected_profit": 466.90,
  "expected_deadhead_miles": 0.0,
  "expected_load_miles": 240.0,
  "expected_deadhead_cost": 175.20,
  "expected_load_cost": 0.0,
  "expected_toll_cost": 0.0,
  "expected_time_cost": 207.90,
  "feasible": true,
  "score": 612.48,
  "rationale": "Sequenced 1 load(s) [1] over 48h horizon. Revenue=$850.00, Cost=$383.10, Profit=$466.90, Deadhead=0mi."
}
```

## Tests

```bash
pytest -q
```

## Configuration
All weights, costs, and constraints are YAML-driven (`config/`). Override the
config directory with `FREIGHTBID_CONFIG_DIR`.

## Benchmark metrics tracked
- Total expected profit per plan
- Profit per mile, profit per driver-hour
- Deadhead miles & deadhead-to-revenue ratio
- Feasibility rate (loads scored / loads ingested)
- Plan utilization (driver hours used / horizon hours)

See `notebooks/experiments.ipynb` for ablation scaffolding.
