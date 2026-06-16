# FreightBid Agent

An AI-powered dispatch and bidding decision system that helps hotshot operators
maximize profit and reduce deadhead through heuristic scoring, route
optimization, benchmarking — and eventually machine learning and agent-based
planning.

**Phase 1** ships a deterministic **Dispatch Brain**: ranks loads and proposes a
single-truck 48-hour plan with explanations. **Phase 2** adds OR-Tools
constraint-programming planners, a three-way benchmark
([results below](#phase-2--or-tools-route-optimization)), and an objective
tuning harness that maps the profit-vs-deadhead
[Pareto frontier](#phase-23--objective-tuning-and-the-pareto-frontier) of
dispatch policies.

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
                 PlanBuilderService, BidRecommenderService, ConfigLoader,
                 ORToolsDistancePlanner, ORToolsProfitAwarePlanner.
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

## Phase 2 — OR-Tools Route Optimization

Phase 2 reframes planning as a **prize-collecting pickup-and-delivery problem**
solved with Google OR-Tools CP routing, benchmarked against the Phase 1
heuristic over 1,000 generated scenarios. Two solver variants isolate the
effect of the objective function:

| Planner | Objective | Avg profit | Avg deadhead | Feasible rate |
|---|---|---|---|---|
| Heuristic (`PlanBuilderService`) | greedy score ranking | $396.38 | 11.3 mi | 88.1% |
| `ORToolsDistancePlanner` (v1) | minimize total miles | $240.14 (**−39.4%**) | **8.9 mi (−20.9%)** | 82.4% |
| `ORToolsProfitAwarePlanner` (v2) | maximize expected profit | **$396.79 (+0.1%)** | 12.0 mi (+6.6%) | **88.1%** |

![Planner comparison](benchmarks/compare_chart.png)

**The ablation story.** The distance objective is a classic mis-specified
proxy: it slashes deadhead by 21% but destroys 39% of profit, because the
cheapest route to drive is rarely the most valuable one to run. The
profit-aware variant fixes the objective rather than the solver — same
constraints, same search budget — and recovers full heuristic-level profit
while retaining the solver's ability to chain multi-load routes.

**Objective formulation (profit-aware).** The solver minimizes the negative of
expected plan profit, in integer cents, derived from the YAML cost model — no
hand-tuned magic weights:

- *Repositioning arcs* cost `miles × 139¢` (fuel + maintenance + driver &
  opportunity time at average speed, from `config/cost_model.yaml`).
- *Loaded arcs* cost 0 — their economics live in the skip penalty.
- *Skipping a load* costs its position-independent static profit
  (revenue − operating cost − time cost), floored at the configured
  `min_expected_profit`: profitable loads are expensive to skip, marginal
  ones are free to drop.

Minimizing (deadhead cost + skipped profit) is equivalent to maximizing
(selected profit − deadhead cost) = expected plan profit.

Every solver plan is **replayed through the same `EvaluateLoadsService` +
feasibility pipeline as the heuristic**, so reported financials come from one
source of truth, and the solver cannot game its own reward. Both planners
share the constraint encoding (pickup-and-delivery pairing, time windows,
HOS-style driver-hours dimension, planning horizon) in
`application/ortools_distance_planner.py`; the profit-aware subclass overrides
only the objective hooks, first-solution strategy, and full-truckload
sequencing. Reproduce with:

```bash
python -m benchmarks.compare_planners --time-limit 0.2 --out benchmarks/compare_results.json
python -m benchmarks.chart_comparison --results benchmarks/compare_results.json
```

## Phase 2.3 — Objective Tuning and the Pareto Frontier

"Maximize profit" is only one dispatch policy. An operator who hates empty
miles (wear, risk, schedule fragility) may happily trade a percent of profit
for far less deadhead. Phase 2.3 turns the profit-aware objective into a
**configurable policy** and maps the tradeoff empirically: a tuning harness
(`benchmarks/tune_objective.py`) sweeps 24 objective configurations over the
same 1,000 scenarios and computes the profit-vs-deadhead **Pareto frontier**
(feasible rate ≥ 85% required; non-dominated configs only).

![Profit vs Deadhead Pareto Frontier](benchmarks/pareto_frontier.png)

Only two knobs exist, both multipliers over the derived cost model — and that
is a finding, not a simplification:

- **`deadhead_cost_multiplier`** — how much above true cost to price an empty
  mile (1.0 = the real cost model).
- **`skip_profit_floor_dollars`** — solver pickiness: the static profit a load
  must clear before skipping it costs anything. Swept only **at or above**
  the business `min_expected_profit` ($50) — an objective floor below the
  replay's acceptance rule would reward the solver for proposing loads the
  feasibility pipeline then rejects.

The objective is `Σ(deadhead miles × D) + Σ(skipped margin × P)`, so scaling
`D` and `P` jointly cannot change which plan wins — only the **D/P ratio**
matters. The harness proves it (`--invariance-check`): tripling both rates
reproduces every plan bit-for-bit; tripling only `D` changes them
(−67% deadhead, −3% profit on the check slice). A naive "skip penalty
multiplier" sweep axis would have silently doubled the grid for zero
information.

**Named dispatch profiles** (`config/objective_profiles.yaml`), placed on the
measured frontier (1,000 scenarios):

| Profile | D × floor | Avg profit | Avg deadhead | Feasible |
|---|---|---|---|---|
| `max_profit` | 1.0 × $50 | **$396.79 (+0.1%)** | 12.0 mi (+6.6%) | 88.1% |
| `balanced` | 1.25 × $50 | $395.90 (−0.1%) | 10.0 mi (−11.7%) | 86.9% |
| **`deadhead_control`** ⭐ | 1.6 × $75 | $392.97 (−0.9%) | **7.4 mi (−34.3%)** | 85.4% |
| `aggressive_deadhead_control` | 2.5 × $100 | $384.91 (−2.9%) | 3.9 mi (−65.4%) | 81.9% ✗ |

*(deltas vs the heuristic baseline; ✗ = fails the ≥ 85% feasibility filter)*

**Findings.**

- **The Phase 2.2 derivation sits exactly at the frontier's max-profit end.**
  Under-pricing deadhead (0.75×) is *dominated* — more deadhead **and** less
  profit — empirical validation that the cost-model-derived 139 ¢/mi was the
  right calibration, not a lucky guess.
- **The recommended knee: `deadhead_control` cuts deadhead 34% for 0.9%
  profit** (largest deadhead reduction costing < 2% of best-config profit).
- **Deadhead aversion has a feasibility cliff.** Past ~1.6× the planner
  increasingly refuses to plan at all (feasible rate slides from 88% toward
  80%) — the floor knob mostly trades feasibility, the multiplier knob trades
  deadhead, and the filter keeps the frontier honest.
- **Runtime is not a tradeoff axis here.** The knee config produces
  bit-identical plans at 0.1 s and 1.0 s per solve (`--time-study`): the
  solver converges in under 100 ms at this instance size (≤ ~20 loads), so
  "fast vs good" is a non-decision until instances grow.

Reproduce with:

```bash
python -m benchmarks.tune_objective --time-limit 0.2 --out benchmarks/tuning_results.json
python -m benchmarks.tune_objective --invariance-check --limit 200
python -m benchmarks.tune_objective --time-study --out benchmarks/tuning_results.json
python -m benchmarks.chart_pareto --results benchmarks/tuning_results.json
```

## Phase 3.1 — Destination Desirability Model (first ML layer)

Phases 2.x optimize *today's* board. But a load that pays well today can still be
a trap if it **delivers into a dead market** — somewhere the next load is far
away or doesn't exist, forcing a long deadhead. Phase 3.1 learns that risk.

The model predicts **`expected_next_deadhead_miles`**: given a candidate
delivery (destination, arrival time, equipment), how far will the truck likely
have to deadhead to its *next* viable load? This becomes a learned
"future-opportunity" signal the planner can price in Phase 3.2
(`future_deadhead_penalty = predict_next_deadhead(...) × deadhead_cost_per_mile`).
**No planner code changes in 3.1** — the `DestinationDesirabilityService` facade
defines the contract; wiring comes later.

**Label (retrospective truth).** After a delivery, search a window
(`8h`) for the nearest viable next load — same equipment, picks up after
arrival, clears a rate bar (`≥ 1.75 $/mi`; "call for rate" loads don't count).
The label is the haversine miles to that load's origin, **censored at 300 mi**
when nothing qualifies (a genuine stranding signal, ~8% of rows).

**Leakage discipline (the hard part).** Two failure modes are designed out:
- *Decision-time features only.* Market-density features (how many loads, and
  how many equipment-matched loads, are posted within 50/100/150 mi of the
  destination *right now*) are read from the same board on which the load
  appears — never from the future board the label looks at. One feature builder
  serves both training and inference, so there's no train/serve skew.
- *Embargo + observability split.* The split is time-based (last 20% is test).
  Labels are computed against the full history (no per-pool truncation), then
  train rows whose label window reaches into the test period are **embargoed**,
  and rows whose window runs off the end of the data are dropped as
  **unobservable**. (An earlier per-pool labeling scheme manufactured fake
  "stranded" labels at each pool's edge and silently wrecked the model — there's
  a regression test for it now.)

**Model — a hurdle (two-part) regressor.** The label is right-censored and
heavy-tailed (a spike at 300 mi over a bulk of short deadheads), so a single
regressor either ignores the spike (good MAE, ~0 R²) or is dragged toward the
mean (good R², poor MAE). Instead:
- a `HistGradientBoostingClassifier` estimates `p = P(no viable next load)`;
- a `HistGradientBoostingRegressor` (absolute-error → conditional median of the
  *non-censored* bulk) estimates the deadhead when a load exists;
- they recombine into a proper expectation: `E[miles] = p·300 + (1−p)·bulk`.

Both halves use native categorical handling (`destination_zone`,
`destination_state`, `equipment_type`, `mode`) — no one-hot.

**Results** (held-out last-20%-by-time test set, 3,594 rows, 8.3% censored):

| Model | MAE | RMSE | MedAE | R² | ≤25 mi | ≤50 mi | top-3 |
|---|---|---|---|---|---|---|---|
| Global mean | 69.8 | 96.8 | 44.5 | −0.00 | 5% | 73% | 37% |
| Zone × daypart | 61.2 | 89.2 | 32.1 | 0.15 | 39% | 66% | 45% |
| **Hurdle GBM** | **49.3** | **88.7** | **13.2** | **0.16** | **64%** | **76%** | **50%** |

The model beats both baselines on **every** metric. Because the target is
censored, MAE / median / bucket-accuracy / **top-3 ranking** (its real job —
ranking destinations) are the headline metrics; R² is reported but secondary.
MedAE of 13 mi vs the baseline's 32 mi, and a 64% ≤25 mi hit rate, mean the
model usually nails the easy "you'll find a reload nearby" calls and reserves
big predictions for genuinely weak destinations.

**Why synthetic data?** Real historical board data isn't available yet, so a
seeded generator manufactures *learnable* structure: strong hubs (Dallas,
Houston, LA…) flood the board while weak markets (Boise, Albuquerque) barely
appear, and each metro has its own **hot-shot equipment mix** — so an `F`
(flatbed) load delivering into a Hot-Shot-heavy market correctly faces a high
expected deadhead (an interaction a zone-only baseline can't see, but the model
can). The schema mirrors the **real Truckstop board** (see Phase 3.0.5 below),
so a future API adapter drops in unchanged.

Reproduce (artifacts are seeded; the `.joblib` and JSONL history are
gitignored, the metadata JSON is committed):

```bash
python -m ml.training.train_destination_model --config config/ml_config.yaml
python -m ml.training.evaluate_destination_model --config config/ml_config.yaml
```

> **Phase 3.0.5 / 3.1.1 — Truckstop feature discovery (grounded).** Real board
> screenshots were captured and folded into the schema:
> `docs/truckstop_feature_discovery.md` holds the observed-field inventory. The
> board's columns confirmed the existing design (load age, "call-for-rate"
> nullable rate, derived rate-per-mile) and added real fields — hot-shot
> equipment codes (`HS`/`F`/`FSD`/`FSDV`), `weight`/`length` (+ usually-blank
> `width`/`height`), `mode` (TL/PTL/LTL), and a `Load Views` competition bucket
> (now the `open_match_within_*` "uncontested onward supply" feature). Crucially,
> the board shows deadhead only to a *user-specified* point (`O-DH` to pickup,
> `D-DH` to a typed destination) — never the open-ended next-load deadhead this
> model predicts, so the ML layer is **additive**. Broker-quality signals
> (`Days-to-Pay`, credit/bond) are observable but describe *load winnability*,
> not onward deadhead, so they're deferred to a future Phase 4 quality model.

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
