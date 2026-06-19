# FreightBid Agent

An AI-powered dispatch and bidding decision engine for hotshot trucking: it
recommends profitable loads, cuts deadhead, and plans routes through heuristic
scoring, OR-Tools optimization, a learned destination-risk model, and a rolling
multi-day dispatch simulation — every recommendation explainable.

![FreightBid Agent rolling-replay A/B: the destination-aware policy lifts cumulative profit while cutting deadhead across 150 sequential episodes](benchmarks/rolling_replay_comparison.png)

## Results at a glance

| Layer | Approach | Headline result |
| --- | --- | --- |
| [Heuristic baseline](#phase-2--or-tools-route-optimization) | rule-based scoring | $396.38 profit · 11.3 mi deadhead · 88.1% feasible |
| [OR-Tools profit-aware](#phase-2--or-tools-route-optimization) | CP-SAT, profit objective | $396.79 profit · 12.0 mi deadhead |
| [OR-Tools deadhead-control](#phase-23--objective-tuning-and-the-pareto-frontier) | tuned objective weights | $392.97 profit · **7.4 mi deadhead (−34.3%)** |
| [ML destination model](#phase-31--destination-desirability-model-first-ml-layer) | Hurdle GBM | MAE 49.3 vs 61.2 zone baseline · ≤50 mi 76% |
| [Destination-aware (one-shot)](#phase-32--destination-aware-planner-closing-the-loop) | model in the planner | −12.9% deadhead at ~free profit |
| [**Rolling replay (sequential)**](#phase-33--rolling-replanning-simulation-measuring-the-sequential-payoff) | multi-day MPC A/B | **+3.9% profit · −4.7% deadhead** (150 episodes) |
| [**Stress test (robustness)**](#phase-34--sequential-policy-stress-testing-is-the-edge-robust) | 18 shifted markets | **0 regressions** · advantage HOLDS 7/18, neutral 11/18 |
| [Winnability dataset (Phase 4.1)](#phase-41--load-quality--winnability-dataset) | seeded outcome simulator | labeled broker-quality + bid-win dataset · 6 processes · leakage-guarded |
| [**Bid-winnability model (Phase 4.2)**](#phase-42--calibrated-bid-winnability-model) | calibrated HGB classifier | **ROC AUC 0.928 · test ECE 0.010** · beats 3 baselines |
| [**EV bid recommender (Phase 4.3)**](#phase-43--expected-value-bid-recommender) | calibrated P(win) × margin → bid ladder | **+32.8% realized profit vs best fixed** · only $39 EV-regret vs a clairvoyant oracle |
| [**Broker-quality stress (Phase 4.5)**](#phase-45--broker-quality-stress-testing-does-the-bid-edge-survive-a-worse-market) | 10 broker-quality shifts · model frozen on baseline | **EV beats best fixed 10/10** · uplift grows to +160% as markets harden · payment risk shown **orthogonal** to bid profit |
| [**Payment-risk model (Phase 5.2)**](#phase-52--calibrated-payment-risk-model) | calibrated HGB `P(default)` + `E[pay_days]` head | **test ECE 0.003** · PR-AUC 0.140 > baselines · pay-days MAE 4.0d · the risk signal 5.1 folds into EV |

Single-truck, synthetic-market simulation: the claim is **sign-stable, explainable**
dispatch gains across markets, not a magic number — see
[What I learned](#what-i-learned) and [Limitations & next work](#limitations--next-work).

## Demo

The CLI ranks loads and proposes a single-truck plan, each with a full
cost-and-bid rationale — rendered straight from the API against `sample_data/`
(regenerate with `python -m benchmarks.render_demo --update-artifacts`).

`freightbid rank sample_data/truck.json` — top loads with target bid + explanation:

![CLI rank demo](benchmarks/demo_rank.svg)

`freightbid plan sample_data/truck.json` — the proposed plan with per-stop economics:

![CLI plan demo](benchmarks/demo_plan.svg)

## Architecture (Hexagonal / Ports & Adapters)

**System** — a thin CLI/API over an application core that depends only on ports;
adapters and planners plug in behind them.

```mermaid
flowchart LR
    CLI[Typer CLI] --> API[FastAPI app]
    API --> APP[Application use-cases<br/>rank · plan · evaluate · bid]
    APP --> DOM[Domain core<br/>Load · TruckState · Plan · policies]
    APP -. ports .-> ADP[Adapters<br/>in-memory / Postgres · Haversine · tolls]
    APP --> PLN[Planners<br/>Heuristic · OR-Tools · destination-aware]
    PLN --> ML[(Destination model<br/>Hurdle GBM)]
```

**Decision flow** — how a load board becomes a measured dispatch decision.

```mermaid
flowchart LR
    GEN[Synthetic load board<br/>market-structured] --> BRD[Visible snapshot<br/>at decision time]
    BRD --> PA[Profit-aware planner]
    BRD --> DA[Destination-aware planner<br/>+ onward-risk model]
    PA --> SIM[Rolling multi-day sim<br/>execute · advance · HOS reset]
    DA --> SIM
    SIM --> MET[Paired metrics<br/>profit · deadhead · CIs]
```

Detailed module layout:

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

## Reproduce

One command regenerates the demo and a reduced rolling A/B **non-destructively**
— it writes only to the gitignored `benchmarks/reproduced/` (so `git status`
stays clean) and prints a run-metadata header plus the results table above:

```bash
python -m benchmarks.reproduce                     # fast smoke (~1-3 min), clean checkout OK
python -m benchmarks.reproduce --update-artifacts  # refresh the committed demo SVGs
python -m benchmarks.reproduce --full              # canonical long benchmark (~70 min)
```

On a fresh clone the gitignored model artifact is absent, so the fast path
quick-trains a small seeded destination model into `benchmarks/reproduced/`
(seconds) just to drive the reduced chart. `--full` regenerates the committed
canonical artifacts (150-episode replay + 18×30 stress sweep).

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

## API response shapes (reference)

The same `sample_data` run as the [demo above](#demo), shown as raw JSON to
document the `/rank` and `/plan` response contracts (truck `101`, 4 loads):

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

## Phase 3.2 — Destination-Aware Planner (closing the loop)

Phase 3.1 *predicted* onward deadhead but changed no decisions. Phase 3.2 wires
that prediction into the optimizer so it actually **influences dispatch**.

The profit-aware planner prices the deadhead needed to *reach* each load, but is
blind to the deadhead a load's **destination** will impose on the *next* load.
`ORToolsDestinationAwarePlanner` subclasses it and changes a **single objective
hook** (`_drop_penalty`): each load's skip penalty becomes its static profit
*net of its destination's expected onward-deadhead cost*, above the floor:

```
skip_penalty = max(0, static_profit − dest_cost − floor) × profit_multiplier
dest_cost    = predict_next_deadhead(...) × deadhead_$_per_mile × weight
```

Folding `dest_cost` in *before* the floor keeps the solver's serve-vs-skip
break-even exact. A load delivering into a strong market keeps its full value; a
load into a weak market loses skip-incentive, so the solver declines
otherwise-profitable freight that would strand the truck. The penalty is
**position-independent** (it depends only on the destination, arrival window and
the visible board), so it stays a per-load disjunction penalty — no
path-dependent arc cost. With `destination_service=None` the planner is
byte-for-byte the profit-aware planner: **the ML signal is a feature flag, not a
fork** (and `destination_weight` scales it).

**The domain ↔ ML boundary (kept honest).** The planner speaks the domain's
trailer vocabulary (`Dry Van`/`Reefer`/`Flatbed`); the model trained on hot-shot
board codes. Two small, deliberately coarse adapters bridge them: an equipment
map (`Flatbed→F`, `Dry Van→FSDV`, everything else→`HS`) and a `_BoardLoad`
wrapper that re-shapes a domain `Load` into the feature builder's board contract
— the same contract a real Truckstop feed would satisfy. The decision-time board
is the prefiltered candidate set (each candidate excluded from its own board).

**A/B result** (1000-scenario suite, OR-Tools 0.2 s/solve, `weight=1.0`):

| Planner | Avg profit | Avg deadhead | Avg loads | Feasible | Median solve |
| --- | --- | --- | --- | --- | --- |
| Heuristic (scoring) | $396.38 | 11.3 mi | 0.91 | 88.1% | 0.2 ms |
| OR-Tools Distance | $240.14 | 8.9 mi | 0.88 | 82.4% | 202 ms |
| OR-Tools Profit-Aware | **$396.79** | 12.0 mi | 0.91 | 88.1% | 202 ms |
| OR-Tools Destination-Aware | $393.23 | **10.5 mi** | 0.89 | 86.1% | 275 ms |

Destination-aware vs. its profit-aware parent: **−0.9% profit, −12.9% deadhead,
−2.4% loads.** (The ~80 ms extra solve time is the per-candidate ML inference.)

**Reading this honestly.** This is a *one-shot* benchmark: each scenario plans
once from a fixed truck position, so the planner can only ever *decline* a load —
it never gets to collect the better next load its choice sets up. Even so, the
trade is favorable: it gives up just **0.9% of immediate profit to cut deadhead
12.9%**, declining loads bound for weak markets that the profit-aware planner
takes blindly. That's the safety behavior we wanted, and it's nearly free here.
The *full* payoff (actually collecting the closer next load) only materializes
under **sequential replanning** — a rolling multi-day simulation is the natural
next step (Phase 3.3) to measure it end-to-end. Charging the full predicted cost
(`weight=1.0`) in a one-shot plan is intentionally the most conservative setting;
`destination_weight` tunes the profit-vs-repositioning trade.

Reproduce (destination-aware column appears only when the gitignored model
artifact exists locally):

```bash
python -m benchmarks.compare_planners --time-limit 0.2 --out benchmarks/compare_results.json
```

## Phase 3.3 — Rolling Replanning Simulation (measuring the sequential payoff)

The Phase 3.2 A/B was *one-shot*: the planner chooses once, so it can only ever
**decline** a stranding load — it never gets to collect the better next load its
choice sets up. That's why it cost 0.9% profit to save deadhead. Phase 3.3 builds
the missing piece: a **rolling-horizon (MPC-style) simulation** that lets a truck
replan over a multi-day week, so the destination signal's *downstream* payoff
shows up end-to-end.

**How it works.** Each **episode** is one synthetic world — a time-stamped stream
of board snapshots from the **Phase 3.1 generator** (the same market structure the
model trained on, so it is tested in-distribution, *not* on the uniform benchmark
generator). The loop is deliberately thin (`simulation/`):

```
observe visible board → planner picks → execute only the first load → advance
truck (position, clock, HOS) → replan at the next snapshot → … until the horizon
```

* **`SnapshotBoard`** answers "what can the truck take right now?": the latest
  snapshot at or before the clock, filtered by equipment, pickup-window expiry,
  radius and consumed loads, then adapted from ML records to domain `Load`s.
* **`TruckSimulator`** owns *no cost math*. The planners already produce a
  position-aware financial replay on `plan.stops[0]` via `EvaluateLoadsService`,
  so the simulator just **lifts those realized numbers** and advances the truck to
  the delivered destination — rolling metrics therefore reconcile exactly with the
  one-shot engine (there is a regression test for this). A simple daily
  Hours-of-Service reset keeps a single truck from running dry after day one.
* **Same world, same truck, every planner.** Only the dispatch policy differs.

**Policy divergence (shadow comparison).** To explain effect size, while the
profit-aware truck runs, the destination-aware planner rides along as a *shadow*:
at each decision it is asked what it *would* pick from the **identical** board and
truck state, **without executing**. The agreement flag yields a
`decision_overlap_rate`. Forced idles (both planners decline an HOS-depleted
board) are excluded — only genuine choice points count.

**A/B result** (150 episodes × 7-day horizon, OR-Tools 0.2 s/decision,
`weight=1.0`; 95% bootstrap CIs):

| Planner | Cumulative profit | Cumulative deadhead | Idle hrs | Loads | Profit/day | Deadhead/load |
| --- | --- | --- | --- | --- | --- | --- |
| OR-Tools Profit-Aware | $1,682.7 `[1562, 1806]` | 123.7 mi `[110, 138]` | 67.8 | 5.6 | $240.4 | 19.6 mi |
| OR-Tools Destination-Aware | **$1,749.0** `[1624, 1878]` | **118.0 mi** `[106, 130]` | 67.7 | 5.6 | **$249.9** | **18.9 mi** |

Destination-aware vs. its profit-aware parent: **+3.9% profit and −4.7% deadhead**
(−3.8% deadhead per load), idle hours and load count flat.

**The headline: rolling flips the one-shot trade-off.** What cost 0.9% profit in
the single-shot benchmark now *earns* +3.9% — because the truck actually collects
the closer next load that pricing onward-deadhead set up. Same model, same weight;
the only change is letting decisions compound. That is exactly the hypothesis this
simulation was built to test.

**Reading this honestly.** The two policies agree **92.4%** of the time (divergence
**7.6%** over 838 genuine decisions), so the effect is carried by a minority of
episodes, and a paired per-episode view (both planners ran the same world) is the
fair lens:

* **Profit:** mean Δ **+$66.3/episode** — better in **34** episodes, tied in **98**,
  worse in **18**.
* **Deadhead:** mean Δ **−5.8 mi/episode** — better in **30**, tied in **98**,
  worse in **22**.

So on the ~1/3 of worlds where the policies diverge, destination-awareness wins
clearly more often than it loses, and never hurts on the other two-thirds. Two
honest caveats: (1) a single-truck, simple-HOS model completes only ~5.6 loads/
week, which bounds how much decisions can compound; (2) the model's *predicted*
onward-deadhead barely tracks the *realized* onward miles in this loop
(correlation ≈ 0.02, MAE ≈ 17.8 mi) — the realized proxy is shaped by HOS timing
and the next snapshot, not destination strength alone — yet the **aggregate
dispatch nudge is still net-positive**. Tightening that signal (and multi-truck /
finer HOS) is the natural Phase 4+ direction. See
`benchmarks/rolling_replay_comparison.png` for the full distribution view.

Reproduce (destination-aware trajectory appears only when the gitignored model
artifact exists locally):

```bash
python -m benchmarks.run_rolling_replay --episodes 150 --out benchmarks/rolling_replay_summary.json
python -m benchmarks.chart_rolling_replay
```

## Phase 3.4 — Sequential Policy Stress Testing (is the edge robust?)

Phase 3.3 showed the destination-aware policy wins **in one synthetic world**. The
question a skeptical reviewer asks next — and the one that separates a portfolio
*toy* from a portfolio *result* — is whether that win is real or an artifact of the
baseline market. Phase 3.4 answers it by replaying the same rolling A/B under **18
shifted market conditions** and asking: does the advantage survive?

**Design.** One *condition* is a perturbation of the baseline world / economics /
HOS. The sweep is **one-factor-at-a-time** (OFAT) — move a single axis, hold the
rest at baseline — plus three **combined stress corners**:

* **market density** — 25 / 120 loads per snapshot (baseline 60)
* **unposted-rate fraction** — 0.35 / 0.50 (baseline 0.15)
* **load-view competition** — 25% / 50% of visible loads skimmed first by rival
  demand, biased toward high-view (contested) loads (`simulation/snapshot_board.py`)
* **fuel / deadhead cost** — $0.95/mi fuel, or 1.5× empty-mile burn — rebuilds the
  *whole* cost stack: the realized evaluator **and** the planners' objective
  weights, so they optimise the same economics they're scored on
* **HOS strictness** — 8 h / 14 h daily drive cap (baseline 11 h)
* **equipment mix** — pin every truck to hot-shot, or to flatbed
* **horizon length** — 3 / 14 days (baseline 7)
* **corners** — thin market (scarce + half-unpriced + half-contested), expensive
  miles under a tight HOS cap, and a worst case combining all of it over 14 days

Two choices keep it honest. (1) **The model artifact never changes** — it is the
same one trained on the baseline distribution, so every condition is an
**inference-time distribution shift, not a retrain** (testing whether a fixed
learned signal generalises). (2) Every condition shares the baseline's seed stream
(**Common Random Numbers**), so a condition's only difference from baseline is the
perturbed parameter — a variance-reduced comparison. Each condition earns a paired
verdict: **HOLDS** (paired profit CI ≥ 0 *and* deadhead no worse), **REGRESSION**
(paired profit CI entirely < 0), otherwise **NEUTRAL**.

**Result** (18 conditions × 30 episodes × 2 planners, OR-Tools 0.2 s/decision,
68.7 min; Δ = destination-aware − profit-aware, 95% paired bootstrap CIs):

| Condition | Shift from baseline | Verdict | Profit Δ% `[95% CI]` | Deadhead Δ% |
| --- | --- | --- | --- | --- |
| `baseline` | the Phase 3.3 world | **HOLDS** | **+3.9** `[+0.3, +7.8]` | −16.4 |
| `density_low` | 25 loads/snapshot | neutral | +0.5 `[−1.5, +2.9]` | +2.3 |
| `density_high` | 120 loads/snapshot | neutral | +1.3 `[−0.8, +4.1]` | −4.8 |
| `unposted_035` | 35% "call for rate" | **HOLDS** | **+8.1** `[+3.5, +13.3]` | −5.5 |
| `unposted_050` | 50% "call for rate" | **HOLDS** | **+5.4** `[+1.5, +10.5]` | −2.6 |
| `competition_025` | 25% of board skimmed | neutral | +5.1 `[−0.0, +11.3]` | −11.2 |
| `competition_050` | 50% of board skimmed | neutral | +2.3 `[−0.3, +5.4]` | −6.8 |
| `fuel_095` | fuel $0.55→$0.95/mi | **HOLDS** | **+9.6** `[+2.7, +16.9]` | −1.8 |
| `deadhead_15` | 1.5× empty-mile burn | neutral | +3.8 `[−0.2, +9.8]` | −10.8 |
| `hos_strict_8` | 8 h daily drive cap | neutral | +2.9 `[−1.6, +7.9]` | −4.4 |
| `hos_relaxed_14` | 14 h daily drive cap | neutral | +1.7 `[−2.8, +6.7]` | −9.2 |
| `equip_hs` | all trucks hot-shot | **HOLDS** | **+3.7** `[+1.2, +6.8]` | −14.0 |
| `equip_f` | all trucks flatbed | neutral | +3.4 `[−0.4, +8.4]` | −9.0 |
| `horizon_3` | 3-day horizon | **HOLDS** | **+6.9** `[+0.9, +14.0]` | −33.1 |
| `horizon_14` | 14-day horizon | **HOLDS** | **+6.7** `[+2.4, +11.7]` | −6.7 |
| `corner_thin_market` | scarce + unpriced + contested | neutral | +3.0 `[−1.8, +9.5]` | −4.3 |
| `corner_expensive_miles` | costly miles + tight HOS | neutral | +4.2 `[−5.1, +12.4]` | −15.1 |
| `corner_worst_case` | everything hostile, 14-day | neutral | −2.8 `[−14.2, +6.6]` | −17.8 |

**The headline: zero regressions across all 18 conditions.** Profit improves in
**17/18** (the lone dip is the deliberately brutal worst-case corner at −2.8%, whose
CI `[−14.2, +6.6]` straddles zero — not a significant loss), and deadhead falls in
**17/18**. The advantage is **significant (HOLDS) in 7** conditions and
**directionally positive but not individually significant (neutral) in 11**.

**Reading this honestly.** At 30 episodes/condition the per-condition CIs are wide,
so most shifts land "neutral": the point estimate favours destination-awareness but
the interval still includes zero. What's compelling is not any single cell — it's
the **consistency of sign**. The edge never reverses, and it is strongest exactly
where the economics say it should be: expensive empty miles (`fuel_095` **+9.6%**),
a sparse *priced* board (`unposted_035` **+8.1%**), and longer horizons that let
decisions compound (`horizon_14` **+6.7%**). Even the worst-case corner — scarce,
contested, expensive, HOS-throttled, over two weeks — still **cuts deadhead 17.8%**
while giving back only a statistically-insignificant slice of profit. The one place
deadhead ticks *up* (`density_low`, +2.3%) is also not significant.

The caveat from Phase 3.3 carries over and bounds the magnitudes: a single truck
with a simple daily-HOS model completes only ~5–6 loads/week, so there is limited
room for a per-load signal to compound. The robustness claim here is deliberately
modest and precise — **sign stability of the destination-aware advantage across a
wide spread of markets**, not a large effect in any one of them. That a fixed
signal trained on one distribution stays net-positive under density, competition,
cost, HOS, equipment, and horizon shifts is the evidence that the Phase 3.1→3.3
loop learned something real rather than overfitting the baseline world. See
`benchmarks/stress_test_comparison.png` for the forest plot.

Reproduce (destination-aware trajectory appears only when the gitignored model
artifact exists locally; otherwise every condition is reported `DEST_SKIPPED`):

```bash
python -m benchmarks.run_stress_test --episodes 30 --out benchmarks/stress_test_summary.json
python -m benchmarks.chart_stress_test
```

## Phase 4.1 — Load Quality & Winnability Dataset

Phases 3.1–3.4 learned one signal: *destination desirability* (will this load's
drop-off strand me?). Phase 4 turns to the **other side of the load** — the broker
and the bid: *will I get paid, and will my bid even win?* Before training that model
(Phase 4.2), this phase defines the **synthetic outcome world** that produces those
labels, and emits a seeded, reproducible **labeled dataset**.

The broker pool (`ml/brokers.py`) mirrors `ml/markets.py`: each broker has **hidden
latent** quality (`true_pay_days`, `true_default_prob`, `rate_bias`) paired with the
**noisy, sometimes-missing observable** columns a dispatcher actually sees on the
board (`credit_bucket` A/B/C/**unknown**, `days_to_pay`, `bonded`,
`quick_pay_available`, broker age). A configurable slice of brokers is `unknown`
(paywalled) — the **missingness is itself a signal**, never imputed.

`ml/data/outcome_simulator.py` realizes six processes — each a hidden latent → a
decision-time signal on the snapshot → an emitted label:

| "Outcome world" goal | Hidden latent (ground truth) | Decision-time signal | Emitted label |
| --- | --- | --- | --- |
| brokers pay quickly | `true_pay_days`, quick-pay pref | `broker_days_to_pay`, `quick_pay_available` | `realized_pay_days` |
| brokers are risky | `true_default_prob` | `broker_credit_bucket`, `bonded`, broker age | `payment_outcome` (paid/late/default) |
| loads highly contested | `contention_intensity` | `load_views` (be-the-first…high) | (drives win + coverage) |
| **which bid prices win** | `reservation_rpm` per load | *none* (you see only the ask) | `won` over a neutral bid grid |
| loads disappear quickly | coverage hazard `λ(contention)` | `load_views`, load age, rpm | `time_to_cover_hours` (censored), `covered` |
| no-rate loads need negotiation | broker target behind `total_rate=None` | `has_posted_rate=False`, `mode` | `negotiation_required`, `negotiated_rate` |

A carrier's ask **wins when it is at or below** the broker's hidden reserve, softened
by a logistic — so **win probability falls as the ask rises**. That is the
economically correct direction and the one that gives the Phase 4.3 EV bid optimizer
a real *more-margin-vs-lower-win-rate* tradeoff.

**Leakage discipline** (the part reviewers check): the latents
(`reservation_rpm`, `contention_intensity`, `true_pay_days`, `true_default_prob`,
`rate_bias`) live **only** in `BrokerProfile`, the simulator, and the outcomes
artifact. They are never an attribute of `LoadSnapshotRecord`, never a key in the
snapshot JSONL, and never a feature — exactly like `ml/data/labeling.py`, labels may
encode the latent world but decision-time code cannot. A dedicated leakage-guard test
asserts this. Broker/quality randomness is also drawn from a **separate per-load
stream**, so the Phase 3.1 destination dataset is byte-identical and the existing
model is untouched.

The build emits three gitignored, byte-reproducible JSONL artifacts under `data/`:
extended **snapshots** (broker columns attached), **outcomes** (realized labels +
hidden ground truth), and **bid trials** (`(bid_rpm, won)` rows ready for 4.2). Base
rates (e.g. default frequency) are intentionally tuned for *learnable signal*, not
calibrated to real-world magnitudes.

```bash
python -m ml.data.build_winnability_dataset            # full seeded build
python -m ml.data.build_winnability_dataset --days 5   # quick smoke build
```

No model is trained here — that is Phase 4.2 (see the
[roadmap](https://github.com/selleckelliott/freightbid_agent/issues)).

## Phase 4.2 — Calibrated Bid-Winnability Model

Phase 4.2 trains a calibrated bid-winnability model estimating
**P(win | load, broker, market, ask)** on the Phase 4.1 dataset. Because the
downstream bid optimizer (Phase 4.3) multiplies this probability by margin to pick a
bid, **probability quality matters more than ranking** — a predicted 70% must win
about 70% of the time. So the evaluation leads with **calibration** metrics (Brier
score, log loss, Expected Calibration Error, reliability curve) alongside the usual
ROC/PR AUC. Scope is deliberately narrow: this phase stops at a loadable,
calibrated `predict_proba` artifact plus its evaluation — **no bid recommendation or
EV optimization** (that is 4.3).

**Three-way grouped time split (test touched once).** Trials are split by
`snapshot_time` into contiguous **train 70% / validation 10% / test 20%** slices.
All six ask-level trials of a load share one snapshot time, so a load's trials never
straddle a boundary (asserted by a test) — no same-load leakage. The validation
slice exists for one reason: to make the calibration decision on held-out data.
*Calibration is selected using the validation split only; the test split is held out
for final reporting.*

**Features are decision-time observables only** (29 of them): the ask
(`bid_rpm`, `ask_to_market_ratio`, `ask_to_posted_ratio` — `NaN` + a
`has_posted_rate` flag when the load posts no rate), load attributes, the **noisy
broker board columns** (credit bucket incl. `unknown`, days-to-pay, bonded,
quick-pay, age), market/time encodings, competition (`load_views`) and load age. The
hidden latents (`reservation_rpm`, `contention_intensity`, `true_*`, `rate_bias`)
never enter — a leakage-guard test asserts no latent name reaches the feature matrix,
and `broker_id` is excluded so the model cannot memorize latent quality by identity.

**Three baselines, then the model** — each beaten on every probability metric on the
untouched test set:

| Model | ROC AUC | PR AUC | Brier ↓ | Log loss ↓ | ECE ↓ |
| --- | --- | --- | --- | --- | --- |
| Global win rate | 0.500 | 0.292 | 0.2067 | 0.6039 | 0.0035 |
| Ask-vs-rate heuristic | 0.673 | 0.427 | 0.1888 | 0.5614 | 0.0088 |
| Broker × market × ask bin | 0.706 | 0.491 | 0.1833 | 0.5478 | 0.0086 |
| **HistGradientBoosting** | **0.928** | **0.843** | **0.0977** | **0.3066** | **0.0102** |

(`HistGradientBoostingClassifier`, native categoricals + NaN handling, early
stopping; fit on the train slice only. Base win rate ≈ 0.29 across all three slices.)

**The calibration decision — and the honest outcome.** The rule: if **validation**
ECE ≤ 0.03, serve the uncalibrated model; otherwise fit isotonic *and* sigmoid
calibrators on the validation slice and serve whichever has the lower validation ECE.
Here the gradient-boosted model — trained on log loss — came out **already
well-calibrated** (validation ECE **0.015**), so the rule correctly **declines to
calibrate** and serves the raw model. Confirmed once on the held-out test set: ECE
**0.010**, reliability bins hugging the diagonal across the full [0, 1] range. The
calibration machinery (a version-robust `calibrate_prefit` helper using
`FrozenEstimator` on scikit-learn ≥ 1.6) is built and tested; reporting "calibration
was unnecessary" is the disciplined result, not a missing step.

![Bid-winnability reliability diagram: predicted vs observed win rate across ten probability bins, hugging the diagonal](ml/artifacts/winnability_reliability.png)

```bash
python -m ml.training.train_winnability_model   # seeded; writes model + metadata + reliability PNG
```

The seeded model `.joblib` is gitignored (regenerable); the metrics
metadata JSON and the reliability diagram are committed. Next, **Phase 4.3** turns
this calibrated probability into an expected-value bid recommendation.

## Phase 4.3 — Expected-Value Bid Recommender

Phase 4.3 consumes the calibrated Phase 4.2 winnability model and turns it into a
**human-reviewable bid ladder**. For any ask, the economics are simple:

```
profit_if_won = ask_amount − estimated_total_cost
EV(ask)       = P(win | ask) × profit_if_won
```

A higher ask lifts `profit_if_won` but sinks `P(win)`, so **expected value peaks in
the interior** — there is a best bid, and bidding past it *loses* money in
expectation. Rather than emit one opaque number, the recommender returns a small
**ladder** so a dispatcher sees the margin-vs-win-probability tradeoff and a written
rationale:

- **conservative** — highest-EV ask that still wins comfortably (`P(win) ≥ 0.70`).
- **target** *(recommended)* — highest-EV ask within 5% of the max EV **and**
  `P(win) ≥ 0.40`; a stable near-optimal bid rather than the knife-edge peak.
- **max-EV** — the raw `argmax EV` ask.
- **stretch** — the most aggressive ask still worth a shot (`P(win) ≥ 0.20`).

A rung is gracefully omitted when nothing qualifies. The default is **target, not
raw max-EV**: the EV curve is flat near its peak, so trading a sliver of expected
value for a meaningfully higher win probability is the better real-world bid.

**Example ladder** (held-out load `L-019241`, 599 loaded miles, market $2.15/mi,
breakeven $1.39/mi):

| Rung | Ask ($/mi) | Ask ($) | P(win) | Profit if won | Expected value |
| --- | --- | --- | --- | --- | --- |
| conservative | 1.83 | $1,094 | 0.87 | $262 | $229 |
| **target ★ / max-EV** | **2.04** | **$1,223** | **0.64** | **$391** | **$252** |
| stretch | 2.15 | $1,288 | 0.37 | $455 | $167 |

Bidding the stretch ask wins $64 more *if* it lands, but its win probability is so
much lower that its expected value ($167) sits below target's ($252). (Here target
and max-EV coincide; they diverge when the EV-maximizing ask wins too rarely to clear
target's stability floor.)

**Swappable model behind a port — and a zero-regression fallback.** The model sits
behind a `WinnabilityPort` (a `ModelWinnabilityAdapter` over the 4.2 artifact, plus a
`NoopWinnabilityAdapter`), mirroring the repo's hexagonal idiom. When **no model is
wired** the port returns `None` and the recommender degrades to today's
cost-plus-margin target (`winnability_available=False`) — a regression test pins this
equivalence, so adding the EV layer changes nothing until a model is present. The
adapter builds features with the **exact** Phase 4.2 `BidQuery` + feature builder, so
there is **no train/serve skew** by construction.

**Candidates are anchored to the market and clamped to the trained support.** The 4.2
model only ever saw asks in `[0.85, 1.25] × market_rate` (the trial grid), so the
recommender generates candidates market-relative, keeps the posted rate and a
breakeven-plus-margin anchor, and **flags any ask outside that envelope as
extrapolated and excludes it from the ladder** — the EV curve is only trusted where
the model is. Guardrails drop candidates below a minimum profit floor.

**Does the EV ladder actually pick better bids?** An **oracle-grounded** offline
benchmark scores 3,792 held-out loads. Each load carries a hidden `reservation_rpm`
(the broker's minimum acceptable rate) from the Phase 4.1 outcome world; the pure
simulator `win_prob(reserve, ask)` gives the *true* acceptance probability, so any ask
scores an honest **oracle-weighted realized profit** `win_prob × profit_if_won` (valid
even off the 6-point training grid). The oracle is **evaluation-only** — a unit test
asserts `reservation_rpm` never reaches the recommender, model, or features.

| Policy | Avg ask ($/mi) | P(win) model / oracle | Realized profit | EV-regret vs oracle ↓ |
| --- | --- | --- | --- | --- |
| Conservative fixed (market ×0.95) | 2.23 | 0.47 / 0.47 | $236 | $116 |
| Posted-rate | 2.39 | 0.32 / 0.35 | $139 | $213 |
| Stretch fixed (market ×1.10) | 2.58 | 0.11 / 0.05 | $38 | $315 |
| **EV recommender (max-EV)** | 2.05 | 0.81 / 0.79 | $314 | $39 |
| **EV recommender (target)** | 2.05 | 0.81 / 0.79 | **$314** | **$39** |

The recommender realizes **$314/load vs $236 for the best fixed policy (+32.8%)**, and
leaves only **$39 of EV-regret** against a clairvoyant oracle that averages $352 — i.e.
it captures ~89% of the achievable expected value with no peek at the hidden reserve.
Its **selected bids stay calibrated**: in the populated probability bands the model's
predicted win rate tracks the oracle's (e.g. 0.75 vs 0.72, 0.85 vs 0.82, 0.94 vs 0.92),
which is what makes the EV arithmetic trustworthy. Naive fixed policies fail in opposite
ways — conservative leaves margin on the table, stretch overbids until win probability
collapses.

![EV bid recommender: realized profit and EV-regret by policy, selected-bid win probability vs oracle, and an example load's ask-vs-P(win)/EV curves with the ladder rungs marked](benchmarks/bid_recommender_comparison.png)

```bash
python -m benchmarks.run_bid_recommender_eval   # oracle-grounded eval -> summary JSON
python -m benchmarks.chart_bid_recommender       # 4-panel comparison PNG
```

**Scope (4.3 engine; 4.4 deferred).** Phase 4.3 shipped the recommender engine, port +
adapters, and the offline benchmark. **Phase 4.3b** (below) surfaces it through the live
API/CLI behind a feature flag. Human-in-the-loop bid approval remains deferred to 4.4. No
auto-bidding, no live Truckstop, no retraining.

### Phase 4.3b — surfacing the recommender (live API/CLI)

Phase 4.3b threads the EV recommender through the live `/rank` + CLI seam **additively**
and **behind a feature flag (default off)** — so the committed demo and every existing
client are byte-for-byte unchanged out of the box:

- **Additive only.** When enabled, the EV ladder + `P(win)`/EV are surfaced as new
  *optional* fields next to the cost-plus-margin bid; the headline `min`/`target`/`max`
  bid is computed exactly as before and **never moves**. Reviewers see the naive margin
  baseline and the EV-optimal ladder side by side.
- **Enable it** by pointing the config at the gitignored 4.2 artifact and flipping the
  flag:

  ```yaml
  # config/bid_recommender.yaml
  model:
    enabled: true                      # default: false
    artifact_path: ml/artifacts/winnability_model.joblib
  ```

- **Graceful no-op fallback.** Flag on but artifact missing ⇒ the app logs a warning,
  serves the margin bid, and reports `winnability_available=false` (the CLI prints an
  explicit *"winnability model unavailable — cost-plus-margin bid"* note). No NaN ever
  reaches the JSON wire.
- **Live-vs-benchmark coarseness (documented limit).** The live `Load` carries no broker
  board or competition columns, so those `BidQuery` features fall back to
  `unknown`/`NaN` (the HGB model handles them natively). Live EV is therefore coarser
  than the full-snapshot offline benchmark; plumbing broker/competition through the live
  board is future work.

## Phase 4.4 — Human-in-the-Loop Bid Approval Workflow

Phases 4.3/4.3b *recommend* a bid. Phase 4.4 puts a **human in the loop** before anything
is "submitted": a recommended bid becomes a reviewable **draft** that a dispatcher drives
through an explicit, audited lifecycle. The state machine, guards, and audit trail live in
the **domain** (pure, clock-injected); a thin service + API + CLI expose it. Deliberately
**narrow** — no auto-bidding, no live Truckstop, no negotiation agent.

```
 create   ┌─────────┐   edit    ┌─────────┐
 ────────▶│ drafted │──────────▶│ edited  │◀──── edit (re-edit) ────┐
          └─────────┘           └─────────┘                         │
             │   │                  │   │                           │
         approve │ reject       approve │ reject                    │
             ▼   ▼                  ▼   ▼                           │
          ┌──────────┐          ┌──────────┐                        │
          │ approved │──────────│ rejected │ (terminal)             │
          └──────────┘  edit ──▶ (back to `edited`: editing an      │
             │   │                approved bid invalidates approval)┘
    submit-mock │ reject
             ▼   ▼
    ┌────────────────┐
    │ submitted_mock │ (terminal, SIMULATED)
    └────────────────┘

 expire: any non-terminal (drafted/edited/approved) ──▶ expired (terminal),
         enforced lazily from the injected clock (no scheduler).
```

- **Domain state machine.** `BidDraft` owns `approve / reject / edit / submit_mock /
  expire`; every illegal transition (or any action on a terminal draft) raises
  `InvalidBidTransition` → HTTP **409**. Rules are unit-tested in isolation.
- **Full audit trail.** Each transition appends an immutable `BidAuditEvent`
  (`at, action, actor_id, from→to, amount_before/after, note`). There's no auth in this
  project, so every action takes an optional **`actor_id`** (default from config; system
  expiry uses `actor_id="system"`).
- **Recommended-vs-adjusted delta.** `recommended_amount` is immutable; an edit moves
  `current_amount` and the draft re-derives `delta_from_recommended` + `delta_percent` and
  stores the `edit_reason` — the preference-learning signal is captured (not yet learned).
- **Lazy, clock-injected expiry.** TTL is config-driven (`approval.draft_ttl_minutes`); the
  service refreshes expiry from the `ClockPort` on every read/action — no background worker.
- **`submitted_mock` is SIMULATED.** It stamps a `MOCK-…` reference for workflow validation
  only — it is **never** a real broker/Truckstop submission. The CLI says so explicitly.
- **Stored in-memory** for the process lifetime (a `BidApprovalRepositoryPort` mirrors the
  load/truck repo idiom); a durable Postgres adapter is deferred.

**API** (`POST /bids` re-runs the recommender for an explicit truck+load — `/rank` is
untouched):

```
POST  /bids                 {truck, load_id, actor_id?}   → draft (status=drafted)
GET   /bids?status=         filter the review queue
GET   /bids/{id}            single draft + audit trail
PATCH /bids/{id}            {amount, reason, actor_id?}    → edited (+ delta)
POST  /bids/{id}/approve    {actor_id?, note?}
POST  /bids/{id}/reject     {actor_id?, note?}
POST  /bids/{id}/submit-mock {actor_id?, note?}            → submitted_mock (SIMULATED)
```

**CLI** (`bids` sub-group):

```bash
freightbid bids create truck.json 42 --actor dispatcher
freightbid bids list --status drafted
freightbid bids edit 1 1875 --reason "hot lane" --actor dispatcher
freightbid bids approve 1 --actor ops-lead
freightbid bids submit-mock 1            # simulated only — prints an explicit note
freightbid bids show 1                   # full draft + audit timeline
```

**Autonomy framing.** 4.4 is the *human-approval* rung of the autonomy ladder
(recommend → **approve** → mock-submit); auto-submission and a negotiation agent are
explicitly later rungs.

**Out of scope (deferred):** automatic / live broker submission, negotiation & multi-agent
autonomy, a guardrail auto-flag-for-review hook (the draft already snapshots the EV fields
so it's a clean add), `WON`/`LOST` real outcomes, and Postgres persistence.

## Phase 4.5 — Broker-Quality Stress Testing (does the bid edge survive a worse market?)

Phase 3.4 proved the *destination-aware dispatch* advantage holds across shifted markets.
Phase 4.5 asks the analogous question for the **bid layer**: when the broker market degrades
— slower pay, more unknown credit, riskier brokers, more no-rate loads, more contention, loads
disappearing faster — does the [EV bid recommender](#phase-43--expected-value-bid-recommender)
**still beat fixed-bid policies**, and does the
[calibrated winnability model](#phase-42--calibrated-bid-winnability-model) **stay trustworthy**?

The method mirrors 3.4 exactly: the calibrated model is **trained once on the baseline world**,
then **held fixed** while it is evaluated on each of 10 broker-quality-shifted worlds defined in
`config/broker_quality_stress.yaml` — distribution shift *at inference*, never retraining (also
forced by reality: the model artifact is gitignored, so the harness always trains in-process in
seconds). Every world reuses the baseline seeds (**Common Random Numbers**), so a condition's
only difference is the one knob it perturbs. Each world is scored with the same oracle-grounded
EV-vs-fixed evaluation as 4.3 — realized profit `= P(win | reserve, ask) × (ask − cost)`.

**Two honest lenses.** A single headline would hide the most important finding, so every world
is reported two ways:

- **EV lens (the headline):** the EV `target` policy's oracle-realized profit vs the *best*
  fixed policy — tagged **HOLDS** (≥ +1%), **NEUTRAL**, or **REGRESSION** (≤ −1%).
- **Calibration lens (model trust):** how far the baseline-trained model's predicted P(win)
  drifts from the world's *true* P(win) on the bids it actually selects, relative to baseline.

| World (shift) | lens | EV target | best fixed | EV uplift | calibration drift |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline | reference | $307 | $248 | **+24.1%** | 0.000 |
| no_rate_heavy (3× call-for-rate) | EV | $308 | $247 | +24.9% | +0.005 |
| sharp_win_curve (steeper accept/reject) | EV | $316 | $251 | +25.9% | −0.029 |
| high_contention (reserve ↓ on hot loads) | EV | $92 | $44 | **+108.8%** | +0.532 |
| tight_brokers (stingier reserve) | EV | $108 | $43 | **+152.0%** | +0.496 |
| slow_pay | calibration | $308 | $248 | +24.5% | +0.002 |
| unknown_credit | calibration | $309 | $248 | +24.8% | +0.010 |
| risky_brokers | calibration | $306 | $248 | +23.7% | +0.003 |
| disappearing_loads | calibration | $307 | $248 | +24.1% | 0.000 |
| degraded_corner (all hostile at once) | both | $137 | $52 | **+161.9%** | +0.446 |

**EV beats best fixed in 10/10 worlds — and the edge is *largest* exactly where the market is
hardest.** When brokers turn stingy or loads get contested (`high_contention`, `tight_brokers`,
`degraded_corner`), everyone's absolute profit collapses, but the fixed policies bid blindly into
a lower reserve and mostly *lose*, while the EV recommender bids **down** to stay in the win
region — so its *relative* advantage explodes to +100–160%.

**But the same shifts that help EV-vs-fixed also wreck model calibration.** Those high-uplift
worlds carry a calibration drift up to **+0.53**: the baseline-trained model turns badly
*over-optimistic* (predicting ~0.55–0.59 P(win) where the true rate is far lower) because the
hidden reserve moved out from under it. The EV policy still wins *relatively*, but it is steering
on a miscalibrated map — a concrete argument for **periodic recalibration / online updating** of
the winnability model under reserve or win-curve shift.

**Payment quality is orthogonal to realized bid profit — by construction, and the sweep proves
it.** Realized profit is `P(win) × (ask − cost)`; it has no payment term, and the oracle reserve
is payment-independent. So slower pay, unknown credit, riskier brokers, and disappearing loads
move *neither* lens meaningfully (uplift stays ~+24%, drift ≈ 0). That is the honest boundary of
today's recommender: it would happily hand a high-EV bid to a broker that **won't pay**. Folding
expected payment into the objective — **risk-adjusted EV** — is the clear next step this phase
motivates, not a flaw it hides.

```bash
python -m benchmarks.run_broker_quality_stress --days 21 --max-loads 500
python -m benchmarks.chart_broker_quality_stress
# quick smoke (≈6 days, 2 worlds): python -m benchmarks.run_broker_quality_stress --fast
```

![Broker-quality stress test: per-world EV uplift over the best fixed policy (EV beats fixed in all ten shifted broker markets, widening as win-economics degrade) alongside calibration drift of the frozen baseline-trained model (concentrated on the reserve/win-curve worlds, near-zero on payment-quality worlds)](benchmarks/broker_quality_stress_comparison.png)

**Out of scope (deliberately narrow):** no new model and no retraining per world (training once
on baseline *is* the design), no live Truckstop, no auto-bidding, no negotiation agent, no new
approval-workflow states, no Postgres, and no risk-adjusted-EV *implementation* yet — only named
as the motivated next step. Approval-delta / simulated-human-edit sensitivity is deferred.

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

## Phase 5.2 — Calibrated Payment-Risk Model

Phase 5.2 is the first brick of **Phase 5 — Risk-Aware Bidding & Recalibration**. It
trains a calibrated **payment-risk** model estimating **P(default | load, broker,
market)** — the chance a broker never pays — plus a secondary **E[pay_days]** head for
how slowly the good payers pay. Phase 4.5 proved payment risk is *orthogonal* to bid
profit today: the recommender would bid into a defaulting broker because realized EV
carries no payment term. This phase builds the missing signal; **folding it into the
objective is Phase 5.1** (risk-adjusted EV). Scope stops at a loadable, calibrated
`predict_proba` artifact and its evaluation — no change to the EV recommender yet.

Default is the **catastrophic, total-loss** outcome and the **minority class** (≈11% of
loads), so — exactly as in [Phase 4.2](#phase-42--calibrated-bid-winnability-model) —
**probability quality matters more than ranking**: 5.1 multiplies margin by
`p_collect = 1 − P(default)`, so a predicted 5% has to mean a ~5% loss rate. Evaluation
leads with **calibration** (Brier, log loss, ECE, reliability curve) over AUC.

**Payment is broker-driven, so the features are ask-free.** Whether a check clears has
nothing to do with the rate a carrier offers, so the feature builder is the Phase 4.2
observable set with **every ask column amputated** (no `bid_rpm`, no ask ratios) — the
broker board columns (credit bucket incl. `unknown`, days-to-pay, bonded, quick-pay,
age), load attributes, market/time encodings, and competition. The hidden latents
(`true_default_prob`, `true_pay_days`, `rate_bias`, …) never enter, and `broker_id` is
excluded so the model can't memorize a broker's latent quality by identity — a
leakage-guard test asserts both. Same three-way **train 70% / validation 10% / test
20%** time split on `snapshot_time`; one load = one outcome, so nothing straddles a
boundary; the test slice is scored once.

**Three baselines, then the model** (untouched test set, base default rate ≈ 0.106):

| Model | ROC AUC | PR AUC ↑ | Brier ↓ | Log loss ↓ | ECE ↓ |
| --- | --- | --- | --- | --- | --- |
| Global default rate | 0.500 | 0.106 | 0.0948 | 0.3381 | 0.0017 |
| Bonded × quick-pay | 0.513 | 0.109 | 0.0948 | 0.3383 | 0.0052 |
| Broker credit bucket | 0.599 | 0.132 | 0.0936 | 0.3313 | 0.0045 |
| **HistGradientBoosting** | 0.594 | **0.140** | **0.0938** | 0.3323 | **0.0034** |

**The honest finding: payment default is mostly *one column*.** The broker **credit
bucket** alone carries almost all of the rankable signal — its baseline (ROC 0.599)
**ties the gradient booster on ranking** (0.594, within noise). The booster earns its
place on the axes that matter here, not ROC: a better **PR-AUC** on the rare positive
(0.140 vs 0.132), the best **log loss / Brier**, and the best **calibration** — and it
folds in days-to-pay, bonded/quick-pay and load context the single-column baseline
can't. For a probability that gets *multiplied into an objective*, "slightly sharper
ranking" is worth less than "trustworthy magnitude," and that is what it buys.

**The calibration decision — and the honest outcome.** Same rule as 4.2: if
**validation** ECE ≤ 0.03 serve the raw model, else fit isotonic *and* sigmoid on the
validation slice and serve the lower-ECE one. Trained on log loss, the booster came out
**already well-calibrated** (validation ECE **0.005**), so the rule correctly **declines
to calibrate**; confirmed once on test — ECE **0.003**, reliability bins on the diagonal
wherever the data is dense (the top probability bins hold only a handful of loads and so
wobble — an honest small-sample artifact, not miscalibration). The version-robust
`calibrate_prefit` machinery is built and tested; "calibration was unnecessary" is the
disciplined result, not a missing step.

A second **E[pay_days]** head (a `HistGradientBoostingRegressor` on non-default rows)
predicts realized days-to-pay at **MAE 4.0 / RMSE 5.1 days** — a lightweight slow-pay
discount input for 5.1, carried as an optional field on the same artifact.

![Payment-risk reliability diagram: predicted vs observed default rate across probability bins, on the diagonal where the data is dense](ml/artifacts/payment_risk_reliability.png)

The model is served behind a `PaymentRiskPort` with **model** and **no-op** adapters: no
artifact ⇒ `estimate()` returns `None` ⇒ callers keep their risk-blind behavior, so
nothing changes until 5.1 opts in (the same "model is optional" contract as 4.3b).

```bash
python -m ml.training.train_payment_risk_model   # seeded; writes model + metadata + reliability PNG
```

The seeded `.joblib` is gitignored (regenerable); the metrics metadata JSON and the
reliability diagram are committed. Next, **Phase 5.1** folds `p_collect` and
`E[pay_days]` into a **risk-adjusted EV** objective so the recommender stops treating a
slow / defaulting broker like a reliable one.

## What I learned
- **Aggregate nudging beats per-load prediction.** Inside the rolling loop the
  destination model's per-decision predicted-vs-realized onward-deadhead
  correlation is weak (~0.02), yet steering the *distribution* of accepted loads
  toward stronger markets still produced **+3.9% profit / −4.7% deadhead**. The
  value is shifting many decisions slightly, not nailing any single one.
- **Sequence changes the verdict.** The same signal looks like a small *penalty*
  in a one-shot benchmark (−0.9% profit) but flips to a *gain* once decisions
  compound over a multi-day horizon — a dispatch policy has to be evaluated the
  way it is actually used.
- **Discipline is what makes the number trustworthy.** Decision-time-only
  features, a split embargo, Common Random Numbers across A/B conditions, and a
  one-shot↔rolling reconciliation test are what separate a real 3.9% from a leak.
- **Honest caveats build credibility.** Documenting the weak in-loop correlation,
  the censored labels, and the single-truck bound makes the result more
  believable, not less.
- **Relative robustness ≠ calibration.** Stress-testing the bid layer across 10
  broker-quality shifts (Phase 4.5), the EV recommender beat fixed bidding in **all
  10** — by the widest margin (up to +160%) exactly where the market hardened — yet
  the *same* shifts left the frozen win-model badly over-optimistic (calibration drift
  up to +0.53). Winning the comparison and trusting the probabilities are two
  separate claims, and a stress test has to score both.
- **The signal can live in one column — and calibration still matters.** Payment
  default turned out to be mostly the broker credit bucket: a one-column baseline ties
  the gradient booster on *ranking* (Phase 5.2). The model still earns its keep because
  the probability gets *multiplied into* the bid objective, where a trustworthy 5% beats
  a slightly sharper ordering — calibration (test ECE 0.003), not AUC, is the deliverable.

## Limitations & next work
**Limitations**
- **Single truck, simple HOS.** One unit with a basic daily drive-hour reset caps
  how much the destination edge can compound; a fleet would amplify (or stress) it.
- **Synthetic market.** The world is generated from market profiles informed by
  real Truckstop board screenshots, but it is not real booking data — magnitudes
  are indicative, not forecasts.
- **Weak in-loop signal.** Predicted onward-deadhead correlates only weakly with
  realized onward-deadhead inside the loop; the gain is aggregate, not per-decision.
- **Censored labels.** Onward-deadhead is capped (300 mi) and ~8% censored, so the
  regressor learns a truncated target.
- **Payment risk is not yet in the bid objective.** Realized bid profit is
  `P(win) × (ask − cost)` with no payment term, so the Phase 4.5 sweep confirmed slower
  pay / unknown credit / risky brokers moved *neither* the EV verdict nor calibration —
  the recommender would happily bid into a broker that won't pay. Phase 5.2 now supplies
  the missing signal — a [calibrated `P(default)` + `E[pay_days]` model](#phase-52--calibrated-payment-risk-model)
  — but **folding it into a risk-adjusted EV objective is Phase 5.1**, so today's
  recommender is still payment-blind.

**Next work**
- **Phase 4 — broker / load quality & winnability.** The
  [winnability dataset](#phase-41--load-quality--winnability-dataset) (4.1), the
  [calibrated bid-winnability model](#phase-42--calibrated-bid-winnability-model)
  (4.2), and the
  [expected-value bid recommender](#phase-43--expected-value-bid-recommender) (4.3)
  are built, the recommender is **surfaced** through the live API/CLI behind a feature
  flag (4.3b), and a [human-in-the-loop bid-approval workflow](#phase-44--human-in-the-loop-bid-approval-workflow)
  (4.4) gates every bid through an audited approve/edit/reject lifecycle, and
  [broker-quality stress tests](#phase-45--broker-quality-stress-testing-does-the-bid-edge-survive-a-worse-market)
  (4.5) show the EV edge **HOLDS across all 10 shifted broker markets**. The open
  thread from 4.5 is **risk-adjusted EV** — folding expected payment into the bid
  objective so the recommender stops treating a slow / defaulting broker like a
  reliable one — plus periodic recalibration of the frozen win-model under market shift.
- **Phase 5 — risk-aware bidding & recalibration (in progress).** The
  [calibrated payment-risk model](#phase-52--calibrated-payment-risk-model) (5.2) is
  built — `P(default)` + `E[pay_days]` from observable broker columns, served behind a
  no-op-safe port; **risk-adjusted EV** (5.1) folds it into the objective, and a
  calibration-drift monitor plus recalibration workflow (5.3–5.4) repair the frozen
  win-model under the market shift Phase 4.5 exposed.
- **Multi-truck dispatch.** Fleet-level assignment so the destination edge can compound.
- **Real Truckstop adapter.** Swap the synthetic board for a live feed behind the existing port.
- **Agent orchestration.** Multi-agent search and negotiation over the planners.

**Quick path (≈90 seconds):** skim the [results table](#results-at-a-glance) and
the [demo](#demo), run `python -m benchmarks.reproduce`, then read
[What I learned](#what-i-learned). The per-phase sections below are the full deep
dive.
