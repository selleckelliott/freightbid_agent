# FreightBid — Production-Readiness Runbook (Phase 7.4)

This runbook is for an operator or reviewer running FreightBid as a service. It covers how to start it,
how to check it is healthy and ready, how to validate config and artifacts, how to smoke-test the full
workflow, and the safety guarantees that hold in every mode.

> **Safety posture (unchanged in Phase 7).** FreightBid is a *decision-support* engine. The compiled
> dispatcher is **shadow-only** and never owns a decision. There is **no live Truckstop integration** and
> **no auto-bidding** — `submit-mock` is a simulated terminal state. Broker **PII is redacted at the
> ingestion boundary** and never persisted or exported. None of the ops surfaces below change that.

---

## 1. Run it

### Local (Python 3.11+)

```bash
pip install -r requirements.txt
export PYTHONPATH=.                       # Windows: $env:PYTHONPATH='.'
uvicorn adapters.inbound.api.app:app --host 0.0.0.0 --port 8000
```

The CLI is a thin client to that API:

```bash
python -m adapters.inbound.cli.main health        # or: freightbid health (if installed)
```

### Docker

```bash
docker build -t freightbid .
docker run -p 8000:8000 freightbid
```

The image runs as a **non-root user** (`freight`, uid 10001) and ships a **HEALTHCHECK** that polls
`/health`. `docker compose up` additionally starts an optional Postgres (FreightBid does **not** require
it — decision export is file-based; see the audit-export section).

---

## 2. Preflight: validate config (no server needed)

Run **before** booting to catch a malformed/missing config file early. Pure disk reads — no server,
no side effects:

```bash
freightbid validate-config                        # validates ./config
freightbid validate-config --config-dir /etc/freightbid
```

Exit code is non-zero and the offending file + error is listed if any of `app_config`,
`bid_recommender`, `bid_approval`, `compiled_dispatcher`, `load_board`, or `objective_profiles` fails to
load. `FREIGHTBID_CONFIG_DIR` overrides the default config directory for both the API and the CLI.

---

## 3. Liveness vs. readiness

| Probe | Endpoint | Meaning |
| --- | --- | --- |
| **Liveness** | `GET /health` | the process is up (`{"status":"ok"}`). Used by the Docker HEALTHCHECK. |
| **Readiness** | `GET /ready` | config + load board + **model-artifact** status, plus engine/provenance. |

```bash
freightbid ready
```

`GET /ready` is **always HTTP 200** and **side-effect-free**. Its `status` field is:

- **`ready`** — every *enabled* dependency is usable.
- **`degraded`** — the engine is serving, but an **enabled** model's artifact is missing, or the
  configured load board is unavailable. The `warnings` list says which. FreightBid still serves in this
  state (rule-based fallbacks + the sandbox board), which is **by design** — every model dependency is
  optional. `degraded` is a heads-up, not an outage.

Because all models are optional, readiness deliberately does **not** return 503 for `degraded`; treat
the `status`/`warnings` fields (not the HTTP code) as the readiness signal.

---

## 4. Artifact availability

`GET /ready` → `checks.artifacts` reports each optional model:

| Model | Enabled by | Artifact (gitignored) |
| --- | --- | --- |
| `winnability` | `bid_recommender.yaml` → `model.enabled` | `ml/artifacts/winnability_model.joblib` |
| `payment_risk` | `bid_recommender.yaml` → `risk_adjusted_ev.enabled` | `ml/artifacts/payment_risk_model.joblib` |
| `compiled_dispatcher` | `compiled_dispatcher.yaml` → `enabled` (shadow-only) | `ml/artifacts/compiled_dispatcher_model.joblib` |

`enabled` is the config flag; `present` is whether the artifact is on disk. The model `.joblib` files are
**gitignored**, so a fresh clone reports `present:false` and the engine runs on rule-based fallbacks — a
fully supported mode. To enable a model: train/obtain the artifact, place it at the path above, flip the
flag, and confirm `freightbid ready` no longer warns about it.

---

## 5. Smoke test the full workflow

End-to-end check against a **running** API:

```bash
freightbid smoke-test                             # uses sample_data/truck.json + loads.json
freightbid smoke-test --truck-file my_truck.json --loads-file my_loads.json
```

Steps: `health → ready → pull → ingest → rank → bid_draft → decisions`. The `pull` step proves Phase 7.2
board connectivity (deterministic sandbox); `rank`/`bid` use the committed sample truck+loads pair (which
is feasibility-aligned) so the source engine has something to rank. **It mutates state** (pulls sandbox
loads, ingests the sample loads, drafts a bid) — run it against a dev/staging instance, not a populated
production queue. It never approves, submits, or bids for real. Non-zero exit if any step fails.

---

## 6. Audit export (Phase 7.3)

Decisions are auditable outside the process. The CLI fetches `GET /decisions` and writes **locally** — a
request can never make the server write to a path:

```bash
freightbid export ./audit --format bundle         # decisions.jsonl + decisions.csv + manifest.json
freightbid export ./won.jsonl --format jsonl --status approved
```

Every record carries provenance (`source_policy_version`, `git_commit`/`git_describe`, `config_hash`,
model-artifact ids, feature-manifest hash) and the bid's full audit trail. This is the file-based audit
trail that makes Postgres optional.

---

## 7. Troubleshooting

| Symptom | Likely cause | Action |
| --- | --- | --- |
| `validate-config` fails on `app_config` | wrong `--config-dir` / missing `cost_model.yaml` etc. | point at the real config dir; check the named file |
| `/ready` `degraded`, warns about a model | `enabled:true` but artifact missing | place the `.joblib` or set the flag back to `false` |
| `/ready` `degraded`, warns about the board | `load_board.yaml` `source: replay` with a missing feed | fix `feed_path` or switch to `source: sandbox` |
| `smoke-test` `rank` returns 0 | truck/loads time-misaligned or infeasible | use the committed sample pair, or align `available_at`/equipment |
| `pull` reports unavailable | sandbox `count: 0`, or replay feed missing/unreadable | restore the feed; the live path fails closed (never crashes) |
| Docker HEALTHCHECK unhealthy | API not listening on 8000 | check container logs; confirm the `uvicorn` process started |

---

## 8. Guarantees that always hold

- Existing endpoints (`/rank`, `/plan`, `/bids`, `/loads`) are **byte-identical** to pre-Phase-7
  behavior; `/ready` and the ops commands are **additive**.
- Readiness and config-validation are **side-effect-free**; only `smoke-test` mutates state, and only the
  ingest/draft paths — never approve/submit/real-bid.
- The compiled dispatcher is **shadow-only**; there is **no live Truckstop** and **no auto-bidding**;
  broker **PII is redacted** at ingestion and never exported.
