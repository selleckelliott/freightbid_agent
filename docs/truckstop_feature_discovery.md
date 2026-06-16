# Truckstop Feature Discovery — Notes & Question Guide

**Phase 3.0.5.** Purpose: find out which decisioning signals Truckstop actually
exposes in its UI, so the FreightBid synthetic data schema, ML features, and a
future API adapter mirror reality instead of imagination.

> **Status: board screenshots received and folded in (Phase 3.1.1).** The
> *Observed fields* section below was extracted from two real Truckstop
> load-search screenshots and now drives the schema. The question guide further
> down is retained for a future live session with an experienced dispatcher
> (dispatcher heuristics + market-intelligence tools are still open).

> The guiding question for the whole session:
>
> **"If I want to build a model that predicts whether a destination will create
> deadhead problems, which fields on Truckstop would tell me that *before* I
> accept the load?"**
>
> The only features the model is allowed to use are ones a dispatcher can see
> **at decision time** (before accepting). Anything only knowable after arrival
> is a label, not a feature.

## Ground rules (read first)

- **No credentials, no scraping.** Don't ask for his login; don't pull data
  through his account in any way the subscription wouldn't allow. Ask for field
  names, what's filterable, what Truckstop calculates, and screenshots.
- **Privacy.** Raw screenshots are **not** committed to this repo. They get
  summarized into the field tables below. If a screenshot shows broker names,
  phone numbers, emails, or MC/DOT numbers, **blur them first**. No PII lands in
  a public portfolio repo.
- **Time-boxed.** Josh offered "a few minutes at work." Lead with the P0
  questions; P1/P2 only if there's time. API/export specifics can come from
  Truckstop's public developer docs later — don't spend his time on those.

---

## Observed fields (from board screenshots) — drives the schema

Extracted from two real Truckstop load-search screenshots (a 4-load Hot Shot
search and a 30-load search with a load-detail panel + map). **Raw images are not
committed** — the detail panel exposed broker PII (name, phone, email, MC/DOT).

### Search-results grid columns

| Column | Meaning | Filterable | TS-calculated | In FreightBid |
| --- | --- | --- | --- | --- |
| `Updated` | time since posted (load age; also a `NEW` badge) | sort | yes | `posted_at` → `load_age_hours` feature ✓ |
| `O-City` / `O-St` | pickup city / state | yes | no | `origin_city` / `origin_state` ✓ |
| `O-DH` | deadhead miles: **search origin → pickup** | via radius | yes | known input, not a model target (see framing) |
| `D-City` / `D-St` | delivery city / state | yes | no | `destination_*` ✓ (drives `destination_zone`) |
| `Distance` | loaded miles | yes | yes | `loaded_miles` ✓ |
| `Rate` | total posted **$** (frequently `N/A`) | yes (min/max) | no | nullable `total_rate` ✓; rate-per-mile **derived** (Pro-only on TS) |
| `Weight` | lbs | yes (0–15k) | no | `weight` ✓ (schema; feasibility) |
| `Length` | ft | yes (0–40) | no | `length` ✓ (schema; feasibility) |
| `Width` / `Height` | ft (usually blank) | yes | no | nullable `width` / `height` ✓ |
| `Pickup` | pickup date / window | yes | no | `pickup_start` / `pickup_end` ✓ |
| `D-DH` | deadhead miles: **delivery → a *user-typed* destination** | via radius | yes | **not** a model input (relative to a fixed endpoint) |
| `Equip` | `HS` / `F` / `FSD` / `FSDV` | yes | no | `equipment_type` ✓ (now hot-shot codes) |
| `Mode` | `TL` / `PTL` / `LTL` | yes | no | `mode` ✓ (schema + categorical feature) |
| `Company` | broker name | yes | no | origin-market name in synth; broker entity deferred |
| `D2P` | broker days-to-pay (e.g. 8, 15, 23, `N/A`) | no | yes | **deferred → Phase 4** quality model |
| `EXP` / `Bond` | broker credit / rating (A/B/R/N; ♦ rating) | no | yes | **deferred → Phase 4** |
| `Load Views` | `Be The First (0)` / `Low (1-9)` / `Med (10-29)` / `High (30+)` — competition | no | yes | `load_views` ✓ → `open_match_within_*` feature |
| `denim` | factorable (denim by truckstop) | filter | yes | deferred |
| `P` / `BIN` | private load / book-it-now | filter | — | deferred |

### Load-detail panel
Posted rate + a "make sure this pays — Get Pro" rate-insight upsell, estimated
fuel cost, pickup window (date **and** time), drop-off (often *Not Available*),
broker MC/DOT/authority requirement + **contact PII**, factoring status
(*Factorable*), free-text **load notes** (dims, appointment hours, "NO TARP",
"NO LANDSTAR", skid counts), and an "is this a quality load posting?" feedback
control.

### Deadhead framing (the key portfolio point)
Truckstop shows `O-DH` (deadhead **to pickup**, known once a truck location is
set) and `D-DH` (deadhead **to a destination the user typed in**). Neither is the
forward *"if I deliver here with no fixed next stop, how far will I deadhead to my
next viable load?"* — which is exactly the label this model predicts. So the ML
layer is **additive** to the board, not a re-display of a number it already shows.

---

## P0 — must answer (these shape the data schema & labels)

> Mostly answered by the *Observed fields* section above; remaining open items
> are the dispatcher heuristics (#3) and the deadhead road-vs-straight-line and
> covered-load nuances.

### 1. Deadhead semantics (most important)
- Where does Truckstop show deadhead, and what is it measured **from** — current
  truck location, or a location you type in?
- Can you set the starting point to a **delivery/destination** city and see
  deadhead to nearby loads (i.e. "if I drop here, what's around me")?
- Is deadhead **road miles or straight-line**? Miles only, or also cost/time?
- Is there a "loads near destination" / backhaul / reload view?

### 2. Load-board fields (screenshot inventory)
Open the **search results** list and **one load detail** page and capture every
field shown. For each, mark: *filterable? / calculated by Truckstop? / shown but
not filterable?* (template below). Specifically confirm:
- Is a **rate always posted**, or are many loads "call for rate"? Roughly what
  fraction have no rate? *(This decides whether a rate-per-mile label filter is
  even usable.)*
- Is **rate-per-mile** shown/auto-calculated, or only total rate?
- **Weight and length** shown? *(Hotshot is weight/length limited — this gates
  feasibility.)*
- **Load age / time posted** shown? Do loads disappear when **covered**?
  *(Tells us if "a viable next load" is actually still available.)*
- Equipment/trailer type, full vs partial, pickup & delivery windows, city/state
  vs exact address.

### 3. Dispatcher heuristics (cheap, high-value — just ask him)
- When choosing a load, what are the **first three things** you look at?
- What makes you **instantly reject** one?
- What signs tell you a load will **leave you stranded**?
- Which **markets/destinations do you avoid**, and which lanes are good for
  hotshot?
- Do you think about deadhead as miles, cost, time, or lost opportunity?

---

## P1 — if there's time (market intelligence & feasibility)

- Does Truckstop show **lane rates / average rate-per-mile** for a lane?
- **Load-to-truck ratio**, demand/supply by market, or a **heat map**?
- **Outbound load volume** from a market (how easy it is to leave a destination)?
- Historical rates, or only current listings?
- Can you enter **when the truck is available** and have it filter by pickup
  feasibility / drive time, or warn when a pickup is impossible?
- Appointment windows vs first-come pickups?

## P2 — later / answer from public docs, not Josh's time

Bidding workflow (in-platform vs call/email, competing bids, covered status),
broker credit / days-to-pay, CSV export / saved searches / reports, and which
fields the **Truckstop API** exposes. These belong to Phase 4; note them but
don't burn the live session on them.

---

## Screenshot inventory template

Fill one block per screen he shows. **Summarize into the tables — do not commit
the raw images.**

```
Screen: <e.g. Search results>
Field            | Filterable? | Truckstop-calculated? | Useful for model? | Notes
---------------- | ----------- | --------------------- | ----------------- | -----
origin city/st   | yes         | no                    | feature (zone)    |
deadhead mi      | (radius)    | yes                   | feature           | from? road/SL?
rate / RPM       | yes         | RPM auto?             | feature/label gate| % missing?
weight / length  | yes         | no                    | feasibility       |
posted / age     | sort        | yes                   | staleness feature |
...
```

Screens worth capturing: search results, one load detail, the deadhead display,
any lane/rate or map/heat-map tool, any reload/backhaul tool, saved
searches/alerts. (Bid + broker-credit screens are P2 — skip unless offered.)

---

## Fields to model (decision-time only) — confirmed in 3.1.1

The screenshots confirmed these are visible **before accepting**, so the model
uses them (✓ = currently a feature; the rest are in the schema for the adapter):

- ✓ loads currently posted near the destination (count within 50 / 100 / 150 mi)
- ✓ equipment-match count among nearby posted loads (hot-shot class)
- ✓ uncontested onward supply near the destination (`open_match_within_*`, from
  `Load Views`)
- ✓ median rate-per-mile of those nearby posted loads (derived; rpm is hidden)
- ✓ load age / staleness of the nearby board
- ✓ arrival time of day / day of week (from the posted delivery window)
- ✓ equipment type, destination market/zone, and load `mode`
- outbound load volume / market strength of the destination market (implicit in
  the density counts; no explicit market-heat field is exposed)
- lane average rate (Pro-only — not exposed on the free board)

## Leakage risks — features to **avoid**

Anything only knowable **after** the truck arrives or after the outcome is
realized. These are labels or post-hoc facts, never features:

- the actual next load eventually taken / its realized deadhead (**this is the
  label**)
- market conditions at arrival time rather than at decision time
- whether *this* load was ultimately covered/accepted
- post-delivery broker payment outcome

---

## Synthesis (from the board screenshots → Phase 3.1.1)

- **Confirmed decision-time fields:** load age (`Updated`), board density &
  per-load competition (`Load Views`), total rate (rate-per-mile is *derived* —
  the board hides it behind Pro), equipment class, weight/length (+ usually-blank
  width/height), mode, and the pickup window. All are visible *before* accepting.
- **Deadhead definition Truckstop uses:** miles relative to a point the user
  specifies — `O-DH` from the search origin to pickup, `D-DH` from the delivery
  city to a typed destination. The board never shows the open-ended *next-load*
  deadhead, so our `expected_next_deadhead_miles` label is genuinely additive.
- **Rate-missing fraction:** a visible share of loads post `Rate = N/A` →
  validates the nullable `total_rate` and `unposted_rate_fraction` (0.15); unrated
  loads stay non-viable for labeling.
- **Schema changes made** (`ml/data/load_history_schema.py` + generator +
  `ml/markets.py`): equipment switched to hot-shot codes `HS/F/FSD/FSDV`; added
  `weight`, `length`, nullable `width`/`height`, `mode` (TL/PTL/LTL), and a
  `load_views` competition bucket.
- **New features unlocked:** `mode` (categorical) and
  `open_match_within_{50,100,150}` — equipment-matched onward loads that are still
  *uncontested* (`Load Views` Be-The-First/Low) near the destination. Model still
  beats both baselines after grounding (MAE 49.3 vs 61.2; ≤25 mi 64% vs 39%).
- **Deferred to a Phase 4 load-quality / winnability model** (observable but not
  predictive of onward *deadhead*): broker `D2P`, `EXP`/`Bond` credit, factorable,
  book-it-now. These describe whether a load pays well and is winnable, a separate
  target from "will this destination strand me."
- **Privacy:** raw screenshots withheld from the repo; the load-detail panel
  exposed broker name/phone/email/MC/DOT.
- **Still open (needs a dispatcher):** road-vs-straight-line deadhead, whether
  covered loads disappear, lane/market-heat tools, and the heuristics in P0 #3.
