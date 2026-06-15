# Truckstop Feature Discovery — Notes & Question Guide

**Phase 3.0.5.** Purpose: find out which decisioning signals Truckstop actually
exposes in its UI, so the FreightBid synthetic data schema, ML features, and a
future API adapter mirror reality instead of imagination.

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

## P0 — must answer (these shape the data schema & labels)

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

## Fields to model (decision-time only) — working hypothesis

To be confirmed/corrected by the answers above. These are candidates the model
may use **because a dispatcher can see them before accepting**:

- loads currently posted near the destination (count within 50 / 100 / 150 mi)
- median rate-per-mile of those nearby posted loads
- equipment-match count among nearby posted loads
- outbound load volume / market strength of the destination market
- lane average rate (if exposed)
- load age / staleness of the nearby board
- deadhead miles to *this* load's pickup
- arrival time of day / day of week (from the posted delivery window)
- equipment type, destination market/zone

## Leakage risks — features to **avoid**

Anything only knowable **after** the truck arrives or after the outcome is
realized. These are labels or post-hoc facts, never features:

- the actual next load eventually taken / its realized deadhead (**this is the
  label**)
- market conditions at arrival time rather than at decision time
- whether *this* load was ultimately covered/accepted
- post-delivery broker payment outcome

---

## Synthesis (fill in after the conversation)

- **Confirmed decision-time fields:** …
- **Deadhead definition Truckstop uses:** …
- **Rate-missing fraction:** …  → labeling rule impact: …
- **Schema changes needed:** …  (e.g. add/rename fields in
  `ml/data/load_history_schema.py`)
- **New features unlocked:** …
- **Dispatcher heuristics worth encoding:** …
