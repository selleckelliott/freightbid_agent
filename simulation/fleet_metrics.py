"""Fleet-episode metrics and cross-episode aggregation (Phase 8.2).

The fleet simulator runs K trucks over one shared board. Each truck accumulates an
ordinary single-truck :class:`~simulation.metrics.RollingReplayMetrics` (so every
per-truck derived figure — utilisation, deadhead ratio, profit/load — is exactly
the single-truck definition, and the K=1 case reconciles with ``run_episode`` for
free). :class:`FleetEpisodeMetrics` then layers the genuinely *fleet*-level signals
on top:

* **fleet totals** — profit, deadhead, loaded miles, loads (sums across trucks);
* **balance** — per-truck profit dispersion (stdev) and a Gini coefficient: did
  the coordinated policy keep every truck earning, or starve some to feed others?
* **mean / min utilisation** — is any truck left idling?
* **destination-market concentration (HHI)** — did the fleet pile every truck into
  one drop-off market (fragile) or spread across markets?
* **contention events** — epochs where two or more free trucks could see the same
  load, i.e. the decision points where coordination can actually matter;
* **artifact-gated payment risk** — when (and only when) a risk scorer is supplied,
  model-scored expected collectible profit and expected default exposure over the
  executed loads. These are *model expectations*, not a realised payment sim.

Cross-episode aggregation reuses the bootstrap helper from ``simulation.metrics``.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from simulation.metrics import RollingReplayMetrics, _stats


@dataclass
class FleetEpisodeMetrics:
    """Fleet-level outcomes for a single multi-truck episode."""

    horizon_hours: float
    truck_metrics: List[RollingReplayMetrics] = field(default_factory=list)
    destination_market_counts: Dict[str, int] = field(default_factory=dict)
    contention_events: int = 0
    # Artifact-gated payment risk: None unless a risk scorer was supplied.
    expected_collectible_profit: Optional[float] = None
    expected_default_exposure: Optional[float] = None

    # --------------------------------------------------------------- fleet totals
    @property
    def truck_count(self) -> int:
        return len(self.truck_metrics)

    @property
    def total_profit(self) -> float:
        return sum(m.total_profit for m in self.truck_metrics)

    @property
    def total_revenue(self) -> float:
        return sum(m.total_revenue for m in self.truck_metrics)

    @property
    def total_cost(self) -> float:
        return sum(m.total_cost for m in self.truck_metrics)

    @property
    def total_deadhead_miles(self) -> float:
        return sum(m.total_deadhead_miles for m in self.truck_metrics)

    @property
    def total_loaded_miles(self) -> float:
        return sum(m.total_loaded_miles for m in self.truck_metrics)

    @property
    def loads_completed(self) -> int:
        return sum(m.loads_completed for m in self.truck_metrics)

    @property
    def total_idle_hours(self) -> float:
        return sum(m.idle_hours for m in self.truck_metrics)

    # ------------------------------------------------------------------- derived
    @property
    def deadhead_ratio(self) -> float:
        return (
            self.total_deadhead_miles / self.total_loaded_miles
            if self.total_loaded_miles
            else 0.0
        )

    @property
    def deadhead_per_load(self) -> float:
        return (
            self.total_deadhead_miles / self.loads_completed
            if self.loads_completed
            else 0.0
        )

    @property
    def profit_per_load(self) -> float:
        return (
            self.total_profit / self.loads_completed if self.loads_completed else 0.0
        )

    @property
    def loads_per_truck(self) -> float:
        return self.loads_completed / self.truck_count if self.truck_count else 0.0

    @property
    def profit_per_day(self) -> float:
        days = self.horizon_hours / 24.0
        return self.total_profit / days if days > 0 else 0.0

    # ----------------------------------------------------------- fleet balance
    @property
    def mean_utilization_rate(self) -> float:
        if not self.truck_metrics:
            return 0.0
        return statistics.fmean(m.utilization_rate for m in self.truck_metrics)

    @property
    def min_utilization_rate(self) -> float:
        if not self.truck_metrics:
            return 0.0
        return min(m.utilization_rate for m in self.truck_metrics)

    @property
    def profit_dispersion(self) -> float:
        """Stdev of per-truck profit (population). 0 with fewer than two trucks.

        Lower is a more *balanced* fleet — the coordinated policy spread work
        rather than starving some trucks to feed others.
        """
        profits = [m.total_profit for m in self.truck_metrics]
        if len(profits) < 2:
            return 0.0
        return statistics.pstdev(profits)

    @property
    def profit_gini(self) -> float:
        """Gini coefficient of per-truck profit in ``[0, 1]`` (0 = perfectly even).

        Computed on profits floored at 0 (a truck cannot have negative collected
        work here; feasible loads are profit-positive). Returns 0 for an empty or
        all-zero fleet.
        """
        profits = sorted(max(0.0, m.total_profit) for m in self.truck_metrics)
        n = len(profits)
        total = sum(profits)
        if n == 0 or total == 0.0:
            return 0.0
        # Gini = (2*sum(i*x_i) / (n*sum(x))) - (n+1)/n, i 1-indexed on sorted x.
        weighted = sum((i + 1) * x for i, x in enumerate(profits))
        return (2.0 * weighted) / (n * total) - (n + 1) / n

    @property
    def destination_hhi(self) -> float:
        """Herfindahl-Hirschman index of executed-load destination markets.

        ``sum(share_i**2)`` over destination states, in ``(0, 1]``: 1.0 means every
        load dropped in a single market (concentrated / fragile onward position),
        lower means the fleet diversified its drop-offs. 0.0 if no loads ran.
        """
        total = sum(self.destination_market_counts.values())
        if total == 0:
            return 0.0
        return sum((c / total) ** 2 for c in self.destination_market_counts.values())


# ---------------------------------------------------------------------------
# Cross-episode aggregation (reuses the simulation.metrics bootstrap helper)
# ---------------------------------------------------------------------------

# Fleet-level scalar accessors aggregated across episodes (label -> callable).
_FLEET_AGGREGATED = {
    "total_profit": lambda m: m.total_profit,
    "total_deadhead_miles": lambda m: m.total_deadhead_miles,
    "total_loaded_miles": lambda m: m.total_loaded_miles,
    "loads_completed": lambda m: float(m.loads_completed),
    "deadhead_ratio": lambda m: m.deadhead_ratio,
    "deadhead_per_load": lambda m: m.deadhead_per_load,
    "profit_per_load": lambda m: m.profit_per_load,
    "profit_per_day": lambda m: m.profit_per_day,
    "mean_utilization_rate": lambda m: m.mean_utilization_rate,
    "min_utilization_rate": lambda m: m.min_utilization_rate,
    "profit_dispersion": lambda m: m.profit_dispersion,
    "profit_gini": lambda m: m.profit_gini,
    "destination_hhi": lambda m: m.destination_hhi,
    "loads_per_truck": lambda m: m.loads_per_truck,
    "contention_events": lambda m: float(m.contention_events),
}


def summarize_fleet_episodes(
    episodes: Sequence[FleetEpisodeMetrics],
) -> Dict[str, object]:
    """Aggregate per-episode fleet metrics into mean +/- SE with bootstrap CIs."""
    import random

    rng = random.Random(12345)
    metrics: Dict[str, Dict[str, float]] = {}
    for label, accessor in _FLEET_AGGREGATED.items():
        metrics[label] = _stats([accessor(m) for m in episodes], rng)

    out: Dict[str, object] = {
        "episode_count": len(episodes),
        "metrics": metrics,
    }

    # Artifact-gated risk is pooled only when present in every episode.
    coll = [
        m.expected_collectible_profit
        for m in episodes
        if m.expected_collectible_profit is not None
    ]
    exp = [
        m.expected_default_exposure
        for m in episodes
        if m.expected_default_exposure is not None
    ]
    if coll and len(coll) == len(episodes):
        out["expected_collectible_profit"] = _stats(coll, rng)
        out["expected_default_exposure"] = _stats(exp, rng)
    else:
        out["expected_collectible_profit"] = None
        out["expected_default_exposure"] = None
    return out
