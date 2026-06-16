"""Rolling-replay metrics and cross-episode aggregation (Phase 3.3).

``RollingReplayMetrics`` accumulates one episode's cumulative outcomes;
:func:`summarize_episodes` aggregates a list of episodes into mean ± standard
error with bootstrap confidence intervals, plus the destination diagnostics:

* ``decision_overlap_rate`` / ``divergence_rate`` — how often a *shadow* planner
  (run on the identical board + truck state, without executing) would pick a
  different load. This explains effect size: near-total agreement implies a
  small cumulative gap is expected; high divergence implies the destination
  signal is genuinely changing decisions.
* predicted vs realized onward-deadhead correlation and signed bias — an
  in-distribution sanity check on the Phase 3.1 model, not a held-out MAE.
"""
from __future__ import annotations

import random
import statistics
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

_BOOTSTRAP_SAMPLES = 2000
_BOOTSTRAP_SEED = 12345


@dataclass
class RollingReplayMetrics:
    """Cumulative outcomes for a single replay episode."""

    total_profit: float = 0.0
    total_revenue: float = 0.0
    total_cost: float = 0.0
    total_deadhead_miles: float = 0.0
    total_loaded_miles: float = 0.0
    loads_completed: int = 0
    idle_hours: float = 0.0
    horizon_hours: float = 0.0
    decision_count: int = 0
    feasible_decision_count: int = 0
    # shadow divergence (counterfactual pick on identical inputs)
    shadow_decision_count: int = 0
    shadow_agreements: int = 0
    # predicted onward-deadhead risk of the loads actually selected
    selected_predicted_onward_sum: float = 0.0
    selected_predicted_onward_count: int = 0

    # ----------------------------------------------------------------- derived
    @property
    def profit_per_day(self) -> float:
        days = self.horizon_hours / 24.0
        return self.total_profit / days if days > 0 else 0.0

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
    def deadhead_ratio(self) -> float:
        return (
            self.total_deadhead_miles / self.total_loaded_miles
            if self.total_loaded_miles
            else 0.0
        )

    @property
    def utilization_rate(self) -> float:
        return (
            max(0.0, 1.0 - self.idle_hours / self.horizon_hours)
            if self.horizon_hours
            else 0.0
        )

    @property
    def feasible_decision_rate(self) -> float:
        return (
            self.feasible_decision_count / self.decision_count
            if self.decision_count
            else 0.0
        )

    @property
    def decision_overlap_rate(self) -> Optional[float]:
        if self.shadow_decision_count == 0:
            return None
        return self.shadow_agreements / self.shadow_decision_count

    @property
    def mean_selected_predicted_onward(self) -> Optional[float]:
        if self.selected_predicted_onward_count == 0:
            return None
        return (
            self.selected_predicted_onward_sum / self.selected_predicted_onward_count
        )


# ---------------------------------------------------------------------------
# Cross-episode aggregation
# ---------------------------------------------------------------------------

def _bootstrap_ci(
    values: Sequence[float], rng: random.Random, alpha: float = 0.05
) -> tuple[float, float]:
    n = len(values)
    if n <= 1:
        v = values[0] if values else 0.0
        return (v, v)
    means: List[float] = []
    for _ in range(_BOOTSTRAP_SAMPLES):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((alpha / 2) * _BOOTSTRAP_SAMPLES)]
    hi = means[int((1 - alpha / 2) * _BOOTSTRAP_SAMPLES) - 1]
    return (lo, hi)


def _stats(values: Sequence[float], rng: random.Random) -> Dict[str, float]:
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "se": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}
    mean = statistics.fmean(values)
    se = statistics.stdev(values) / (n**0.5) if n > 1 else 0.0
    ci_low, ci_high = _bootstrap_ci(values, rng)
    return {
        "mean": mean,
        "se": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "median": statistics.median(values),
        "n": n,
    }


# Metric accessors aggregated across episodes (label -> callable).
_AGGREGATED: Dict[str, Callable[[RollingReplayMetrics], float]] = {
    "total_profit": lambda m: m.total_profit,
    "total_deadhead_miles": lambda m: m.total_deadhead_miles,
    "total_loaded_miles": lambda m: m.total_loaded_miles,
    "loads_completed": lambda m: float(m.loads_completed),
    "idle_hours": lambda m: m.idle_hours,
    "profit_per_day": lambda m: m.profit_per_day,
    "deadhead_per_load": lambda m: m.deadhead_per_load,
    "profit_per_load": lambda m: m.profit_per_load,
    "deadhead_ratio": lambda m: m.deadhead_ratio,
    "utilization_rate": lambda m: m.utilization_rate,
    "feasible_decision_rate": lambda m: m.feasible_decision_rate,
    "decision_count": lambda m: float(m.decision_count),
}


def summarize_episodes(episodes: Sequence[RollingReplayMetrics]) -> Dict[str, object]:
    """Aggregate per-episode metrics into mean ± SE with bootstrap CIs."""
    rng = random.Random(_BOOTSTRAP_SEED)
    out: Dict[str, object] = {"episode_count": len(episodes)}
    metrics: Dict[str, Dict[str, float]] = {}
    for label, accessor in _AGGREGATED.items():
        metrics[label] = _stats([accessor(m) for m in episodes], rng)
    out["metrics"] = metrics

    # Divergence vs shadow planner (pooled across episodes).
    shadow_decisions = sum(m.shadow_decision_count for m in episodes)
    shadow_agree = sum(m.shadow_agreements for m in episodes)
    if shadow_decisions:
        overlap = shadow_agree / shadow_decisions
        out["divergence"] = {
            "shadow_decision_count": shadow_decisions,
            "decision_overlap_rate": overlap,
            "divergence_rate": 1.0 - overlap,
        }
    else:
        out["divergence"] = None

    # Predicted onward-risk of selected loads (pooled).
    sel_sum = sum(m.selected_predicted_onward_sum for m in episodes)
    sel_n = sum(m.selected_predicted_onward_count for m in episodes)
    out["mean_selected_predicted_onward"] = (sel_sum / sel_n) if sel_n else None
    return out
