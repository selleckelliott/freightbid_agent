"""Fleet dispatch greedy-vs-coordinated benchmark runner (Phase 8.3).

Replays many independent synthetic worlds, one per episode, for a whole FLEET
under two dispatch policies -- uncoordinated **greedy** and coordinated
**fleet-aware** (CP-SAT max-profit matching) -- across several named conditions
(a heterogeneous baseline plus homogeneous-contention and reused stress axes).

For every episode both arms face the **same world** (same seed, same snapshot
stream) and the **same** starting fleet; only the dispatch policy differs. Each
condition reports both arms' fleet summaries (profit, deadhead, utilisation,
per-truck profit balance, destination HHI, contention), the paired
fleet-aware-vs-greedy profit/deadhead deltas, and a robustness verdict.

This runner is a thin CLI over ``simulation.fleet_experiment.run_fleet_condition``
(the same engine the tests reconcile against). It writes
``benchmarks/fleet_dispatch_summary.json`` (committed) for the chart script and
the README.

Examples
--------
    # quick smoke (5 episodes, short solves)
    python -m benchmarks.run_fleet_dispatch --episodes 5

    # canonical run
    python -m benchmarks.run_fleet_dispatch --episodes 40 \
        --out benchmarks/fleet_dispatch_summary.json
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from adapters.inbound.api.container import build_container
from simulation.experiment import ConditionSpec, WorldDefaults
from simulation.fleet_experiment import run_fleet_condition

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "fleet_dispatch.yaml"
DEFAULT_OUT = ROOT / "benchmarks" / "fleet_dispatch_summary.json"

_SPEC_KEYS = (
    "loads_per_snapshot_mean",
    "unposted_rate_fraction",
    "competition_take_rate",
    "fuel_cost_per_mile",
    "deadhead_fuel_multiplier",
    "daily_drive_hours",
    "force_equipment",
    "horizon_days",
)


def _parse_start(value: str) -> datetime:
    dt = datetime.fromisoformat(str(value))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _spec_from_entry(entry: Dict[str, Any]) -> ConditionSpec:
    kwargs: Dict[str, Any] = {
        "name": entry["name"],
        "rationale": entry.get("rationale", ""),
    }
    for key in _SPEC_KEYS:
        if key in entry and entry[key] is not None:
            kwargs[key] = entry[key]
    if "competition_take_rate" not in kwargs:
        kwargs["competition_take_rate"] = 0.0
    return ConditionSpec(**kwargs)


@dataclass
class BenchmarkSettings:
    base: WorldDefaults
    specs: List[ConditionSpec]
    episode_count: int
    fleet_size: int
    base_seed: int
    time_limit: float
    max_candidates: Optional[int]


def load_config(
    path: Path,
    *,
    episodes: Optional[int] = None,
    fleet_size: Optional[int] = None,
    horizon_days: Optional[int] = None,
    time_limit: Optional[float] = None,
) -> BenchmarkSettings:
    """Parse the fleet-dispatch YAML into a ``BenchmarkSettings`` (shared by the
    runner ``main`` and the committed-summary reconciliation test)."""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    world = cfg["world"]
    fleet_cfg = cfg.get("fleet", {})
    episodes_cfg = cfg["episodes"]
    solver_cfg = cfg.get("solver", {})

    episode_count = episodes or int(episodes_cfg["count"])
    fleet = fleet_size or int(fleet_cfg.get("size", 5))
    hz = horizon_days or int(world["horizon_days"])
    base_seed = int(episodes_cfg["base_seed"])
    tl = time_limit if time_limit is not None else float(
        solver_cfg.get("time_limit_seconds", 0.5)
    )
    max_candidates = solver_cfg.get("max_candidates_per_truck")
    if max_candidates is not None:
        max_candidates = int(max_candidates)

    base = WorldDefaults(
        start_date=_parse_start(world["start_date"]),
        snapshots_per_day=int(world["snapshots_per_day"]),
        loads_per_snapshot_mean=float(world["loads_per_snapshot_mean"]),
        unposted_rate_fraction=float(world["unposted_rate_fraction"]),
        max_post_age_hours=float(world["max_post_age_hours"]),
        horizon_days=hz,
        radius_mi=float(world["radius_mi"]),
        daily_drive_hours=float(world["daily_drive_hours"]),
    )
    specs = [_spec_from_entry(e) for e in cfg["conditions"]]
    return BenchmarkSettings(
        base=base,
        specs=specs,
        episode_count=episode_count,
        fleet_size=fleet,
        base_seed=base_seed,
        time_limit=tl,
        max_candidates=max_candidates,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--episodes", type=int, default=None, help="Override episode count.")
    parser.add_argument("--fleet-size", type=int, default=None)
    parser.add_argument("--horizon-days", type=int, default=None)
    parser.add_argument("--time-limit", type=float, default=None,
                        help="CP-SAT matching time limit per epoch (seconds).")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    settings = load_config(
        Path(args.config),
        episodes=args.episodes,
        fleet_size=args.fleet_size,
        horizon_days=args.horizon_days,
        time_limit=args.time_limit,
    )
    base = settings.base
    episode_count = settings.episode_count
    fleet_size = settings.fleet_size
    horizon_days = base.horizon_days
    base_seed = settings.base_seed
    time_limit = settings.time_limit
    max_candidates = settings.max_candidates

    container = build_container(ROOT / "config")
    specs = settings.specs

    print("=" * 78)
    print(f"Fleet dispatch benchmark: {len(specs)} conditions x {episode_count} "
          f"episodes, fleet={fleet_size}, {horizon_days}d horizon "
          f"(CP-SAT {time_limit:.2f}s/epoch)")
    print("=" * 78)

    t_start = time.perf_counter()
    conditions: List[Dict[str, Any]] = []
    for spec in specs:
        c_start = time.perf_counter()
        result = run_fleet_condition(
            spec,
            container=container,
            base=base,
            fleet_size=fleet_size,
            episode_count=episode_count,
            base_seed=base_seed,
            solver_time_limit=time_limit,
            max_candidates_per_truck=max_candidates,
        )
        conditions.append(result.to_dict())
        _print_condition(result, time.perf_counter() - c_start)
    elapsed = time.perf_counter() - t_start

    summary: Dict[str, Any] = {
        "config": {
            "episode_count": episode_count,
            "fleet_size": fleet_size,
            "horizon_days": horizon_days,
            "snapshots_per_day": base.snapshots_per_day,
            "loads_per_snapshot_mean": base.loads_per_snapshot_mean,
            "radius_mi": base.radius_mi,
            "daily_drive_hours": base.daily_drive_hours,
            "solver_time_limit_seconds": time_limit,
            "max_candidates_per_truck": max_candidates,
            "base_seed": base_seed,
        },
        "headline": "fleet_aware vs greedy (coordination value)",
        "conditions": conditions,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Volatile fields confined to the written artifact's top level so the
    # per-condition bodies stay byte-stable / reconcilable across runs.
    written = dict(summary)
    written["elapsed_seconds"] = elapsed
    written["generated_at"] = datetime.now(timezone.utc).isoformat()
    out_path.write_text(json.dumps(written, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {out_path} ({elapsed:.1f}s).")


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def _ci(stat: Dict[str, float]) -> str:
    return f"{stat['mean']:.1f} [{stat['ci_low']:.1f}, {stat['ci_high']:.1f}]"


def _print_condition(result: Any, elapsed: float) -> None:
    g = result.greedy["metrics"]
    f = result.fleet_aware["metrics"]
    kind = "homogeneous" if result.homogeneous else "heterogeneous"
    print(f"\n[{result.verdict}] {result.name}  ({kind}, fleet={result.fleet_size}, "
          f"n={result.episode_count}, {elapsed:.1f}s)")
    print(f"  Fleet profit    greedy ${_ci(g['total_profit'])}  ->  "
          f"fleet-aware ${_ci(f['total_profit'])}  "
          f"({result.headline_delta['profit_pct']:+.1f}%)")
    print(f"  Deadhead mi     greedy {_ci(g['total_deadhead_miles'])}  ->  "
          f"fleet-aware {_ci(f['total_deadhead_miles'])}  "
          f"({result.headline_delta['deadhead_pct']:+.1f}%, lower is better)")
    print(f"  Loads           greedy {_ci(g['loads_completed'])}  ->  "
          f"fleet-aware {_ci(f['loads_completed'])}")
    print(f"  Mean util       greedy {g['mean_utilization_rate']['mean']:.3f}  ->  "
          f"fleet-aware {f['mean_utilization_rate']['mean']:.3f}")
    print(f"  Profit balance  greedy disp {g['profit_dispersion']['mean']:.0f} / "
          f"gini {g['profit_gini']['mean']:.3f}  ->  fleet-aware disp "
          f"{f['profit_dispersion']['mean']:.0f} / gini {f['profit_gini']['mean']:.3f}")
    print(f"  Contention      greedy {g['contention_events']['mean']:.1f}  ->  "
          f"fleet-aware {f['contention_events']['mean']:.1f} events/episode")
    print(f"  Paired profit   {result.paired_profit['mean']:+.1f} "
          f"[{result.paired_profit['ci_low']:+.1f}, "
          f"{result.paired_profit['ci_high']:+.1f}] (fleet-aware - greedy)")


if __name__ == "__main__":
    main()
