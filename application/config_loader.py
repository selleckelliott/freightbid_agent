from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

from domain.policies.constraints import (
    BiddingConstraints,
    CostModel,
    PlanningConstraints,
)
from domain.policies.ortools_objective_weights import ORToolsObjectiveWeights
from domain.policies.scoring_weights import BidPolicy, ScoringWeights


@dataclass
class AppConfig:
    cost_model: CostModel
    scoring_weights: ScoringWeights
    planning_constraints: PlanningConstraints
    bidding_constraints: BiddingConstraints
    bid_policy: BidPolicy
    ortools_objective_weights: ORToolsObjectiveWeights
    average_speed_mph: float = 50.0


@dataclass(frozen=True)
class ObjectiveProfile:
    """A named dispatch policy for the profit-aware planner (Phase 2.3).

    Profiles are defined as multipliers over the derived cost model in
    ``config/objective_profiles.yaml`` and resolved to concrete
    ``ORToolsObjectiveWeights`` at load time.
    """

    name: str
    description: str
    deadhead_cost_multiplier: float
    skip_profit_floor_dollars: float
    weights: ORToolsObjectiveWeights
    solver_time_limit_seconds: float | None = None


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(config_dir: str | Path) -> AppConfig:
    cdir = Path(config_dir)
    cost = _load_yaml(cdir / "cost_model.yaml")
    weights = _load_yaml(cdir / "weights.yaml")
    constraints = _load_yaml(cdir / "constraints.yaml")

    cost_model = CostModel(**cost["cost_model"])
    average_speed_mph = constraints.get("average_speed_mph", 50.0)

    return AppConfig(
        cost_model=cost_model,
        scoring_weights=ScoringWeights(**weights["scoring"]),
        bid_policy=BidPolicy(**weights.get("bid_policy", {})),
        planning_constraints=PlanningConstraints(**constraints["planning"]),
        bidding_constraints=BiddingConstraints(**constraints["bidding"]),
        ortools_objective_weights=ORToolsObjectiveWeights.from_cost_model(
            cost_model,
            average_speed_mph,
            **weights.get("ortools_objective", {}),
        ),
        average_speed_mph=average_speed_mph,
    )


def load_objective_profiles(config_dir: str | Path) -> Dict[str, ObjectiveProfile]:
    """Load named objective profiles, resolving each to concrete weights.

    Reads ``objective_profiles.yaml`` alongside the cost model/constraints in
    ``config_dir`` so every profile's per-mile rate is derived from the same
    business cost model the planners and evaluator share.
    """
    cdir = Path(config_dir)
    doc = _load_yaml(cdir / "objective_profiles.yaml")
    cost_model = CostModel(**_load_yaml(cdir / "cost_model.yaml")["cost_model"])
    constraints = _load_yaml(cdir / "constraints.yaml")
    average_speed_mph = constraints.get("average_speed_mph", 50.0)

    profiles: Dict[str, ObjectiveProfile] = {}
    for name, spec in (doc.get("profiles") or {}).items():
        multiplier = float(spec.get("deadhead_cost_multiplier", 1.0))
        floor = float(spec["skip_profit_floor_dollars"])
        profiles[name] = ObjectiveProfile(
            name=name,
            description=spec.get("description", ""),
            deadhead_cost_multiplier=multiplier,
            skip_profit_floor_dollars=floor,
            weights=ORToolsObjectiveWeights.from_cost_model(
                cost_model,
                average_speed_mph,
                deadhead_cost_multiplier=multiplier,
                skip_profit_floor_dollars=floor,
            ),
            solver_time_limit_seconds=spec.get("solver_time_limit_seconds"),
        )
    if not profiles:
        raise ValueError(f"No profiles defined in {cdir / 'objective_profiles.yaml'}")
    return profiles
