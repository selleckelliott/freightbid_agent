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


@dataclass(frozen=True)
class BidRecommenderConfig:
    """Phase 4.3 EV bid-recommender policy (config/bid_recommender.yaml).

    Pure business/decision-time policy — none of these knobs touch model training.
    """

    # -- candidate generation ----------------------------------------------
    anchor_multipliers: tuple[float, ...] = (
        0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25,
    )
    cost_per_loaded_mile: float = 1.39
    min_profit_dollars: float = 75.0
    min_margin_rpm: float = 0.20
    max_anchor_multiplier: float = 1.30
    max_candidate_count: int = 16
    min_rate_per_mile: float = 1.00
    max_rate_per_mile: float = 6.00
    trained_ask_ratio_min: float = 0.85
    trained_ask_ratio_max: float = 1.25
    # -- ladder thresholds -------------------------------------------------
    conservative_min_win_prob: float = 0.70
    target_min_win_prob: float = 0.40
    target_ev_tolerance: float = 0.95
    stretch_min_win_prob: float = 0.20
    # -- no-model fallback -------------------------------------------------
    fallback_target_margin: float = 0.20
    # -- model artifact ----------------------------------------------------
    enabled: bool = False
    model_path: str = "ml/artifacts/winnability_model.joblib"
    metadata_path: str = "ml/artifacts/winnability_model_metadata.json"


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


def load_bid_recommender_config(config_dir: str | Path) -> BidRecommenderConfig:
    """Load the Phase 4.3 EV bid-recommender policy from ``bid_recommender.yaml``.

    Missing keys fall back to the ``BidRecommenderConfig`` defaults, so a partial
    file still loads.
    """
    doc = _load_yaml(Path(config_dir) / "bid_recommender.yaml")
    cg = doc.get("candidate_generation", {}) or {}
    ladder = doc.get("ladder", {}) or {}
    model = doc.get("model", {}) or {}
    d = BidRecommenderConfig()
    return BidRecommenderConfig(
        anchor_multipliers=tuple(cg.get("anchor_multipliers", d.anchor_multipliers)),
        cost_per_loaded_mile=float(cg.get("cost_per_loaded_mile", d.cost_per_loaded_mile)),
        min_profit_dollars=float(cg.get("min_profit_dollars", d.min_profit_dollars)),
        min_margin_rpm=float(cg.get("min_margin_rpm", d.min_margin_rpm)),
        max_anchor_multiplier=float(cg.get("max_anchor_multiplier", d.max_anchor_multiplier)),
        max_candidate_count=int(cg.get("max_candidate_count", d.max_candidate_count)),
        min_rate_per_mile=float(cg.get("min_rate_per_mile", d.min_rate_per_mile)),
        max_rate_per_mile=float(cg.get("max_rate_per_mile", d.max_rate_per_mile)),
        trained_ask_ratio_min=float(cg.get("trained_ask_ratio_min", d.trained_ask_ratio_min)),
        trained_ask_ratio_max=float(cg.get("trained_ask_ratio_max", d.trained_ask_ratio_max)),
        conservative_min_win_prob=float(
            ladder.get("conservative_min_win_prob", d.conservative_min_win_prob)
        ),
        target_min_win_prob=float(ladder.get("target_min_win_prob", d.target_min_win_prob)),
        target_ev_tolerance=float(ladder.get("target_ev_tolerance", d.target_ev_tolerance)),
        stretch_min_win_prob=float(ladder.get("stretch_min_win_prob", d.stretch_min_win_prob)),
        fallback_target_margin=float(
            doc.get("fallback", {}).get("target_margin", d.fallback_target_margin)
        ),
        enabled=bool(model.get("enabled", d.enabled)),
        model_path=str(model.get("artifact_path", d.model_path)),
        metadata_path=str(model.get("metadata_path", d.metadata_path)),
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
