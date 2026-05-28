from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

from domain.policies.constraints import (
    BiddingConstraints,
    CostModel,
    PlanningConstraints,
)
from domain.policies.scoring_weights import BidPolicy, ScoringWeights


@dataclass
class AppConfig:
    cost_model: CostModel
    scoring_weights: ScoringWeights
    planning_constraints: PlanningConstraints
    bidding_constraints: BiddingConstraints
    bid_policy: BidPolicy
    average_speed_mph: float = 50.0


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(config_dir: str | Path) -> AppConfig:
    cdir = Path(config_dir)
    cost = _load_yaml(cdir / "cost_model.yaml")
    weights = _load_yaml(cdir / "weights.yaml")
    constraints = _load_yaml(cdir / "constraints.yaml")

    return AppConfig(
        cost_model=CostModel(**cost["cost_model"]),
        scoring_weights=ScoringWeights(**weights["scoring"]),
        bid_policy=BidPolicy(**weights.get("bid_policy", {})),
        planning_constraints=PlanningConstraints(**constraints["planning"]),
        bidding_constraints=BiddingConstraints(**constraints["bidding"]),
        average_speed_mph=constraints.get("average_speed_mph", 50.0),
    )
