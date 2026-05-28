from dataclasses import dataclass
from pathlib import Path

from adapters.outbound.clock import SystemClock
from adapters.outbound.distance.haversine import HaversineDistanceProvider
from adapters.outbound.memory.load_repository import InMemoryLoadRepository
from adapters.outbound.memory.truck_repository import InMemoryTruckRepository
from adapters.outbound.tolls.flat_rate import FlatRateTollEstimator
from application.bid_recommender import BidRecommenderService
from application.config_loader import AppConfig, load_config
from application.evaluate_loads import EvaluateLoadsService
from application.plan_builder import PlanBuilderService
from application.recommend_loads import RecommendLoadsService
from domain.scoring.heuristic_scoring import HeuristicScoringStrategy
from ports.clock import ClockPort
from ports.load_repository import LoadRepositoryPort
from ports.truck_repository import TruckRepositoryPort

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"


@dataclass
class Container:
    config: AppConfig
    load_repo: LoadRepositoryPort
    truck_repo: TruckRepositoryPort
    clock: ClockPort
    evaluator: EvaluateLoadsService
    scoring: HeuristicScoringStrategy
    recommender: RecommendLoadsService
    planner: PlanBuilderService
    bid_recommender: BidRecommenderService


def build_container(
    config_dir: Path | str = DEFAULT_CONFIG_DIR,
    load_repo: LoadRepositoryPort | None = None,
    truck_repo: TruckRepositoryPort | None = None,
) -> Container:
    config = load_config(config_dir)

    load_repo = load_repo or InMemoryLoadRepository()
    truck_repo = truck_repo or InMemoryTruckRepository()
    clock = SystemClock()

    distance = HaversineDistanceProvider()
    tolls = FlatRateTollEstimator()

    evaluator = EvaluateLoadsService(
        distance_provider=distance,
        toll_estimator=tolls,
        cost_model=config.cost_model,
        average_speed_mph=config.average_speed_mph,
        load_unload_hours=config.planning_constraints.average_load_unload_hours,
    )
    scoring = HeuristicScoringStrategy(config.scoring_weights, config.cost_model)
    recommender = RecommendLoadsService(scoring, config.planning_constraints, evaluator)
    planner = PlanBuilderService(scoring, config.planning_constraints, evaluator)
    bid_recommender = BidRecommenderService(
        config.bid_policy, config.bidding_constraints
    )

    return Container(
        config=config,
        load_repo=load_repo,
        truck_repo=truck_repo,
        clock=clock,
        evaluator=evaluator,
        scoring=scoring,
        recommender=recommender,
        planner=planner,
        bid_recommender=bid_recommender,
    )
