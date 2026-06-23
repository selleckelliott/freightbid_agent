import logging
from dataclasses import dataclass
from pathlib import Path

from adapters.outbound.clock import SystemClock
from adapters.outbound.compiled_dispatcher.noop_compiled_dispatcher import NoopCompiledDispatcher
from adapters.outbound.compiled_dispatcher.sklearn_compiled_dispatcher import (
    SklearnCompiledDispatcher,
)
from adapters.outbound.distance.haversine import HaversineDistanceProvider
from adapters.outbound.memory.bid_repository import InMemoryBidApprovalRepository
from adapters.outbound.memory.load_repository import InMemoryLoadRepository
from adapters.outbound.memory.truck_repository import InMemoryTruckRepository
from adapters.outbound.payment_risk.model_adapter import ModelPaymentRiskAdapter
from adapters.outbound.tolls.flat_rate import FlatRateTollEstimator
from adapters.outbound.winnability.model_adapter import ModelWinnabilityAdapter
from application.bid_approval_service import BidApprovalService
from application.bid_recommender import BidRecommenderService
from application.config_loader import (
    AppConfig,
    BidApprovalConfig,
    BidRecommenderConfig,
    CompiledDispatcherConfig,
    load_bid_approval_config,
    load_bid_recommender_config,
    load_compiled_dispatcher_config,
    load_config,
)
from application.ev_bid_recommender import EVBidRecommender
from application.evaluate_loads import EvaluateLoadsService
from application.ortools_distance_planner import ORToolsDistancePlanner
from application.ortools_profit_aware_planner import ORToolsProfitAwarePlanner
from application.plan_builder import PlanBuilderService
from application.recommend_loads import RecommendLoadsService
from application.services.shadow_compiled_dispatcher_service import (
    ShadowCompiledDispatcherService,
)
from domain.scoring.heuristic_scoring import HeuristicScoringStrategy
from ml.models.compiled_dispatcher_model import (
    default_feature_manifest,
    feature_manifest_hash,
)
from ports.bid_repository import BidApprovalRepositoryPort
from ports.clock import ClockPort
from ports.compiled_dispatcher import REASON_DISABLED, REASON_NO_ARTIFACT
from ports.load_repository import LoadRepositoryPort
from ports.truck_repository import TruckRepositoryPort

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_DIR = ROOT / "config"


def _build_ev_recommender(bid_cfg: BidRecommenderConfig) -> EVBidRecommender | None:
    """Wire the EV recommender only when the flag is on AND the artifact exists.

    Returns ``None`` (so ``BidRecommenderService`` keeps its cost-plus-margin behavior)
    when winnability is disabled, or enabled but the gitignored 4.2 artifact is absent —
    the latter logged as a warning rather than raised, so the app always boots.
    """
    if not bid_cfg.enabled:
        return None
    artifact = Path(bid_cfg.model_path)
    if not artifact.is_absolute():
        artifact = ROOT / artifact
    if not artifact.exists():
        logger.warning(
            "winnability model enabled but artifact %s not found; "
            "serving cost-plus-margin bids",
            artifact,
        )
        return None
    adapter = ModelWinnabilityAdapter.from_artifact(artifact)
    logger.info("winnability model loaded from %s; EV surfacing enabled", artifact)
    payment = _build_payment_adapter(bid_cfg)
    return EVBidRecommender(adapter, bid_cfg, payment=payment)


def _build_payment_adapter(bid_cfg: BidRecommenderConfig) -> ModelPaymentRiskAdapter | None:
    """Wire the Phase 5.2 payment-risk adapter only when risk-adjusted EV is on AND the
    artifact exists.

    Returns ``None`` (so ``EVBidRecommender`` ranks by raw EV — risk-blind) when the
    flag is off, or on but the gitignored 5.2 artifact is absent — the latter logged as
    a warning rather than raised, mirroring the winnability wiring above.
    """
    if not bid_cfg.risk_adjusted_ev_enabled:
        return None
    artifact = Path(bid_cfg.payment_model_path)
    if not artifact.is_absolute():
        artifact = ROOT / artifact
    if not artifact.exists():
        logger.warning(
            "risk-adjusted EV enabled but payment-risk artifact %s not found; "
            "serving risk-blind EV bids",
            artifact,
        )
        return None
    adapter = ModelPaymentRiskAdapter.from_artifact(artifact)
    logger.info("payment-risk model loaded from %s; risk-adjusted EV enabled", artifact)
    return adapter


def _build_compiled_dispatcher_shadow(
    cfg: CompiledDispatcherConfig,
) -> ShadowCompiledDispatcherService:
    """Wire the Phase 6.4 shadow compiled dispatcher — fail-closed at every step.

    Default off, missing artifact, or a load failure all degrade to the :class:`NoopCompiledDispatcher`
    (so the shadow service reports ``compiled_available=False`` with a reason and the source
    recommendation is untouched). When enabled with a present artifact, the loaded model is gated on a
    feature-manifest-hash match against the current inference contract — a mismatch is reported as
    unavailable (``manifest_mismatch``) rather than served.
    """
    if not cfg.enabled:
        return ShadowCompiledDispatcherService(NoopCompiledDispatcher(REASON_DISABLED))
    artifact = Path(cfg.artifact_path)
    if not artifact.is_absolute():
        artifact = ROOT / artifact
    if not artifact.exists():
        logger.warning(
            "compiled dispatcher enabled but artifact %s not found; shadow disabled", artifact
        )
        return ShadowCompiledDispatcherService(NoopCompiledDispatcher(REASON_NO_ARTIFACT))
    expected_hash = feature_manifest_hash(default_feature_manifest())
    try:
        adapter = SklearnCompiledDispatcher.from_artifact(
            artifact, expected_manifest_hash=expected_hash
        )
    except Exception:  # noqa: BLE001 — a bad artifact must never block app boot
        logger.warning(
            "failed to load compiled dispatcher artifact %s; shadow disabled", artifact,
            exc_info=True,
        )
        return ShadowCompiledDispatcherService(NoopCompiledDispatcher(REASON_NO_ARTIFACT))
    logger.info("compiled dispatcher loaded from %s; shadow mode enabled", artifact)
    return ShadowCompiledDispatcherService(adapter)


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
    ortools_distance_planner: ORToolsDistancePlanner
    ortools_profit_aware_planner: ORToolsProfitAwarePlanner
    bid_recommender: BidRecommenderService
    bid_recommender_config: BidRecommenderConfig
    bid_repo: BidApprovalRepositoryPort
    bid_approval_config: BidApprovalConfig
    bid_approval_service: BidApprovalService
    compiled_dispatcher_config: CompiledDispatcherConfig
    compiled_dispatcher_shadow: ShadowCompiledDispatcherService


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
    ortools_distance_planner = ORToolsDistancePlanner(
        distance_provider=distance,
        evaluate_loads_service=evaluator,
        constraints=config.planning_constraints,
        average_speed_mph=config.average_speed_mph,
        load_unload_hours=config.planning_constraints.average_load_unload_hours,
    )
    ortools_profit_aware_planner = ORToolsProfitAwarePlanner(
        distance_provider=distance,
        evaluate_loads_service=evaluator,
        constraints=config.planning_constraints,
        objective_weights=config.ortools_objective_weights,
        average_speed_mph=config.average_speed_mph,
        load_unload_hours=config.planning_constraints.average_load_unload_hours,
    )
    bid_recommender_config = load_bid_recommender_config(config_dir)
    ev_recommender = _build_ev_recommender(bid_recommender_config)
    bid_recommender = BidRecommenderService(
        config.bid_policy,
        config.bidding_constraints,
        ev_recommender=ev_recommender,
    )

    bid_repo = InMemoryBidApprovalRepository()
    bid_approval_config = load_bid_approval_config(config_dir)
    bid_approval_service = BidApprovalService(
        bid_repo=bid_repo,
        load_repo=load_repo,
        truck_repo=truck_repo,
        evaluator=evaluator,
        bid_recommender=bid_recommender,
        clock=clock,
        config=bid_approval_config,
    )

    compiled_dispatcher_config = load_compiled_dispatcher_config(config_dir)
    compiled_dispatcher_shadow = _build_compiled_dispatcher_shadow(compiled_dispatcher_config)

    return Container(
        config=config,
        load_repo=load_repo,
        truck_repo=truck_repo,
        clock=clock,
        evaluator=evaluator,
        scoring=scoring,
        recommender=recommender,
        planner=planner,
        ortools_distance_planner=ortools_distance_planner,
        ortools_profit_aware_planner=ortools_profit_aware_planner,
        bid_recommender=bid_recommender,
        bid_recommender_config=bid_recommender_config,
        bid_repo=bid_repo,
        bid_approval_config=bid_approval_config,
        bid_approval_service=bid_approval_service,
        compiled_dispatcher_config=compiled_dispatcher_config,
        compiled_dispatcher_shadow=compiled_dispatcher_shadow,
    )
