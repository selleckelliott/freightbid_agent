import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query

from adapters.inbound.api.container import build_container
from adapters.inbound.api.mappers import (
    bid_draft_to_dto,
    bid_range_to_dto,
    load_from_dto,
    truck_from_dto,
)
from adapters.inbound.api.schemas import (
    BidActionRequest,
    BidDraftDTO,
    BidQueueResponse,
    CompiledShadowDTO,
    CreateBidDraftRequest,
    EditBidRequest,
    IngestRequest,
    IngestResponse,
    PlanResponse,
    PlanStopDTO,
    PullRequest,
    PullResponse,
    RankRequest,
    RankResponse,
    RankedLoad,
)
from application.bid_approval_service import BidDraftNotFound, LoadNotFoundForBid
from domain.enums.bid_approval_status import BidApprovalStatus
from domain.models.bid_draft import InvalidBidTransition
from ports.compiled_dispatcher import SHADOW_ONLY

CONFIG_DIR = Path(os.environ.get("FREIGHTBID_CONFIG_DIR", Path(__file__).resolve().parents[3] / "config"))


def _compiled_shadow_banner(container) -> "CompiledShadowDTO | None":
    """Additive Phase 6.4 surface: ``None`` when the compiled dispatcher is OFF, else an availability
    banner. Never affects the ``ranked`` items — the source engine still decides (``shadow_only``).
    """
    if not container.compiled_dispatcher_config.enabled:
        return None
    avail = container.compiled_dispatcher_shadow.availability()
    return CompiledShadowDTO(
        compiled_available=avail.available,
        shadow_only=SHADOW_ONLY,
        reason=avail.reason,
        artifact_path=avail.artifact_path,
        feature_manifest_hash=avail.feature_manifest_hash,
    )


def create_app(container=None) -> FastAPI:
    container = container or build_container(CONFIG_DIR)
    app = FastAPI(title="FreightBid Dispatch Brain", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/loads", response_model=IngestResponse)
    def ingest_loads(req: IngestRequest):
        loads = [load_from_dto(l) for l in req.loads]
        container.load_repo.add_many(loads)
        return IngestResponse(accepted=len(loads))

    @app.post("/loads/pull", response_model=PullResponse)
    def pull_loads(req: PullRequest | None = None):
        """Phase 7.2 thin live wiring: pull external-style rows from the configured sandbox/replay
        board, validate via the 7.1 contract, and ingest the accepted loads. Additive — the synthetic
        ``POST /loads`` ingress is unchanged."""
        req = req or PullRequest()
        report = container.load_board_ingest.pull(limit=req.limit, replace=req.replace)
        return PullResponse(
            source=report.source,
            available=report.available,
            fetched=report.fetched,
            accepted=report.accepted,
            rejected=report.rejected,
            load_ids=report.load_ids,
            errors=report.errors,
            reason=report.reason,
            replaced=report.replaced,
        )

    @app.get("/loads")
    def list_loads():
        return [l.__dict__ for l in container.load_repo.list_all()]

    @app.delete("/loads")
    def clear_loads():
        container.load_repo.clear()
        return {"status": "cleared"}

    @app.post("/rank", response_model=RankResponse)
    def rank(req: RankRequest):
        truck = truck_from_dto(req.truck)
        container.truck_repo.upsert(truck)

        all_loads = container.load_repo.list_all()
        if req.load_ids:
            allowed = set(req.load_ids)
            all_loads = [l for l in all_loads if l.load_id in allowed]
        if not all_loads:
            raise HTTPException(status_code=400, detail="No loads to rank. Ingest loads first.")

        ranked = container.recommender.recommend_loads(all_loads, truck, top_n=req.top_n)

        items = []
        for evaluation, score in ranked:
            bid = container.bid_recommender.recommend(
                evaluation, decided_at=truck.available_at
            )
            items.append(
                RankedLoad(
                    load_id=score.load_id,
                    score=score.score,
                    expected_profit=score.expected_profit,
                    expected_revenue=score.expected_revenue,
                    rate_per_mile=score.rate_per_mile,
                    deadhead_miles=score.deadhead_miles,
                    driver_hours=score.driver_hours,
                    pickup_eta=evaluation.pickup_eta,
                    delivery_eta=evaluation.delivery_eta,
                    rationale=score.rationale or "",
                    bid=bid_range_to_dto(bid),
                )
            )
        return RankResponse(
            truck_id=truck.truck_id,
            ranked=items,
            compiled_shadow=_compiled_shadow_banner(container),
        )

    @app.post("/plan", response_model=PlanResponse)
    def plan(req: RankRequest):
        truck = truck_from_dto(req.truck)
        container.truck_repo.upsert(truck)

        all_loads = container.load_repo.list_all()
        if req.load_ids:
            allowed = set(req.load_ids)
            all_loads = [l for l in all_loads if l.load_id in allowed]
        if not all_loads:
            raise HTTPException(status_code=400, detail="No loads to plan. Ingest loads first.")

        plan = container.planner.build_plan(all_loads, truck)
        return PlanResponse(
            plan_id=plan.plan_id,
            truck_id=plan.truck_id,
            horizon_hours=plan.horizon_hours,
            stops=[PlanStopDTO(**s.__dict__) for s in plan.stops],
            expected_revenue=plan.expected_revenue,
            expected_cost=plan.expected_cost,
            expected_profit=plan.expected_profit,
            expected_deadhead_miles=plan.expected_deadhead_miles,
            expected_load_miles=plan.expected_load_miles,
            expected_deadhead_cost=plan.expected_deadhead_cost,
            expected_load_cost=plan.expected_load_cost,
            expected_toll_cost=plan.expected_toll_cost,
            expected_time_cost=plan.expected_time_cost,
            feasible=plan.feasible,
            score=plan.score,
            rationale=plan.rationale,
        )

    # -- Phase 4.4: human-in-the-loop bid approval workflow -------------------

    @app.post("/bids", response_model=BidDraftDTO)
    def create_bid_draft(req: CreateBidDraftRequest):
        truck = truck_from_dto(req.truck)
        try:
            draft = container.bid_approval_service.create_draft(
                truck, req.load_id, actor_id=req.actor_id
            )
        except LoadNotFoundForBid as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return bid_draft_to_dto(draft)

    @app.get("/bids", response_model=BidQueueResponse)
    def list_bid_drafts(status: str | None = Query(default=None)):
        status_filter = None
        if status is not None:
            try:
                status_filter = BidApprovalStatus(status)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"unknown status '{status}'")
        drafts = container.bid_approval_service.list_drafts(status_filter)
        return BidQueueResponse(bids=[bid_draft_to_dto(d) for d in drafts])

    @app.get("/bids/{bid_id}", response_model=BidDraftDTO)
    def get_bid_draft(bid_id: int):
        try:
            draft = container.bid_approval_service.get_draft(bid_id)
        except BidDraftNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return bid_draft_to_dto(draft)

    @app.patch("/bids/{bid_id}", response_model=BidDraftDTO)
    def edit_bid_draft(bid_id: int, req: EditBidRequest):
        try:
            draft = container.bid_approval_service.edit_draft(
                bid_id, req.amount, reason=req.reason, actor_id=req.actor_id
            )
        except BidDraftNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except InvalidBidTransition as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return bid_draft_to_dto(draft)

    @app.post("/bids/{bid_id}/approve", response_model=BidDraftDTO)
    def approve_bid_draft(bid_id: int, req: BidActionRequest | None = None):
        req = req or BidActionRequest()
        try:
            draft = container.bid_approval_service.approve_draft(
                bid_id, actor_id=req.actor_id, note=req.note
            )
        except BidDraftNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except InvalidBidTransition as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return bid_draft_to_dto(draft)

    @app.post("/bids/{bid_id}/reject", response_model=BidDraftDTO)
    def reject_bid_draft(bid_id: int, req: BidActionRequest | None = None):
        req = req or BidActionRequest()
        try:
            draft = container.bid_approval_service.reject_draft(
                bid_id, actor_id=req.actor_id, note=req.note
            )
        except BidDraftNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except InvalidBidTransition as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return bid_draft_to_dto(draft)

    @app.post("/bids/{bid_id}/submit-mock", response_model=BidDraftDTO)
    def submit_mock_bid_draft(bid_id: int, req: BidActionRequest | None = None):
        req = req or BidActionRequest()
        try:
            draft = container.bid_approval_service.submit_mock_draft(
                bid_id, actor_id=req.actor_id, note=req.note
            )
        except BidDraftNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except InvalidBidTransition as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return bid_draft_to_dto(draft)

    return app


app = create_app()
