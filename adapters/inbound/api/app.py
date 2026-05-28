import os
from pathlib import Path

from fastapi import FastAPI, HTTPException

from adapters.inbound.api.container import build_container
from adapters.inbound.api.mappers import load_from_dto, truck_from_dto
from adapters.inbound.api.schemas import (
    BidRangeDTO,
    IngestRequest,
    IngestResponse,
    PlanResponse,
    PlanStopDTO,
    RankRequest,
    RankResponse,
    RankedLoad,
)

CONFIG_DIR = Path(os.environ.get("FREIGHTBID_CONFIG_DIR", Path(__file__).resolve().parents[3] / "config"))


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
            bid = container.bid_recommender.recommend(evaluation)
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
                    bid=BidRangeDTO(**bid.__dict__),
                )
            )
        return RankResponse(truck_id=truck.truck_id, ranked=items)

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

    return app


app = create_app()
