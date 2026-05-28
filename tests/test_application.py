def test_evaluator_populates_costs_and_etas(container, sample_loads, sample_truck):
    evals = container.evaluator.evaluate_loads(sample_loads, sample_truck)
    assert len(evals) == len(sample_loads)
    for e in evals:
        assert e.total_cost > 0
        assert e.deadhead_cost >= 0
        assert e.load_cost > 0
        assert e.time_cost > 0
        assert e.pickup_eta is not None
        assert e.delivery_eta is not None
        assert e.expected_profit == e.expected_revenue - e.total_cost


def test_recommender_returns_topn_with_rationale(container, sample_loads, sample_truck):
    ranked = container.recommender.recommend_loads(sample_loads, sample_truck, top_n=10)
    assert len(ranked) >= 1
    assert len(ranked) <= 10
    scores = [s.score for _, s in ranked]
    assert scores == sorted(scores, reverse=True)
    for _, score in ranked:
        assert score.rationale and "score=" in score.rationale


def test_recommender_filters_wrong_trailer(container, sample_loads, sample_truck):
    ranked = container.recommender.recommend_loads(sample_loads, sample_truck, top_n=10)
    assert 4 not in [s.load_id for _, s in ranked]


def test_plan_builder_sequences_loads(container, sample_loads, sample_truck):
    plan = container.planner.build_plan(sample_loads, sample_truck)
    assert plan.feasible
    assert len(plan.stops) >= 1
    assert plan.expected_revenue > 0
    assert abs(plan.expected_profit - (plan.expected_revenue - plan.expected_cost)) < 1e-6
    assert plan.rationale


def test_bid_recommender_orders_min_target_max(container, sample_loads, sample_truck):
    eval_one = container.evaluator.evaluate_one(sample_loads[0], sample_truck)
    bid = container.bid_recommender.recommend(eval_one)
    assert bid.min_bid <= bid.target_bid <= bid.max_bid
    assert bid.breakeven == eval_one.total_cost
    assert "target margin" in bid.rationale


def test_inmemory_load_repo_roundtrip(container, sample_loads):
    container.load_repo.clear()
    container.load_repo.add_many(sample_loads)
    assert len(container.load_repo.list_all()) == len(sample_loads)
    assert container.load_repo.get(sample_loads[0].load_id) is not None
