"""Phase 8.2 - FleetEpisodeMetrics math + cross-episode aggregation.

Pure unit tests: ``FleetEpisodeMetrics`` is built from hand-made per-truck
``RollingReplayMetrics`` so the fleet-balance / concentration / aggregation math is
verified independently of the simulator.
"""
from simulation.fleet_metrics import FleetEpisodeMetrics, summarize_fleet_episodes
from simulation.metrics import RollingReplayMetrics


def _rrm(profit=0.0, deadhead=0.0, loaded=0.0, loads=0, idle=0.0, horizon=24.0):
    return RollingReplayMetrics(
        total_profit=profit,
        total_revenue=profit + 0.0,
        total_deadhead_miles=deadhead,
        total_loaded_miles=loaded,
        loads_completed=loads,
        idle_hours=idle,
        horizon_hours=horizon,
    )


def _fleet(trucks, **kw):
    return FleetEpisodeMetrics(horizon_hours=24.0, truck_metrics=trucks, **kw)


# ------------------------------------------------------------------ fleet totals
def test_fleet_totals_sum_across_trucks():
    m = _fleet([
        _rrm(profit=300.0, deadhead=40.0, loaded=400.0, loads=2),
        _rrm(profit=100.0, deadhead=60.0, loaded=200.0, loads=1),
    ])
    assert m.truck_count == 2
    assert m.total_profit == 400.0
    assert m.total_deadhead_miles == 100.0
    assert m.total_loaded_miles == 600.0
    assert m.loads_completed == 3
    assert m.deadhead_ratio == 100.0 / 600.0
    assert m.deadhead_per_load == 100.0 / 3
    assert m.profit_per_load == 400.0 / 3
    assert m.loads_per_truck == 1.5


# ------------------------------------------------------------------- utilization
def test_mean_and_min_utilization():
    m = _fleet([
        _rrm(idle=0.0, horizon=10.0),   # util 1.0
        _rrm(idle=5.0, horizon=10.0),   # util 0.5
    ])
    assert m.mean_utilization_rate == 0.75
    assert m.min_utilization_rate == 0.5


# --------------------------------------------------------------------- balance
def test_profit_dispersion():
    assert _fleet([_rrm(profit=100.0), _rrm(profit=300.0)]).profit_dispersion == 100.0
    # A perfectly balanced fleet has zero dispersion.
    assert _fleet([_rrm(profit=200.0), _rrm(profit=200.0)]).profit_dispersion == 0.0
    # Fewer than two trucks -> dispersion undefined -> 0.
    assert _fleet([_rrm(profit=200.0)]).profit_dispersion == 0.0


def test_profit_gini():
    uneven = _fleet([_rrm(profit=100.0), _rrm(profit=300.0)])
    assert abs(uneven.profit_gini - 0.25) < 1e-9
    even = _fleet([_rrm(profit=200.0), _rrm(profit=200.0)])
    assert even.profit_gini == 0.0
    # All-zero / empty fleets are defined as perfectly even (0).
    assert _fleet([_rrm(profit=0.0), _rrm(profit=0.0)]).profit_gini == 0.0
    assert _fleet([]).profit_gini == 0.0


# --------------------------------------------------- destination concentration
def test_destination_hhi():
    one_market = _fleet([_rrm()], destination_market_counts={"TX": 2})
    assert one_market.destination_hhi == 1.0
    split = _fleet([_rrm()], destination_market_counts={"TX": 1, "CA": 1})
    assert split.destination_hhi == 0.5
    skewed = _fleet([_rrm()], destination_market_counts={"TX": 3, "CA": 1})
    assert abs(skewed.destination_hhi - 0.625) < 1e-9
    assert _fleet([_rrm()]).destination_hhi == 0.0  # no loads run


# ------------------------------------------------------- artifact-gated risk
def test_risk_fields_default_none():
    m = _fleet([_rrm(profit=100.0)])
    assert m.expected_collectible_profit is None
    assert m.expected_default_exposure is None


# ------------------------------------------------------- cross-episode summary
def test_summarize_fleet_episodes():
    eps = [
        _fleet([_rrm(profit=300.0, loads=2), _rrm(profit=100.0, loads=1)]),
        _fleet([_rrm(profit=200.0, loads=1), _rrm(profit=200.0, loads=2)]),
    ]
    summary = summarize_fleet_episodes(eps)
    assert summary["episode_count"] == 2
    assert summary["metrics"]["total_profit"]["mean"] == 400.0
    assert summary["metrics"]["loads_completed"]["mean"] == 3.0
    assert "profit_dispersion" in summary["metrics"]
    assert "destination_hhi" in summary["metrics"]
    # No risk fields present -> pooled risk is None.
    assert summary["expected_collectible_profit"] is None


def test_summarize_pools_risk_when_present_in_all_episodes():
    a = _fleet([_rrm(profit=100.0)])
    b = _fleet([_rrm(profit=100.0)])
    a.expected_collectible_profit = 80.0
    a.expected_default_exposure = 20.0
    b.expected_collectible_profit = 120.0
    b.expected_default_exposure = 30.0
    summary = summarize_fleet_episodes([a, b])
    assert summary["expected_collectible_profit"]["mean"] == 100.0
    assert summary["expected_default_exposure"]["mean"] == 25.0
