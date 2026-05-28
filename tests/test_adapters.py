from adapters.outbound.distance.haversine import HaversineDistanceProvider
from adapters.outbound.tolls.flat_rate import FlatRateTollEstimator


def test_haversine_dallas_houston():
    d = HaversineDistanceProvider().miles_between(32.7767, -96.7970, 29.7604, -95.3698)
    assert 200 < d < 260


def test_tolls_average_for_known_states():
    t = FlatRateTollEstimator().estimate(500, "TX", "OK")
    assert t == 500 * (0 + 0.08) / 2


def test_tolls_high_in_ny_nj():
    t = FlatRateTollEstimator().estimate(100, "NY", "NJ")
    assert t == 100 * 0.18
