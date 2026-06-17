"""Tests for the decision-time SnapshotBoard + ML->domain load adapter."""
from datetime import datetime, timedelta, timezone

from ml.data.load_history_schema import LoadSnapshotRecord
from simulation.snapshot_board import (
    EQUIPMENT_ML_TO_DOMAIN,
    SnapshotBoard,
    record_int_id,
)

T0 = datetime(2026, 1, 1, 8, 0, tzinfo=timezone.utc)
DENVER = (39.74, -104.99)
LA = (34.05, -118.24)


def make_record(seq=1, snapshot_time=T0, origin=DENVER, equipment="HS",
                total_rate=1200.0, pickup_offset_h=4.0, pickup_len_h=4.0,
                miles=300.0, load_views="low"):
    o_lat, o_lon = origin
    pickup_start = snapshot_time + timedelta(hours=pickup_offset_h)
    pickup_end = pickup_start + timedelta(hours=pickup_len_h)
    dropoff_start = pickup_end + timedelta(hours=miles / 50.0)
    dropoff_end = dropoff_start + timedelta(hours=3.0)
    return LoadSnapshotRecord(
        snapshot_time=snapshot_time,
        load_id=f"L-{seq:06d}",
        origin_city="Origin",
        origin_state="CO",
        origin_lat=o_lat,
        origin_lon=o_lon,
        destination_city="Dest",
        destination_state="TX",
        destination_lat=32.78,
        destination_lon=-96.80,
        pickup_start=pickup_start,
        pickup_end=pickup_end,
        dropoff_start=dropoff_start,
        dropoff_end=dropoff_end,
        equipment_type=equipment,
        loaded_miles=miles,
        posted_at=snapshot_time - timedelta(hours=2.0),
        total_rate=total_rate,
        load_views=load_views,
    )


def test_record_int_id_roundtrips_sequence():
    assert record_int_id("L-000042") == 42
    assert record_int_id("L-001000") == 1000


def test_visible_load_is_adapted_to_domain():
    board = SnapshotBoard([make_record(seq=7)])
    loads = board.visible_loads_at(T0, *DENVER, 250.0, "HS")
    assert len(loads) == 1
    load = loads[0]
    assert load.load_id == 7
    assert load.equipment_type == EQUIPMENT_ML_TO_DOMAIN["HS"]
    assert load.miles == 300.0
    assert load.total_rate == 1200.0
    # native record recoverable for the simulator
    assert board.record_for(7).load_id == "L-000007"


def test_call_for_rate_loads_are_dropped():
    board = SnapshotBoard([make_record(total_rate=None)])
    assert board.visible_loads_at(T0, *DENVER, 250.0, "HS") == []


def test_equipment_filter_is_exact():
    board = SnapshotBoard([make_record(seq=1, equipment="HS"),
                           make_record(seq=2, equipment="F")])
    hs = board.visible_loads_at(T0, *DENVER, 250.0, "HS")
    assert [l.load_id for l in hs] == [1]


def test_expired_pickup_window_is_hidden():
    # query 10h after the snapshot; a load whose pickup closed at +6h is gone.
    t = T0 + timedelta(hours=10)
    board = SnapshotBoard([
        make_record(seq=1, pickup_offset_h=2.0, pickup_len_h=4.0),   # ends +6h
        make_record(seq=2, pickup_offset_h=2.0, pickup_len_h=20.0),  # ends +22h
    ])
    visible = board.visible_loads_at(t, *DENVER, 250.0, "HS")
    assert [l.load_id for l in visible] == [2]


def test_radius_filter_excludes_far_origins():
    board = SnapshotBoard([make_record(seq=1, origin=DENVER),
                           make_record(seq=2, origin=LA)])
    near = board.visible_loads_at(T0, *DENVER, 250.0, "HS")
    assert [l.load_id for l in near] == [1]


def test_consumed_loads_are_not_revisited():
    board = SnapshotBoard([make_record(seq=1)])
    assert board.visible_loads_at(T0, *DENVER, 250.0, "HS")
    board.mark_consumed(1)
    assert board.visible_loads_at(T0, *DENVER, 250.0, "HS") == []


def test_active_and_next_snapshot_clock():
    s1 = T0
    s2 = T0 + timedelta(hours=3)
    board = SnapshotBoard([make_record(seq=1, snapshot_time=s1),
                           make_record(seq=2, snapshot_time=s2)])
    assert board.active_snapshot_time(s1 + timedelta(hours=1)) == s1
    assert board.active_snapshot_time(s2 + timedelta(hours=1)) == s2
    assert board.active_snapshot_time(s1 - timedelta(hours=1)) is None
    assert board.next_snapshot_after(s1) == s2
    assert board.next_snapshot_after(s2) is None


# --------------------------------------------------------- competition thinning
def test_competition_take_rate_zero_is_noop():
    records = [make_record(seq=i) for i in range(1, 21)]
    board = SnapshotBoard(records, competition_take_rate=0.0, competition_seed=1)
    assert board.competition_taken_ids == set()
    assert len(board.visible_loads_at(T0, *DENVER, 250.0, "HS")) == 20


def test_competition_thinning_removes_expected_fraction():
    records = [make_record(seq=i) for i in range(1, 101)]
    board = SnapshotBoard(records, competition_take_rate=0.4, competition_seed=7)
    assert len(board.competition_taken_ids) == 40
    visible = board.visible_loads_at(T0, *DENVER, 250.0, "HS")
    assert len(visible) == 60
    assert all(not board.is_competition_taken(l.load_id) for l in visible)


def test_competition_thinning_is_deterministic_per_seed():
    records = [make_record(seq=i) for i in range(1, 51)]
    a = SnapshotBoard(records, competition_take_rate=0.5, competition_seed=99)
    b = SnapshotBoard(records, competition_take_rate=0.5, competition_seed=99)
    assert a.competition_taken_ids == b.competition_taken_ids
    other = SnapshotBoard(records, competition_take_rate=0.5, competition_seed=100)
    assert other.competition_taken_ids != a.competition_taken_ids


def test_competition_thinning_is_fair_across_planners():
    # Two boards built for the same world with identical thinning args (one per
    # planner trajectory) must expose exactly the same visible board.
    records = [make_record(seq=i) for i in range(1, 41)]
    board_a = SnapshotBoard(records, competition_take_rate=0.3, competition_seed=5)
    board_b = SnapshotBoard(records, competition_take_rate=0.3, competition_seed=5)
    seen_a = {l.load_id for l in board_a.visible_loads_at(T0, *DENVER, 250.0, "HS")}
    seen_b = {l.load_id for l in board_b.visible_loads_at(T0, *DENVER, 250.0, "HS")}
    assert seen_a == seen_b


def test_competition_thinning_is_view_biased():
    # 50 heavily-viewed loads + 50 "be the first" loads; with a 50% take rate the
    # contested high-view loads should be removed far more often.
    high = [make_record(seq=i, load_views="high") for i in range(1, 51)]
    quiet = [make_record(seq=i, load_views="be_the_first") for i in range(51, 101)]
    board = SnapshotBoard(high + quiet, competition_take_rate=0.5, competition_seed=3)
    taken = board.competition_taken_ids
    taken_high = sum(1 for i in range(1, 51) if i in taken)
    taken_quiet = sum(1 for i in range(51, 101) if i in taken)
    assert taken_high > taken_quiet
    # the bias should be strong, not marginal
    assert taken_high >= 2 * taken_quiet
