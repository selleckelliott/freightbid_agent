"""Tests for the synthetic load-history generator (Phase 3.1)."""
from datetime import datetime, timezone

from ml.data.load_history_schema import read_jsonl, write_jsonl
from ml.data.synthetic_history_generator import GeneratorParams, generate_history
from ml.markets import HOTSHOT_EQUIPMENT, LOAD_MODES, LOAD_VIEW_BUCKETS


def _params(**overrides) -> GeneratorParams:
    base = dict(
        start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
        days=3,
        snapshots_per_day=4,
        loads_per_snapshot_mean=20.0,
        unposted_rate_fraction=0.2,
        max_post_age_hours=12.0,
        seed=7,
    )
    base.update(overrides)
    return GeneratorParams(**base)


def test_generates_records():
    records = generate_history(_params())
    assert len(records) > 0


def test_seed_is_deterministic():
    a = generate_history(_params())
    b = generate_history(_params())
    assert [r.load_id for r in a] == [r.load_id for r in b]
    assert [r.total_rate for r in a] == [r.total_rate for r in b]


def test_required_fields_valid_and_positive():
    for r in generate_history(_params()):
        assert r.loaded_miles > 0
        assert r.equipment_type in HOTSHOT_EQUIPMENT
        assert r.pickup_start < r.dropoff_start
        assert r.posted_at <= r.snapshot_time
        if r.total_rate is not None:
            assert r.total_rate > 0
        # Phase 3.1.1 board fields.
        assert r.weight > 0
        assert r.length > 0
        assert r.mode in LOAD_MODES
        assert r.load_views in LOAD_VIEW_BUCKETS
        assert r.width is None or r.width > 0
        assert r.height is None or r.height > 0


def test_some_loads_have_no_posted_rate():
    records = generate_history(_params(unposted_rate_fraction=0.5))
    assert any(r.total_rate is None for r in records)
    assert any(r.total_rate is not None for r in records)


def test_jsonl_roundtrip(tmp_path):
    records = generate_history(_params())
    path = tmp_path / "history.jsonl"
    written = write_jsonl(records, path)
    restored = read_jsonl(path)

    assert written == len(records) == len(restored)
    assert restored[0].load_id == records[0].load_id
    assert restored[0].snapshot_time == records[0].snapshot_time
    assert restored[3].total_rate == records[3].total_rate
    assert restored[3].equipment_type == records[3].equipment_type
    # Phase 3.1.1 board fields survive the round-trip.
    assert restored[3].weight == records[3].weight
    assert restored[3].mode == records[3].mode
    assert restored[3].load_views == records[3].load_views
    assert restored[3].width == records[3].width
