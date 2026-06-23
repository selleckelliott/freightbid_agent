"""Phase 7.2 — sandbox/replay load board + the thin live ``pull`` ingest flow.

Covers: sandbox determinism, that boards emit *raw* external dicts (anti-corruption boundary) which the
7.1 contract accepts cleanly, replay cursor paging + fail-closed on a missing/unreadable feed, the
``LoadBoardIngestService`` (pull / replace / unavailable no-op), container wiring, the additive
``POST /loads/pull`` endpoint, and a regression guard that the synthetic ``POST /loads`` ingress is
unchanged.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from adapters.inbound.api.app import create_app
from adapters.inbound.api.container import build_container
from adapters.outbound.load_board.replay import RecordedLoadBoardReplayAdapter
from adapters.outbound.load_board.sandbox import SandboxLoadBoardAdapter
from adapters.outbound.memory.load_repository import InMemoryLoadRepository
from application.config_loader import LoadBoardConfig, load_load_board_config
from application.ingestion.board_ingest import LoadBoardIngestService
from application.ingestion.import_contract import validate_loads
from domain.models.load import Load
from ports.load_board import LoadBoardUnavailable

ROOT = Path(__file__).resolve().parents[1]
FEED = ROOT / "sample_data" / "external" / "recorded_feed.json"


def _dummy_load(load_id: int) -> Load:
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return Load(
        load_id=load_id, weight=10000.0, created_at=now,
        origin_city="X", origin_state="TX", origin_latitude=30.0, origin_longitude=-95.0,
        destination_city="Y", destination_state="GA", destination_latitude=33.0, destination_longitude=-84.0,
        pickup_window_start=now, pickup_window_end=now, delivery_window_start=now, delivery_window_end=now,
        miles=100.0, total_rate=500.0, equipment_type="Dry Van",
    )


# --------------------------------------------------------------------------- #
# Sandbox board
# --------------------------------------------------------------------------- #
def test_sandbox_is_deterministic():
    a = SandboxLoadBoardAdapter(seed=7, count=12).fetch_raw()
    b = SandboxLoadBoardAdapter(seed=7, count=12).fetch_raw()
    c = SandboxLoadBoardAdapter(seed=99, count=12).fetch_raw()
    assert a.count == 12
    assert a.rows == b.rows  # same seed -> byte-identical rows
    assert a.rows != c.rows  # different seed -> different rows


def test_sandbox_emits_raw_external_dicts_not_domain_loads():
    batch = SandboxLoadBoardAdapter(seed=7, count=3).fetch_raw()
    for row in batch.rows:
        assert isinstance(row, dict)
        assert not isinstance(row, Load)
        # external dialect keys (aliases), not the domain field names
        assert "posting_id" in row
        assert "trip_miles" in row
        assert "load_id" not in row


def test_sandbox_rows_validate_clean_through_contract():
    batch = SandboxLoadBoardAdapter(seed=7, count=12).fetch_raw()
    result = validate_loads(batch.rows)
    assert result.accepted == 12
    assert result.rejected == 0
    load = result.loads[0]
    assert load.miles > 0
    assert load.total_rate > 0
    assert len(load.origin_state) == 2 and load.origin_state.isupper()
    assert load.equipment_type[0].isupper()  # normalized, not a raw code like "v"


def test_sandbox_fetch_respects_limit():
    board = SandboxLoadBoardAdapter(seed=7, count=12)
    assert board.fetch_raw(limit=5).count == 5
    assert board.fetch_raw(limit=0).count == 0
    assert board.fetch_raw().count == 12


def test_sandbox_count_zero_is_unavailable():
    board = SandboxLoadBoardAdapter(seed=7, count=0)
    avail = board.availability()
    assert avail.available is False
    assert avail.reason == "no_feed"
    assert board.fetch_raw().count == 0


# --------------------------------------------------------------------------- #
# Replay board
# --------------------------------------------------------------------------- #
def test_replay_reads_recorded_feed_and_validates():
    board = RecordedLoadBoardReplayAdapter(FEED)
    assert board.availability().available is True
    assert board.total_rows == 5
    result = validate_loads(board.fetch_raw().rows)
    assert result.accepted == 5
    assert result.rejected == 0


def test_replay_cursor_pages_then_exhausts_and_resets():
    board = RecordedLoadBoardReplayAdapter(FEED)
    p1 = board.fetch_raw(limit=2)
    p2 = board.fetch_raw(limit=2)
    p3 = board.fetch_raw(limit=2)
    assert (p1.count, p2.count, p3.count) == (2, 2, 1)
    assert p1.exhausted is False and p3.exhausted is True
    # cursor advanced; a further pull yields nothing
    assert board.fetch_raw().count == 0
    # reset rewinds the replay
    board.reset()
    assert board.fetch_raw(limit=2).count == 2


def test_replay_missing_feed_fails_closed():
    board = RecordedLoadBoardReplayAdapter(ROOT / "sample_data" / "external" / "does_not_exist.json")
    avail = board.availability()
    assert avail.available is False
    assert avail.reason == "no_feed"
    try:
        board.fetch_raw()
        assert False, "expected LoadBoardUnavailable"
    except LoadBoardUnavailable as exc:
        assert exc.reason == "no_feed"


def test_replay_unreadable_feed_fails_closed(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ this is not valid json", encoding="utf-8")
    board = RecordedLoadBoardReplayAdapter(bad)
    assert board.availability().available is False
    assert board.availability().reason == "no_feed"


# --------------------------------------------------------------------------- #
# Ingest service (board -> validate -> repo)
# --------------------------------------------------------------------------- #
def test_ingest_service_pull_persists_accepted_loads():
    repo = InMemoryLoadRepository()
    svc = LoadBoardIngestService(SandboxLoadBoardAdapter(seed=7, count=12), repo)
    report = svc.pull(limit=5)
    assert report.available is True
    assert report.fetched == 5
    assert report.accepted == 5
    assert report.rejected == 0
    assert len(repo.list_all()) == 5
    assert report.load_ids == [l.load_id for l in repo.list_all()]


def test_ingest_service_replace_clears_first():
    repo = InMemoryLoadRepository()
    repo.add_many([_dummy_load(7777)])
    svc = LoadBoardIngestService(SandboxLoadBoardAdapter(seed=7, count=12), repo)

    # without replace, the sentinel survives alongside the pulled loads
    svc.pull(limit=3)
    assert repo.get(7777) is not None
    assert len(repo.list_all()) == 4

    # with replace, the repo is cleared before ingest
    report = svc.pull(replace=True)
    assert report.replaced is True
    assert repo.get(7777) is None
    assert len(repo.list_all()) == 12


def test_ingest_service_unavailable_board_is_noop():
    repo = InMemoryLoadRepository()
    repo.add_many([_dummy_load(7777)])
    svc = LoadBoardIngestService(SandboxLoadBoardAdapter(seed=7, count=0), repo)
    report = svc.pull()
    assert report.available is False
    assert report.reason == "no_feed"
    assert report.accepted == 0
    # repo untouched — the sentinel is still there
    assert repo.get(7777) is not None
    assert len(repo.list_all()) == 1


# --------------------------------------------------------------------------- #
# Container wiring + config
# --------------------------------------------------------------------------- #
def test_container_wires_sandbox_load_board():
    container = build_container(ROOT / "config")
    assert container.load_board_config.source == "sandbox"
    assert container.load_board.availability().available is True
    assert container.load_board_ingest is not None


def test_config_loader_defaults_and_override(tmp_path):
    # default (real config dir) -> sandbox
    cfg = load_load_board_config(ROOT / "config")
    assert isinstance(cfg, LoadBoardConfig)
    assert cfg.source == "sandbox"

    # a replay override parses
    (tmp_path / "load_board.yaml").write_text(
        "load_board:\n  source: replay\n  feed_path: feeds/x.json\n", encoding="utf-8"
    )
    override = load_load_board_config(tmp_path)
    assert override.source == "replay"
    assert override.feed_path == "feeds/x.json"


# --------------------------------------------------------------------------- #
# API: additive /loads/pull, and the synthetic /loads ingress unchanged
# --------------------------------------------------------------------------- #
def _client():
    return TestClient(create_app(build_container(ROOT / "config")))


def test_pull_endpoint_ingests_and_lists():
    c = _client()
    r = c.post("/loads/pull", json={"replace": True})
    assert r.status_code == 200
    data = r.json()
    assert data["source"] == "sandbox"
    assert data["available"] is True
    assert data["accepted"] == data["fetched"] > 0
    assert data["rejected"] == 0

    listed = c.get("/loads")
    assert listed.status_code == 200
    assert len(listed.json()) == data["accepted"]


def test_pull_endpoint_accepts_empty_body():
    c = _client()
    r = c.post("/loads/pull")
    assert r.status_code == 200
    assert r.json()["accepted"] > 0


def test_existing_loads_post_is_unchanged():
    """Regression: the synthetic ingress is byte-identical (Phase 7 anti-corruption guarantee)."""
    c = _client()
    payload = json.loads((ROOT / "sample_data" / "loads.json").read_text())
    r = c.post("/loads", json=payload)
    assert r.status_code == 200
    assert r.json() == {"accepted": len(payload["loads"])}
