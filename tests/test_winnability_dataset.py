"""Tests for the Phase 4.1 winnability dataset builder + leakage guard."""
import dataclasses
from dataclasses import fields

from ml.brokers import HIDDEN_BROKER_FIELDS
from ml.config import load_ml_config
from ml.data.build_winnability_dataset import build_winnability_dataset
from ml.data.load_history_schema import LoadSnapshotRecord, read_jsonl
from ml.data.outcome_schema import read_bid_trials, read_outcomes

# Latent ground-truth that may appear in labels but must never reach a snapshot.
_LATENT_NAMES = set(HIDDEN_BROKER_FIELDS) | {"reservation_rpm", "contention_intensity"}


def _cfg_into(tmp_path):
    cfg = load_ml_config()
    outcomes = dataclasses.replace(
        cfg.outcomes,
        snapshot_path=str(tmp_path / "snap.jsonl"),
        outcomes_path=str(tmp_path / "out.jsonl"),
        trials_path=str(tmp_path / "trials.jsonl"),
    )
    return dataclasses.replace(cfg, outcomes=outcomes)


def test_build_emits_three_artifacts_and_round_trips(tmp_path):
    cfg = _cfg_into(tmp_path)
    summary = build_winnability_dataset(cfg, days=4)

    snaps = read_jsonl(tmp_path / "snap.jsonl")
    outs = read_outcomes(tmp_path / "out.jsonl")
    trials = read_bid_trials(tmp_path / "trials.jsonl")

    assert summary["snapshots"] == len(snaps) > 0
    assert summary["outcomes"] == len(outs) == len(snaps)
    assert summary["trials"] == len(trials) > 0
    # A snapshot carries observable broker columns; an outcome reconstructs.
    assert snaps[0].broker_id is not None
    assert outs[0].load_id == snaps[0].load_id


def test_build_is_byte_reproducible(tmp_path):
    cfg1 = _cfg_into(tmp_path / "a")
    cfg2 = _cfg_into(tmp_path / "b")
    build_winnability_dataset(cfg1, days=4)
    build_winnability_dataset(cfg2, days=4)
    for name in ("snap.jsonl", "out.jsonl", "trials.jsonl"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_no_latent_fields_on_snapshot_record_or_jsonl(tmp_path):
    # (a) the dataclass itself exposes no latent attribute
    record_fields = {f.name for f in fields(LoadSnapshotRecord)}
    assert not (record_fields & _LATENT_NAMES)

    # (b) no latent key appears in a serialized snapshot line
    cfg = _cfg_into(tmp_path)
    build_winnability_dataset(cfg, days=4)
    snaps = read_jsonl(tmp_path / "snap.jsonl")
    keys = set(snaps[0].to_json_dict().keys())
    assert not (keys & _LATENT_NAMES)


def test_all_six_processes_represented(tmp_path):
    cfg = _cfg_into(tmp_path)
    build_winnability_dataset(cfg, days=6)
    outs = read_outcomes(tmp_path / "out.jsonl")
    trials = read_bid_trials(tmp_path / "trials.jsonl")

    payment = {o.payment_outcome for o in outs}
    assert "paid" in payment                          # brokers pay
    assert payment & {"late", "default"}              # brokers are risky
    assert {o.covered for o in outs} == {True, False}  # some loads disappear, some linger
    assert any(o.cover_censored for o in outs)        # censoring exercised
    assert {o.negotiation_required for o in outs} == {True, False}  # no-rate negotiation
    assert {t.won for t in trials} == {True, False}   # winnable vs not
    assert any(o.contention_intensity > 0 for o in outs)  # contention realized
