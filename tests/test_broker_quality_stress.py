"""Tests for the Phase 4.5 broker-quality stress sweep.

Fast, pure tests for config validation / lens derivation / verdict thresholds /
config perturbation, plus one small seeded end-to-end smoke that proves the sweep
is deterministic and that the baseline (training) world reproduces the Phase 4.3
ordering (EV target beats the best fixed policy).
"""
import json
from dataclasses import replace

import pytest

from application.config_loader import load_bid_recommender_config
from benchmarks.run_broker_quality_stress import (
    DEFAULT_CONDITIONS,
    VERDICT_HOLDS,
    VERDICT_NEUTRAL,
    VERDICT_REGRESSION,
    Condition,
    load_conditions,
    run_sweep,
    world_cfg,
    _verdict,
)
from ml.config import load_ml_config
from ml.data.build_winnability_dataset import build_winnability_dataset


def _tiny_cfg():
    """A small synthetic world so the dataset-building smokes stay fast (seconds)."""
    cfg = load_ml_config()
    return replace(
        cfg,
        synthetic_data=replace(cfg.synthetic_data, loads_per_snapshot_mean=10.0,
                               snapshots_per_day=4),
    )



# --------------------------------------------------------------------------- #
# Config validation
# --------------------------------------------------------------------------- #

def test_default_conditions_parse_and_have_rationales():
    conditions = load_conditions(DEFAULT_CONDITIONS)
    names = [c.name for c in conditions]
    assert "baseline" in names
    assert len(set(names)) == len(names)  # unique
    for c in conditions:
        if c.overrides:  # baseline may omit a rationale, shifts must justify themselves
            assert c.rationale, f"{c.name} is missing a rationale"


def test_baseline_has_no_overrides_and_is_reference_lens():
    baseline = next(c for c in load_conditions(DEFAULT_CONDITIONS) if c.name == "baseline")
    assert baseline.overrides == {}
    assert baseline.lens() == "reference"


def _write(tmp_path, body):
    path = tmp_path / "conds.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_unknown_top_level_key_rejected(tmp_path):
    path = _write(tmp_path, "conditions:\n  - name: bad\n    nonsense: 1\n")
    with pytest.raises(ValueError, match="unknown keys"):
        load_conditions(path)


def test_unknown_knob_in_section_rejected(tmp_path):
    path = _write(
        tmp_path,
        "conditions:\n  - name: bad\n    brokers:\n      not_a_real_knob: 1\n",
    )
    with pytest.raises(ValueError, match="unknown knobs"):
        load_conditions(path)


def test_io_path_fields_are_not_overridable(tmp_path):
    path = _write(
        tmp_path,
        "conditions:\n  - name: bad\n    outcomes:\n      outcomes_path: x.jsonl\n",
    )
    with pytest.raises(ValueError, match="unknown knobs"):
        load_conditions(path)


def test_non_mapping_section_rejected(tmp_path):
    path = _write(tmp_path, "conditions:\n  - name: bad\n    brokers: 5\n")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_conditions(path)


# --------------------------------------------------------------------------- #
# Lens derivation
# --------------------------------------------------------------------------- #

def test_lens_classification():
    ev = Condition("c", "", {"synthetic_data": {"unposted_rate_fraction": 0.4}})
    cal = Condition("c", "", {"brokers": {"unknown_credit_fraction": 0.4}})
    both = Condition(
        "c", "",
        {"synthetic_data": {"unposted_rate_fraction": 0.4},
         "brokers": {"default_prob_worst": 0.3}},
    )
    assert ev.lens() == "ev"
    assert cal.lens() == "calibration"
    assert both.lens() == "both"
    # An outcomes EV knob is EV; an outcomes payment/coverage knob is calibration.
    assert Condition("c", "", {"outcomes": {"reservation_center_mult": 0.9}}).lens() == "ev"
    assert Condition("c", "", {"outcomes": {"late_pay_threshold_days": 30.0}}).lens() == "calibration"


def test_named_conditions_have_expected_lenses():
    by_name = {c.name: c for c in load_conditions(DEFAULT_CONDITIONS)}
    assert by_name["no_rate_heavy"].lens() == "ev"
    assert by_name["high_contention"].lens() == "ev"
    assert by_name["slow_pay"].lens() == "calibration"
    assert by_name["unknown_credit"].lens() == "calibration"
    assert by_name["degraded_corner"].lens() == "both"


# --------------------------------------------------------------------------- #
# Verdict thresholds (+/-1% band)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "uplift,expected",
    [
        (5.0, VERDICT_HOLDS),
        (1.0, VERDICT_HOLDS),       # inclusive lower edge
        (0.5, VERDICT_NEUTRAL),
        (0.0, VERDICT_NEUTRAL),
        (-0.5, VERDICT_NEUTRAL),
        (-1.0, VERDICT_REGRESSION),  # inclusive edge
        (-7.3, VERDICT_REGRESSION),
    ],
)
def test_verdict_band(uplift, expected):
    assert _verdict(uplift) == expected


# --------------------------------------------------------------------------- #
# Config perturbation
# --------------------------------------------------------------------------- #

def test_world_cfg_applies_overrides_and_redirects_io(tmp_path):
    cfg = load_ml_config()
    overrides = {
        "synthetic_data": {"unposted_rate_fraction": 0.45},
        "brokers": {"default_prob_worst": 0.32},
        "outcomes": {"reservation_center_mult": 0.92},
    }
    world = world_cfg(cfg, tmp_path, overrides)

    assert world.synthetic_data.unposted_rate_fraction == 0.45
    assert world.brokers.default_prob_worst == 0.32
    assert world.outcomes.reservation_center_mult == 0.92
    # Untouched knobs and all seeds are preserved (Common Random Numbers).
    assert world.outcomes.win_logistic_scale_rpm == cfg.outcomes.win_logistic_scale_rpm
    assert world.synthetic_data.seed == cfg.synthetic_data.seed
    assert world.brokers.seed == cfg.brokers.seed
    assert world.outcomes.seed == cfg.outcomes.seed
    # IO is redirected under the temp dir.
    assert str(tmp_path) in world.outcomes.outcomes_path
    assert str(tmp_path) in world.outcomes.snapshot_path


def test_perturbed_world_builds_a_valid_dataset(tmp_path):
    world = world_cfg(_tiny_cfg(), tmp_path, {"synthetic_data": {"unposted_rate_fraction": 0.5}})
    summary = build_winnability_dataset(world, days=2)
    assert summary["snapshots"] > 0
    assert summary["trials"] > 0
    assert (tmp_path / "out.jsonl").exists()


# --------------------------------------------------------------------------- #
# End-to-end smoke: deterministic + baseline reproduces 4.3 ordering
# --------------------------------------------------------------------------- #

def test_sweep_is_deterministic_and_baseline_holds():
    cfg = _tiny_cfg()
    bid_cfg = load_bid_recommender_config("config")
    conditions = [
        Condition("baseline", "", {}),
        Condition("no_rate_heavy", "", {"synthetic_data": {"unposted_rate_fraction": 0.45}}),
    ]

    kwargs = dict(days=3, max_loads=40)
    first = run_sweep(cfg, bid_cfg, conditions, **kwargs)
    second = run_sweep(cfg, bid_cfg, conditions, **kwargs)

    # Determinism: identical seeds -> byte-identical records.
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)

    base = next(r for r in first if r["name"] == "baseline")
    assert base["lens"] == "reference"
    assert base["calibration_drift_vs_baseline"] == 0.0
    # Baseline = the training world; the EV target must beat the best fixed policy
    # (reproduces the Phase 4.3 ordering -> a positive uplift).
    assert base["target_realized_profit"] > base["best_fixed_realized_profit"]
    assert base["uplift_pct"] > 0.0

    # Every record carries both lenses' fields.
    for r in first:
        assert set(r) >= {
            "name", "lens", "verdict", "uplift_pct", "n_loads",
            "target_realized_profit", "best_fixed_realized_profit",
            "model_win_prob", "oracle_win_prob", "calibration_gap",
            "calibration_drift_vs_baseline",
        }
