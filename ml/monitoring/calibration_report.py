"""Config loading + multi-world aggregation/formatting for the calibration monitor (5.3).

Keeps YAML / presentation concerns out of :mod:`ml.monitoring.calibration_drift` (the pure
metric core). Provides:

* :func:`load_calibration_config` — read ``config/calibration_monitor.yaml`` into a
  :class:`~ml.monitoring.calibration_drift.CalibrationThresholds` + a small ``extras`` dict
  (which target to monitor, where the stress worlds live).
* :func:`report_to_dict` — a :class:`CalibrationReport` as a JSON-ready dict.
* :func:`severity_tally`, :func:`headline`, :func:`format_table` — roll a set of per-world
  reports into the committed summary + a console view.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml

from ml.monitoring.calibration_drift import (
    ALERT,
    OK,
    WATCH,
    CalibrationReport,
    CalibrationThresholds,
)

_DEFAULT_CONDITIONS = "config/broker_quality_stress.yaml"


def load_calibration_config(path: str | Path) -> Tuple[CalibrationThresholds, Dict[str, Any]]:
    """Parse the ``calibration_monitor:`` block into thresholds + extras.

    Unknown keys are ignored; every threshold falls back to its dataclass default, so a
    partial (or empty) config still yields a usable monitor. ``extras`` carries the
    ``target`` (which model to monitor) and the ``conditions`` path (the stress worlds).
    """
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    block = doc.get("calibration_monitor", doc) or {}
    defaults = CalibrationThresholds()
    thresholds = CalibrationThresholds(
        ece_watch=float(block.get("ece_watch_threshold", defaults.ece_watch)),
        ece_alert=float(block.get("ece_alert_threshold", defaults.ece_alert)),
        bias_watch=float(block.get("bias_watch_threshold", defaults.bias_watch)),
        bias_alert=float(block.get("bias_alert_threshold", defaults.bias_alert)),
        min_samples=int(block.get("min_samples", defaults.min_samples)),
        n_bins=int(block.get("n_bins", defaults.n_bins)),
    )
    extras = {
        "target": str(block.get("target", "winnability")),
        "conditions": str(block.get("conditions", _DEFAULT_CONDITIONS)),
    }
    return thresholds, extras


def report_to_dict(report: CalibrationReport) -> Dict[str, Any]:
    """The report as a plain JSON-serializable dict."""
    return asdict(report)


def severity_tally(reports: Iterable[CalibrationReport]) -> Dict[str, int]:
    """Count of ``OK`` / ``WATCH`` / ``ALERT`` verdicts across ``reports``."""
    tally = {OK: 0, WATCH: 0, ALERT: 0}
    for r in reports:
        tally[r.severity] = tally.get(r.severity, 0) + 1
    return tally


def headline(tally: Dict[str, int], total: int) -> str:
    """One-line summary of the severity tally."""
    return (
        f"Calibration monitor: {tally.get(ALERT, 0)} ALERT, {tally.get(WATCH, 0)} WATCH, "
        f"{tally.get(OK, 0)} OK across {total} worlds."
    )


_SEVERITY_TAG = {OK: "OK    ", WATCH: "WATCH ", ALERT: "ALERT "}


def format_table(records: List[Dict[str, Any]]) -> str:
    """Render per-world monitor records as a fixed-width console table."""
    header = (
        f"{'world':<19} {'sev':<6} {'n':>6} {'ece':>7} {'bias':>8} "
        f"{'ece_drift':>10} {'bias_drift':>11}"
    )
    lines = [header, "-" * len(header)]
    for r in records:
        tag = _SEVERITY_TAG.get(r["severity"], r["severity"])
        ece = "  n/a" if r.get("ece") is None else f"{r['ece']:.4f}"
        bias = "    n/a" if r.get("bias") is None else f"{r['bias']:+.4f}"
        ed = r.get("ece_drift")
        bd = r.get("bias_drift")
        ed_s = "      n/a" if ed is None else f"{ed:+.4f}"
        bd_s = "       n/a" if bd is None else f"{bd:+.4f}"
        flag = " (low-n)" if r.get("insufficient_data") else ""
        lines.append(
            f"{r['name']:<19} {tag} {r.get('n', 0):>6} {ece:>7} {bias:>8} "
            f"{ed_s:>10} {bd_s:>11}{flag}"
        )
    return "\n".join(lines)
