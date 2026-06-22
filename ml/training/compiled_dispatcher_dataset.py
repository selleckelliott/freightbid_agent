"""Loading + splitting for the compiled-dispatcher dataset (Phase 6.3).

Reads the Phase 6.2 structured rows (``data/compiled_dispatcher_dataset.jsonl``) and produces a
deterministic, action-stratified train / validation / test split. Rows are sorted by ``scenario_id``
first so the split is reproducible regardless of file order. Stratification keeps the minority
``no_bid`` / ``approval_required`` actions present in every slice; when a class is too small to
stratify a three-way split (tiny in-process test batches), it falls back to a plain shuffled split.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sklearn.model_selection import train_test_split

Row = Dict[str, Any]


def load_rows(path: str | Path, *, limit: Optional[int] = None) -> List[Row]:
    rows: List[Row] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _action_labels(rows: Sequence[Row]) -> List[str]:
    return [r["targets"]["decision"] for r in rows]


def _can_stratify(labels: Sequence[str], min_per_class: int = 4) -> bool:
    counts = Counter(labels)
    return len(counts) >= 2 and min(counts.values()) >= min_per_class


def split_rows(
    rows: Sequence[Row],
    *,
    seed: int = 63,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> Tuple[List[Row], List[Row], List[Row]]:
    """Deterministic action-stratified ``(train, validation, test)`` split."""
    ordered = sorted(rows, key=lambda r: r["scenario_id"])
    labels = _action_labels(ordered)
    strat = labels if _can_stratify(labels) else None

    train_val, test = train_test_split(
        ordered, test_size=test_fraction, random_state=seed, stratify=strat
    )
    tv_labels = _action_labels(train_val)
    strat2 = tv_labels if _can_stratify(tv_labels) else None
    rel_val = val_fraction / (1.0 - test_fraction)
    train, val = train_test_split(
        train_val, test_size=rel_val, random_state=seed, stratify=strat2
    )
    return train, val, test
