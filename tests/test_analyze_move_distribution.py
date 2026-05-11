"""analyze_move_distribution.py — CSV/heatmap helpers (no GPU checkpoint required)."""

import csv
import os

import numpy as np
import pytest

from board import NUM_VALID, VALID_POSITIONS


def test_write_csv_roundtrip(tmp_path):
    from analyze_move_distribution import write_csv

    counts = np.zeros(NUM_VALID, dtype=np.int64)
    counts[0] = 10
    counts[5] = 3
    p = os.path.join(tmp_path, "m.csv")
    write_csv(counts, p)
    with open(p, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == NUM_VALID
    assert rows[0]["compact_idx"] == "0"
    assert int(rows[0]["count"]) == 10
    assert rows[0]["row"] == str(VALID_POSITIONS[0][0])
    assert rows[0]["col"] == str(VALID_POSITIONS[0][1])


def test_entropy_print_smoke():
    """Distribution entropy helper used in main()."""
    counts = np.ones(NUM_VALID, dtype=np.int64)
    pvec = counts.astype(np.float64) / counts.sum()
    ent = -np.sum(pvec * np.log(pvec + 1e-20))
    assert ent == pytest.approx(np.log(NUM_VALID), rel=1e-5)
