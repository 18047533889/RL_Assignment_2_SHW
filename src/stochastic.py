"""
stochastic.py — Stochastic placement after a player picks an aim cell.

**Quick read:** ``resolve_placement`` samples the spec (1/2 direct, 1/2 uniform 8-neighbour
branch; invalid/occupied neighbour => forfeit). ``forfeit_probability`` and
``placement_distribution`` are closed-form mirrors for tests.
``forfeit_probability_map`` returns a full (12,12) plane for the observation encoder.
"""

from __future__ import annotations

import numpy as np

from board import ROWS, COLS, VALID_MASK, is_valid

NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1),           (0, 1),
              (1, -1),  (1, 0),  (1, 1)]


def resolve_placement(board: np.ndarray, r: int, c: int,
                      rng: np.random.Generator | None = None
                      ) -> tuple[int, int] | None:
    if rng is None:
        rng = np.random.default_rng()

    if rng.random() < 0.5:
        return (r, c)

    idx = rng.integers(0, 8)
    dr, dc = NEIGHBOURS[idx]
    nr, nc = r + dr, c + dc

    if not is_valid(nr, nc):
        return None
    if board[nr, nc] != 0:
        return None

    return (nr, nc)


def forfeit_probability(board: np.ndarray, r: int, c: int) -> float:
    bad = 0
    for dr, dc in NEIGHBOURS:
        nr, nc = r + dr, c + dc
        if not is_valid(nr, nc) or board[nr, nc] != 0:
            bad += 1
    return 0.5 * bad / 8.0


def forfeit_probability_map(board: np.ndarray) -> np.ndarray:
    fp = np.zeros((ROWS, COLS), dtype=np.float32)
    for r in range(ROWS):
        for c in range(COLS):
            if VALID_MASK[r, c] and board[r, c] == 0:
                fp[r, c] = forfeit_probability(board, r, c)
    return fp


def placement_distribution(board: np.ndarray, r: int, c: int
                           ) -> dict[tuple[int, int] | None, float]:
    dist: dict[tuple[int, int] | None, float] = {(r, c): 0.5}
    forfeit_p = 0.0
    for dr, dc in NEIGHBOURS:
        nr, nc = r + dr, c + dc
        if not is_valid(nr, nc) or board[nr, nc] != 0:
            forfeit_p += 1.0 / 16.0
        else:
            dist[(nr, nc)] = dist.get((nr, nc), 0.0) + 1.0 / 16.0
    if forfeit_p > 0:
        dist[None] = forfeit_p
    return dist
