"""
board.py — Board representation for the triangular Super Tic-Tac-Toe.

**Quick read:** 96 valid triangle cells embedded in a 12x12 grid. The RL action space
uses a **compact 96-dim** representation via ``POS_TO_INDEX`` / ``INDEX_TO_POS``.
``compact_to_rc`` / ``rc_to_compact`` convert between compact indices and (row, col).
``get_level`` / ``crosses_level_boundary`` implement column-win geometry for ``rules.py``.

The physical board is an inverted triangle of 6 sub-boards (each 4x4):
    Level 1 (top):  1 block  -> rows 0-3,  cols 4-7
    Level 2 (mid):  2 blocks -> rows 4-7,  cols 2-9
    Level 3 (bot):  3 blocks -> rows 8-11, cols 0-11

We embed this into a 12x12 grid.  Cells outside the triangle are invalid.
Total valid cells: 16 + 32 + 48 = 96.
"""

from __future__ import annotations

import numpy as np

ROWS = 12
COLS = 12
GRID_SIZE = ROWS * COLS  # 144

LEVEL_BOUNDS: list[tuple[int, int, int, int]] = [
    (0, 4, 4, 8),   # level 1: rows [0,4), cols [4,8)
    (4, 8, 2, 10),  # level 2: rows [4,8), cols [2,10)
    (8, 12, 0, 12), # level 3: rows [8,12), cols [0,12)
]

LEVEL_ROW_RANGES: list[tuple[int, int]] = [
    (0, 4), (4, 8), (8, 12),
]


def _build_valid_mask() -> np.ndarray:
    mask = np.zeros((ROWS, COLS), dtype=np.bool_)
    for r_lo, r_hi, c_lo, c_hi in LEVEL_BOUNDS:
        mask[r_lo:r_hi, c_lo:c_hi] = True
    return mask


VALID_MASK: np.ndarray = _build_valid_mask()
VALID_POSITIONS: list[tuple[int, int]] = [
    (r, c) for r in range(ROWS) for c in range(COLS) if VALID_MASK[r, c]
]
NUM_VALID = len(VALID_POSITIONS)  # 96

POS_TO_INDEX: dict[tuple[int, int], int] = {
    pos: idx for idx, pos in enumerate(VALID_POSITIONS)
}
INDEX_TO_POS: dict[int, tuple[int, int]] = {
    idx: pos for idx, pos in enumerate(VALID_POSITIONS)
}


def rc_to_flat(r: int, c: int) -> int:
    return r * COLS + c


def flat_to_rc(flat: int) -> tuple[int, int]:
    return divmod(flat, COLS)


def compact_to_rc(compact: int) -> tuple[int, int]:
    return INDEX_TO_POS[compact]


def rc_to_compact(r: int, c: int) -> int:
    return POS_TO_INDEX[(r, c)]


def compact_to_flat(compact: int) -> int:
    r, c = INDEX_TO_POS[compact]
    return rc_to_flat(r, c)


def flat_to_compact(flat: int) -> int:
    r, c = flat_to_rc(flat)
    return POS_TO_INDEX[(r, c)]


COMPACT_TO_FLAT: np.ndarray = np.array(
    [compact_to_flat(i) for i in range(NUM_VALID)], dtype=np.int64
)
FLAT_TO_COMPACT: np.ndarray = np.full(GRID_SIZE, -1, dtype=np.int64)
for i in range(NUM_VALID):
    FLAT_TO_COMPACT[COMPACT_TO_FLAT[i]] = i


def is_valid(r: int, c: int) -> bool:
    return 0 <= r < ROWS and 0 <= c < COLS and VALID_MASK[r, c]


def get_level(r: int) -> int:
    if r < 4:
        return 1
    if r < 8:
        return 2
    return 3


def level_boundary_rows() -> list[int]:
    return [3, 4, 7, 8]


def crosses_level_boundary(rows: list[int]) -> bool:
    levels = {get_level(r) for r in rows}
    return len(levels) > 1


def empty_board() -> np.ndarray:
    return np.zeros((ROWS, COLS), dtype=np.int8)


def get_valid_mask_flat() -> np.ndarray:
    return VALID_MASK.flatten().astype(np.bool_)
