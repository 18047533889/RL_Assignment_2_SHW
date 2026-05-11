"""
rules.py — Win detection, threat counting, and threat heatmaps (vectorised).

**Quick read:** all line / window enumeration is precomputed once at module load as
flat-index arrays (row/col 4-windows, diag 5-windows, plus a cross-level mask for
columns). The public API is unchanged from the loop-based version; every primitive
now runs as pure numpy fancy-indexing + scatter-add, and batched variants are
provided so tactical-opponent code can score ~800 candidate boards in one shot
instead of per-board Python loops.

``check_win`` returns the same ``("row"|"col"|"diag", list-of-(r,c))`` tuple and
still prefers row → col → diag (tests lock this order). ``threat_heatmap`` /
``partial_threat_heatmap`` / ``row_threat_heatmap`` / ``col_threat_heatmap`` all
return ``(12, 12) float32``; single-board calls delegate to the batched path with
B = 1.

Threat definitions (unchanged):
  - Row threat:  3-of-4 window + 1 empty → +1 at the empty cell.
  - Column threat: 3-of-4 window + 1 empty, **cross-level required** → +1.
  - Diagonal threat: 4-of-5 window + 1 empty → +1.
  - Partial threats use one-less-mine + 0 opp in the same windows and spread to all
    empty cells in the window.
  - row / col pressure: opp==0 and mine>=1 in 4-window (col requires cross-level)
    → add ``mine`` to each empty cell.

Win detection (unchanged):
  - Row: 4 consecutive same-colour cells in a row.
  - Column: 4 consecutive same-colour cells spanning >1 pyramid level.
  - Diagonal: 5 consecutive same-colour cells.
"""

from __future__ import annotations

import numpy as np

from board import ROWS, COLS, GRID_SIZE, VALID_MASK, get_level


# ---------------------------------------------------------------------------
# Precomputed line windows (flat indices into a 144-cell flattened board).
# ---------------------------------------------------------------------------


def _build_row_windows() -> np.ndarray:
    """Return (N, 4) flat indices for every 4-cell row window inside the triangle."""
    windows: list[list[int]] = []
    for r in range(ROWS):
        valid_cols = [c for c in range(COLS) if VALID_MASK[r, c]]
        if len(valid_cols) < 4:
            continue
        for start in range(len(valid_cols) - 3):
            seg = valid_cols[start:start + 4]
            windows.append([r * COLS + c for c in seg])
    return np.asarray(windows, dtype=np.int64) if windows else np.zeros((0, 4), dtype=np.int64)


def _build_col_windows() -> tuple[np.ndarray, np.ndarray]:
    """Return (windows (N, 4), cross-level (N,) bool) for every column 4-window."""
    windows: list[list[int]] = []
    cross: list[bool] = []
    for c in range(COLS):
        valid_rows = [r for r in range(ROWS) if VALID_MASK[r, c]]
        if len(valid_rows) < 4:
            continue
        for start in range(len(valid_rows) - 3):
            seg = valid_rows[start:start + 4]
            windows.append([r * COLS + c for r in seg])
            cross.append(len({get_level(r) for r in seg}) > 1)
    if not windows:
        return np.zeros((0, 4), dtype=np.int64), np.zeros((0,), dtype=bool)
    return np.asarray(windows, dtype=np.int64), np.asarray(cross, dtype=bool)


def _build_diag_windows() -> np.ndarray:
    """Return (N, 5) flat indices for every diagonal 5-window (both directions)."""
    windows: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for r in range(ROWS):
        for c in range(COLS):
            if not VALID_MASK[r, c]:
                continue
            for dr, dc in ((1, 1), (1, -1)):
                pos: list[int] = []
                rr, cc = r, c
                while 0 <= rr < ROWS and 0 <= cc < COLS and VALID_MASK[rr, cc]:
                    pos.append(rr * COLS + cc)
                    rr += dr
                    cc += dc
                if len(pos) < 5:
                    continue
                key = tuple(pos)
                if key in seen:
                    continue
                seen.add(key)
                for start in range(len(pos) - 4):
                    windows.append(pos[start:start + 5])
    return np.asarray(windows, dtype=np.int64) if windows else np.zeros((0, 5), dtype=np.int64)


_ROW_WIN_IDX: np.ndarray = _build_row_windows()
_COL_WIN_IDX, _COL_CROSS_LEVEL = _build_col_windows()
_DIAG_WIN_IDX: np.ndarray = _build_diag_windows()


# ---------------------------------------------------------------------------
# Batched primitives. All accept a (B, 144) int8 (or int) board buffer.
# ---------------------------------------------------------------------------


def _gather_window_cells(boards_flat: np.ndarray, windows: np.ndarray) -> np.ndarray:
    """(B, 144) + (W, K) -> (B, W, K) cell values via a single fancy-index op."""
    return boards_flat[:, windows]


def _mine_opp_empty(cells: np.ndarray, player: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mine = (cells == player).sum(axis=-1)
    opp = ((cells != 0) & (cells != player)).sum(axis=-1)
    empty = (cells == 0).sum(axis=-1)
    return mine, opp, empty


def _flat_boards(board: np.ndarray) -> np.ndarray:
    """Return a (1, 144) view of a single (12, 12) board without copying."""
    if board.ndim == 2:
        return board.reshape(1, GRID_SIZE)
    if board.ndim == 1:
        return board.reshape(1, GRID_SIZE)
    return board


def check_win_batch(boards_flat: np.ndarray, player: int) -> np.ndarray:
    """(B, 144) int -> (B,) bool: True iff player has any winning line."""
    B = boards_flat.shape[0]
    has_win = np.zeros(B, dtype=bool)
    if _ROW_WIN_IDX.shape[0] > 0:
        cells = _gather_window_cells(boards_flat, _ROW_WIN_IDX)
        has_win |= (cells == player).all(axis=-1).any(axis=-1)
    if _COL_WIN_IDX.shape[0] > 0:
        cells = _gather_window_cells(boards_flat, _COL_WIN_IDX)
        col_wins = (cells == player).all(axis=-1) & _COL_CROSS_LEVEL[None, :]
        has_win |= col_wins.any(axis=-1)
    if _DIAG_WIN_IDX.shape[0] > 0:
        cells = _gather_window_cells(boards_flat, _DIAG_WIN_IDX)
        has_win |= (cells == player).all(axis=-1).any(axis=-1)
    return has_win


def count_threats_batch(boards_flat: np.ndarray, player: int) -> np.ndarray:
    """(B, 144) -> (B,) int: number of (3-of-4 row / cross-level col / 4-of-5 diag) threats."""
    B = boards_flat.shape[0]
    total = np.zeros(B, dtype=np.int64)
    if _ROW_WIN_IDX.shape[0] > 0:
        cells = _gather_window_cells(boards_flat, _ROW_WIN_IDX)
        mine, _opp, empty = _mine_opp_empty(cells, player)
        total += ((mine == 3) & (empty == 1)).sum(axis=-1, dtype=np.int64)
    if _COL_WIN_IDX.shape[0] > 0:
        cells = _gather_window_cells(boards_flat, _COL_WIN_IDX)
        mine, _opp, empty = _mine_opp_empty(cells, player)
        mask = (mine == 3) & (empty == 1) & _COL_CROSS_LEVEL[None, :]
        total += mask.sum(axis=-1, dtype=np.int64)
    if _DIAG_WIN_IDX.shape[0] > 0:
        cells = _gather_window_cells(boards_flat, _DIAG_WIN_IDX)
        mine, _opp, empty = _mine_opp_empty(cells, player)
        total += ((mine == 4) & (empty == 1)).sum(axis=-1, dtype=np.int64)
    return total


def _scatter_single_empty(
    hmap: np.ndarray,
    cells: np.ndarray,
    windows: np.ndarray,
    mask: np.ndarray,
    value: float = 1.0,
) -> None:
    """For each (b, w) in mask, scatter ``value`` to the unique empty cell of the window."""
    if not mask.any():
        return
    b_idx, w_idx = np.nonzero(mask)
    selected = cells[b_idx, w_idx]  # (M, K)
    empty_pos = (selected == 0).argmax(axis=-1)  # (M,) — mask guarantees exactly one empty
    flat_targets = windows[w_idx, empty_pos]
    np.add.at(hmap, (b_idx, flat_targets), value)


def _scatter_all_empty(
    hmap: np.ndarray,
    cells: np.ndarray,
    windows: np.ndarray,
    mask: np.ndarray,
    weight_bw: np.ndarray | None = None,
) -> None:
    """For each (b, w) in mask, add ``weight_bw[b, w]`` (default 1.0) to every empty cell in the window."""
    if not mask.any():
        return
    b_idx, w_idx = np.nonzero(mask)  # (M,)
    selected = cells[b_idx, w_idx]  # (M, K)
    empty_mask = (selected == 0)
    targets = windows[w_idx]  # (M, K)
    M, K = targets.shape
    b_broadcast = np.broadcast_to(b_idx[:, None], (M, K))
    if weight_bw is None:
        values = np.ones((M, K), dtype=hmap.dtype)
    else:
        w_vals = weight_bw[b_idx, w_idx].astype(hmap.dtype)  # (M,)
        values = np.broadcast_to(w_vals[:, None], (M, K))
    flat_b = b_broadcast[empty_mask]
    flat_t = targets[empty_mask]
    flat_v = values[empty_mask]
    np.add.at(hmap, (flat_b, flat_t), flat_v)


def threat_heatmap_batch(boards_flat: np.ndarray, player: int) -> np.ndarray:
    """(B, 144) -> (B, 144) float32: +1 at the empty cell of each 3-of-4 / 4-of-5 threat window."""
    B = boards_flat.shape[0]
    hmap = np.zeros((B, GRID_SIZE), dtype=np.float32)

    if _ROW_WIN_IDX.shape[0] > 0:
        cells = _gather_window_cells(boards_flat, _ROW_WIN_IDX)
        mine, _opp, empty = _mine_opp_empty(cells, player)
        mask = (mine == 3) & (empty == 1)
        _scatter_single_empty(hmap, cells, _ROW_WIN_IDX, mask)

    if _COL_WIN_IDX.shape[0] > 0:
        cells = _gather_window_cells(boards_flat, _COL_WIN_IDX)
        mine, _opp, empty = _mine_opp_empty(cells, player)
        mask = (mine == 3) & (empty == 1) & _COL_CROSS_LEVEL[None, :]
        _scatter_single_empty(hmap, cells, _COL_WIN_IDX, mask)

    if _DIAG_WIN_IDX.shape[0] > 0:
        cells = _gather_window_cells(boards_flat, _DIAG_WIN_IDX)
        mine, _opp, empty = _mine_opp_empty(cells, player)
        mask = (mine == 4) & (empty == 1)
        _scatter_single_empty(hmap, cells, _DIAG_WIN_IDX, mask)

    return hmap


def partial_threat_heatmap_batch(boards_flat: np.ndarray, player: int) -> np.ndarray:
    """(B, 144) -> (B, 144) float32: +1 at every empty cell of each (mine = near, opp = 0) window."""
    B = boards_flat.shape[0]
    hmap = np.zeros((B, GRID_SIZE), dtype=np.float32)

    if _ROW_WIN_IDX.shape[0] > 0:
        cells = _gather_window_cells(boards_flat, _ROW_WIN_IDX)
        mine, opp, _empty = _mine_opp_empty(cells, player)
        mask = (mine == 2) & (opp == 0)
        _scatter_all_empty(hmap, cells, _ROW_WIN_IDX, mask)

    if _COL_WIN_IDX.shape[0] > 0:
        cells = _gather_window_cells(boards_flat, _COL_WIN_IDX)
        mine, opp, _empty = _mine_opp_empty(cells, player)
        mask = (mine == 2) & (opp == 0) & _COL_CROSS_LEVEL[None, :]
        _scatter_all_empty(hmap, cells, _COL_WIN_IDX, mask)

    if _DIAG_WIN_IDX.shape[0] > 0:
        cells = _gather_window_cells(boards_flat, _DIAG_WIN_IDX)
        mine, opp, _empty = _mine_opp_empty(cells, player)
        mask = (mine == 3) & (opp == 0)
        _scatter_all_empty(hmap, cells, _DIAG_WIN_IDX, mask)

    return hmap


def row_threat_heatmap_batch(boards_flat: np.ndarray, player: int) -> np.ndarray:
    """(B, 144) -> (B, 144) float32: for each (opp = 0, mine >= 1) row 4-window add ``mine`` to every empty cell."""
    B = boards_flat.shape[0]
    hmap = np.zeros((B, GRID_SIZE), dtype=np.float32)
    if _ROW_WIN_IDX.shape[0] == 0:
        return hmap
    cells = _gather_window_cells(boards_flat, _ROW_WIN_IDX)
    mine, opp, _empty = _mine_opp_empty(cells, player)
    mask = (opp == 0) & (mine >= 1)
    if mask.any():
        _scatter_all_empty(hmap, cells, _ROW_WIN_IDX, mask, weight_bw=mine)
    return hmap


def col_threat_heatmap_batch(boards_flat: np.ndarray, player: int) -> np.ndarray:
    """(B, 144) -> (B, 144) float32: cross-level (opp = 0, mine >= 1) col 4-windows, weight = mine."""
    B = boards_flat.shape[0]
    hmap = np.zeros((B, GRID_SIZE), dtype=np.float32)
    if _COL_WIN_IDX.shape[0] == 0:
        return hmap
    cells = _gather_window_cells(boards_flat, _COL_WIN_IDX)
    mine, opp, _empty = _mine_opp_empty(cells, player)
    mask = (opp == 0) & (mine >= 1) & _COL_CROSS_LEVEL[None, :]
    if mask.any():
        _scatter_all_empty(hmap, cells, _COL_WIN_IDX, mask, weight_bw=mine)
    return hmap


# ---------------------------------------------------------------------------
# Single-board public API (delegates to batch with B = 1).
# ---------------------------------------------------------------------------


def _flat_to_pairs(flat_indices: np.ndarray) -> list[tuple[int, int]]:
    return [(int(p // COLS), int(p % COLS)) for p in flat_indices]


def check_row_win(board: np.ndarray, player: int) -> list[tuple[int, int]] | None:
    if _ROW_WIN_IDX.shape[0] == 0:
        return None
    flat = board.reshape(-1)
    cells = flat[_ROW_WIN_IDX]  # (N, 4)
    wins = (cells == player).all(axis=-1)
    if not wins.any():
        return None
    idx = int(np.argmax(wins))
    return _flat_to_pairs(_ROW_WIN_IDX[idx])


def check_col_win(board: np.ndarray, player: int) -> list[tuple[int, int]] | None:
    if _COL_WIN_IDX.shape[0] == 0:
        return None
    flat = board.reshape(-1)
    cells = flat[_COL_WIN_IDX]
    wins = (cells == player).all(axis=-1) & _COL_CROSS_LEVEL
    if not wins.any():
        return None
    idx = int(np.argmax(wins))
    return _flat_to_pairs(_COL_WIN_IDX[idx])


def check_diag_win(board: np.ndarray, player: int) -> list[tuple[int, int]] | None:
    if _DIAG_WIN_IDX.shape[0] == 0:
        return None
    flat = board.reshape(-1)
    cells = flat[_DIAG_WIN_IDX]
    wins = (cells == player).all(axis=-1)
    if not wins.any():
        return None
    idx = int(np.argmax(wins))
    return _flat_to_pairs(_DIAG_WIN_IDX[idx])


def check_win(board: np.ndarray, player: int) -> tuple[str, list[tuple[int, int]]] | None:
    result = check_row_win(board, player)
    if result is not None:
        return ("row", result)
    result = check_col_win(board, player)
    if result is not None:
        return ("col", result)
    result = check_diag_win(board, player)
    if result is not None:
        return ("diag", result)
    return None


def count_threats(board: np.ndarray, player: int) -> int:
    return int(count_threats_batch(_flat_boards(board), player)[0])


def threat_heatmap(board: np.ndarray, player: int) -> np.ndarray:
    return threat_heatmap_batch(_flat_boards(board), player)[0].reshape(ROWS, COLS)


def partial_threat_heatmap(board: np.ndarray, player: int) -> np.ndarray:
    return partial_threat_heatmap_batch(_flat_boards(board), player)[0].reshape(ROWS, COLS)


def row_threat_heatmap(board: np.ndarray, player: int) -> np.ndarray:
    return row_threat_heatmap_batch(_flat_boards(board), player)[0].reshape(ROWS, COLS)


def col_threat_heatmap(board: np.ndarray, player: int) -> np.ndarray:
    return col_threat_heatmap_batch(_flat_boards(board), player)[0].reshape(ROWS, COLS)


def is_draw(board: np.ndarray) -> bool:
    return not np.any(VALID_MASK & (board == 0))
