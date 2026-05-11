"""
``rules.py``: row/column/diagonal wins and draws; column must span levels; diagonal needs five.

Matches assignment spec: column wins require multiple pyramid levels; row/diag rules per helpers.
"""

import numpy as np
import pytest
from board import empty_board, VALID_MASK
from rules import check_row_win, check_col_win, check_diag_win, check_win, is_draw, threat_heatmap, partial_threat_heatmap, row_threat_heatmap, col_threat_heatmap

P1, P2 = 1, 2


def place(board, positions, player):
    """Place several stones for constructing win patterns."""
    for r, c in positions:
        board[r, c] = player


class TestRowWin:
    """Horizontal four: enough stones, blocking, return value."""

    def test_4_in_row_level1(self):
        """L1 row of four."""
        b = empty_board()
        place(b, [(0, 4), (0, 5), (0, 6), (0, 7)], P1)
        assert check_row_win(b, P1) is not None

    def test_4_in_row_level3(self):
        """L3 row of four."""
        b = empty_board()
        place(b, [(8, 3), (8, 4), (8, 5), (8, 6)], P2)
        assert check_row_win(b, P2) is not None

    def test_3_in_row_not_enough(self):
        """Three in a row is not a win."""
        b = empty_board()
        place(b, [(0, 4), (0, 5), (0, 6)], P1)
        assert check_row_win(b, P1) is None

    def test_4_in_row_broken_by_opponent(self):
        """Opponent in the middle breaks the segment."""
        b = empty_board()
        place(b, [(8, 0), (8, 1), (8, 3), (8, 4)], P1)
        b[8, 2] = P2
        assert check_row_win(b, P1) is None

    def test_row_win_returns_positions(self):
        """``check_row_win`` returns four cells."""
        b = empty_board()
        place(b, [(10, 0), (10, 1), (10, 2), (10, 3)], P1)
        result = check_row_win(b, P1)
        assert len(result) == 4


class TestColWin:
    """Vertical four must span levels; same-level four is invalid."""

    def test_4_in_col_cross_level_l1_l2(self):
        """Four in one column spanning L1–L2."""
        b = empty_board()
        place(b, [(2, 4), (3, 4), (4, 4), (5, 4)], P1)
        result = check_col_win(b, P1)
        assert result is not None

    def test_4_in_col_cross_level_l2_l3(self):
        """Four in one column spanning L2–L3."""
        b = empty_board()
        place(b, [(6, 3), (7, 3), (8, 3), (9, 3)], P1)
        result = check_col_win(b, P1)
        assert result is not None

    def test_4_in_col_same_level_INVALID(self):
        """Four in one column on L3 only: invalid."""
        b = empty_board()
        place(b, [(8, 0), (9, 0), (10, 0), (11, 0)], P1)
        result = check_col_win(b, P1)
        assert result is None

    def test_4_in_col_same_level2_INVALID(self):
        """Four in one column on L2 only: invalid."""
        b = empty_board()
        place(b, [(4, 3), (5, 3), (6, 3), (7, 3)], P1)
        result = check_col_win(b, P1)
        assert result is None

    def test_4_in_col_same_level1_INVALID(self):
        """Four in one column on L1 only: invalid."""
        b = empty_board()
        place(b, [(0, 5), (1, 5), (2, 5), (3, 5)], P1)
        result = check_col_win(b, P1)
        assert result is None

    def test_col_spanning_all_3_levels(self):
        """Four rows spanning all three levels: valid column win."""
        b = empty_board()
        place(b, [(3, 4), (4, 4), (5, 4), (6, 4)], P1)
        result = check_col_win(b, P1)
        assert result is not None

    def test_col_3_not_enough(self):
        """Three vertical is not enough."""
        b = empty_board()
        place(b, [(3, 4), (4, 4), (5, 4)], P1)
        result = check_col_win(b, P1)
        assert result is None

    def test_col_five_consecutive_same_column_still_wins(self):
        """Five in one column (sliding window): still a column win; segment is four consecutive cells."""
        b = empty_board()
        place(b, [(2, 4), (3, 4), (4, 4), (5, 4), (6, 4)], P1)
        result = check_col_win(b, P1)
        assert result is not None
        assert len(result) == 4
        assert all(b[r, c] == P1 for r, c in result)


class TestDiagWin:
    """Diagonal: five in a row along precomputed chains; four is not enough."""

    def test_5_on_diagonal_down_right(self):
        """Five along one diagonal direction."""
        b = empty_board()
        place(b, [(0, 4), (1, 5), (2, 6), (3, 7), (4, 8)], P1)
        result = check_diag_win(b, P1)
        assert result is not None

    def test_5_on_diagonal_down_left(self):
        """Five along the other diagonal direction."""
        b = empty_board()
        place(b, [(4, 9), (5, 8), (6, 7), (7, 6), (8, 5)], P2)
        result = check_diag_win(b, P2)
        assert result is not None

    def test_4_on_diagonal_not_enough(self):
        """Four on diagonal: not a win."""
        b = empty_board()
        place(b, [(0, 4), (1, 5), (2, 6), (3, 7)], P1)
        result = check_diag_win(b, P1)
        assert result is None

    def test_diag_5_in_level3(self):
        """Four on L3 diag fails; another five-cell pattern wins."""
        b = empty_board()
        place(b, [(8, 0), (9, 1), (10, 2), (11, 3)], P1)
        assert check_diag_win(b, P1) is None
        b[8, 0] = 0
        place(b, [(7, 8), (8, 7), (9, 6), (10, 5), (11, 4)], P1)
        result = check_diag_win(b, P1)
        assert result is not None


class TestRowWinLineEndpoints:
    """Four in a row at the ends of a long horizontal valid strip (``_scan_line`` edges)."""

    def test_l3_row_four_at_left_end_of_strip(self):
        """L3 row 8: win flush to the left valid column."""
        b = empty_board()
        place(b, [(8, 0), (8, 1), (8, 2), (8, 3)], P1)
        assert check_row_win(b, P1) is not None

    def test_l3_row_four_at_right_end_of_strip(self):
        """L3 row 8: win flush to the right valid column."""
        b = empty_board()
        place(b, [(8, 8), (8, 9), (8, 10), (8, 11)], P2)
        assert check_row_win(b, P2) is not None


class TestCheckWin:
    """``check_win`` dispatches row/col/diag and tags kind.

    Order in ``rules.check_win`` is row → column → diagonal; tests below lock observable behaviour.
    """

    def test_row_detected_via_check_win(self):
        """Detected as row."""
        b = empty_board()
        place(b, [(0, 4), (0, 5), (0, 6), (0, 7)], P1)
        result = check_win(b, P1)
        assert result is not None
        assert result[0] == "row"

    def test_col_detected_via_check_win(self):
        """Detected as col."""
        b = empty_board()
        place(b, [(3, 5), (4, 5), (5, 5), (6, 5)], P2)
        result = check_win(b, P2)
        assert result is not None
        assert result[0] == "col"

    def test_diag_detected_via_check_win(self):
        """Diag win (or overlap with row still yields a win)."""
        b = empty_board()
        place(b, [(0, 4), (1, 5), (2, 6), (3, 7), (4, 8)], P1)
        result = check_win(b, P1)
        assert result is not None
        assert result[0] in ("row", "diag")

    def test_no_win_empty_board(self):
        """Empty board: no win."""
        b = empty_board()
        assert check_win(b, P1) is None
        assert check_win(b, P2) is None

    def test_no_cross_player_win(self):
        """Interleaved two-and-two: no four for either player."""
        b = empty_board()
        place(b, [(0, 4), (0, 5)], P1)
        place(b, [(0, 6), (0, 7)], P2)
        assert check_win(b, P1) is None
        assert check_win(b, P2) is None

    def test_check_win_prefers_row_when_row_and_column_both_satisfied(self):
        """Same player satisfies row and column; implementation checks row first."""
        b = empty_board()
        place(b, [(8, 0), (8, 1), (8, 2), (8, 3)], P1)
        place(b, [(2, 4), (3, 4), (4, 4), (5, 4)], P1)
        kind, cells = check_win(b, P1)
        assert kind == "row"
        assert len(cells) == 4


class TestIsDraw:
    """Draw: all valid cells full; one empty -> not draw."""

    def test_empty_not_draw(self):
        """Empty board is not a draw."""
        assert not is_draw(empty_board())

    def test_full_board_draw(self):
        """All valid cells occupied."""
        b = empty_board()
        for r in range(12):
            for c in range(12):
                if VALID_MASK[r, c]:
                    b[r, c] = P1 if (r + c) % 2 == 0 else P2
        assert is_draw(b)

    def test_one_empty_not_draw(self):
        """One valid cell empty."""
        b = empty_board()
        for r in range(12):
            for c in range(12):
                if VALID_MASK[r, c]:
                    b[r, c] = P1
        b[0, 4] = 0
        assert not is_draw(b)

    def test_winning_position_is_not_draw(self):
        """Terminal win with many empty cells: must not classify as draw."""
        b = empty_board()
        place(b, [(8, 0), (8, 1), (8, 2), (8, 3)], P1)
        assert not is_draw(b)


class TestThreatHeatmap:
    """``threat_heatmap`` returns (12,12) float32 with non-negative values."""

    def test_shape_and_dtype(self):
        b = empty_board()
        h = threat_heatmap(b, P1)
        assert h.shape == (12, 12)
        assert h.dtype == np.float32

    def test_empty_board_all_zero(self):
        b = empty_board()
        h = threat_heatmap(b, P1)
        assert h.sum() == pytest.approx(0.0)

    def test_three_in_row_creates_threat(self):
        b = empty_board()
        place(b, [(8, 0), (8, 1), (8, 2)], P1)
        h = threat_heatmap(b, P1)
        assert h.max() > 0

    def test_non_negative(self):
        b = empty_board()
        place(b, [(8, 0), (8, 1), (8, 2)], P1)
        place(b, [(9, 3), (9, 4), (9, 5)], P2)
        h1 = threat_heatmap(b, P1)
        h2 = threat_heatmap(b, P2)
        assert (h1 >= 0).all()
        assert (h2 >= 0).all()


class TestPartialThreatHeatmap:

    def test_shape_and_dtype(self):
        b = empty_board()
        h = partial_threat_heatmap(b, P1)
        assert h.shape == (12, 12)
        assert h.dtype == np.float32

    def test_empty_board_all_zero(self):
        b = empty_board()
        h = partial_threat_heatmap(b, P1)
        assert h.sum() == pytest.approx(0.0)

    def test_two_in_row_creates_partial_threat(self):
        b = empty_board()
        place(b, [(8, 0), (8, 1)], P1)
        h = partial_threat_heatmap(b, P1)
        assert h[8, 2] > 0 or h[8, 3] > 0

    def test_two_blocked_by_opponent_no_partial(self):
        b = empty_board()
        place(b, [(8, 0), (8, 1)], P1)
        b[8, 2] = P2
        b[8, 3] = P2
        h = partial_threat_heatmap(b, P1)
        assert h[8, 2] == 0 and h[8, 3] == 0

    def test_non_negative(self):
        b = empty_board()
        place(b, [(8, 0), (8, 1)], P1)
        h = partial_threat_heatmap(b, P1)
        assert (h >= 0).all()


class TestRowThreatHeatmap:

    def test_shape_and_dtype(self):
        b = empty_board()
        h = row_threat_heatmap(b, P1)
        assert h.shape == (12, 12)
        assert h.dtype == np.float32

    def test_empty_board_all_zero(self):
        b = empty_board()
        h = row_threat_heatmap(b, P1)
        assert h.sum() == pytest.approx(0.0)

    def test_two_in_row_creates_heat(self):
        b = empty_board()
        place(b, [(8, 0), (8, 1)], P1)
        h = row_threat_heatmap(b, P1)
        assert h[8, 2] > 0 or h[8, 3] > 0

    def test_three_in_row_higher_heat(self):
        b = empty_board()
        place(b, [(8, 0), (8, 1), (8, 2)], P1)
        h = row_threat_heatmap(b, P1)
        assert h[8, 3] >= 3

    def test_blocked_by_opponent_no_heat(self):
        b = empty_board()
        place(b, [(8, 0), (8, 1)], P1)
        b[8, 2] = P2
        b[8, 3] = P2
        h = row_threat_heatmap(b, P1)
        assert h[8, 2] == 0 and h[8, 3] == 0

    def test_non_negative(self):
        b = empty_board()
        place(b, [(8, 0), (8, 1), (8, 2)], P1)
        h = row_threat_heatmap(b, P1)
        assert (h >= 0).all()


class TestColThreatHeatmap:

    def test_shape_and_dtype(self):
        b = empty_board()
        h = col_threat_heatmap(b, P1)
        assert h.shape == (12, 12)
        assert h.dtype == np.float32

    def test_empty_board_all_zero(self):
        b = empty_board()
        h = col_threat_heatmap(b, P1)
        assert h.sum() == pytest.approx(0.0)

    def test_cross_level_col_creates_heat(self):
        b = empty_board()
        place(b, [(3, 4), (4, 4)], P1)
        h = col_threat_heatmap(b, P1)
        assert h[5, 4] > 0 or h[2, 4] > 0

    def test_same_level_col_no_heat(self):
        b = empty_board()
        place(b, [(8, 0), (9, 0), (10, 0)], P1)
        h = col_threat_heatmap(b, P1)
        assert h[11, 0] == 0, "same-level column should not produce col_threat heat"

    def test_non_negative(self):
        b = empty_board()
        place(b, [(3, 4), (4, 4), (5, 4)], P1)
        h = col_threat_heatmap(b, P1)
        assert (h >= 0).all()
