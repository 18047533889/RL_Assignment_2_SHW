"""
``board.py``: 12×12 triangle, three-level bounds, flat indexing, valid mask.

Geometry constants and ``rc_to_flat`` / ``flat_to_rc`` / level helpers stay consistent.
"""

import numpy as np
import pytest
from board import (
    ROWS, COLS, GRID_SIZE, VALID_MASK, VALID_POSITIONS, NUM_VALID,
    POS_TO_INDEX, INDEX_TO_POS, LEVEL_BOUNDS,
    COMPACT_TO_FLAT, FLAT_TO_COMPACT,
    rc_to_flat, flat_to_rc, is_valid, get_level, crosses_level_boundary,
    empty_board, get_valid_mask_flat, level_boundary_rows,
    compact_to_rc, rc_to_compact, compact_to_flat, flat_to_compact,
)


class TestGridDimensions:
    """Global size and 96 valid cells."""

    def test_grid_size(self):
        """12×12, 144 cells."""
        assert ROWS == 12
        assert COLS == 12
        assert GRID_SIZE == 144

    def test_valid_mask_shape(self):
        """Mask shape matches board."""
        assert VALID_MASK.shape == (12, 12)

    def test_total_valid_cells(self):
        """96 playable cells."""
        assert NUM_VALID == 96
        assert VALID_MASK.sum() == 96


class TestLevelGeometry:
    """Per-level cell counts and bounds; ``VALID_MASK`` equals union of level rectangles."""

    def test_level1_has_16_cells(self):
        """L1: 16 cells."""
        count = VALID_MASK[0:4, 4:8].sum()
        assert count == 16

    def test_level2_has_32_cells(self):
        """L2: 32 cells."""
        count = VALID_MASK[4:8, 2:10].sum()
        assert count == 32

    def test_level3_has_48_cells(self):
        """L3: 48 cells."""
        count = VALID_MASK[8:12, 0:12].sum()
        assert count == 48

    def test_no_valid_cells_outside_triangle(self):
        """Union of ``LEVEL_BOUNDS`` equals ``VALID_MASK``."""
        mask_copy = np.zeros((12, 12), dtype=bool)
        for r_lo, r_hi, c_lo, c_hi in LEVEL_BOUNDS:
            mask_copy[r_lo:r_hi, c_lo:c_hi] = True
        assert np.array_equal(VALID_MASK, mask_copy)

    def test_level1_boundaries(self):
        """L1 inside/outside examples."""
        assert not is_valid(0, 3)
        assert is_valid(0, 4)
        assert is_valid(3, 7)
        assert not is_valid(0, 8)

    def test_level2_boundaries(self):
        """L2 boundary cells."""
        assert not is_valid(4, 1)
        assert is_valid(4, 2)
        assert is_valid(7, 9)
        assert not is_valid(4, 10)

    def test_level3_boundaries(self):
        """L3 spans full bottom width."""
        assert is_valid(8, 0)
        assert is_valid(11, 11)


class TestCoordinateConversion:
    """Row/col <-> flat index; 96 legal bijection tables."""

    def test_rc_to_flat_origin(self):
        """(0,0) -> 0."""
        assert rc_to_flat(0, 0) == 0

    def test_rc_to_flat_last(self):
        """(11,11) -> 143."""
        assert rc_to_flat(11, 11) == 143

    def test_flat_to_rc_roundtrip(self):
        """Roundtrip all cells."""
        for r in range(ROWS):
            for c in range(COLS):
                assert flat_to_rc(rc_to_flat(r, c)) == (r, c)

    def test_valid_positions_biject_to_indices(self):
        """``POS_TO_INDEX`` and ``INDEX_TO_POS`` invert."""
        assert len(POS_TO_INDEX) == 96
        assert len(INDEX_TO_POS) == 96
        for idx, pos in INDEX_TO_POS.items():
            assert POS_TO_INDEX[pos] == idx


class TestGetLevel:
    """Row index -> pyramid level 1/2/3."""

    def test_level_boundary_rows_documentation(self):
        """Row indices between pyramid bands (pairs with ``get_level``)."""
        assert level_boundary_rows() == [3, 4, 7, 8]

    @pytest.mark.parametrize("row,expected", [
        (0, 1), (3, 1), (4, 2), (7, 2), (8, 3), (11, 3),
    ])
    def test_level_assignment(self, row, expected):
        """Boundary rows map correctly."""
        assert get_level(row) == expected


class TestCrossesLevelBoundary:
    """Whether four rows span multiple levels (column-win rule)."""

    def test_single_level(self):
        """All rows in one level: no cross."""
        assert not crosses_level_boundary([0, 1, 2, 3])
        assert not crosses_level_boundary([4, 5, 6, 7])

    def test_cross_l1_l2(self):
        """Spans L1 and L2."""
        assert crosses_level_boundary([2, 3, 4, 5])

    def test_cross_l2_l3(self):
        """Spans L2 and L3."""
        assert crosses_level_boundary([6, 7, 8, 9])

    def test_cross_all_three(self):
        """Four rows cover all three levels."""
        assert crosses_level_boundary([3, 4, 8, 9])


class TestEmptyBoard:
    """Empty board factory."""

    def test_shape(self):
        """Shape and dtype."""
        b = empty_board()
        assert b.shape == (12, 12)
        assert b.dtype == np.int8

    def test_all_zeros(self):
        """All zeros."""
        assert empty_board().sum() == 0


class TestGetValidMaskFlat:
    """144-d flat mask matches 2D."""

    def test_length(self):
        """Length and sum."""
        flat = get_valid_mask_flat()
        assert flat.shape == (144,)
        assert flat.sum() == 96

    def test_consistency(self):
        """Cell-wise match to ``VALID_MASK``."""
        flat = get_valid_mask_flat()
        for r in range(ROWS):
            for c in range(COLS):
                assert flat[rc_to_flat(r, c)] == VALID_MASK[r, c]


class TestIsValid:
    """Out of bounds vs all valid positions."""

    def test_out_of_bounds(self):
        """Outside board -> false."""
        assert not is_valid(-1, 0)
        assert not is_valid(0, -1)
        assert not is_valid(12, 0)
        assert not is_valid(0, 12)

    def test_all_valid_positions(self):
        """Every ``VALID_POSITIONS`` entry is valid."""
        for r, c in VALID_POSITIONS:
            assert is_valid(r, c)


class TestCompactIndexing:
    """Compact 0..95 <-> (row, col) and compact <-> flat conversions."""

    def test_compact_to_rc_roundtrip(self):
        for i in range(NUM_VALID):
            r, c = compact_to_rc(i)
            assert rc_to_compact(r, c) == i

    def test_compact_to_flat_roundtrip(self):
        for i in range(NUM_VALID):
            f = compact_to_flat(i)
            assert flat_to_compact(f) == i

    def test_compact_to_rc_matches_index_to_pos(self):
        for i in range(NUM_VALID):
            assert compact_to_rc(i) == INDEX_TO_POS[i]

    def test_compact_to_flat_array_length(self):
        assert len(COMPACT_TO_FLAT) == NUM_VALID

    def test_flat_to_compact_array_length(self):
        assert len(FLAT_TO_COMPACT) == GRID_SIZE

    def test_flat_to_compact_invalid_is_minus_one(self):
        for f in range(GRID_SIZE):
            r, c = flat_to_rc(f)
            if not VALID_MASK[r, c]:
                assert FLAT_TO_COMPACT[f] == -1
