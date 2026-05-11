"""
``augment.py``: vertical-axis symmetry — action indices, observation tensors, and legal masks stay consistent.

Ensures ``flip_action`` is an involution, valid cells map to valid cells, and geometry matches the board (for the augment learner).
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from augment import FLIP_TABLE, flip_action, flip_mask, flip_obs
from board import COLS, NUM_VALID, ROWS, VALID_MASK, VALID_POSITIONS, POS_TO_INDEX


class TestFlipAction:
    """Compact action index: left-right permutation and column mirror geometry."""

    def test_roundtrip_all(self):
        """Double flip returns the original action."""
        for idx in range(NUM_VALID):
            assert flip_action(flip_action(idx)) == idx

    def test_valid_maps_to_valid(self):
        """Valid cells map to valid cells under flip."""
        for idx, (r, c) in enumerate(VALID_POSITIONS):
            f2 = flip_action(idx)
            r2, c2 = VALID_POSITIONS[f2]
            assert VALID_MASK[r2, c2], f"({r},{c})->({r2},{c2}) not valid"

    def test_column_mirror(self):
        """Row fixed; column mirrors about the centre."""
        for idx, (r, c) in enumerate(VALID_POSITIONS):
            f2 = flip_action(idx)
            r2, c2 = VALID_POSITIONS[f2]
            assert r2 == r
            assert c2 == COLS - 1 - c

    def test_center_column_fixed_points(self):
        """Centre columns swap pairwise when both legal."""
        for r in range(ROWS):
            if VALID_MASK[r, 5] and VALID_MASK[r, 6]:
                i5 = POS_TO_INDEX[(r, 5)]
                i6 = POS_TO_INDEX[(r, 6)]
                assert flip_action(i5) == i6
                assert flip_action(i6) == i5

    def test_flip_table_is_permutation(self):
        """``FLIP_TABLE`` is a permutation of 0..95."""
        assert len(set(FLIP_TABLE.tolist())) == NUM_VALID


class TestFlipObs:
    """Observation 7×12×12: channel flip and static legal-plane invariance."""

    def test_shape_preserved(self):
        """Shape unchanged."""
        obs = np.random.rand(7, ROWS, COLS).astype(np.float32)
        flipped = flip_obs(obs)
        assert flipped.shape == obs.shape

    def test_double_flip_identity(self):
        """Double flip is identity."""
        obs = np.random.rand(7, ROWS, COLS).astype(np.float32)
        assert np.array_equal(flip_obs(flip_obs(obs)), obs)

    def test_valid_mask_channel_invariant(self):
        """Channel 2 = global legal mask stays identical after flip."""
        obs = np.zeros((7, ROWS, COLS), dtype=np.float32)
        obs[2] = VALID_MASK.astype(np.float32)
        flipped = flip_obs(obs)
        np.testing.assert_array_equal(flipped[2], obs[2])

    def test_batch_shape(self):
        """Batch dimension flips with the rest."""
        batch = np.random.rand(4, 7, ROWS, COLS).astype(np.float32)
        flipped = flip_obs(batch)
        assert flipped.shape == batch.shape
        assert np.array_equal(flip_obs(flip_obs(batch)), batch)


class TestFlipMask:
    """96-d legal mask permutes consistently with ``flip_action``."""

    def test_shape_preserved(self):
        """Length stays 96."""
        mask = np.zeros(NUM_VALID, dtype=np.int8)
        for i in range(10):
            mask[i] = 1
        flipped = flip_mask(mask)
        assert flipped.shape == mask.shape

    def test_double_flip_identity(self):
        """Double flip is identity."""
        mask = np.random.randint(0, 2, size=NUM_VALID).astype(np.int8)
        assert np.array_equal(flip_mask(flip_mask(mask)), mask)

    def test_valid_count_preserved(self):
        """Count of legal bits unchanged."""
        mask = np.random.randint(0, 2, size=NUM_VALID).astype(np.int8)
        flipped = flip_mask(mask)
        assert flipped.sum() == mask.sum()

    def test_consistent_with_flip_action(self):
        """Ones in the mask align with ``flip_action`` geometry."""
        mask = np.zeros(NUM_VALID, dtype=np.int8)
        chosen = [(0, 4), (4, 2), (8, 0), (8, 11)]
        for r, c in chosen:
            mask[POS_TO_INDEX[(r, c)]] = 1
        flipped = flip_mask(mask)
        for r, c in chosen:
            expected_idx = POS_TO_INDEX[(r, COLS - 1 - c)]
            assert flipped[expected_idx] == 1

    def test_batch_shape(self):
        """Batched masks."""
        batch = np.random.randint(0, 2, size=(8, NUM_VALID)).astype(np.int8)
        flipped = flip_mask(batch)
        assert flipped.shape == batch.shape
        assert np.array_equal(flip_mask(flip_mask(batch)), batch)


class TestJointConsistency:
    """Same board: flipped obs and mask still describe the mirrored position."""

    def test_obs_and_mask_agree(self):
        """Two pieces placed; after flip, pieces and empty bits match mirrored coords."""
        board = np.zeros((ROWS, COLS), dtype=np.int8)
        board[0, 4] = 1
        board[4, 9] = 2
        obs = np.zeros((7, ROWS, COLS), dtype=np.float32)
        obs[0] = (board == 1).astype(np.float32)
        obs[1] = (board == 2).astype(np.float32)
        obs[2] = VALID_MASK.astype(np.float32)
        mask = np.zeros(NUM_VALID, dtype=np.int8)
        for idx, (r, c) in enumerate(VALID_POSITIONS):
            if board[r, c] == 0:
                mask[idx] = 1

        fobs = flip_obs(obs)
        fmask = flip_mask(mask)

        assert fobs[0, 0, 7] == 1.0
        assert fobs[1, 4, 2] == 1.0
        assert fmask[POS_TO_INDEX[(0, 7)]] == 0
        assert fmask[POS_TO_INDEX[(4, 2)]] == 0
