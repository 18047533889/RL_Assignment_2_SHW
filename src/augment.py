"""
augment.py — Left-right symmetry (mirror columns) for observations, masks, and actions.

**Quick read:** ``FLIP_TABLE`` precomputes **compact 96-dim** index → mirrored compact index.
Used by ``augment_learner`` to flip obs, mask, logits, and discrete actions together.

The board is symmetric about col=5.5 (left-right mirror).  Flipping columns
preserves VALID_MASK exactly, so we can double effective training data by
mirroring (obs, action_mask, action) jointly with zero extra sampling cost.

Usage in learner batch preprocessing:
    With 50% probability, apply flip_obs / flip_mask / flip_action to an
    entire trajectory (all timesteps together — never mix flipped and
    unflipped within the same episode fragment).
"""

from __future__ import annotations

import numpy as np

from board import COLS, NUM_VALID, VALID_POSITIONS, POS_TO_INDEX

_FLIP_TABLE: np.ndarray = np.empty(NUM_VALID, dtype=np.int64)
for _idx, (_r, _c) in enumerate(VALID_POSITIONS):
    _mc = COLS - 1 - _c
    _FLIP_TABLE[_idx] = POS_TO_INDEX[(_r, _mc)]

FLIP_TABLE: np.ndarray = _FLIP_TABLE


def flip_action(compact: int) -> int:
    return int(FLIP_TABLE[compact])


def flip_obs(obs: np.ndarray) -> np.ndarray:
    return np.flip(obs, axis=-1).copy()


def flip_mask(mask: np.ndarray) -> np.ndarray:
    return mask[..., FLIP_TABLE].copy()
