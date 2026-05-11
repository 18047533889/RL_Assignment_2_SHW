"""
self_play.py — League self-play opponent pool (no Ray/RLlib dependency).

Manages frozen opponent snapshots, weighted selection, win-rate tracking,
and threshold-based snapshotting. Pure Python + PyTorch state_dict copies.
"""

from __future__ import annotations

import copy
from collections import deque

import numpy as np

from config import OPPONENT_MODULE_IDS, SELF_PLAY_CONFIG


SCRIPTED_TYPES = [
    "random_legal",
    "heuristic",
    "line_rusher",
    "center_biased",
    "edge_explorer",
    "row_rusher",
    "col_rusher",
    "greedy_tactical",
    "lookahead_scripted",
    "pure_defender",
]

# Per-type sampling weights inside the scripted branch. Tuned post-1800 to push the agent
# against defenders it was failing to break through while keeping attacker/rusher diversity.
SCRIPTED_WEIGHTS = {
    "random_legal":       0.3,
    "heuristic":          2.0,
    "line_rusher":        0.8,
    "center_biased":      1.0,
    "edge_explorer":      0.6,
    "row_rusher":         0.8,
    "col_rusher":         0.8,
    "greedy_tactical":    2.2,
    "lookahead_scripted": 2.8,
    "pure_defender":      2.0,
}
_SCRIPTED_W = np.array([SCRIPTED_WEIGHTS[t] for t in SCRIPTED_TYPES], dtype=np.float64)
_SCRIPTED_W /= _SCRIPTED_W.sum()


class OpponentPool:
    """League-style frozen opponent pool with round-robin snapshotting."""

    def __init__(self, num_slots: int = len(OPPONENT_MODULE_IDS)):
        self._slots: list[dict | None] = [None] * num_slots
        self._snapshot_rr = 0
        self._win_history: deque[int] = deque(maxlen=100)
        self._loss_history: deque[int] = deque(maxlen=100)
        self._draw_history: deque[int] = deque(maxlen=100)
        self._per_slot_wins: list[deque] = [deque(maxlen=50) for _ in range(num_slots)]
        self._per_slot_games: list[deque] = [deque(maxlen=50) for _ in range(num_slots)]
        self._last_opponent_slot: int | None = None
        self._pfsp_min_games = 10

    def initialize(self, main_state_dict: dict) -> None:
        for i in range(len(self._slots)):
            self._slots[i] = copy.deepcopy(main_state_dict)

    def save_snapshot(self, main_state_dict: dict) -> int:
        idx = self._snapshot_rr % len(self._slots)
        self._slots[idx] = copy.deepcopy(main_state_dict)
        self._snapshot_rr += 1
        return idx

    def should_snapshot(self) -> bool:
        wr = self.win_rate()
        if wr is None:
            return False
        return wr >= SELF_PLAY_CONFIG["win_rate_threshold"]

    def _pfsp_sample_history_slot(self, rng: np.random.Generator) -> int:
        n = len(self._slots)
        if n <= 1:
            return 0
        weights = np.ones(n, dtype=np.float64)
        for i in range(n):
            games = sum(self._per_slot_games[i])
            if games >= self._pfsp_min_games:
                wr = sum(self._per_slot_wins[i]) / games
                weights[i] = (1.0 - wr) ** 2 + 0.05
            else:
                weights[i] = 1.0
        weights /= weights.sum()
        return int(rng.choice(n, p=weights))

    def get_opponent_state(self, rng: np.random.Generator) -> tuple[dict | None, str | None]:
        p0 = float(SELF_PLAY_CONFIG["initial_snapshot_prob"])
        p1 = float(SELF_PLAY_CONFIG["history_snapshot_prob"])
        p2 = float(SELF_PLAY_CONFIG["current_self_prob"])
        p3 = float(SELF_PLAY_CONFIG.get("scripted_prob", 0.0))
        s = p0 + p1 + p2 + p3
        if s <= 0:
            return self._slots[0], None
        p0, p1, p2, p3 = p0 / s, p1 / s, p2 / s, p3 / s

        branch = rng.choice(4, p=[p0, p1, p2, p3])
        if branch == 0:
            self._last_opponent_slot = 0
            return self._slots[0], None
        if branch == 1:
            idx = self._pfsp_sample_history_slot(rng)
            self._last_opponent_slot = idx
            return self._slots[idx], None
        if branch == 2:
            idx = int(rng.integers(0, len(self._slots)))
            self._last_opponent_slot = idx
            return self._slots[idx], None
        self._last_opponent_slot = None
        scripted_type = SCRIPTED_TYPES[int(rng.choice(len(SCRIPTED_TYPES), p=_SCRIPTED_W))]
        return None, scripted_type

    def record_outcome(self, outcome: str) -> None:
        win = 1 if outcome == "main_win" else 0
        if outcome == "main_win":
            self._win_history.append(1)
            self._loss_history.append(0)
            self._draw_history.append(0)
        elif outcome == "main_loss":
            self._win_history.append(0)
            self._loss_history.append(1)
            self._draw_history.append(0)
        else:
            self._win_history.append(0)
            self._loss_history.append(0)
            self._draw_history.append(1)
        if self._last_opponent_slot is not None:
            idx = self._last_opponent_slot
            if 0 <= idx < len(self._per_slot_wins):
                self._per_slot_wins[idx].append(win)
                self._per_slot_games[idx].append(1)

    def win_rate(self) -> float | None:
        w = sum(self._win_history)
        l = sum(self._loss_history)
        d = sum(self._draw_history)
        total = w + l + d
        if total == 0:
            return None
        return w / total

    @property
    def snapshot_count(self) -> int:
        return self._snapshot_rr
