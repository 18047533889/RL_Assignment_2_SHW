"""
env.py — PettingZoo AEC environment for triangular Super Tic-Tac-Toe.

**Quick read:** ``_get_obs`` builds ego-centric 7-plane observations. ``step`` maps
compact action (0..95) -> aim (r,c) -> ``stochastic.resolve_placement`` -> win/draw/cap.
``env_creator`` is the RLlib entry (optional ``config["seed"]``).

Observations: (7,12,12) float32:
  ch0: my pieces, ch1: opponent pieces, ch2: valid mask,
  ch3: opponent last move (one-hot), ch4: my threat heatmap,
  ch5: opponent threat heatmap, ch6: forfeit probability map.
Actions: 0..95 (compact) with ``action_mask`` marking empty valid cells.

Step shaping (all zero-sum, multiplied by ``_shaping_multiplier`` except forfeit):
  - Successful placement: +0.01 base.
  - Blocking opponent threats (reducing their ``count_threats``): +0.10 per threat removed.
  - Creating own threats (increasing own ``count_threats``): +0.06 per threat added.
  - Persistence: +0.08 when aim cell is on own threat_heatmap (4-in-row completer) and
    placement succeeded — rewards repeat attempts on the same winning line.
  - Partial block: +0.04 × min(opp_partial_before[landing], 3) — occupying a cell in an
    opponent 2-of-4+0 window (before opp has formed a 3-in-row threat).
  - Waste: -0.015 when placement creates/blocks nothing and is Chebyshev distance > 2
    from any other own piece. Opening-exempt: skipped while own_count < 3 (first 3
    own placements), so legitimate opening spacing plays aren't punished.
  - Forfeit (stochastic miss or illegal aim): -0.015 (not annealed).
  - Terminal win: +1.0 (overrides all shaping); loss: -1.0.
  - Shaping annealing: ``_shaping_multiplier`` decays per ``SHAPING_SCHEDULE`` in config.py.

Curriculum opening: with ``random_opening_prob`` probability, the first
``random_opening_steps`` steps of an episode use uniform random legal actions instead
of the policy. This exposes the agent to diverse board states.

Forfeit injection: with ``forfeit_injection_prob`` per step (max once per episode),
the current agent's action is discarded and forced to a FORFEIT. The board does
not change but the FORFEIT penalty is applied. Exposes the agent to "just forfeited"
states so value-function can learn recovery (vs the naturally zero ``ff_rate``).
Per-step flags are exposed via ``_last_step_forfeit_injected`` and
``_last_step_blocked`` for rollout-level metric aggregation.

RLlib packs obs as ``{"observations", "action_mask"}``. Finished agents must take
``step(None)`` (PettingZoo); handled via ``_was_dead_step``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector

from board import (
    ROWS, COLS, GRID_SIZE, NUM_VALID, VALID_MASK, VALID_POSITIONS,
    empty_board, compact_to_rc, POS_TO_INDEX,
)
from rules import (
    check_win, count_threats, threat_heatmap, partial_threat_heatmap,
    row_threat_heatmap, col_threat_heatmap, is_draw,
    check_win_batch, count_threats_batch,
    threat_heatmap_batch, partial_threat_heatmap_batch,
    row_threat_heatmap_batch, col_threat_heatmap_batch,
)
from stochastic import resolve_placement, forfeit_probability_map, placement_distribution

PLAYER_IDS = ["player_0", "player_1"]
PLAYER_MAP = {"player_0": 1, "player_1": 2}
PLAYER_INV = {1: "player_0", 2: "player_1"}

OBS_CHANNELS = 7
MAX_STEPS = 200
FORFEIT_PENALTY = -0.015
PLACEMENT_BONUS = 0.01
BLOCK_BONUS = 0.10
THREAT_BONUS = 0.06
PERSIST_BONUS = 0.08
PARTIAL_BLOCK_BONUS = 0.04
PARTIAL_BLOCK_CAP = 3.0
WASTE_PENALTY = 0.015
WASTE_DISTANCE = 2


def heuristic_action(board: np.ndarray, player: int, rng: np.random.Generator) -> int:
    opp = 3 - player
    opp_hmap = threat_heatmap(board, opp)
    my_hmap = threat_heatmap(board, player)
    opp_partial = partial_threat_heatmap(board, opp)
    my_partial = partial_threat_heatmap(board, player)

    best_score = -1.0
    best_actions: list[int] = []
    for idx, (r, c) in enumerate(VALID_POSITIONS):
        if board[r, c] != 0:
            continue
        score = (opp_hmap[r, c] * 5.0
                 + my_hmap[r, c] * 3.0
                 + opp_partial[r, c] * 1.0
                 + my_partial[r, c] * 0.5)
        if score > best_score:
            best_score = score
            best_actions = [idx]
        elif score == best_score:
            best_actions.append(idx)

    if best_score > 0 and best_actions:
        return int(rng.choice(best_actions))

    legal = [idx for idx, (r, c) in enumerate(VALID_POSITIONS) if board[r, c] == 0]
    if not legal:
        return 0
    return int(rng.choice(legal))


def random_legal_action(board: np.ndarray, rng: np.random.Generator) -> int:
    legal = [idx for idx, (r, c) in enumerate(VALID_POSITIONS) if board[r, c] == 0]
    if not legal:
        return 0
    return int(rng.choice(legal))


_VALID_FLAT: np.ndarray = VALID_MASK.reshape(-1).astype(bool)


def _opp_winning_per_board(next_boards_flat: np.ndarray, opp_threat_hmap: np.ndarray) -> np.ndarray:
    """(B, 144) board + (B, 144) opp threat heatmap -> (B,) max P(opp wins in one ply).

    Closed-form for the standard 1/2 direct + 1/16 × 8-neighbor placement kernel:
    P(opp wins | opp aims at (tr, tc)) = 0.5 * W[tr, tc] + (1/16) * Σ W[neighbour]
    where W = (opp_threat > 0) & empty & valid -- cells where placing opp completes a line.
    The max is taken over all empty+valid aim cells.
    """
    B = next_boards_flat.shape[0]
    empty_valid = (next_boards_flat == 0) & _VALID_FLAT[None, :]
    W = ((opp_threat_hmap > 0) & empty_valid).astype(np.float32)
    W2d = W.reshape(B, ROWS, COLS)
    pad = np.pad(W2d, ((0, 0), (1, 1), (1, 1)))
    neighbor_sum = (
        pad[:, 0:ROWS, 0:COLS]     + pad[:, 0:ROWS, 1:COLS + 1]     + pad[:, 0:ROWS, 2:COLS + 2]
        + pad[:, 1:ROWS + 1, 0:COLS]                                + pad[:, 1:ROWS + 1, 2:COLS + 2]
        + pad[:, 2:ROWS + 2, 0:COLS] + pad[:, 2:ROWS + 2, 1:COLS + 1] + pad[:, 2:ROWS + 2, 2:COLS + 2]
    )
    opp_win_aim = 0.5 * W2d + (1.0 / 16.0) * neighbor_sum
    aim_mask = empty_valid.reshape(B, ROWS, COLS)
    opp_win_aim = np.where(aim_mask, opp_win_aim, 0.0)
    return opp_win_aim.reshape(B, -1).max(axis=-1)


def _tactical_ev_scores(
    board: np.ndarray,
    player: int,
    *,
    opp_reply_weight: float,
) -> np.ndarray:
    """Batched EV score (shape (NUM_VALID,)) for every compact action.

    Mirrors the old ``_expected_tactical_value`` semantics with two differences:
    - The inner opp-reply scan is replaced by an exact closed-form over all
      empty+valid aim cells (no scan_cap truncation — exact beats approximate).
    - All heatmap / win / count-threat evaluations run once in numpy for the full
      outcome batch instead of per-(action, outcome) Python loops.

    Illegal (occupied) actions receive -inf so argmax-with-eps naturally skips them.
    ``placement_distribution`` is still called once per legal aim so tests that
    monkey-patch it keep working.
    """
    opp = 3 - player
    board_flat = board.reshape(-1).astype(np.int8)
    scores = np.full(NUM_VALID, -np.inf, dtype=np.float64)

    act_ids: list[int] = []
    flat_outs: list[int] = []
    probs_list: list[float] = []
    forfeit_contrib = np.zeros(NUM_VALID, dtype=np.float64)
    legal_mask = np.zeros(NUM_VALID, dtype=bool)

    for idx, (r, c) in enumerate(VALID_POSITIONS):
        if board[r, c] != 0:
            continue
        legal_mask[idx] = True
        dist = placement_distribution(board, r, c)
        for outcome, prob in dist.items():
            if outcome is None:
                forfeit_contrib[idx] += prob * -60.0
            else:
                act_ids.append(idx)
                flat_outs.append(outcome[0] * COLS + outcome[1])
                probs_list.append(prob)

    scores[legal_mask] = forfeit_contrib[legal_mask]
    if not act_ids:
        return scores

    B = len(act_ids)
    a_idx = np.asarray(act_ids, dtype=np.int64)
    flat_out = np.asarray(flat_outs, dtype=np.int64)
    probs = np.asarray(probs_list, dtype=np.float64)

    next_boards = np.broadcast_to(board_flat, (B, GRID_SIZE)).copy()
    next_boards[np.arange(B), flat_out] = player

    wins = check_win_batch(next_boards, player)
    my_hmap = threat_heatmap_batch(next_boards, player)
    opp_hmap = threat_heatmap_batch(next_boards, opp)
    my_partial = partial_threat_heatmap_batch(next_boards, player)
    opp_partial = partial_threat_heatmap_batch(next_boards, opp)
    row_hmap = row_threat_heatmap_batch(next_boards, player)
    col_hmap = col_threat_heatmap_batch(next_boards, player)
    my_cnt_after = count_threats_batch(next_boards, player).astype(np.float64)
    opp_cnt_after = count_threats_batch(next_boards, opp).astype(np.float64)

    my_cnt_before = float(count_threats(board, player))
    opp_cnt_before = float(count_threats(board, opp))

    b_rows = np.arange(B)
    continuation = (
        my_hmap[b_rows, flat_out] * 20.0
        + my_partial[b_rows, flat_out] * 8.0
        + row_hmap[b_rows, flat_out] * 2.0
        + col_hmap[b_rows, flat_out] * 2.0
    )
    blocking = (opp_cnt_before - opp_cnt_after) * 80.0 + opp_partial[b_rows, flat_out] * 6.0
    pressure = (my_cnt_after - my_cnt_before) * 40.0
    safety = -(opp_cnt_after * 18.0) - opp_hmap[b_rows, flat_out] * 12.0
    centrality = _CENTRALITY_FLAT[flat_out] * 0.15
    position_score = continuation + blocking + pressure + safety + centrality

    opp_winning = _opp_winning_per_board(next_boards, opp_hmap)
    ev = np.where(wins, 1000.0, position_score - opp_winning * opp_reply_weight)

    np.add.at(scores, a_idx, probs * ev)
    return scores


def _argmax_tie_break(scores: np.ndarray, rng: np.random.Generator) -> int:
    legal = np.isfinite(scores)
    if not legal.any():
        return 0
    masked = np.where(legal, scores, -np.inf)
    best_val = masked.max()
    candidates = np.flatnonzero(legal & (scores >= best_val - 1e-9))
    return int(rng.choice(candidates))


def line_rusher_action(board: np.ndarray, player: int, rng: np.random.Generator) -> int:
    my_partial = partial_threat_heatmap(board, player)
    my_threat = threat_heatmap(board, player)

    best_score = -1.0
    best_actions: list[int] = []
    for idx, (r, c) in enumerate(VALID_POSITIONS):
        if board[r, c] != 0:
            continue
        score = my_threat[r, c] * 10.0 + my_partial[r, c] * 3.0
        if score > best_score:
            best_score = score
            best_actions = [idx]
        elif score == best_score:
            best_actions.append(idx)

    if best_score > 0 and best_actions:
        return int(rng.choice(best_actions))

    legal = [idx for idx, (r, c) in enumerate(VALID_POSITIONS) if board[r, c] == 0]
    if not legal:
        return 0
    return int(rng.choice(legal))


def row_rusher_action(board: np.ndarray, player: int, rng: np.random.Generator) -> int:
    my_hmap = row_threat_heatmap(board, player)
    best_score = -1.0
    best_actions: list[int] = []
    for idx, (r, c) in enumerate(VALID_POSITIONS):
        if board[r, c] != 0:
            continue
        score = my_hmap[r, c]
        if score > best_score:
            best_score = score
            best_actions = [idx]
        elif score == best_score:
            best_actions.append(idx)
    if best_score > 0 and best_actions:
        return int(rng.choice(best_actions))
    legal = [idx for idx, (r, c) in enumerate(VALID_POSITIONS) if board[r, c] == 0]
    if not legal:
        return 0
    return int(rng.choice(legal))


def col_rusher_action(board: np.ndarray, player: int, rng: np.random.Generator) -> int:
    my_hmap = col_threat_heatmap(board, player)
    best_score = -1.0
    best_actions: list[int] = []
    for idx, (r, c) in enumerate(VALID_POSITIONS):
        if board[r, c] != 0:
            continue
        score = my_hmap[r, c]
        if score > best_score:
            best_score = score
            best_actions = [idx]
        elif score == best_score:
            best_actions.append(idx)
    if best_score > 0 and best_actions:
        return int(rng.choice(best_actions))
    legal = [idx for idx, (r, c) in enumerate(VALID_POSITIONS) if board[r, c] == 0]
    if not legal:
        return 0
    return int(rng.choice(legal))


_CENTER_R, _CENTER_C = ROWS / 2.0, COLS / 2.0
_CENTER_SCORES: dict[int, float] = {}
_CENTRALITY_FLAT: np.ndarray = np.zeros(GRID_SIZE, dtype=np.float32)
for _idx, (_r, _c) in enumerate(VALID_POSITIONS):
    _dist = ((_r - _CENTER_R) ** 2 + (_c - _CENTER_C) ** 2) ** 0.5
    _val = max(0.0, 10.0 - _dist)
    _CENTER_SCORES[_idx] = _val
    _CENTRALITY_FLAT[_r * COLS + _c] = _val
_EDGE_SCORES: dict[int, float] = {_idx: _dist for _idx, ((_r, _c), _dist) in
    enumerate(zip(VALID_POSITIONS, [
        ((_r - _CENTER_R) ** 2 + (_c - _CENTER_C) ** 2) ** 0.5
        for _r, _c in VALID_POSITIONS
    ]))}


def center_biased_action(board: np.ndarray, player: int, rng: np.random.Generator) -> int:
    opp = 3 - player
    opp_hmap = threat_heatmap(board, opp)

    best_score = -999.0
    best_actions: list[int] = []
    for idx, (r, c) in enumerate(VALID_POSITIONS):
        if board[r, c] != 0:
            continue
        score = _CENTER_SCORES[idx] + opp_hmap[r, c] * 5.0
        if score > best_score:
            best_score = score
            best_actions = [idx]
        elif score == best_score:
            best_actions.append(idx)

    if best_actions:
        return int(rng.choice(best_actions))
    return 0


def edge_explorer_action(board: np.ndarray, player: int, rng: np.random.Generator) -> int:
    opp = 3 - player
    opp_hmap = threat_heatmap(board, opp)

    best_score = -999.0
    best_actions: list[int] = []
    for idx, (r, c) in enumerate(VALID_POSITIONS):
        if board[r, c] != 0:
            continue
        score = _EDGE_SCORES[idx] + opp_hmap[r, c] * 5.0
        if score > best_score:
            best_score = score
            best_actions = [idx]
        elif score == best_score:
            best_actions.append(idx)

    if best_actions:
        return int(rng.choice(best_actions))
    return 0


def greedy_tactical_action(board: np.ndarray, player: int, rng: np.random.Generator) -> int:
    scores = _tactical_ev_scores(board, player, opp_reply_weight=400.0)
    return _argmax_tie_break(scores, rng)


def lookahead_scripted_action(board: np.ndarray, player: int, rng: np.random.Generator) -> int:
    """Stronger than ``greedy_tactical_action``: same EV model with higher penalty on opp one-ply wins."""
    scores = _tactical_ev_scores(board, player, opp_reply_weight=620.0)
    return _argmax_tie_break(scores, rng)


_VALID_ROWS: np.ndarray = np.array([r for r, _ in VALID_POSITIONS], dtype=np.int64)
_VALID_COLS: np.ndarray = np.array([c for _, c in VALID_POSITIONS], dtype=np.int64)


def pure_defender_action(board: np.ndarray, player: int, rng: np.random.Generator) -> int:
    """Defense-first scripted opponent: massively prioritises blocking opp threats and
    2-of-4 partials over own attack. Used to force the main agent to learn how to
    break through a committed defender rather than just out-rush weak opponents.

    Vectorized: all scoring done in numpy, no per-cell Python loop.
    """
    opp = 3 - player
    opp_threat = threat_heatmap(board, opp)
    opp_partial = partial_threat_heatmap(board, opp)
    my_threat = threat_heatmap(board, player)
    my_partial = partial_threat_heatmap(board, player)

    score_grid = (opp_threat * 100.0
                  + opp_partial * 30.0
                  + my_threat * 20.0
                  + my_partial * 5.0)
    legal = (board[_VALID_ROWS, _VALID_COLS] == 0)
    scores = np.where(legal, score_grid[_VALID_ROWS, _VALID_COLS], -np.inf)

    if not legal.any():
        return 0
    best_val = scores.max()
    if best_val > 0:
        candidates = np.flatnonzero(scores >= best_val - 1e-9)
        return int(rng.choice(candidates))
    legal_idx = np.flatnonzero(legal)
    return int(rng.choice(legal_idx))


class SuperTicTacToeEnv(AECEnv):
    metadata = {"name": "super_ttt_v1", "is_parallelizable": False}

    def __init__(
        self,
        seed: int | None = None,
        random_opening_prob: float = 0.0,
        random_opening_steps: int = 0,
        forfeit_injection_prob: float = 0.0,
    ):
        super().__init__()
        self.possible_agents = list(PLAYER_IDS)
        self.agents = list(PLAYER_IDS)
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.board = empty_board()
        self._rng = np.random.default_rng(seed)

        self.rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos: dict[str, dict[str, Any]] = {a: {} for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self._last_action: int | None = None
        self._last_placement: tuple[int, int] | None = None
        self._step_count = 0
        self._last_move_plane = np.zeros((ROWS, COLS), dtype=np.float32)
        self._zero_plane = np.zeros((ROWS, COLS), dtype=np.float32)
        self._random_opening_prob = random_opening_prob
        self._random_opening_steps = random_opening_steps
        self._use_random_opening = False
        self._shaping_multiplier = 1.0
        self._forfeit_injection_prob = forfeit_injection_prob
        self._injected_forfeit_done = False
        self._last_step_forfeit_injected = False
        self._last_step_blocked = False
        self._last_step_persist = False
        self._last_step_partial_block = False
        self._last_step_waste = False
        self._last_step_main_succeeded = False

    def observation_space(self, agent: str) -> spaces.Dict:
        return spaces.Dict({
            "observations": spaces.Box(0.0, 1.0, shape=(OBS_CHANNELS, ROWS, COLS), dtype=np.float32),
            "action_mask": spaces.Box(0, 1, shape=(NUM_VALID,), dtype=np.int8),
        })

    def action_space(self, agent: str) -> spaces.Discrete:
        return spaces.Discrete(NUM_VALID)

    def _get_obs(self, agent: str) -> dict[str, np.ndarray]:
        pid = PLAYER_MAP[agent]
        opp = 3 - pid
        obs = np.zeros((OBS_CHANNELS, ROWS, COLS), dtype=np.float32)
        obs[0] = (self.board == pid).astype(np.float32)
        obs[1] = (self.board == opp).astype(np.float32)
        obs[2] = VALID_MASK.astype(np.float32)
        obs[3] = self._last_move_plane
        my_hmap = threat_heatmap(self.board, pid)
        opp_hmap = threat_heatmap(self.board, opp)
        max_val = max(my_hmap.max(), opp_hmap.max(), 1.0)
        obs[4] = my_hmap / max_val
        obs[5] = opp_hmap / max_val
        fp = forfeit_probability_map(self.board)
        obs[6] = fp
        return {"observations": obs, "action_mask": self._action_mask()}

    def _action_mask(self) -> np.ndarray:
        rows, cols = zip(*VALID_POSITIONS)
        return (self.board[rows, cols] == 0).astype(np.int8)

    def observe(self, agent: str) -> dict[str, np.ndarray]:
        return self._get_obs(agent)

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.agents = list(PLAYER_IDS)
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self.board = empty_board()
        self.rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self._last_action = None
        self._last_placement = None
        self._step_count = 0
        self._last_move_plane = self._zero_plane
        self._use_random_opening = (
            self._random_opening_prob > 0
            and self._rng.random() < self._random_opening_prob
        )
        self._injected_forfeit_done = False
        self._last_step_forfeit_injected = False
        self._last_step_blocked = False
        self._last_step_persist = False
        self._last_step_partial_block = False
        self._last_step_waste = False
        self._last_step_main_succeeded = False

    def step(self, action):
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(None)
            return

        self._last_step_forfeit_injected = False
        self._last_step_blocked = False
        self._last_step_persist = False
        self._last_step_partial_block = False
        self._last_step_waste = False
        self._last_step_main_succeeded = False

        if self._use_random_opening and self._step_count < self._random_opening_steps:
            action = random_legal_action(self.board, self._rng)

        force_forfeit = False
        if (self._forfeit_injection_prob > 0.0
                and not self._injected_forfeit_done
                and self._rng.random() < self._forfeit_injection_prob):
            force_forfeit = True
            self._injected_forfeit_done = True
            self._last_step_forfeit_injected = True

        pid = PLAYER_MAP[agent]
        r, c = compact_to_rc(action)

        self.rewards = {a: 0.0 for a in self.agents}
        self._last_action = action
        self._last_placement = None

        self._step_count += 1

        opp_agent = PLAYER_INV[3 - pid]
        sm = self._shaping_multiplier
        if force_forfeit or not VALID_MASK[r, c] or self.board[r, c] != 0:
            self._last_placement = None
            self.rewards[agent] = FORFEIT_PENALTY
            self.rewards[opp_agent] = -FORFEIT_PENALTY
            self._last_move_plane = self._zero_plane
        else:
            opp_pid = 3 - pid
            my_threat_hmap_before = threat_heatmap(self.board, pid)
            opp_partial_hmap_before = partial_threat_heatmap(self.board, opp_pid)
            aim_is_own_threat = bool(my_threat_hmap_before[r, c] > 0)
            opp_threats_before = count_threats(self.board, opp_pid)
            my_threats_before = count_threats(self.board, pid)

            result = resolve_placement(self.board, r, c, self._rng)
            if result is not None:
                pr, pc = result
                self.board[pr, pc] = pid
                self._last_placement = (pr, pc)
                self._last_step_main_succeeded = True
                shaping = PLACEMENT_BONUS * sm
                opp_threats_after = count_threats(self.board, opp_pid)
                my_threats_after = count_threats(self.board, pid)
                blocked = opp_threats_before - opp_threats_after
                if blocked > 0:
                    shaping += BLOCK_BONUS * blocked * sm
                    self._last_step_blocked = True
                created = my_threats_after - my_threats_before
                if created > 0:
                    shaping += THREAT_BONUS * created * sm

                # Persistence: reward aiming at own threat cell (4-in-row completer),
                # regardless of exact landing cell. Pays only when placement succeeded.
                if aim_is_own_threat:
                    shaping += PERSIST_BONUS * sm
                    self._last_step_persist = True

                # Partial block: landed on a cell that was in opp's 2-of-4 windows.
                partial_count = float(opp_partial_hmap_before[pr, pc])
                if partial_count > 0:
                    shaping += PARTIAL_BLOCK_BONUS * min(partial_count, PARTIAL_BLOCK_CAP) * sm
                    self._last_step_partial_block = True

                # Waste: no tactical value and far from any of own existing pieces.
                # Opening exemption: first 3 own pieces never count as waste —
                # legitimate spacing plays in the opening land far from existing
                # pieces and shouldn't be penalised. Waste only applies from the
                # 4th own placement onward (when a shape has begun forming).
                if blocked == 0 and created == 0 and partial_count == 0:
                    own_mask = self.board == pid
                    own_mask[pr, pc] = False
                    own_count = int(own_mask.sum())
                    if own_count >= 3:
                        own_rs, own_cs = np.where(own_mask)
                        dist = int(np.maximum(np.abs(own_rs - pr), np.abs(own_cs - pc)).min())
                        if dist > WASTE_DISTANCE:
                            shaping += -WASTE_PENALTY * sm
                            self._last_step_waste = True

                self.rewards[agent] = shaping
                self.rewards[opp_agent] = -shaping
                plane = np.zeros((ROWS, COLS), dtype=np.float32)
                plane[pr, pc] = 1.0
                self._last_move_plane = plane
            else:
                self.rewards[agent] = FORFEIT_PENALTY
                self.rewards[opp_agent] = -FORFEIT_PENALTY
                self._last_move_plane = self._zero_plane

        win_info = check_win(self.board, pid) if self._last_placement else None
        if win_info is not None:
            self.rewards[agent] = 1.0
            self.rewards[opp_agent] = -1.0
            self.terminations = {a: True for a in self.agents}
            win_dict = {"winner": agent, "win_type": win_info[0],
                        "win_cells": win_info[1]}
            self.infos = {a: dict(win_dict) for a in self.agents}
        elif is_draw(self.board):
            self.terminations = {a: True for a in self.agents}
            self.infos = {a: {"draw": True} for a in self.agents}
        elif self._step_count >= MAX_STEPS:
            self.truncations = {a: True for a in self.agents}
            self.infos = {a: {"draw": True} for a in self.agents}

        self._cumulative_rewards[agent] = 0
        self._accumulate_rewards()
        self.agent_selection = self._agent_selector.next()

    def render(self) -> str:
        symbols = {0: ".", 1: "X", 2: "O"}
        lines = []
        for r in range(ROWS):
            row_str = ""
            for c in range(COLS):
                if VALID_MASK[r, c]:
                    row_str += symbols[self.board[r, c]] + " "
                else:
                    row_str += "  "
            lines.append(row_str.rstrip())
        return "\n".join(lines)


def env_creator(config: dict | None = None) -> SuperTicTacToeEnv:
    config = config or {}
    return SuperTicTacToeEnv(
        seed=config.get("seed"),
        random_opening_prob=config.get("random_opening_prob", 0.0),
        random_opening_steps=config.get("random_opening_steps", 0),
        forfeit_injection_prob=config.get("forfeit_injection_prob", 0.0),
    )
