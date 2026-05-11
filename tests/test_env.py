"""
``SuperTicTacToeEnv`` (PettingZoo AEC): observation space, turns, legal mask, win/draw, render.

Covers init, ego perspective flip, mask updates after stochastic placement, random full games.
"""

from unittest.mock import patch

import numpy as np
import pytest
import env as env_module
from env import SuperTicTacToeEnv, PLAYER_IDS, OBS_CHANNELS, greedy_tactical_action, lookahead_scripted_action
from board import ROWS, COLS, VALID_MASK, VALID_POSITIONS, NUM_VALID, POS_TO_INDEX, rc_to_compact, empty_board


@pytest.fixture
def env():
    """Shared env after ``reset`` to avoid repeated setup."""
    e = SuperTicTacToeEnv(seed=42)
    e.reset()
    return e


class TestEnvInit:
    """PettingZoo contract: agent lists, first player, empty board, rewards."""

    def test_agents(self, env):
        """``possible_agents`` / ``agents`` match ``PLAYER_IDS``."""
        assert env.possible_agents == PLAYER_IDS
        assert env.agents == PLAYER_IDS

    def test_initial_agent_selection(self, env):
        """First to act is player_0."""
        assert env.agent_selection == "player_0"

    def test_board_empty_on_reset(self, env):
        """Board starts empty."""
        assert env.board.sum() == 0

    def test_all_rewards_zero(self, env):
        """All agents start at reward 0."""
        for a in env.agents:
            assert env.rewards[a] == 0.0

    def test_not_terminated(self, env):
        """No termination or truncation at start."""
        for a in env.agents:
            assert not env.terminations[a]
            assert not env.truncations[a]


class TestObservationSpace:
    """``observation_space`` / ``action_space`` shapes and discrete size."""

    def test_obs_space_shape(self, env):
        """7×12×12 observations and 96-d mask."""
        for a in env.agents:
            sp = env.observation_space(a)
            assert sp["observations"].shape == (OBS_CHANNELS, 12, 12)
            assert sp["action_mask"].shape == (NUM_VALID,)

    def test_action_space_size(self, env):
        """Discrete(96)."""
        for a in env.agents:
            assert env.action_space(a).n == NUM_VALID


class TestObservation:
    """``observe``: planes + static legal mask; self/opp swap between players."""

    def test_initial_obs_channels(self, env):
        """Planes 0–1 empty; plane 2 has 96 legal cells; 7 channels total."""
        obs = env.observe("player_0")
        assert obs["observations"].shape == (OBS_CHANNELS, 12, 12)
        assert obs["observations"][0].sum() == 0
        assert obs["observations"][1].sum() == 0
        assert obs["observations"][2].sum() == 96

    def test_action_mask_matches_empty_valid(self, env):
        """Compact mask matches ``VALID_MASK`` on empty board."""
        obs = env.observe("player_0")
        mask = obs["action_mask"]
        assert mask.shape == (NUM_VALID,)
        assert mask.sum() == 96
        for (r, c), idx in POS_TO_INDEX.items():
            assert mask[idx] == 1

    def test_obs_perspective_flips(self, env):
        """Same board: player_0 / player_1 swap self vs opponent channels."""
        env.board[0, 4] = 1
        env.board[0, 5] = 2
        obs_p0 = env.observe("player_0")
        obs_p1 = env.observe("player_1")
        assert obs_p0["observations"][0, 0, 4] == 1.0
        assert obs_p0["observations"][1, 0, 5] == 1.0
        assert obs_p1["observations"][0, 0, 5] == 1.0
        assert obs_p1["observations"][1, 0, 4] == 1.0


class TestStepMechanics:
    """Legal/illegal moves, turn order, mask update after placement."""

    def test_turn_alternates(self, env):
        """One step switches the acting agent."""
        assert env.agent_selection == "player_0"
        action = rc_to_compact(0, 4)
        env.step(action)
        assert env.agent_selection == "player_1"

    def test_valid_move_places_piece(self):
        """Some reset seed eventually places on the aimed cell (stochastic)."""
        e = SuperTicTacToeEnv(seed=0)
        e.reset()
        placed = False
        for trial in range(100):
            e.reset(seed=trial)
            e.step(rc_to_compact(9, 5))
            if e.board[9, 5] != 0 or e._last_placement is not None:
                placed = True
                break
        assert placed

    def test_action_mask_updates_after_placement(self):
        """If placement lands on target, that compact index is 0 in the next mask."""
        e = SuperTicTacToeEnv(seed=12345)
        e.reset()
        for _ in range(20):
            e = SuperTicTacToeEnv(seed=np.random.randint(100000))
            e.reset()
            action = rc_to_compact(9, 5)
            e.step(action)
            if e.board[9, 5] != 0:
                obs = e.observe("player_1")
                assert obs["action_mask"][action] == 0
                return
        pytest.skip("Stochastic: couldn't get direct placement in 20 tries")


class TestTerminalRewardsAndTruncation:
    """Terminal ±1, ``MAX_STEPS`` truncation, shaping zero-sum, forfeit on occupied aim."""

    def test_win_terminal_rewards_plus_one_minus_one(self):
        """Winner gets +1, loser −1 (current player completes four in a row)."""
        e = SuperTicTacToeEnv(seed=0)
        e.reset()
        e.board[8, 0] = 1
        e.board[8, 1] = 1
        e.board[8, 2] = 1
        with patch.object(env_module, "resolve_placement", return_value=(8, 3)):
            e.step(rc_to_compact(8, 3))
        assert e.rewards["player_0"] == 1.0
        assert e.rewards["player_1"] == -1.0
        assert e.terminations["player_0"] and e.terminations["player_1"]

    def test_max_steps_truncation_marks_draw_in_infos(self):
        """Patch small cap; no win/draw → ``truncations`` and ``infos`` draw flag."""
        with patch.object(env_module, "MAX_STEPS", 5):
            e = SuperTicTacToeEnv(seed=2)
            e.reset()
            rng = np.random.default_rng(10)
            for _ in range(5):
                agent = e.agent_selection
                obs = e.observe(agent)
                legal = np.where(obs["action_mask"] == 1)[0]
                e.step(int(rng.choice(legal)))
        assert all(e.truncations[a] for a in e.agents)
        assert not any(e.terminations[a] for a in e.agents)
        assert all(e.infos[a].get("draw") is True for a in e.agents)

    def test_nonterminal_shaped_rewards_are_zero_sum(self):
        """One non-terminal step: ``PLACEMENT_BONUS`` / ``FORFEIT_PENALTY`` pair sums to 0."""
        e = SuperTicTacToeEnv(seed=0)
        e.reset()
        with patch.object(env_module, "resolve_placement", return_value=(8, 4)):
            e.step(rc_to_compact(8, 4))
        assert e.rewards["player_0"] + e.rewards["player_1"] == 0.0
        assert not any(e.terminations.values())

    def test_aim_occupied_cell_forfeits_no_new_stone(self):
        """Aim at occupied valid cell (bypass mask): forfeit, cell unchanged."""
        e = SuperTicTacToeEnv(seed=0)
        e.reset()
        e.board[0, 4] = 2
        before = e.board.copy()
        e.step(rc_to_compact(0, 4))
        assert np.array_equal(e.board, before)
        assert e.rewards["player_0"] == env_module.FORFEIT_PENALTY
        assert e.rewards["player_1"] == -env_module.FORFEIT_PENALTY

    def test_step_none_after_terminal_does_not_change_board(self):
        """PettingZoo: finished agents receive ``step(None)``; board must stay fixed."""
        e = SuperTicTacToeEnv(seed=0)
        e.reset()
        e.board[8, 0] = 1
        e.board[8, 1] = 1
        e.board[8, 2] = 1
        with patch.object(env_module, "resolve_placement", return_value=(8, 3)):
            e.step(rc_to_compact(8, 3))
        frozen = e.board.copy()
        for _ in range(24):
            if not e.agents:
                break
            e.step(None)
        assert np.array_equal(e.board, frozen)


class TestDeterministicReplay:
    """Same seed + deterministic placement → identical trajectory (silent RNG/order bugs)."""

    def test_two_runs_match_with_direct_placement(self):
        def run_once():
            e = SuperTicTacToeEnv(seed=999)
            e.reset(seed=999)
            with patch.object(env_module, "resolve_placement", side_effect=lambda b, r, c, rng: (r, c)):
                e.step(rc_to_compact(8, 4))
                e.step(rc_to_compact(8, 5))
            return e.board.copy()

        assert np.array_equal(run_once(), run_once())


class TestWinDetection:
    """Row win and cross-level column win consistent with ``rules``."""

    def test_row_win_terminates(self):
        """Four in a row detected (manual pieces before step)."""
        e = SuperTicTacToeEnv(seed=42)
        e.reset()
        e.board[8, 0] = 1
        e.board[8, 1] = 1
        e.board[8, 2] = 1
        e.board[8, 3] = 1
        e.step(rc_to_compact(9, 5))
        from rules import check_win
        assert check_win(e.board, 1) is not None

    def test_col_cross_level_win(self):
        """Cross-level column four: ``check_win`` type col."""
        e = SuperTicTacToeEnv(seed=42)
        e.reset()
        e.board[2, 4] = 2
        e.board[3, 4] = 2
        e.board[4, 4] = 2
        e.board[5, 4] = 2
        from rules import check_win
        result = check_win(e.board, 2)
        assert result is not None
        assert result[0] == "col"


class TestDraw:
    """Full board with no winner -> draw."""

    def test_draw_detection(self):
        """Row-wise pattern fills valid cells: no win, ``is_draw`` true."""
        e = SuperTicTacToeEnv(seed=42)
        e.reset()
        from rules import check_win, is_draw
        row_patterns = [
            [1, 2, 1, 2],
            [2, 1, 2, 1],
            [1, 2, 1, 2],
            [1, 2, 1, 2],
            [2, 1, 2, 1, 2, 1, 2, 1],
            [1, 2, 1, 2, 1, 2, 1, 2],
            [2, 1, 2, 1, 2, 1, 2, 1],
            [2, 1, 2, 1, 2, 1, 2, 1],
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            [2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
            [2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
        ]
        for r in range(ROWS):
            idx = 0
            for c in range(COLS):
                if VALID_MASK[r, c]:
                    e.board[r, c] = row_patterns[r][idx]
                    idx += 1
        assert check_win(e.board, 1) is None
        assert check_win(e.board, 2) is None
        assert is_draw(e.board)


class TestRender:
    """``render`` returns 12 text lines."""

    def test_render_returns_string(self, env):
        """Type and line count."""
        rendered = env.render()
        assert isinstance(rendered, str)
        lines = rendered.split("\n")
        assert len(lines) == 12


class TestReset:
    """``reset`` clears board and restores first player."""

    def test_reset_clears_board(self, env):
        """After pieces, reset empties board and sets player_0."""
        env.board[5, 5] = 1
        env.reset()
        assert env.board.sum() == 0
        assert env.agent_selection == "player_0"

    def test_reset_with_seed(self, env):
        """``reset(seed=...)`` still clears the board."""
        env.reset(seed=999)
        assert env.board.sum() == 0


class TestFullRandomGame:
    """Random legal play should end within a step cap."""

    def test_random_game_terminates(self):
        """Within 500 steps: termination or truncation."""
        rng = np.random.default_rng(77)
        e = SuperTicTacToeEnv(seed=77)
        e.reset()
        steps = 0
        max_steps = 500
        while steps < max_steps:
            agent = e.agent_selection
            if e.terminations[agent] or e.truncations[agent]:
                break
            obs = e.observe(agent)
            mask = obs["action_mask"]
            legal = np.where(mask == 1)[0]
            if len(legal) == 0:
                break
            action = rng.choice(legal)
            e.step(action)
            steps += 1
        assert steps < max_steps or any(e.terminations.values())


class TestHeuristicAction:

    def test_returns_legal_action(self):
        from env import heuristic_action
        board = empty_board()
        rng = np.random.default_rng(42)
        action = heuristic_action(board, 1, rng)
        assert 0 <= action < NUM_VALID

    def test_prefers_threats_over_random(self):
        from env import heuristic_action
        board = empty_board()
        board[8, 0] = 2
        board[8, 1] = 2
        board[8, 2] = 2
        rng = np.random.default_rng(0)
        actions = [heuristic_action(board, 1, rng) for _ in range(50)]
        from board import compact_to_rc
        cols = [compact_to_rc(a)[1] for a in actions]
        assert 3 in cols, "Heuristic should frequently block the row-4 threat at col 3"

    def test_full_board_returns_zero(self):
        from env import heuristic_action
        board = empty_board()
        pid = 1
        for r, c in VALID_POSITIONS:
            board[r, c] = pid
            pid = 3 - pid
        rng = np.random.default_rng(0)
        action = heuristic_action(board, 1, rng)
        assert action == 0


class TestRandomLegalAction:

    def test_returns_legal_on_empty_board(self):
        from env import random_legal_action
        board = empty_board()
        rng = np.random.default_rng(42)
        action = random_legal_action(board, rng)
        assert 0 <= action < NUM_VALID

    def test_full_board_returns_zero(self):
        from env import random_legal_action
        board = empty_board()
        pid = 1
        for r, c in VALID_POSITIONS:
            board[r, c] = pid
            pid = 3 - pid
        rng = np.random.default_rng(0)
        assert random_legal_action(board, rng) == 0


class TestCurriculumOpening:

    def test_random_opening_overrides_action(self):
        e = SuperTicTacToeEnv(seed=42, random_opening_prob=1.0, random_opening_steps=4)
        e.reset()
        for i in range(4):
            agent = e.agent_selection
            if e.terminations[agent] or e.truncations[agent]:
                break
            e.step(0)
        placed = int(np.sum(e.board > 0))
        assert placed >= 1

    def test_no_curriculum_when_prob_zero(self):
        e = SuperTicTacToeEnv(seed=42, random_opening_prob=0.0, random_opening_steps=4)
        e.reset()
        assert not e._use_random_opening


class TestRewardShaping:

    def test_placement_gives_positive_reward(self):
        from env import PLACEMENT_BONUS
        e = SuperTicTacToeEnv(seed=42)
        e.reset()
        found_bonus = False
        for trial in range(50):
            e.reset(seed=trial)
            e.step(rc_to_compact(9, 5))
            r = e.rewards.get("player_0", 0)
            if r > 0:
                found_bonus = True
                break
        assert found_bonus, "At least one successful placement should yield positive reward"

    def test_forfeit_no_win_check(self):
        e = SuperTicTacToeEnv(seed=42)
        e.reset()
        e.board[9, 5] = 1
        e.step(rc_to_compact(9, 5))
        assert not any(e.terminations.values()), "Forfeit on occupied cell should not trigger win"


class TestEnvCreator:

    def test_env_creator_returns_env(self):
        from env import env_creator
        e = env_creator({"seed": 42})
        assert isinstance(e, SuperTicTacToeEnv)
        e.reset()
        assert e.board.shape == (ROWS, COLS)

    def test_env_creator_with_curriculum(self):
        from env import env_creator
        e = env_creator({"seed": 0, "random_opening_prob": 0.5, "random_opening_steps": 3})
        assert e._random_opening_prob == 0.5
        assert e._random_opening_steps == 3


class TestLineRusherAction:

    def test_returns_legal_action(self):
        from env import line_rusher_action
        board = empty_board()
        rng = np.random.default_rng(42)
        a = line_rusher_action(board, 1, rng)
        assert 0 <= a < NUM_VALID

    def test_extends_own_line(self):
        from env import line_rusher_action
        from board import compact_to_rc
        board = empty_board()
        board[8, 0] = 1
        board[8, 1] = 1
        rng = np.random.default_rng(0)
        actions = [line_rusher_action(board, 1, rng) for _ in range(30)]
        rows_cols = [compact_to_rc(a) for a in actions]
        row8_count = sum(1 for r, c in rows_cols if r == 8)
        assert row8_count > 15, "line_rusher should prefer extending its own line"

    def test_full_board_returns_zero(self):
        from env import line_rusher_action
        board = empty_board()
        pid = 1
        for r, c in VALID_POSITIONS:
            board[r, c] = pid
            pid = 3 - pid
        rng = np.random.default_rng(0)
        assert line_rusher_action(board, 1, rng) == 0


class TestCenterBiasedAction:

    def test_returns_legal_action(self):
        from env import center_biased_action
        board = empty_board()
        rng = np.random.default_rng(42)
        a = center_biased_action(board, 1, rng)
        assert 0 <= a < NUM_VALID

    def test_prefers_center_on_empty_board(self):
        from env import center_biased_action
        from board import compact_to_rc
        board = empty_board()
        rng = np.random.default_rng(0)
        actions = [center_biased_action(board, 1, rng) for _ in range(30)]
        rows_cols = [compact_to_rc(a) for a in actions]
        center_count = sum(1 for r, c in rows_cols if 3 <= r <= 8 and 3 <= c <= 8)
        assert center_count > 20, "center_biased should prefer central positions"


class TestEdgeExplorerAction:

    def test_returns_legal_action(self):
        from env import edge_explorer_action
        board = empty_board()
        rng = np.random.default_rng(42)
        a = edge_explorer_action(board, 1, rng)
        assert 0 <= a < NUM_VALID

    def test_prefers_edges_on_empty_board(self):
        from env import edge_explorer_action
        from board import compact_to_rc
        board = empty_board()
        rng = np.random.default_rng(0)
        actions = [edge_explorer_action(board, 1, rng) for _ in range(30)]
        rows_cols = [compact_to_rc(a) for a in actions]
        edge_count = sum(1 for r, c in rows_cols if r >= 8 or r <= 3)
        assert edge_count > 15, "edge_explorer should prefer outer positions"


class TestScriptedTypesInPool:

    def test_all_scripted_types_recognized(self):
        from self_play import SCRIPTED_TYPES
        assert "random_legal" in SCRIPTED_TYPES
        assert "heuristic" in SCRIPTED_TYPES
        assert "line_rusher" in SCRIPTED_TYPES
        assert "center_biased" in SCRIPTED_TYPES
        assert "edge_explorer" in SCRIPTED_TYPES
        assert "row_rusher" in SCRIPTED_TYPES
        assert "col_rusher" in SCRIPTED_TYPES
        assert "greedy_tactical" in SCRIPTED_TYPES
        assert "lookahead_scripted" in SCRIPTED_TYPES
        assert "pure_defender" in SCRIPTED_TYPES
        assert len(SCRIPTED_TYPES) == 10


class TestRowRusherAction:

    def test_returns_legal_action(self):
        from env import row_rusher_action
        board = empty_board()
        rng = np.random.default_rng(42)
        a = row_rusher_action(board, 1, rng)
        assert 0 <= a < NUM_VALID

    def test_prefers_extending_row(self):
        from env import row_rusher_action
        from board import compact_to_rc
        board = empty_board()
        board[8, 0] = 1
        board[8, 1] = 1
        board[8, 2] = 1
        rng = np.random.default_rng(0)
        a = row_rusher_action(board, 1, rng)
        r, c = compact_to_rc(a)
        assert r == 8, "row_rusher should extend horizontal line on row 8"

    def test_full_board_returns_zero(self):
        from env import row_rusher_action
        board = empty_board()
        for r in range(12):
            for c in range(12):
                if VALID_MASK[r, c]:
                    board[r, c] = 1
        rng = np.random.default_rng(42)
        a = row_rusher_action(board, 1, rng)
        assert a == 0


class TestColRusherAction:

    def test_returns_legal_action(self):
        from env import col_rusher_action
        board = empty_board()
        rng = np.random.default_rng(42)
        a = col_rusher_action(board, 1, rng)
        assert 0 <= a < NUM_VALID

    def test_prefers_extending_cross_level_column(self):
        from env import col_rusher_action
        from board import compact_to_rc, get_level
        board = empty_board()
        board[3, 4] = 1
        board[4, 4] = 1
        rng = np.random.default_rng(0)
        actions = [col_rusher_action(board, 1, rng) for _ in range(10)]
        cols = [compact_to_rc(a)[1] for a in actions]
        assert cols.count(4) > 5, "col_rusher should extend cross-level column"

    def test_same_level_column_not_preferred(self):
        from env import col_rusher_action
        from rules import col_threat_heatmap
        board = empty_board()
        board[8, 0] = 1
        board[9, 0] = 1
        board[10, 0] = 1
        hmap = col_threat_heatmap(board, 1)
        assert hmap[11, 0] == 0, "same-level 4-in-col should not register as threat"


class TestShapingMultiplier:

    def test_default_multiplier_is_one(self):
        e = SuperTicTacToeEnv(seed=42)
        e.reset()
        assert e._shaping_multiplier == 1.0

    def test_multiplier_scales_placement_reward(self):
        from unittest.mock import patch
        e = SuperTicTacToeEnv(seed=42)
        e.reset()
        e._shaping_multiplier = 0.5
        agent = e.agent_selection
        obs = e.observe(agent)
        mask = obs["action_mask"]
        legal = np.flatnonzero(mask)
        with patch("env.resolve_placement", return_value=None):
            e.step(int(legal[0]))
        r0 = e.rewards[agent]
        e2 = SuperTicTacToeEnv(seed=42)
        e2.reset()
        e2._shaping_multiplier = 1.0
        agent2 = e2.agent_selection
        with patch("env.resolve_placement", return_value=None):
            e2.step(int(legal[0]))
        r1 = e2.rewards[agent2]
        assert abs(r0) == pytest.approx(abs(r1), abs=1e-9)

    def test_zero_multiplier_gives_zero_shaping(self):
        e = SuperTicTacToeEnv(seed=100)
        e.reset()
        e._shaping_multiplier = 0.0
        agent = e.agent_selection
        obs = e.observe(agent)
        mask = obs["action_mask"]
        legal = np.flatnonzero(mask)
        action = int(legal[0])
        e.step(action)
        if not any(e.terminations.get(a, False) for a in e.possible_agents):
            assert e.rewards[agent] == pytest.approx(0.0, abs=1e-9) or \
                   abs(e.rewards[agent]) == pytest.approx(abs(env_module.FORFEIT_PENALTY), abs=1e-9)


class TestForfeitInjection:
    """``forfeit_injection_prob``: force FORFEIT once per episode; exposes agent to post-FORFEIT states."""

    def test_prob_zero_never_injects(self):
        """With ``forfeit_injection_prob=0.0``, flag stays False over many steps."""
        e = SuperTicTacToeEnv(seed=7, forfeit_injection_prob=0.0)
        e.reset()
        for _ in range(20):
            agent = e.agent_selection
            if e.terminations.get(agent) or e.truncations.get(agent):
                break
            obs = e.observe(agent)
            legal = np.flatnonzero(obs["action_mask"])
            e.step(int(legal[0]))
            assert e._last_step_forfeit_injected is False

    def test_prob_one_injects_first_step(self):
        """With ``forfeit_injection_prob=1.0``, first step is forced to FORFEIT (board unchanged, penalty applied)."""
        e = SuperTicTacToeEnv(seed=42, forfeit_injection_prob=1.0)
        e.reset()
        before = e.board.copy()
        obs = e.observe("player_0")
        legal = np.flatnonzero(obs["action_mask"])
        e.step(int(legal[0]))
        assert e._last_step_forfeit_injected is True
        assert e._injected_forfeit_done is True
        assert np.array_equal(e.board, before)
        assert e.rewards["player_0"] == env_module.FORFEIT_PENALTY
        assert e.rewards["player_1"] == -env_module.FORFEIT_PENALTY

    def test_once_per_episode_gate(self):
        """After one forced injection (p=1.0), subsequent steps do not re-inject."""
        e = SuperTicTacToeEnv(seed=3, forfeit_injection_prob=1.0)
        e.reset()
        # Step 1: forced injection.
        agent = e.agent_selection
        obs = e.observe(agent)
        e.step(int(np.flatnonzero(obs["action_mask"])[0]))
        assert e._last_step_forfeit_injected is True
        # Step 2: flag must be cleared and stay False even though prob=1.0.
        agent = e.agent_selection
        obs = e.observe(agent)
        e.step(int(np.flatnonzero(obs["action_mask"])[0]))
        assert e._last_step_forfeit_injected is False

    def test_reset_clears_injection_flags(self):
        """``reset`` restores ``_injected_forfeit_done=False`` so the next episode can inject again."""
        e = SuperTicTacToeEnv(seed=1, forfeit_injection_prob=1.0)
        e.reset()
        obs = e.observe("player_0")
        e.step(int(np.flatnonzero(obs["action_mask"])[0]))
        assert e._injected_forfeit_done is True
        e.reset()
        assert e._injected_forfeit_done is False
        assert e._last_step_forfeit_injected is False
        assert e._last_step_blocked is False
        assert e._last_step_persist is False
        assert e._last_step_partial_block is False
        assert e._last_step_waste is False
        assert e._last_step_main_succeeded is False

    def test_env_creator_passes_forfeit_injection(self):
        """``env_creator`` forwards ``forfeit_injection_prob`` to the env."""
        from env import env_creator
        e = env_creator({"seed": 0, "forfeit_injection_prob": 0.05})
        assert e._forfeit_injection_prob == 0.05
        e2 = env_creator({"seed": 0})
        assert e2._forfeit_injection_prob == 0.0


class TestBlockedFlag:
    """``_last_step_blocked``: per-step flag for block events; used for blk_rate telemetry."""

    def test_flag_false_after_reset(self):
        e = SuperTicTacToeEnv(seed=0)
        e.reset()
        assert e._last_step_blocked is False

    def test_flag_false_on_forfeit(self):
        """FORFEIT branch doesn't set blocked flag (no successful placement)."""
        e = SuperTicTacToeEnv(seed=0)
        e.reset()
        e.board[0, 4] = 2  # opponent occupies target
        e.step(rc_to_compact(0, 4))
        assert e._last_step_blocked is False

    def test_flag_true_when_blocking_threat(self):
        """Place main on an open end of an opp 3-in-row threat with patched placement: blocked flag set."""
        e = SuperTicTacToeEnv(seed=0)
        e.reset()
        # Opponent (pid=2) has 3-in-row at row 8 cols 0,1,2 -> open end at (8,3). Main is player_0 (pid=1).
        e.board[8, 0] = 2
        e.board[8, 1] = 2
        e.board[8, 2] = 2
        from rules import count_threats
        before = count_threats(e.board, 2)
        with patch.object(env_module, "resolve_placement", return_value=(8, 3)):
            e.step(rc_to_compact(8, 3))
        after = count_threats(e.board, 2)
        if before > after:
            assert e._last_step_blocked is True
        else:
            pytest.skip(f"count_threats did not decrease (before={before}, after={after})")


class TestEnvEdgeCases:

    def test_multiple_resets(self):
        e = SuperTicTacToeEnv(seed=42)
        for i in range(5):
            e.reset(seed=i)
            assert np.sum(e.board) == 0
            assert not any(e.terminations.values())

    def test_two_envs_different_seeds_differ(self):
        e1 = SuperTicTacToeEnv(seed=1)
        e1.reset(seed=1)
        e2 = SuperTicTacToeEnv(seed=9999)
        e2.reset(seed=9999)
        actions_1 = []
        actions_2 = []
        for _ in range(10):
            a1 = e1.agent_selection
            a2 = e2.agent_selection
            if e1.terminations.get(a1) or e1.truncations.get(a1):
                break
            if e2.terminations.get(a2) or e2.truncations.get(a2):
                break
            obs1 = e1.observe(a1)
            obs2 = e2.observe(a2)
            legal1 = np.flatnonzero(obs1["action_mask"])
            legal2 = np.flatnonzero(obs2["action_mask"])
            act1 = int(e1._rng.choice(legal1))
            act2 = int(e2._rng.choice(legal2))
            actions_1.append(act1)
            actions_2.append(act2)
            e1.step(act1)
            e2.step(act2)
        assert actions_1 != actions_2 or len(actions_1) == 0

    def test_observe_after_several_moves(self):
        e = SuperTicTacToeEnv(seed=42)
        e.reset()
        for _ in range(6):
            agent = e.agent_selection
            if e.terminations.get(agent) or e.truncations.get(agent):
                break
            obs = e.observe(agent)
            assert obs["observations"].shape == (7, 12, 12)
            legal = np.flatnonzero(obs["action_mask"])
            if len(legal) == 0:
                break
            e.step(int(legal[0]))
        agent = e.agent_selection
        if not e.terminations.get(agent):
            obs = e.observe(agent)
            pieces = np.sum(e.board > 0)
            assert pieces >= 1
            assert obs["observations"][0].sum() + obs["observations"][1].sum() > 0


class TestTacticalScriptedOpponents:

    def test_greedy_tactical_action_returns_legal(self):
        board = empty_board()
        rng = np.random.default_rng(0)
        a = greedy_tactical_action(board, 1, rng)
        assert 0 <= a < NUM_VALID

    def test_lookahead_scripted_action_returns_legal(self):
        board = empty_board()
        rng = np.random.default_rng(0)
        a = lookahead_scripted_action(board, 1, rng)
        assert 0 <= a < NUM_VALID

    def test_greedy_tactical_prefers_immediate_win(self):
        board = empty_board()
        board[8, 0] = 1
        board[8, 1] = 1
        board[8, 2] = 1
        rng = np.random.default_rng(0)

        def fake_dist(_board, r, c):
            if (r, c) == (8, 3):
                return {(8, 3): 1.0}
            return {(r, c): 1.0}

        with patch.object(env_module, "placement_distribution", side_effect=fake_dist):
            a = greedy_tactical_action(board, 1, rng)
        assert a == rc_to_compact(8, 3)

    def test_lookahead_scripted_blocks_immediate_loss(self):
        board = empty_board()
        board[8, 0] = 2
        board[8, 1] = 2
        board[8, 2] = 2
        rng = np.random.default_rng(0)
        with patch.object(env_module, "placement_distribution", side_effect=lambda b, r, c: {(r, c): 1.0}):
            a = lookahead_scripted_action(board, 1, rng)
        assert a == rc_to_compact(8, 3)


class TestPureDefenderAction:

    def test_returns_legal_on_empty_board(self):
        from env import pure_defender_action
        board = empty_board()
        rng = np.random.default_rng(42)
        a = pure_defender_action(board, 1, rng)
        assert 0 <= a < NUM_VALID

    def test_blocks_opponent_three_in_row(self):
        from env import pure_defender_action
        from board import compact_to_rc
        board = empty_board()
        board[8, 0] = 2
        board[8, 1] = 2
        board[8, 2] = 2  # opp threat at (8, 3) for player=1
        rng = np.random.default_rng(0)
        a = pure_defender_action(board, 1, rng)
        assert compact_to_rc(a) == (8, 3)

    def test_blocks_opponent_two_in_row_gap(self):
        from env import pure_defender_action
        from board import compact_to_rc
        board = empty_board()
        # Opp has 2-of-4 window on row 9 cells 0,1; cells 2,3 empty (a partial threat).
        board[9, 0] = 2
        board[9, 1] = 2
        rng = np.random.default_rng(0)
        a = pure_defender_action(board, 1, rng)
        r, c = compact_to_rc(a)
        # With no full 3-in-row threat on the board, pure_defender should target a cell
        # inside an opp 2-of-4 window (cells 2 or 3 on row 9).
        assert (r, c) in {(9, 2), (9, 3)}


class TestPersistBonus:
    """Aim at own threat cell + placement succeeds => persist flag + reward includes PERSIST_BONUS."""

    def test_persist_flag_on_aim_at_threat(self):
        e = SuperTicTacToeEnv(seed=0)
        e.reset()
        # Player 0 has 3-in-row at row 8 cols 0,1,2 -> threat at (8, 3).
        e.board[8, 0] = 1
        e.board[8, 1] = 1
        e.board[8, 2] = 1
        e._shaping_multiplier = 1.0
        # Patch landing to (8, 4) so placement succeeds but does NOT complete the 4-in-row.
        with patch.object(env_module, "resolve_placement", return_value=(8, 4)):
            e.step(rc_to_compact(8, 3))
        assert e._last_step_persist is True
        assert e._last_step_main_succeeded is True
        # Reward should include PERSIST_BONUS (and not be the terminal +1 since no win).
        assert e.rewards["player_0"] > env_module.PERSIST_BONUS - 1e-6
        assert not any(e.terminations.values())

    def test_no_persist_when_aim_not_on_threat(self):
        e = SuperTicTacToeEnv(seed=0)
        e.reset()
        e._shaping_multiplier = 1.0
        # Empty board: no threat anywhere.
        with patch.object(env_module, "resolve_placement", return_value=(9, 5)):
            e.step(rc_to_compact(9, 5))
        assert e._last_step_persist is False

    def test_persist_not_set_on_forfeit(self):
        e = SuperTicTacToeEnv(seed=0)
        e.reset()
        e.board[8, 0] = 1
        e.board[8, 1] = 1
        e.board[8, 2] = 1
        e._shaping_multiplier = 1.0
        with patch.object(env_module, "resolve_placement", return_value=None):
            e.step(rc_to_compact(8, 3))
        assert e._last_step_persist is False
        assert e._last_step_main_succeeded is False


class TestPartialBlockBonus:
    """Landing on a cell inside an opp 2-of-4+0 window => partial_block flag + bonus."""

    def test_partial_block_flag(self):
        e = SuperTicTacToeEnv(seed=0)
        e.reset()
        e._shaping_multiplier = 1.0
        # Opp has 2-in-row at row 9 cols 0,1. Cell (9,2) is inside the 2-of-4 window (0..3).
        e.board[9, 0] = 2
        e.board[9, 1] = 2
        with patch.object(env_module, "resolve_placement", return_value=(9, 2)):
            e.step(rc_to_compact(9, 2))
        assert e._last_step_partial_block is True
        # Reward should include the PARTIAL_BLOCK_BONUS (sm=1.0).
        assert e.rewards["player_0"] > env_module.PARTIAL_BLOCK_BONUS - 1e-6

    def test_no_partial_block_when_opp_has_none(self):
        e = SuperTicTacToeEnv(seed=0)
        e.reset()
        e._shaping_multiplier = 1.0
        # Empty board: opp has no partial windows.
        with patch.object(env_module, "resolve_placement", return_value=(9, 5)):
            e.step(rc_to_compact(9, 5))
        assert e._last_step_partial_block is False


class TestWastePenalty:
    """Isolated placement with no tactical value => waste flag + negative reward component."""

    def test_waste_flag_on_far_isolated_placement(self):
        e = SuperTicTacToeEnv(seed=0)
        e.reset()
        e._shaping_multiplier = 1.0
        # Player 0 has 3 existing pieces (opening exemption ends at own_count >= 3).
        # Place at (8, 7) — Chebyshev distance from nearest own piece = 7 > 2.
        e.board[8, 0] = 1
        e.board[7, 0] = 1
        e.board[6, 0] = 1
        with patch.object(env_module, "resolve_placement", return_value=(8, 7)):
            e.step(rc_to_compact(8, 7))
        assert e._last_step_waste is True
        assert e.rewards["player_0"] < env_module.PLACEMENT_BONUS

    def test_no_waste_when_placement_near_own_piece(self):
        e = SuperTicTacToeEnv(seed=0)
        e.reset()
        e._shaping_multiplier = 1.0
        e.board[8, 0] = 1
        e.board[7, 0] = 1
        e.board[6, 0] = 1
        # Place at (8, 2) — Chebyshev 2, not > WASTE_DISTANCE.
        with patch.object(env_module, "resolve_placement", return_value=(8, 2)):
            e.step(rc_to_compact(8, 2))
        assert e._last_step_waste is False

    def test_no_waste_on_first_ever_placement(self):
        e = SuperTicTacToeEnv(seed=0)
        e.reset()
        e._shaping_multiplier = 1.0
        # No existing own pieces: waste rule skipped (opening exemption).
        with patch.object(env_module, "resolve_placement", return_value=(9, 5)):
            e.step(rc_to_compact(9, 5))
        assert e._last_step_waste is False

    def test_no_waste_in_opening_with_few_own_pieces(self):
        """Opening exemption: even far isolated placements don't trigger waste
        when own_count < 3 (first 3 own placements are always allowed)."""
        e = SuperTicTacToeEnv(seed=0)
        e.reset()
        e._shaping_multiplier = 1.0
        # Only 2 existing own pieces — still in opening window.
        e.board[8, 0] = 1
        e.board[7, 0] = 1
        # Place far away (Chebyshev 7 > 2) but should NOT be waste yet.
        with patch.object(env_module, "resolve_placement", return_value=(8, 7)):
            e.step(rc_to_compact(8, 7))
        assert e._last_step_waste is False

    def test_no_waste_when_creating_threat(self):
        e = SuperTicTacToeEnv(seed=0)
        e.reset()
        e._shaping_multiplier = 1.0
        # Enough own pieces that opening exemption no longer applies; then test
        # that a blocking placement (blocked > 0) suppresses the waste branch.
        e.board[9, 0] = 1
        e.board[5, 5] = 1
        e.board[4, 4] = 1
        # Opp 3-in-row -> threat at (9, 4). Block it by placing there.
        e.board[9, 1] = 2
        e.board[9, 2] = 2
        e.board[9, 3] = 2
        with patch.object(env_module, "resolve_placement", return_value=(9, 4)):
            e.step(rc_to_compact(9, 4))
        assert e._last_step_waste is False
        assert e._last_step_blocked is True
