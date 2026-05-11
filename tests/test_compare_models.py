"""
Tests for ``compare_models.py``: architecture detection, obs/mask builders,
game play, and comparison logic.
"""

import json
import os
import tempfile

import numpy as np
import pytest

from board import ROWS, COLS, NUM_VALID, VALID_MASK, VALID_POSITIONS, empty_board
from compare_models import (
    ARCH_OLD, ARCH_NEW,
    detect_architecture,
    build_obs_old, build_obs_new,
    get_action_mask_compact, get_action_mask_flat,
    play_one_game, make_heuristic_agent, make_random_agent,
    run_comparison,
)
from network import get_device


class TestDetectArchitecture:

    def test_old_arch_from_version_info(self, tmp_path):
        info = {"architecture": {"obs_channels": 3, "action_space": 144}}
        (tmp_path / "version_info.json").write_text(json.dumps(info))
        assert detect_architecture(str(tmp_path)) == ARCH_OLD

    def test_new_arch_from_version_info(self, tmp_path):
        info = {"architecture": {"obs_channels": 7, "action_space": 96}}
        (tmp_path / "version_info.json").write_text(json.dumps(info))
        assert detect_architecture(str(tmp_path)) == ARCH_NEW

    def test_old_arch_string(self, tmp_path):
        info = {"architecture": "old"}
        (tmp_path / "version_info.json").write_text(json.dumps(info))
        assert detect_architecture(str(tmp_path)) == ARCH_OLD

    def test_default_new_when_no_files(self, tmp_path):
        assert detect_architecture(str(tmp_path)) == ARCH_NEW

    def test_fallback_to_registry(self, tmp_path):
        ver_dir = tmp_path / "v1.0"
        ver_dir.mkdir()
        reg = {"versions": [{"version": "1.0", "architecture": "old"}]}
        (tmp_path / "registry.json").write_text(json.dumps(reg))
        assert detect_architecture(str(ver_dir)) == ARCH_OLD


class TestObsBuilders:

    def test_build_obs_old_shape(self):
        board = empty_board()
        obs = build_obs_old(board, 1)
        assert obs.shape == (3, ROWS, COLS)
        assert obs.dtype == np.float32

    def test_build_obs_old_channels(self):
        board = empty_board()
        board[9, 5] = 1
        board[9, 6] = 2
        obs = build_obs_old(board, 1)
        assert obs[0, 9, 5] == 1.0
        assert obs[1, 9, 6] == 1.0
        assert obs[2, 9, 5] == 0.0
        assert obs[2, 9, 7] == 1.0

    def test_build_obs_old_ego_perspective(self):
        board = empty_board()
        board[9, 5] = 1
        obs_p1 = build_obs_old(board, 1)
        obs_p2 = build_obs_old(board, 2)
        assert obs_p1[0, 9, 5] == 1.0
        assert obs_p2[1, 9, 5] == 1.0

    def test_build_obs_new_shape(self):
        board = empty_board()
        lmp = np.zeros((ROWS, COLS), dtype=np.float32)
        obs = build_obs_new(board, 1, lmp)
        assert obs.shape == (7, ROWS, COLS)
        assert obs.dtype == np.float32

    def test_build_obs_new_last_move_plane(self):
        board = empty_board()
        lmp = np.zeros((ROWS, COLS), dtype=np.float32)
        lmp[9, 5] = 1.0
        obs = build_obs_new(board, 1, lmp)
        assert obs[3, 9, 5] == 1.0
        assert obs[3, 0, 0] == 0.0


class TestActionMasks:

    def test_compact_mask_empty_board(self):
        board = empty_board()
        mask = get_action_mask_compact(board)
        assert mask.shape == (NUM_VALID,)
        assert mask.sum() == NUM_VALID

    def test_compact_mask_occupied_cell(self):
        board = empty_board()
        board[9, 5] = 1
        mask = get_action_mask_compact(board)
        assert mask.sum() == NUM_VALID - 1

    def test_flat_mask_empty_board(self):
        board = empty_board()
        mask = get_action_mask_flat(board)
        assert mask.shape == (ROWS * COLS,)
        assert mask.sum() == NUM_VALID

    def test_flat_mask_occupied_cell(self):
        board = empty_board()
        board[9, 5] = 1
        mask = get_action_mask_flat(board)
        assert mask.sum() == NUM_VALID - 1


class TestPlayOneGame:

    def test_game_terminates(self):
        a = make_random_agent()
        b = make_random_agent()
        rng = np.random.default_rng(42)
        device = get_device()
        result = play_one_game(a, b, rng, device, max_steps=200)
        assert "winner" in result
        assert "draw" in result
        assert "steps" in result
        assert result["steps"] <= 200

    def test_game_has_winner_or_draw(self):
        a = make_random_agent()
        b = make_random_agent()
        rng = np.random.default_rng(42)
        device = get_device()
        result = play_one_game(a, b, rng, device, max_steps=500)
        if result["draw"]:
            assert result["winner"] is None
        else:
            assert result["winner"] in (0, 1)
            assert result["pid"] in (1, 2)

    def test_short_max_steps_produces_draw(self):
        a = make_random_agent()
        b = make_random_agent()
        rng = np.random.default_rng(0)
        device = get_device()
        result = play_one_game(a, b, rng, device, max_steps=2)
        assert result["steps"] <= 2


class TestRunComparison:

    def test_comparison_totals(self):
        a = make_random_agent()
        b = make_heuristic_agent()
        results = run_comparison(a, b, num_games=10, seed=42)
        assert results["a_wins"] + results["b_wins"] + results["draws"] == 10
        assert results["num_games"] == 10

    def test_seat_alternation(self):
        a = make_random_agent()
        b = make_random_agent()
        results = run_comparison(a, b, num_games=10, seed=42)
        a_first = results["a_wins_as_first"]
        a_second = results["a_wins_as_second"]
        b_first = results["b_wins_as_first"]
        b_second = results["b_wins_as_second"]
        assert a_first + a_second == results["a_wins"]
        assert b_first + b_second == results["b_wins"]

    def test_result_keys(self):
        a = make_random_agent()
        b = make_random_agent()
        results = run_comparison(a, b, num_games=4, seed=0)
        expected_keys = {
            "agent_a", "agent_b", "num_games",
            "a_wins", "b_wins", "draws",
            "a_win_rate", "b_win_rate", "draw_rate",
            "a_wins_as_first", "a_wins_as_second",
            "b_wins_as_first", "b_wins_as_second",
            "elapsed_seconds", "seed",
        }
        assert expected_keys <= set(results.keys())
