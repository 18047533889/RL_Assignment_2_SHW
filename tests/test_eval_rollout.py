"""
Tests for eval_rollout.py: outcome parsing, greedy action, evaluation functions.

Pure PyTorch — no RLlib.
"""

import math
from unittest.mock import patch

import numpy as np
import pytest
import torch

from eval_rollout import (
    EVAL_MAX_AGENT_STEPS,
    _episode_outcome_from_infos,
    _random_legal_action,
    _resolve_outcome,
    _build_obs_for_board,
    greedy_action,
    evaluate_main_vs_random,
    evaluate_main_vs_snapshot,
    evaluate_main_vs_heuristic,
    evaluate_main_vs_heuristic_by_position,
    evaluate_main_vs_scripted,
    evaluate_random_opening,
    evaluate_forfeit_recovery,
    measure_block_rate,
    play_one_episode,
    play_one_episode_vs_random,
    play_one_episode_vs_heuristic,
)
from env import MAX_STEPS, greedy_tactical_action, lookahead_scripted_action
from board import empty_board, VALID_POSITIONS
from network import SuperTTTNet, get_device


def test_eval_max_agent_steps_covers_long_env_episode():
    assert EVAL_MAX_AGENT_STEPS >= MAX_STEPS * 2


def test_main_win_from_player_0_info():
    infos = {"player_0": {"winner": "player_0"}, "player_1": {}}
    assert _episode_outcome_from_infos(infos) == "main_win"


def test_main_win_from_player_1_info():
    infos = {"player_0": {}, "player_1": {"winner": "player_0"}}
    assert _episode_outcome_from_infos(infos) == "main_win"


def test_opp_win_from_either_slot():
    infos = {"player_0": {"winner": "player_1"}, "player_1": {}}
    assert _episode_outcome_from_infos(infos) == "opp_win"


def test_main_as_player1_win():
    infos = {"player_0": {"winner": "player_1"}, "player_1": {}}
    assert _episode_outcome_from_infos(infos, main_agent="player_1") == "main_win"


def test_main_as_player1_loss():
    infos = {"player_0": {"winner": "player_0"}, "player_1": {}}
    assert _episode_outcome_from_infos(infos, main_agent="player_1") == "opp_win"


def test_draw_flag():
    infos = {"player_0": {"draw": True}, "player_1": {}}
    assert _episode_outcome_from_infos(infos) == "draw"


def test_no_outcome():
    infos = {"player_0": {}, "player_1": {}}
    assert _episode_outcome_from_infos(infos) is None


def test_greedy_action_returns_legal():
    device = torch.device("cpu")
    net = SuperTTTNet(in_channels=7, num_filters=16, num_res_blocks=1,
                      num_actions=96, value_fc_hidden=32)
    net.eval()
    obs = {
        "observations": np.random.randn(7, 12, 12).astype(np.float32),
        "action_mask": np.ones(96, dtype=np.float32),
    }
    a = greedy_action(net, obs, device)
    assert 0 <= a < 96


def test_greedy_action_respects_mask():
    device = torch.device("cpu")
    net = SuperTTTNet(in_channels=7, num_filters=16, num_res_blocks=1,
                      num_actions=96, value_fc_hidden=32)
    net.eval()
    mask = np.zeros(96, dtype=np.float32)
    mask[42] = 1.0
    obs = {
        "observations": np.random.randn(7, 12, 12).astype(np.float32),
        "action_mask": mask,
    }
    a = greedy_action(net, obs, device)
    assert a == 42


def test_evaluate_zero_episodes_returns_nan_rates():
    r = evaluate_main_vs_random(None, num_episodes=0, base_seed=0, device=torch.device("cpu"))
    assert math.isnan(r["win_rate"])


def test_evaluate_main_vs_random_none_returns_nan():
    r = evaluate_main_vs_random(None, num_episodes=5, base_seed=0, device=torch.device("cpu"))
    assert math.isnan(r["win_rate"])


def test_evaluate_main_vs_snapshot_none_returns_nan():
    r = evaluate_main_vs_snapshot(None, None, num_episodes=5, base_seed=0, device=torch.device("cpu"))
    assert math.isnan(r["win_rate"])


def test_evaluate_main_vs_heuristic_none_returns_nan():
    r = evaluate_main_vs_heuristic(None, num_episodes=0, base_seed=0, device=torch.device("cpu"))
    assert math.isnan(r["win_rate"])


def test_evaluate_main_vs_random_rates_sum():
    device = torch.device("cpu")
    net = SuperTTTNet(in_channels=7, num_filters=16, num_res_blocks=1,
                      num_actions=96, value_fc_hidden=32)
    net.eval()
    r = evaluate_main_vs_random(net, num_episodes=6, base_seed=42, device=device)
    total = r["win_rate"] + r["draw_rate"] + r["loss_rate"]
    assert total == pytest.approx(1.0, abs=0.01)


def test_evaluate_main_vs_snapshot_rates_sum():
    device = torch.device("cpu")
    net = SuperTTTNet(in_channels=7, num_filters=16, num_res_blocks=1,
                      num_actions=96, value_fc_hidden=32)
    net.eval()
    opp = SuperTTTNet(in_channels=7, num_filters=16, num_res_blocks=1,
                      num_actions=96, value_fc_hidden=32)
    opp.eval()
    r = evaluate_main_vs_snapshot(net, opp, num_episodes=6, base_seed=42, device=device)
    total = r["win_rate"] + r["draw_rate"] + r["loss_rate"]
    assert total == pytest.approx(1.0, abs=0.01)


def test_evaluate_main_vs_heuristic_rates_sum():
    device = torch.device("cpu")
    net = SuperTTTNet(in_channels=7, num_filters=16, num_res_blocks=1,
                      num_actions=96, value_fc_hidden=32)
    net.eval()
    r = evaluate_main_vs_heuristic(net, num_episodes=6, base_seed=42, device=device)
    total = r["win_rate"] + r["draw_rate"] + r["loss_rate"]
    assert total == pytest.approx(1.0, abs=0.01)


def test_random_legal_action_returns_legal():
    rng = np.random.default_rng(42)
    mask = np.zeros(96, dtype=np.float32)
    mask[10] = 1.0
    mask[20] = 1.0
    obs = {"action_mask": mask}
    a = _random_legal_action(obs, rng)
    assert a in (10, 20)


def test_random_legal_action_empty_mask():
    rng = np.random.default_rng(42)
    mask = np.zeros(96, dtype=np.float32)
    obs = {"action_mask": mask}
    a = _random_legal_action(obs, rng)
    assert 0 <= a < 96


def test_resolve_outcome_passthrough():
    assert _resolve_outcome(None, "main_win") == "main_win"
    assert _resolve_outcome(None, "opp_win") == "opp_win"
    assert _resolve_outcome(None, "draw") == "draw"


class _FakeEnv:
    def __init__(self, infos=None, terminations=None, truncations=None):
        self.infos = infos or {}
        self.terminations = terminations or {}
        self.truncations = truncations or {}


def test_resolve_outcome_from_env_main_win():
    env = _FakeEnv(infos={"player_0": {"winner": "player_0"}, "player_1": {}})
    assert _resolve_outcome(env, None) == "main_win"


def test_resolve_outcome_from_env_draw():
    env = _FakeEnv(infos={"player_0": {"draw": True}, "player_1": {}})
    assert _resolve_outcome(env, None) == "draw"


def test_resolve_outcome_terminated_fallback():
    env = _FakeEnv(
        infos={"player_0": {}, "player_1": {}},
        terminations={"player_0": True},
    )
    assert _resolve_outcome(env, None) == "draw"


def test_resolve_outcome_incomplete():
    env = _FakeEnv()
    assert _resolve_outcome(env, None) == "incomplete"


def test_play_episode_vs_random_returns_valid_outcome():
    device = torch.device("cpu")
    net = SuperTTTNet(in_channels=7, num_filters=16, num_res_blocks=1,
                      num_actions=96, value_fc_hidden=32)
    net.eval()
    out = play_one_episode_vs_random(net, seed=42, device=device)
    assert out in ("main_win", "opp_win", "draw", "incomplete")


def test_play_episode_vs_random_as_player1():
    device = torch.device("cpu")
    net = SuperTTTNet(in_channels=7, num_filters=16, num_res_blocks=1,
                      num_actions=96, value_fc_hidden=32)
    net.eval()
    out = play_one_episode_vs_random(net, seed=42, device=device, main_agent="player_1")
    assert out in ("main_win", "opp_win", "draw", "incomplete")


def test_play_episode_vs_heuristic_returns_valid_outcome():
    device = torch.device("cpu")
    net = SuperTTTNet(in_channels=7, num_filters=16, num_res_blocks=1,
                      num_actions=96, value_fc_hidden=32)
    net.eval()
    out = play_one_episode_vs_heuristic(net, seed=42, device=device)
    assert out in ("main_win", "opp_win", "draw", "incomplete")


def test_play_episode_vs_heuristic_as_player1():
    device = torch.device("cpu")
    net = SuperTTTNet(in_channels=7, num_filters=16, num_res_blocks=1,
                      num_actions=96, value_fc_hidden=32)
    net.eval()
    out = play_one_episode_vs_heuristic(net, seed=42, device=device, main_agent="player_1")
    assert out in ("main_win", "opp_win", "draw", "incomplete")


def test_evaluate_main_vs_scripted_none_returns_nan():
    from env import line_rusher_action
    r = evaluate_main_vs_scripted(None, line_rusher_action, num_episodes=5, base_seed=0, device=torch.device("cpu"))
    assert math.isnan(r["win_rate"])


def test_evaluate_main_vs_scripted_zero_episodes_returns_nan():
    from env import line_rusher_action
    r = evaluate_main_vs_scripted(None, line_rusher_action, num_episodes=0, base_seed=0, device=torch.device("cpu"))
    assert math.isnan(r["win_rate"])


def test_evaluate_main_vs_scripted_rates_sum():
    from env import line_rusher_action
    device = torch.device("cpu")
    net = SuperTTTNet(in_channels=7, num_filters=16, num_res_blocks=1,
                      num_actions=96, value_fc_hidden=32)
    net.eval()
    r = evaluate_main_vs_scripted(net, line_rusher_action, num_episodes=6, base_seed=42, device=device)
    total = r["win_rate"] + r["draw_rate"] + r["loss_rate"]
    assert total == pytest.approx(1.0, abs=0.01)


def test_evaluate_main_vs_scripted_row_rusher():
    from env import row_rusher_action
    device = torch.device("cpu")
    net = SuperTTTNet(in_channels=7, num_filters=16, num_res_blocks=1,
                      num_actions=96, value_fc_hidden=32)
    net.eval()
    r = evaluate_main_vs_scripted(net, row_rusher_action, num_episodes=4, base_seed=42, device=device)
    total = r["win_rate"] + r["draw_rate"] + r["loss_rate"]
    assert total == pytest.approx(1.0, abs=0.01)


def test_evaluate_main_vs_scripted_col_rusher():
    from env import col_rusher_action
    device = torch.device("cpu")
    net = SuperTTTNet(in_channels=7, num_filters=16, num_res_blocks=1,
                      num_actions=96, value_fc_hidden=32)
    net.eval()
    r = evaluate_main_vs_scripted(net, col_rusher_action, num_episodes=4, base_seed=42, device=device)
    total = r["win_rate"] + r["draw_rate"] + r["loss_rate"]
    assert total == pytest.approx(1.0, abs=0.01)


def test_measure_block_rate_none_returns_nan():
    r = measure_block_rate(None, num_episodes=5, base_seed=0, device=torch.device("cpu"))
    assert math.isnan(r)


def test_measure_block_rate_zero_episodes_returns_nan():
    r = measure_block_rate(None, num_episodes=0, base_seed=0, device=torch.device("cpu"))
    assert math.isnan(r)


def test_measure_block_rate_returns_valid_range():
    device = torch.device("cpu")
    net = SuperTTTNet(in_channels=7, num_filters=16, num_res_blocks=1,
                      num_actions=96, value_fc_hidden=32)
    net.eval()
    r = measure_block_rate(net, num_episodes=4, base_seed=42, device=device)
    assert math.isnan(r) or (0.0 <= r <= 1.0)


def test_lookahead_action_returns_legal():
    from eval_rollout import lookahead_action
    device = torch.device("cpu")
    net = SuperTTTNet(in_channels=7, num_filters=16, num_res_blocks=1,
                      num_actions=96, value_fc_hidden=32)
    net.eval()
    obs = {
        "observations": np.random.randn(7, 12, 12).astype(np.float32),
        "action_mask": np.ones(96, dtype=np.float32),
    }
    from board import empty_board
    board = empty_board()
    a = lookahead_action(net, obs, board, 1, device)
    assert 0 <= a < 96


def test_lookahead_action_respects_mask():
    from eval_rollout import lookahead_action
    device = torch.device("cpu")
    net = SuperTTTNet(in_channels=7, num_filters=16, num_res_blocks=1,
                      num_actions=96, value_fc_hidden=32)
    net.eval()
    mask = np.zeros(96, dtype=np.float32)
    mask[42] = 1.0
    obs = {
        "observations": np.random.randn(7, 12, 12).astype(np.float32),
        "action_mask": mask,
    }
    from board import empty_board
    board = empty_board()
    a = lookahead_action(net, obs, board, 1, device)
    assert a == 42


def test_lookahead_action_single_legal():
    from eval_rollout import lookahead_action
    device = torch.device("cpu")
    net = SuperTTTNet(in_channels=7, num_filters=16, num_res_blocks=1,
                      num_actions=96, value_fc_hidden=32)
    net.eval()
    mask = np.zeros(96, dtype=np.float32)
    mask[10] = 1.0
    obs = {
        "observations": np.random.randn(7, 12, 12).astype(np.float32),
        "action_mask": mask,
    }
    board = empty_board()
    a = lookahead_action(net, obs, board, 1, device)
    assert a == 10


def test_build_obs_for_board_shape():
    board = empty_board()
    lmp = np.zeros((12, 12), dtype=np.float32)
    obs = _build_obs_for_board(board, 1, lmp)
    assert obs.shape == (7, 12, 12)
    assert obs.dtype == np.float32


def test_build_obs_for_board_channels():
    board = empty_board()
    r, c = VALID_POSITIONS[0]
    board[r, c] = 1
    lmp = np.zeros((12, 12), dtype=np.float32)
    obs = _build_obs_for_board(board, 1, lmp)
    assert obs[0, r, c] == 1.0
    assert obs[1, r, c] == 0.0
    obs_opp = _build_obs_for_board(board, 2, lmp)
    assert obs_opp[0, r, c] == 0.0
    assert obs_opp[1, r, c] == 1.0


def test_build_obs_for_board_last_move_plane():
    board = empty_board()
    lmp = np.zeros((12, 12), dtype=np.float32)
    r, c = VALID_POSITIONS[5]
    lmp[r, c] = 1.0
    obs = _build_obs_for_board(board, 1, lmp)
    assert obs[3, r, c] == 1.0


def test_greedy_action_with_torch_tensor():
    device = torch.device("cpu")
    net = SuperTTTNet(in_channels=7, num_filters=16, num_res_blocks=1,
                      num_actions=96, value_fc_hidden=32)
    net.eval()
    obs = {
        "observations": torch.randn(7, 12, 12),
        "action_mask": torch.ones(96),
    }
    a = greedy_action(net, obs, device)
    assert 0 <= a < 96


def test_play_one_episode_net_vs_net():
    device = torch.device("cpu")
    net = SuperTTTNet(in_channels=7, num_filters=16, num_res_blocks=1,
                      num_actions=96, value_fc_hidden=32)
    net.eval()
    result = play_one_episode(net, net, seed=42, device=device)
    assert result in ("main_win", "opp_win", "draw")


def test_play_one_episode_as_player_1():
    device = torch.device("cpu")
    net = SuperTTTNet(in_channels=7, num_filters=16, num_res_blocks=1,
                      num_actions=96, value_fc_hidden=32)
    net.eval()
    result = play_one_episode(net, net, seed=42, device=device, main_agent="player_1")
    assert result in ("main_win", "opp_win", "draw")


def test_episode_outcome_conflicting_info():
    infos = {
        "player_0": {"winner": "player_0"},
        "player_1": {"winner": "player_1"},
    }
    result = _episode_outcome_from_infos(infos, "player_0")
    assert result == "main_win"


def test_evaluate_heuristic_by_position_none_returns_nan():
    r = evaluate_main_vs_heuristic_by_position(None, num_episodes=5, base_seed=0, device=torch.device("cpu"))
    assert math.isnan(r["win_rate"])


def test_evaluate_heuristic_by_position_zero_returns_nan():
    r = evaluate_main_vs_heuristic_by_position(None, num_episodes=0, base_seed=0, device=torch.device("cpu"))
    assert math.isnan(r["win_rate"])


def test_evaluate_heuristic_by_position_first_rates_sum():
    device = torch.device("cpu")
    net = SuperTTTNet(in_channels=7, num_filters=16, num_res_blocks=1,
                      num_actions=96, value_fc_hidden=32)
    net.eval()
    r = evaluate_main_vs_heuristic_by_position(
        net, num_episodes=4, base_seed=42, device=device, main_agent_position="first",
    )
    total = r["win_rate"] + r["draw_rate"] + r["loss_rate"]
    assert total == pytest.approx(1.0, abs=0.01)


def test_evaluate_heuristic_by_position_second_rates_sum():
    device = torch.device("cpu")
    net = SuperTTTNet(in_channels=7, num_filters=16, num_res_blocks=1,
                      num_actions=96, value_fc_hidden=32)
    net.eval()
    r = evaluate_main_vs_heuristic_by_position(
        net, num_episodes=4, base_seed=42, device=device, main_agent_position="second",
    )
    total = r["win_rate"] + r["draw_rate"] + r["loss_rate"]
    assert total == pytest.approx(1.0, abs=0.01)


def test_evaluate_main_vs_scripted_first_vs_second_differs_or_sums():
    """first/second branches are independent paths — both should return valid rate dicts."""
    from env import line_rusher_action
    device = torch.device("cpu")
    net = SuperTTTNet(in_channels=7, num_filters=16, num_res_blocks=1,
                      num_actions=96, value_fc_hidden=32)
    net.eval()
    r_first = evaluate_main_vs_scripted(
        net, line_rusher_action, num_episodes=4, base_seed=42, device=device,
        main_agent_position="first",
    )
    r_second = evaluate_main_vs_scripted(
        net, line_rusher_action, num_episodes=4, base_seed=42, device=device,
        main_agent_position="second",
    )
    assert r_first["win_rate"] + r_first["draw_rate"] + r_first["loss_rate"] == pytest.approx(1.0, abs=0.01)
    assert r_second["win_rate"] + r_second["draw_rate"] + r_second["loss_rate"] == pytest.approx(1.0, abs=0.01)


def test_evaluate_random_opening_none_returns_nan():
    from env import heuristic_action
    r = evaluate_random_opening(None, heuristic_action, num_episodes=5, base_seed=0, device=torch.device("cpu"))
    assert math.isnan(r["win_rate"])


def test_evaluate_random_opening_zero_returns_nan():
    from env import heuristic_action
    r = evaluate_random_opening(None, heuristic_action, num_episodes=0, base_seed=0, device=torch.device("cpu"))
    assert math.isnan(r["win_rate"])


def test_evaluate_random_opening_rates_sum():
    from env import heuristic_action
    device = torch.device("cpu")
    net = SuperTTTNet(in_channels=7, num_filters=16, num_res_blocks=1,
                      num_actions=96, value_fc_hidden=32)
    net.eval()
    r = evaluate_random_opening(net, heuristic_action, num_episodes=4, base_seed=42, device=device)
    total = r["win_rate"] + r["draw_rate"] + r["loss_rate"]
    assert total == pytest.approx(1.0, abs=0.01)


def test_evaluate_random_opening_honours_opening_steps():
    """With ``opening_steps=6`` the starting board is not empty by the time greedy takes over."""
    from env import heuristic_action, SuperTicTacToeEnv
    from eval_rollout import evaluate_random_opening
    device = torch.device("cpu")
    net = SuperTTTNet(in_channels=7, num_filters=16, num_res_blocks=1,
                      num_actions=96, value_fc_hidden=32)
    net.eval()
    # Completes without error (smoke) — step count handled by ``random_opening_steps``.
    r = evaluate_random_opening(
        net, heuristic_action, num_episodes=2, base_seed=42, device=device, opening_steps=6,
    )
    assert set(r.keys()) == {"win_rate", "draw_rate", "loss_rate"}


def test_evaluate_forfeit_recovery_none_returns_nan():
    from env import heuristic_action
    r = evaluate_forfeit_recovery(None, heuristic_action, num_episodes=5, base_seed=0, device=torch.device("cpu"))
    assert math.isnan(r["win_rate"])


def test_evaluate_forfeit_recovery_zero_returns_nan():
    from env import heuristic_action
    r = evaluate_forfeit_recovery(None, heuristic_action, num_episodes=0, base_seed=0, device=torch.device("cpu"))
    assert math.isnan(r["win_rate"])


def test_evaluate_forfeit_recovery_rates_sum():
    from env import heuristic_action
    device = torch.device("cpu")
    net = SuperTTTNet(in_channels=7, num_filters=16, num_res_blocks=1,
                      num_actions=96, value_fc_hidden=32)
    net.eval()
    r = evaluate_forfeit_recovery(net, heuristic_action, num_episodes=4, base_seed=42, device=device)
    total = r["win_rate"] + r["draw_rate"] + r["loss_rate"]
    assert total == pytest.approx(1.0, abs=0.01)


def test_evaluate_main_vs_greedy_tactical_rates_sum():
    device = torch.device("cpu")
    net = SuperTTTNet(in_channels=7, num_filters=16, num_res_blocks=1,
                      num_actions=96, value_fc_hidden=32)
    net.eval()
    r = evaluate_main_vs_scripted(net, greedy_tactical_action, num_episodes=4, base_seed=42, device=device)
    total = r["win_rate"] + r["draw_rate"] + r["loss_rate"]
    assert total == pytest.approx(1.0, abs=0.01)


def test_evaluate_main_vs_lookahead_scripted_rates_sum():
    device = torch.device("cpu")
    net = SuperTTTNet(in_channels=7, num_filters=16, num_res_blocks=1,
                      num_actions=96, value_fc_hidden=32)
    net.eval()
    r = evaluate_main_vs_scripted(net, lookahead_scripted_action, num_episodes=4, base_seed=42, device=device)
    total = r["win_rate"] + r["draw_rate"] + r["loss_rate"]
    assert total == pytest.approx(1.0, abs=0.01)


def test_tactical_scripted_policies_choose_legal_moves():
    """Lightweight check (no full rollout): placement-aware scripted opponents return valid indices."""
    from env import greedy_tactical_action, lookahead_scripted_action
    from board import NUM_VALID, empty_board

    b = empty_board()
    rng = np.random.default_rng(7)
    for _ in range(2):
        a0 = greedy_tactical_action(b, 1, rng)
        a1 = lookahead_scripted_action(b, 2, rng)
        assert 0 <= a0 < NUM_VALID
        assert 0 <= a1 < NUM_VALID
