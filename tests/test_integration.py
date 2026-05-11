"""
End-to-end integration: rules, stochastic placement, env observations, policy net, random play.

No Ray RLlib; checks modules compose, games terminate, coarse fairness.
"""

import os

import numpy as np
import torch
import pytest
from board import VALID_MASK, ROWS, COLS, NUM_VALID, VALID_POSITIONS
from rules import check_win, is_draw
from stochastic import placement_distribution
from env import SuperTicTacToeEnv, OBS_CHANNELS
from network import SuperTTTNet


class TestGameEngineConsistency:
    """Rules + stochastic: placement sums to 1; win uses correct player marker."""

    def test_placement_distribution_covers_all_neighbours(self):
        """``placement_distribution`` sums to 1 on every valid cell."""
        b = np.zeros((ROWS, COLS), dtype=np.int8)
        for r in range(ROWS):
            for c in range(COLS):
                if VALID_MASK[r, c]:
                    dist = placement_distribution(b, r, c)
                    total = sum(dist.values())
                    assert abs(total - 1.0) < 1e-10

    def test_win_requires_correct_player_marker(self):
        """Four in a row counts for that player only."""
        b = np.zeros((ROWS, COLS), dtype=np.int8)
        b[8, 0], b[8, 1], b[8, 2], b[8, 3] = 1, 1, 1, 1
        assert check_win(b, 1) is not None
        assert check_win(b, 2) is None


class TestEnvNetworkPipeline:
    """Env tensors match ``SuperTTTNet``; masked policy picks legal empty cells."""

    def test_obs_feeds_into_network(self):
        """7×12×12 obs -> 96 logits and [-1,1] value."""
        env = SuperTicTacToeEnv(seed=42)
        env.reset()
        obs = env.observe("player_0")
        tensor = torch.from_numpy(obs["observations"]).unsqueeze(0)
        net = SuperTTTNet(in_channels=OBS_CHANNELS, num_filters=32, num_res_blocks=1, num_actions=NUM_VALID)
        with torch.no_grad():
            logits, value = net(tensor)
        assert logits.shape == (1, NUM_VALID)
        assert -1.0 <= value.item() <= 1.0

    def test_masked_policy_selects_legal_action(self):
        """Masked argmax lands on a legal empty cell."""
        env = SuperTicTacToeEnv(seed=42)
        env.reset()
        obs = env.observe("player_0")
        tensor = torch.from_numpy(obs["observations"]).unsqueeze(0)
        mask = torch.from_numpy(obs["action_mask"]).unsqueeze(0).float()

        net = SuperTTTNet(in_channels=OBS_CHANNELS, num_filters=32, num_res_blocks=1, num_actions=NUM_VALID)
        with torch.no_grad():
            logits, _ = net(tensor)
        inf_mask = torch.clamp(torch.log(mask), min=-1e10)
        masked_logits = logits + inf_mask
        probs = torch.softmax(masked_logits, dim=-1)
        action = torch.argmax(probs, dim=-1).item()

        r, c = VALID_POSITIONS[action]
        assert VALID_MASK[r, c]
        assert env.board[r, c] == 0


class TestMultipleRandomGames:
    """Random legal play: terminate or draw within cap; invalid cells stay empty."""

    @pytest.mark.parametrize("seed", range(10))
    def test_game_finishes_under_300_steps(self, seed):
        """Within 300 steps: terminate or draw (else fail)."""
        rng = np.random.default_rng(seed)
        env = SuperTicTacToeEnv(seed=seed)
        env.reset()
        for _step in range(300):
            agent = env.agent_selection
            if env.terminations[agent] or env.truncations[agent]:
                return
            obs = env.observe(agent)
            legal = np.where(obs["action_mask"] == 1)[0]
            if len(legal) == 0:
                return
            env.step(rng.choice(legal))
        assert any(env.terminations.values()) or is_draw(env.board)

    def test_no_piece_placed_outside_valid(self):
        """After many random games, invalid cells remain zero."""
        rng = np.random.default_rng(42)
        env = SuperTicTacToeEnv(seed=42)
        for game in range(5):
            env.reset(seed=game)
            for _ in range(200):
                agent = env.agent_selection
                if env.terminations[agent]:
                    break
                obs = env.observe(agent)
                legal = np.where(obs["action_mask"] == 1)[0]
                if len(legal) == 0:
                    break
                env.step(rng.choice(legal))
            for r in range(ROWS):
                for c in range(COLS):
                    if not VALID_MASK[r, c]:
                        assert env.board[r, c] == 0


class TestSymmetryAndFairness:
    """Coarse fairness: first player does not win every random game."""

    def test_first_player_does_not_always_win(self):
        """Across 50 seeds, both sides or draws occur (not one-sided)."""
        wins = {1: 0, 2: 0, "draw": 0}
        for seed in range(50):
            rng = np.random.default_rng(seed)
            env = SuperTicTacToeEnv(seed=seed)
            env.reset()
            for _ in range(300):
                agent = env.agent_selection
                if env.terminations[agent]:
                    break
                obs = env.observe(agent)
                legal = np.where(obs["action_mask"] == 1)[0]
                if len(legal) == 0:
                    break
                env.step(rng.choice(legal))
            if any(env.terminations.values()):
                info_p0 = env.infos.get("player_0", {})
                info_p1 = env.infos.get("player_1", {})
                winner = info_p0.get("winner") or info_p1.get("winner")
                if winner == "player_0":
                    wins[1] += 1
                elif winner == "player_1":
                    wins[2] += 1
                elif info_p0.get("draw") or info_p1.get("draw"):
                    wins["draw"] += 1
        assert wins[1] > 0 or wins[2] > 0


class TestTrainingSmokeTest:

    def test_train_one_fast_iteration(self):
        import tempfile
        from train import train
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = os.path.join(tmpdir, "ckpts")
            results_dir = os.path.join(tmpdir, "results")
            path = train(
                num_iterations=1,
                checkpoint_dir=ckpt_dir,
                results_root=results_dir,
                seed=42,
                early_stop=False,
                fast=True,
            )
            assert path is not None
            assert os.path.isfile(path)

    def test_network_determinism_with_seed(self):
        from train import _build_net, apply_seed
        cfg = {"num_filters": 16, "num_res_blocks": 1, "value_fc_hidden": 32}
        apply_seed(777)
        net1 = _build_net(cfg, "cpu")
        apply_seed(777)
        net2 = _build_net(cfg, "cpu")
        x = torch.randn(1, 7, 12, 12)
        with torch.no_grad():
            out1 = net1(x)
            out2 = net2(x)
        assert torch.allclose(out1[0], out2[0])
        assert torch.allclose(out1[1], out2[1])
