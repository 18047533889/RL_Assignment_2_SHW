"""Tests for _ppo_update, _collect_rollouts, and rollout slot machinery in train.py."""

import numpy as np
import torch
import pytest

from train import (
    _ppo_update, _build_net, _collect_rollouts,
    _EnvSlot, _init_slot, _advance_slot_until_pause_or_done,
    _finalise_episode, _flush_finished,
)
from self_play import OpponentPool


_TINY_CFG = {"num_filters": 16, "num_res_blocks": 1, "value_fc_hidden": 32}


def _make_tiny_net(device="cpu"):
    return _build_net(_TINY_CFG, device)


def _make_synthetic_batch(n=64, device="cpu"):
    obs = torch.randn(n, 7, 12, 12, device=device)
    action_mask = torch.ones(n, 96, device=device)
    actions = torch.randint(0, 96, (n,), device=device)
    logits = torch.randn(n, 96, device=device)
    dist = torch.distributions.Categorical(logits=logits)
    log_probs = dist.log_prob(actions)
    values = torch.randn(n, device=device) * 0.1
    rewards = torch.randn(n, device=device) * 0.05
    dones = torch.zeros(n, dtype=torch.bool, device=device)
    dones[-1] = True
    return {
        "obs": obs,
        "action_mask": action_mask,
        "actions": actions,
        "old_log_probs": log_probs,
        "old_logits": logits,
        "values": values,
        "rewards": rewards,
        "dones": dones,
    }


class TestPPOUpdate:
    def test_returns_all_metric_keys(self):
        net = _make_tiny_net()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        batch = _make_synthetic_batch()
        result = _ppo_update(
            net, opt, batch,
            clip_param=0.2, vf_coeff=1.0, entropy_coeff=0.01,
            num_epochs=1, minibatch_size=32, grad_clip=0.5,
            gamma=0.99, lam=0.95, device=torch.device("cpu"),
        )
        expected_keys = {
            "policy_loss", "value_loss", "entropy",
            "clip_fraction", "approx_kl", "grad_norm", "explained_variance",
        }
        assert set(result.keys()) == expected_keys

    def test_losses_are_finite(self):
        net = _make_tiny_net()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        batch = _make_synthetic_batch()
        result = _ppo_update(
            net, opt, batch,
            clip_param=0.2, vf_coeff=1.0, entropy_coeff=0.01,
            num_epochs=2, minibatch_size=32, grad_clip=0.5,
            gamma=0.99, lam=0.95, device=torch.device("cpu"),
        )
        for k, v in result.items():
            assert np.isfinite(v), f"{k} is not finite: {v}"

    def test_clip_fraction_in_range(self):
        net = _make_tiny_net()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        batch = _make_synthetic_batch()
        result = _ppo_update(
            net, opt, batch,
            clip_param=0.2, vf_coeff=1.0, entropy_coeff=0.01,
            num_epochs=1, minibatch_size=32, grad_clip=0.5,
            gamma=0.99, lam=0.95, device=torch.device("cpu"),
        )
        assert 0.0 <= result["clip_fraction"] <= 1.0

    def test_entropy_is_positive(self):
        net = _make_tiny_net()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        batch = _make_synthetic_batch()
        result = _ppo_update(
            net, opt, batch,
            clip_param=0.2, vf_coeff=1.0, entropy_coeff=0.01,
            num_epochs=1, minibatch_size=64, grad_clip=0.5,
            gamma=0.99, lam=0.95, device=torch.device("cpu"),
        )
        assert result["entropy"] > 0

    def test_grad_norm_positive(self):
        net = _make_tiny_net()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        batch = _make_synthetic_batch()
        result = _ppo_update(
            net, opt, batch,
            clip_param=0.2, vf_coeff=1.0, entropy_coeff=0.01,
            num_epochs=1, minibatch_size=64, grad_clip=0.5,
            gamma=0.99, lam=0.95, device=torch.device("cpu"),
        )
        assert result["grad_norm"] > 0

    def test_augment_path_does_not_crash(self):
        net = _make_tiny_net()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        batch = _make_synthetic_batch()
        result = _ppo_update(
            net, opt, batch,
            clip_param=0.2, vf_coeff=1.0, entropy_coeff=0.01,
            num_epochs=2, minibatch_size=32, grad_clip=0.5,
            gamma=0.99, lam=0.95, device=torch.device("cpu"),
            augment=True,
        )
        assert "policy_loss" in result

    def test_minibatch_larger_than_batch(self):
        net = _make_tiny_net()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        batch = _make_synthetic_batch(n=16)
        result = _ppo_update(
            net, opt, batch,
            clip_param=0.2, vf_coeff=1.0, entropy_coeff=0.01,
            num_epochs=1, minibatch_size=256, grad_clip=0.5,
            gamma=0.99, lam=0.95, device=torch.device("cpu"),
        )
        assert np.isfinite(result["policy_loss"])

    def test_weights_change_after_update(self):
        net = _make_tiny_net()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        batch = _make_synthetic_batch()
        w_before = next(net.parameters()).clone()
        _ppo_update(
            net, opt, batch,
            clip_param=0.2, vf_coeff=1.0, entropy_coeff=0.01,
            num_epochs=2, minibatch_size=32, grad_clip=0.5,
            gamma=0.99, lam=0.95, device=torch.device("cpu"),
        )
        w_after = next(net.parameters())
        assert not torch.equal(w_before, w_after)


class TestCollectRollouts:
    def test_smoke_returns_correct_keys(self):
        net = _make_tiny_net()
        opp_net = _make_tiny_net()
        opp_net.eval()
        pool = OpponentPool(num_slots=2)
        pool.initialize(net.state_dict())
        rng = np.random.default_rng(42)
        batch = _collect_rollouts(
            net, opp_net, pool,
            num_steps=32, device=torch.device("cpu"),
            rng=rng,
            curriculum_cfg={"random_opening_prob": 0.0, "random_opening_steps": 0},
            num_collectors=2,
        )
        for key in ("obs", "action_mask", "actions", "old_log_probs", "values", "rewards", "dones", "old_logits"):
            assert key in batch, f"missing key: {key}"

    def test_tensor_shapes(self):
        net = _make_tiny_net()
        opp_net = _make_tiny_net()
        opp_net.eval()
        pool = OpponentPool(num_slots=2)
        pool.initialize(net.state_dict())
        rng = np.random.default_rng(123)
        batch = _collect_rollouts(
            net, opp_net, pool,
            num_steps=32, device=torch.device("cpu"),
            rng=rng,
            curriculum_cfg={"random_opening_prob": 0.0, "random_opening_steps": 0},
            num_collectors=2,
        )
        n = batch["obs"].shape[0]
        assert n >= 32
        assert batch["obs"].shape == (n, 7, 12, 12)
        assert batch["action_mask"].shape == (n, 96)
        assert batch["actions"].shape == (n,)
        assert batch["rewards"].shape == (n,)
        assert batch["dones"].shape == (n,)

    def test_actions_in_valid_range(self):
        net = _make_tiny_net()
        opp_net = _make_tiny_net()
        opp_net.eval()
        pool = OpponentPool(num_slots=2)
        pool.initialize(net.state_dict())
        rng = np.random.default_rng(7)
        batch = _collect_rollouts(
            net, opp_net, pool,
            num_steps=64, device=torch.device("cpu"),
            rng=rng,
            curriculum_cfg={"random_opening_prob": 0.0, "random_opening_steps": 0},
            num_collectors=2,
        )
        assert batch["actions"].min().item() >= 0
        assert batch["actions"].max().item() < 96

    def test_has_episode_endings(self):
        net = _make_tiny_net()
        opp_net = _make_tiny_net()
        opp_net.eval()
        pool = OpponentPool(num_slots=2)
        pool.initialize(net.state_dict())
        rng = np.random.default_rng(99)
        batch = _collect_rollouts(
            net, opp_net, pool,
            num_steps=128, device=torch.device("cpu"),
            rng=rng,
            curriculum_cfg={"random_opening_prob": 0.0, "random_opening_steps": 0},
            num_collectors=4,
        )
        assert batch["dones"].any()

    def test_scripted_opponents_work(self):
        net = _make_tiny_net()
        opp_net = _make_tiny_net()
        opp_net.eval()
        pool = OpponentPool(num_slots=2)
        pool.initialize(net.state_dict())
        rng = np.random.default_rng(55)
        batch = _collect_rollouts(
            net, opp_net, pool,
            num_steps=64, device=torch.device("cpu"),
            rng=rng,
            curriculum_cfg={"random_opening_prob": 0.0, "random_opening_steps": 0},
            num_collectors=4,
            shaping_multiplier=0.5,
        )
        assert batch["obs"].shape[0] >= 64


class TestEnvSlot:
    def test_init_defaults(self):
        s = _EnvSlot()
        assert s.finished is True
        assert s.ep_done is False
        assert s.ep_obs == []
        assert s.waiting_for_main is False

    def test_init_slot_creates_env(self):
        net = _make_tiny_net()
        opp_net = _make_tiny_net()
        opp_net.eval()
        pool = OpponentPool(num_slots=2)
        pool.initialize(net.state_dict())
        rng = np.random.default_rng(42)
        s = _EnvSlot()
        _init_slot(s, pool, opp_net, rng, {"random_opening_prob": 0.0, "random_opening_steps": 0}, torch.device("cpu"))
        assert s.env is not None
        assert s.finished is False
        assert s.ep_obs == []

    def test_advance_reaches_main_turn(self):
        net = _make_tiny_net()
        opp_net = _make_tiny_net()
        opp_net.eval()
        pool = OpponentPool(num_slots=2)
        pool.initialize(net.state_dict())
        rng = np.random.default_rng(42)
        s = _EnvSlot()
        _init_slot(s, pool, opp_net, rng, {"random_opening_prob": 0.0, "random_opening_steps": 0}, torch.device("cpu"))
        _advance_slot_until_pause_or_done(s)
        assert s.waiting_for_main or s.waiting_for_opp or s.finished


class TestFlushFinished:
    def test_flush_returns_step_count(self):
        net = _make_tiny_net()
        opp_net = _make_tiny_net()
        opp_net.eval()
        pool = OpponentPool(num_slots=2)
        pool.initialize(net.state_dict())
        rng = np.random.default_rng(42)
        s = _EnvSlot()
        s.finished = True
        s.ep_obs = []
        obs_l, mask_l, act_l, lp_l, val_l, rew_l, done_l, logits_l = [], [], [], [], [], [], [], []
        added = _flush_finished(
            [s], pool,
            obs_l, mask_l, act_l, lp_l, val_l, rew_l, done_l, logits_l,
            rng, opp_net,
            {"random_opening_prob": 0.0, "random_opening_steps": 0},
            torch.device("cpu"),
        )
        assert added == 0
