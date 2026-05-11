"""Tests for train.py helpers: schedule, net builder, checkpoint, GAE, CSV float, seed."""

import os
import random
import tempfile

import numpy as np
import pytest
import torch

from train import (
    _piecewise_schedule, _build_net, _save_checkpoint,
    _compute_gae, _csv_float, apply_seed,
)
from config import PPO_CONFIG, MODEL_CONFIG
from self_play import OpponentPool


def test_piecewise_schedule_initial():
    schedule = [(0, 3e-4), (1000, 1e-4), (2000, 3e-5)]
    assert _piecewise_schedule(schedule, 0) == pytest.approx(3e-4)


def test_piecewise_schedule_mid():
    schedule = [(0, 3e-4), (1000, 1e-4), (2000, 3e-5)]
    assert _piecewise_schedule(schedule, 1500) == pytest.approx(1e-4)


def test_piecewise_schedule_final():
    schedule = [(0, 3e-4), (1000, 1e-4), (2000, 3e-5)]
    assert _piecewise_schedule(schedule, 5000) == pytest.approx(3e-5)


def test_piecewise_schedule_empty():
    assert _piecewise_schedule([], 100) == pytest.approx(0.0)


def test_build_net_default():
    net = _build_net(MODEL_CONFIG, "cpu")
    assert isinstance(net, torch.nn.Module)
    params = sum(p.numel() for p in net.parameters())
    assert params > 1_000_000


def test_build_net_fast():
    cfg = {"num_filters": 64, "num_res_blocks": 2, "value_fc_hidden": 256}
    net = _build_net(cfg, "cpu")
    params = sum(p.numel() for p in net.parameters())
    assert params < 1_000_000


def test_save_checkpoint_creates_file():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = {"num_filters": 16, "num_res_blocks": 1, "value_fc_hidden": 32}
        net = _build_net(cfg, "cpu")
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        pool = OpponentPool(num_slots=2)
        pool.initialize(net.state_dict())
        path = _save_checkpoint(tmp, net, opt, 10, 5000, pool)
        assert os.path.isfile(path)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        assert "main_net" in ckpt
        assert "optimizer" in ckpt
        assert ckpt["iteration"] == 10
        assert ckpt["total_env_steps"] == 5000


def test_compute_gae_known_values():
    rewards = torch.tensor([0.0, 0.0, 1.0])
    values = torch.tensor([0.5, 0.6, 0.8])
    dones = torch.tensor([False, False, True])
    adv, ret = _compute_gae(rewards, values, dones, gamma=0.99, lam=0.95, device=torch.device("cpu"))
    assert adv.shape == (3,)
    assert ret.shape == (3,)
    assert torch.isclose(adv[-1], torch.tensor(0.2), atol=1e-4)
    assert torch.allclose(ret, adv + values, atol=1e-4)


def test_compute_gae_single_step():
    rewards = torch.tensor([1.0])
    values = torch.tensor([0.0])
    dones = torch.tensor([True])
    adv, ret = _compute_gae(rewards, values, dones, gamma=0.99, lam=0.95, device=torch.device("cpu"))
    assert torch.isclose(adv[0], torch.tensor(1.0), atol=1e-4)


def test_compute_gae_episode_boundary():
    rewards = torch.tensor([0.0, 1.0, 0.0, -1.0])
    values = torch.tensor([0.1, 0.2, 0.3, 0.4])
    dones = torch.tensor([False, True, False, True])
    adv, ret = _compute_gae(rewards, values, dones, gamma=0.99, lam=0.95, device=torch.device("cpu"))
    assert adv.shape == (4,)


def test_csv_float_valid():
    assert _csv_float(3.14) == pytest.approx(3.14)
    assert _csv_float("2.5") == pytest.approx(2.5)


def test_csv_float_empty():
    assert _csv_float(None) == ""
    assert _csv_float("") == ""


def test_csv_float_invalid():
    assert _csv_float("abc") == ""


def test_apply_seed_deterministic():
    apply_seed(123)
    a = random.random()
    b = np.random.random()
    c = torch.rand(1).item()
    apply_seed(123)
    assert random.random() == a
    assert np.random.random() == b
    assert torch.rand(1).item() == c


def test_save_checkpoint_includes_opponent_pool():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = {"num_filters": 16, "num_res_blocks": 1, "value_fc_hidden": 32}
        net = _build_net(cfg, "cpu")
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        pool = OpponentPool(num_slots=3)
        pool.initialize(net.state_dict())
        pool.save_snapshot(net.state_dict())
        path = _save_checkpoint(tmp, net, opt, 5, 1000, pool)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        assert "opponent_pool" in ckpt
        assert len(ckpt["opponent_pool"]["slots"]) == 3
        assert ckpt["opponent_pool"]["snapshot_rr"] == 1


def test_piecewise_schedule_single_element():
    schedule = [(500, 0.1)]
    assert _piecewise_schedule(schedule, 0) == pytest.approx(0.1)
    assert _piecewise_schedule(schedule, 500) == pytest.approx(0.1)
    assert _piecewise_schedule(schedule, 1000) == pytest.approx(0.1)


def test_piecewise_schedule_step_before_breakpoint():
    schedule = [(0, 3e-4), (1000, 1e-4)]
    assert _piecewise_schedule(schedule, 999) == pytest.approx(3e-4)
    assert _piecewise_schedule(schedule, 1000) == pytest.approx(1e-4)


def test_compute_gae_all_zero_rewards():
    rewards = torch.zeros(5)
    values = torch.ones(5) * 0.5
    dones = torch.tensor([False, False, False, False, True])
    adv, ret = _compute_gae(rewards, values, dones, gamma=0.99, lam=0.95, device=torch.device("cpu"))
    assert adv.shape == (5,)
    assert torch.all(torch.isfinite(adv))


def test_compute_gae_negative_rewards():
    rewards = torch.tensor([-1.0, -0.5, -0.1])
    values = torch.tensor([0.0, 0.0, 0.0])
    dones = torch.tensor([False, False, True])
    adv, ret = _compute_gae(rewards, values, dones, gamma=0.99, lam=0.95, device=torch.device("cpu"))
    assert adv[0] < 0


def test_compute_gae_all_done():
    rewards = torch.tensor([1.0, -1.0, 0.5])
    values = torch.tensor([0.0, 0.0, 0.0])
    dones = torch.tensor([True, True, True])
    adv, ret = _compute_gae(rewards, values, dones, gamma=0.99, lam=0.95, device=torch.device("cpu"))
    assert torch.isclose(adv[0], torch.tensor(1.0), atol=1e-4)
    assert torch.isclose(adv[1], torch.tensor(-1.0), atol=1e-4)
    assert torch.isclose(adv[2], torch.tensor(0.5), atol=1e-4)


def test_checkpoint_restore_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = {"num_filters": 16, "num_res_blocks": 1, "value_fc_hidden": 32}
        net = _build_net(cfg, "cpu")
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        pool = OpponentPool(num_slots=2)
        pool.initialize(net.state_dict())
        path = _save_checkpoint(tmp, net, opt, 42, 99999, pool)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        net2 = _build_net(cfg, "cpu")
        net2.load_state_dict(ckpt["main_net"])
        x = torch.randn(1, 7, 12, 12)
        with torch.no_grad():
            out1 = net(x)
            out2 = net2(x)
        assert torch.allclose(out1[0], out2[0])
        assert torch.allclose(out1[1], out2[1])
