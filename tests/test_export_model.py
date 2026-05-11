"""Tests for export_model.py: checkpoint extraction and flat state_dict export."""

import os
import tempfile

import torch
import pytest

from export_model import export_checkpoint
from network import SuperTTTNet
from train import _build_net, _save_checkpoint
from self_play import OpponentPool


def _make_checkpoint(tmp, num_filters=16, num_res_blocks=1, value_fc_hidden=32):
    cfg = {"num_filters": num_filters, "num_res_blocks": num_res_blocks,
           "value_fc_hidden": value_fc_hidden}
    net = _build_net(cfg, "cpu")
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    pool = OpponentPool(num_slots=2)
    pool.initialize(net.state_dict())
    path = _save_checkpoint(tmp, net, opt, 1, 100, pool)
    return path, net


def _make_default_checkpoint(tmp):
    from config import MODEL_CONFIG
    from env import OBS_CHANNELS
    from board import NUM_VALID
    net = SuperTTTNet(
        in_channels=OBS_CHANNELS,
        num_filters=int(MODEL_CONFIG["num_filters"]),
        num_res_blocks=int(MODEL_CONFIG["num_res_blocks"]),
        num_actions=NUM_VALID,
        value_fc_hidden=int(MODEL_CONFIG["value_fc_hidden"]),
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    pool = OpponentPool(num_slots=2)
    pool.initialize(net.state_dict())
    path = _save_checkpoint(tmp, net, opt, 1, 100, pool)
    return path, net


def test_export_creates_file():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path, _ = _make_default_checkpoint(tmp)
        out = os.path.join(tmp, "exported.pt")
        result = export_checkpoint(ckpt_path, out, map_location="cpu")
        assert os.path.isfile(result)


def test_exported_is_valid_state_dict():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path, orig_net = _make_default_checkpoint(tmp)
        out = os.path.join(tmp, "exported.pt")
        export_checkpoint(ckpt_path, out, map_location="cpu")
        sd = torch.load(out, map_location="cpu", weights_only=True)
        assert isinstance(sd, dict)
        assert "initial_conv.weight" in sd


def test_export_bare_state_dict():
    with tempfile.TemporaryDirectory() as tmp:
        from config import MODEL_CONFIG
        from env import OBS_CHANNELS
        from board import NUM_VALID
        net = SuperTTTNet(
            in_channels=OBS_CHANNELS,
            num_filters=int(MODEL_CONFIG["num_filters"]),
            num_res_blocks=int(MODEL_CONFIG["num_res_blocks"]),
            num_actions=NUM_VALID,
            value_fc_hidden=int(MODEL_CONFIG["value_fc_hidden"]),
        )
        bare_path = os.path.join(tmp, "bare.pt")
        torch.save(net.state_dict(), bare_path)
        out = os.path.join(tmp, "exported.pt")
        export_checkpoint(bare_path, out, map_location="cpu")
        sd = torch.load(out, map_location="cpu", weights_only=True)
        assert len(sd) == len(net.state_dict())


def test_export_missing_keys_raises():
    with tempfile.TemporaryDirectory() as tmp:
        bad_path = os.path.join(tmp, "bad.pt")
        torch.save({"fake_key": torch.zeros(1)}, bad_path)
        out = os.path.join(tmp, "exported.pt")
        with pytest.raises(RuntimeError, match="Too many missing"):
            export_checkpoint(bad_path, out, map_location="cpu")
