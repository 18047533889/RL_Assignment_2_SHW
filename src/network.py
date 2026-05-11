"""
network.py — Standalone twin-head residual CNN (PyTorch only, no RLlib).

**Quick read:** Same backbone/head shapes as ``action_mask_model.ActionMaskModel`` so
``export_model.py`` can load RL checkpoints with ``strict=False``. Served by ``server.py``;
hyperparameters come from ``config.MODEL_CONFIG``.

Architecture (typical):
    Input:  (batch, 7, 12, 12)   — 7-channel ego-centric observation
    Backbone: initial conv + ``num_res_blocks`` residual blocks (192 filters, SE attention)
    Policy head: 1×1 conv → flatten → linear → 96 logits (compact action space)
    Value head:  1×1 conv → flatten → FC(value_fc_hidden) → FC(1, tanh)

Supports Apple MPS GPU acceleration when available.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_device() -> torch.device:
    """Prefer MPS on Apple Silicon, then CUDA, else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = x.mean(dim=(2, 3))
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y


class ResidualBlock(nn.Module):
    """Two conv-BN layers with ReLU, residual skip, and SE channel attention."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + residual)


class SuperTTTNet(nn.Module):
    """Policy + value network over 12×12 planes; logits length ``num_actions`` (default 96)."""

    def __init__(
        self,
        in_channels: int = 7,
        num_filters: int = 192,
        num_res_blocks: int = 8,
        num_actions: int = 96,
        value_fc_hidden: int = 768,
    ):
        super().__init__()
        self.num_actions = num_actions

        self.initial_conv = nn.Conv2d(in_channels, num_filters, 3,
                                      padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(num_filters)

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_filters) for _ in range(num_res_blocks)]
        )

        self.policy_conv = nn.Conv2d(num_filters, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 12 * 12, num_actions)

        self.value_conv = nn.Conv2d(num_filters, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        flat_v = 1 * 12 * 12
        self.value_fc1 = nn.Linear(flat_v, value_fc_hidden)
        self.value_fc2 = nn.Linear(value_fc_hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(policy_logits, value)`` for batch of observations."""
        h = F.relu(self.initial_bn(self.initial_conv(x)))
        h = self.res_blocks(h)

        p = F.relu(self.policy_bn(self.policy_conv(h)))
        p = p.view(p.size(0), -1)
        logits = self.policy_fc(p)

        v = F.relu(self.value_bn(self.value_conv(h)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return logits, value.squeeze(-1)
