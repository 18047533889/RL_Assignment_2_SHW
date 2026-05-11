"""
augment_learner.py — Vertical-axis symmetry augmentation for PPO minibatches.

Pure PyTorch: no RLlib dependency. Called from the training loop to flip 50%
of minibatches in-place using augment.FLIP_TABLE.
"""

from __future__ import annotations

import torch

from augment import FLIP_TABLE

_FLIP_TABLE_T: torch.Tensor | None = None


def _get_flip_table(device: torch.device) -> torch.Tensor:
    global _FLIP_TABLE_T
    if _FLIP_TABLE_T is None or _FLIP_TABLE_T.device != device:
        _FLIP_TABLE_T = torch.from_numpy(FLIP_TABLE).long().to(device)
    return _FLIP_TABLE_T


def flip_batch_inplace(batch: dict) -> None:
    obs = batch.get("obs")
    action_mask = batch.get("action_mask")
    actions = batch.get("actions")
    old_logits = batch.get("old_logits")

    if obs is None:
        return

    device = obs.device
    ft = _get_flip_table(device)

    batch["obs"] = torch.flip(obs, dims=[-1])

    if action_mask is not None:
        batch["action_mask"] = action_mask[:, ft]

    if old_logits is not None:
        batch["old_logits"] = old_logits[:, ft]

    if actions is not None:
        batch["actions"] = ft[actions.long()]

    if old_logits is not None and actions is not None:
        flipped_logits = batch["old_logits"]
        flipped_actions = batch["actions"]
        flipped_mask = batch.get("action_mask")
        if flipped_mask is not None:
            inf_mask = torch.clamp(torch.log(flipped_mask.float() + 1e-10), min=-1e10)
            flipped_logits = flipped_logits + inf_mask
        dist = torch.distributions.Categorical(logits=flipped_logits)
        batch["old_log_probs"] = dist.log_prob(flipped_actions)
