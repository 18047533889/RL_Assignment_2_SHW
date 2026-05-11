"""
Tests for augment_learner.py: batch flipping, action/logit permutation, log_prob recompute.
"""

import numpy as np
import torch
import pytest

from augment_learner import flip_batch_inplace, _get_flip_table
from augment import FLIP_TABLE


def _make_batch(n=8, device="cpu"):
    obs = torch.randn(n, 7, 12, 12, device=device)
    action_mask = torch.ones(n, 96, device=device)
    actions = torch.randint(0, 96, (n,), device=device)
    old_logits = torch.randn(n, 96, device=device)
    dist = torch.distributions.Categorical(logits=old_logits)
    old_log_probs = dist.log_prob(actions)
    return {
        "obs": obs,
        "action_mask": action_mask,
        "actions": actions,
        "old_logits": old_logits,
        "old_log_probs": old_log_probs,
    }


def test_flip_reverses_obs_columns():
    batch = _make_batch()
    orig_obs = batch["obs"].clone()
    flip_batch_inplace(batch)
    expected = torch.flip(orig_obs, dims=[-1])
    assert torch.allclose(batch["obs"], expected)


def test_flip_permutes_action_mask():
    batch = _make_batch()
    orig_mask = batch["action_mask"].clone()
    flip_batch_inplace(batch)
    ft = _get_flip_table(torch.device("cpu"))
    expected = orig_mask[:, ft]
    assert torch.allclose(batch["action_mask"], expected)


def test_flip_permutes_actions():
    batch = _make_batch()
    orig_actions = batch["actions"].clone()
    flip_batch_inplace(batch)
    ft = _get_flip_table(torch.device("cpu"))
    expected = ft[orig_actions.long()]
    assert torch.equal(batch["actions"], expected)


def test_flip_permutes_logits():
    batch = _make_batch()
    orig_logits = batch["old_logits"].clone()
    flip_batch_inplace(batch)
    ft = _get_flip_table(torch.device("cpu"))
    expected = orig_logits[:, ft]
    assert torch.allclose(batch["old_logits"], expected)


def test_double_flip_identity():
    batch = _make_batch()
    orig_obs = batch["obs"].clone()
    orig_actions = batch["actions"].clone()
    flip_batch_inplace(batch)
    flip_batch_inplace(batch)
    assert torch.allclose(batch["obs"], orig_obs)
    assert torch.equal(batch["actions"], orig_actions)


def test_flip_recomputes_log_probs():
    batch = _make_batch()
    flip_batch_inplace(batch)
    dist = torch.distributions.Categorical(logits=batch["old_logits"])
    expected_lp = dist.log_prob(batch["actions"])
    assert torch.allclose(batch["old_log_probs"], expected_lp, atol=1e-5)


def test_flip_table_is_permutation():
    ft = FLIP_TABLE
    assert len(ft) == 96
    assert len(set(ft.tolist())) == 96


def test_flip_noop_when_obs_is_none():
    batch = {"obs": None}
    flip_batch_inplace(batch)
    assert batch["obs"] is None


def test_flip_without_optional_keys():
    batch = {"obs": torch.randn(4, 7, 12, 12)}
    orig = batch["obs"].clone()
    flip_batch_inplace(batch)
    expected = torch.flip(orig, dims=[-1])
    assert torch.allclose(batch["obs"], expected)
