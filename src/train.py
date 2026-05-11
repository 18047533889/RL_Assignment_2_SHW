"""
train.py — PPO + league self-play on Apple Silicon MPS.

Uses TorchRL GAE (generalized_advantage_estimate) for advantage computation,
manual rollout collection (env on CPU, net on MPS), and clip-PPO updates.

Usage:
    python src/train.py [--iterations N] [--checkpoint-dir DIR]
    python src/train.py --restore PATH/TO/checkpoint.pt --iterations 800
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

_ASSIGNMENT2_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F
from torchrl.objectives.value.functional import generalized_advantage_estimate as torchrl_gae

from network import SuperTTTNet
from env import (
    SuperTicTacToeEnv, heuristic_action, random_legal_action,
    line_rusher_action, center_biased_action, edge_explorer_action,
    row_rusher_action, col_rusher_action, greedy_tactical_action,
    lookahead_scripted_action, pure_defender_action,
    PLAYER_MAP, OBS_CHANNELS, FORFEIT_PENALTY,
)
from board import NUM_VALID
from self_play import OpponentPool, SCRIPTED_TYPES
from eval_rollout import (
    EVAL_MAX_AGENT_STEPS,
    evaluate_main_vs_random,
    evaluate_main_vs_snapshot,
    evaluate_main_vs_heuristic,
    evaluate_main_vs_heuristic_by_position,
    evaluate_main_vs_scripted,
    evaluate_random_opening,
    evaluate_forfeit_recovery,
    measure_block_rate,
    greedy_action,
)
from config import (
    PPO_CONFIG,
    MODEL_CONFIG,
    TRAINING_CONFIG,
    DEVICE,
    CURRICULUM_CONFIG,
    SHAPING_SCHEDULE,
)
from training_viz import (
    append_metrics_csv,
    collect_run_meta,
    log_print,
    open_run_directory,
    plot_metrics,
    write_latest_pointer,
    write_run_meta_json,
)


def apply_seed(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed % (2**31)))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_net(model_cfg: dict, device: str | torch.device) -> SuperTTTNet:
    net = SuperTTTNet(
        in_channels=OBS_CHANNELS,
        num_filters=int(model_cfg.get("num_filters", 192)),
        num_res_blocks=int(model_cfg.get("num_res_blocks", 8)),
        num_actions=NUM_VALID,
        value_fc_hidden=int(model_cfg.get("value_fc_hidden", 768)),
    )
    return net.to(device)


def _piecewise_schedule(schedule: list[tuple], step: int) -> float:
    if not schedule:
        return 0.0
    val = schedule[0][1]
    for s, v in schedule:
        if step >= s:
            val = v
        else:
            break
    return val


class _EnvSlot:
    __slots__ = (
        "env", "main_agent", "scripted_type", "opp_rng",
        "ep_obs", "ep_mask", "ep_actions", "ep_log_probs",
        "ep_values", "ep_rewards", "ep_logits", "ep_done",
        "waiting_for_main", "waiting_for_opp",
        "pending_obs", "pending_mask",
        "pending_opp_obs", "pending_opp_mask",
        "finished", "_outcome",
    )

    def __init__(self):
        self.env: SuperTicTacToeEnv | None = None
        self.main_agent: str = "player_0"
        self.scripted_type: str | None = None
        self.opp_rng: np.random.Generator | None = None
        self.ep_obs: list = []
        self.ep_mask: list = []
        self.ep_actions: list = []
        self.ep_log_probs: list = []
        self.ep_values: list = []
        self.ep_rewards: list = []
        self.ep_logits: list = []
        self.ep_done: bool = False
        self.waiting_for_main: bool = False
        self.waiting_for_opp: bool = False
        self.pending_obs: np.ndarray | None = None
        self.pending_mask: np.ndarray | None = None
        self.pending_opp_obs: dict | None = None
        self.pending_opp_mask: np.ndarray | None = None
        self.finished: bool = True
        self._outcome: str = "draw"


def _init_slot(
    slot: _EnvSlot,
    opponent_pool: OpponentPool,
    opp_net: SuperTTTNet,
    rng: np.random.Generator,
    curriculum_cfg: dict,
    device: torch.device,
    shaping_multiplier: float = 1.0,
) -> None:
    opp_state, scripted_type = opponent_pool.get_opponent_state(rng)
    if opp_state is not None:
        opp_net.load_state_dict(opp_state)
        opp_net.eval()
    slot.scripted_type = scripted_type

    main_is_p0 = rng.random() < 0.5
    slot.main_agent = "player_0" if main_is_p0 else "player_1"

    env_seed = int(rng.integers(0, 2**31))
    slot.env = SuperTicTacToeEnv(
        seed=env_seed,
        random_opening_prob=curriculum_cfg.get("random_opening_prob", 0.0),
        random_opening_steps=curriculum_cfg.get("random_opening_steps", 0),
        forfeit_injection_prob=curriculum_cfg.get("forfeit_injection_prob", 0.0),
    )
    slot.env.reset(seed=env_seed)
    slot.env._shaping_multiplier = shaping_multiplier
    slot.opp_rng = np.random.default_rng(env_seed + 999)
    slot.ep_obs = []
    slot.ep_mask = []
    slot.ep_actions = []
    slot.ep_log_probs = []
    slot.ep_values = []
    slot.ep_rewards = []
    slot.ep_logits = []
    slot.ep_done = False
    slot.waiting_for_main = False
    slot.pending_obs = None
    slot.pending_mask = None
    slot.finished = False


def _advance_slot_until_pause_or_done(
    slot: _EnvSlot,
) -> None:
    env = slot.env
    while True:
        agent = env.agent_selection
        if env.terminations.get(agent) or env.truncations.get(agent):
            done_any = any(
                env.terminations.get(ag, False) or env.truncations.get(ag, False)
                for ag in env.possible_agents
            )
            if done_any and not slot.ep_done:
                _finalise_episode(slot)
                return
            env.step(None)
            if all(
                env.terminations.get(ag, False) or env.truncations.get(ag, False)
                for ag in env.possible_agents
            ):
                if not slot.ep_done:
                    _finalise_episode(slot)
                return
            continue

        obs = env.observe(agent)

        if agent == slot.main_agent:
            slot.waiting_for_main = True
            slot.waiting_for_opp = False
            slot.pending_obs = obs["observations"]
            slot.pending_mask = obs["action_mask"]
            return

        if slot.scripted_type == "random_legal":
            m = obs["action_mask"]
            legal = np.flatnonzero(m)
            a = int(slot.opp_rng.choice(legal)) if len(legal) > 0 else 0
        elif slot.scripted_type == "heuristic":
            pid = PLAYER_MAP[agent]
            a = heuristic_action(env.board, pid, slot.opp_rng)
        elif slot.scripted_type == "line_rusher":
            pid = PLAYER_MAP[agent]
            a = line_rusher_action(env.board, pid, slot.opp_rng)
        elif slot.scripted_type == "center_biased":
            pid = PLAYER_MAP[agent]
            a = center_biased_action(env.board, pid, slot.opp_rng)
        elif slot.scripted_type == "edge_explorer":
            pid = PLAYER_MAP[agent]
            a = edge_explorer_action(env.board, pid, slot.opp_rng)
        elif slot.scripted_type == "row_rusher":
            pid = PLAYER_MAP[agent]
            a = row_rusher_action(env.board, pid, slot.opp_rng)
        elif slot.scripted_type == "col_rusher":
            pid = PLAYER_MAP[agent]
            a = col_rusher_action(env.board, pid, slot.opp_rng)
        elif slot.scripted_type == "greedy_tactical":
            pid = PLAYER_MAP[agent]
            a = greedy_tactical_action(env.board, pid, slot.opp_rng)
        elif slot.scripted_type == "lookahead_scripted":
            pid = PLAYER_MAP[agent]
            a = lookahead_scripted_action(env.board, pid, slot.opp_rng)
        elif slot.scripted_type == "pure_defender":
            pid = PLAYER_MAP[agent]
            a = pure_defender_action(env.board, pid, slot.opp_rng)
        else:
            slot.waiting_for_opp = True
            slot.waiting_for_main = False
            slot.pending_opp_obs = obs
            return
        env.step(a)
        if slot.ep_rewards:
            slot.ep_rewards[-1] += env.rewards.get(slot.main_agent, 0.0)

        done_any = any(
            env.terminations.get(ag, False) or env.truncations.get(ag, False)
            for ag in env.possible_agents
        )
        if done_any:
            _finalise_episode(slot)
            return


def _finalise_episode(slot: _EnvSlot) -> None:
    env = slot.env
    i0 = env.infos.get("player_0", {})
    i1 = env.infos.get("player_1", {})
    winner = i0.get("winner") or i1.get("winner")
    is_draw_flag = i0.get("draw", False) or i1.get("draw", False)

    if is_draw_flag or winner is None:
        slot._outcome = "draw"
    elif winner == slot.main_agent:
        slot._outcome = "main_win"
    else:
        slot._outcome = "main_loss"

    if slot.ep_rewards:
        step_reward = env.rewards.get(slot.main_agent, 0.0)
        final_reward = {"draw": 0.0, "main_win": 1.0, "main_loss": -1.0}[slot._outcome]
        slot.ep_rewards[-1] = step_reward if abs(step_reward) > abs(final_reward) else final_reward

    slot.ep_done = True
    slot.waiting_for_main = False
    slot.finished = True


def _flush_finished(
    slots: list[_EnvSlot],
    opponent_pool: OpponentPool,
    obs_list: list, action_mask_list: list, action_list: list,
    log_prob_list: list, value_list: list, reward_list: list,
    done_list: list, logits_list: list,
    rng: np.random.Generator, opp_net: SuperTTTNet,
    curriculum_cfg: dict, device: torch.device,
    shaping_multiplier: float = 1.0,
    side_stats: dict | None = None,
) -> int:
    added = 0
    for s in slots:
        if s.finished and s.ep_obs:
            opponent_pool.record_outcome(s._outcome)
            n_ep = len(s.ep_obs)
            obs_list.extend(s.ep_obs)
            action_mask_list.extend(s.ep_mask)
            action_list.extend(s.ep_actions)
            log_prob_list.extend(s.ep_log_probs)
            value_list.extend(s.ep_values)
            reward_list.extend(s.ep_rewards)
            logits_list.extend(s.ep_logits)
            for j in range(n_ep):
                done_list.append(j == n_ep - 1 and s.ep_done)
            added += n_ep
            if side_stats is not None:
                key = "first" if s.main_agent == "player_0" else "second"
                side_stats[f"{key}_n"] += 1
                side_stats[f"{key}_len_total"] += n_ep
                if s._outcome == "main_win":
                    side_stats[f"{key}_wins"] += 1
                elif s._outcome == "main_loss":
                    side_stats[f"{key}_losses"] += 1
                else:
                    side_stats[f"{key}_draws"] += 1
            _init_slot(s, opponent_pool, opp_net, rng, curriculum_cfg, device, shaping_multiplier)
            _advance_slot_until_pause_or_done(s)
    return added


def _collect_rollouts(
    main_net: SuperTTTNet,
    opp_net: SuperTTTNet,
    opponent_pool: OpponentPool,
    *,
    num_steps: int,
    device: torch.device,
    rng: np.random.Generator,
    curriculum_cfg: dict,
    num_collectors: int = 8,
    shaping_multiplier: float = 1.0,
) -> dict:
    obs_list: list[np.ndarray] = []
    action_mask_list: list[np.ndarray] = []
    action_list: list[int] = []
    log_prob_list: list[float] = []
    value_list: list[float] = []
    reward_list: list[float] = []
    done_list: list[bool] = []
    logits_list: list[np.ndarray] = []

    main_steps_count = 0
    main_success_count = 0
    forfeit_injected_count = 0
    blocked_count = 0
    persist_count = 0
    partial_block_count = 0
    waste_count = 0

    side_stats: dict[str, int] = {
        "first_n": 0, "first_wins": 0, "first_losses": 0,
        "first_draws": 0, "first_len_total": 0,
        "second_n": 0, "second_wins": 0, "second_losses": 0,
        "second_draws": 0, "second_len_total": 0,
    }

    steps_collected = 0
    n_envs = max(1, num_collectors)
    slots = [_EnvSlot() for _ in range(n_envs)]

    for s in slots:
        _init_slot(s, opponent_pool, opp_net, rng, curriculum_cfg, device, shaping_multiplier)
        _advance_slot_until_pause_or_done(s)

    while steps_collected < num_steps:
        steps_collected += _flush_finished(
            slots, opponent_pool,
            obs_list, action_mask_list, action_list,
            log_prob_list, value_list, reward_list,
            done_list, logits_list,
            rng, opp_net, curriculum_cfg, device, shaping_multiplier,
            side_stats=side_stats,
        )

        opp_indices = [i for i, s in enumerate(slots) if s.waiting_for_opp]
        if opp_indices:
            opp_obs_np = np.stack([
                slots[i].pending_opp_obs["observations"].astype(np.float32)
                for i in opp_indices
            ])
            opp_mask_np = np.stack([
                slots[i].pending_opp_obs["action_mask"].astype(np.float32)
                for i in opp_indices
            ])
            o_t = torch.from_numpy(opp_obs_np).to(device)
            m_t = torch.from_numpy(opp_mask_np).to(device)
            with torch.inference_mode():
                logits_b, _ = opp_net(o_t)
                inf_m = torch.clamp(torch.log(m_t + 1e-10), min=-1e10)
                opp_actions = torch.argmax(logits_b + inf_m, dim=-1).cpu().numpy()

            for idx_in_batch, slot_idx in enumerate(opp_indices):
                s = slots[slot_idx]
                s.env.step(int(opp_actions[idx_in_batch]))
                if s.ep_rewards:
                    s.ep_rewards[-1] += s.env.rewards.get(s.main_agent, 0.0)
                s.waiting_for_opp = False
                s.pending_opp_obs = None
                done_any = any(
                    s.env.terminations.get(ag, False) or s.env.truncations.get(ag, False)
                    for ag in s.env.possible_agents
                )
                if done_any:
                    _finalise_episode(s)
                else:
                    _advance_slot_until_pause_or_done(s)

            steps_collected += _flush_finished(
                slots, opponent_pool,
                obs_list, action_mask_list, action_list,
                log_prob_list, value_list, reward_list,
                done_list, logits_list,
                rng, opp_net, curriculum_cfg, device, shaping_multiplier,
                side_stats=side_stats,
            )

        main_indices = [i for i, s in enumerate(slots) if s.waiting_for_main]
        if not main_indices:
            if all(s.finished or s.waiting_for_opp for s in slots):
                if all(s.finished for s in slots):
                    continue
                continue
            continue

        batch_obs = np.stack([slots[i].pending_obs for i in main_indices])
        batch_mask = np.stack([slots[i].pending_mask for i in main_indices])

        o_t = torch.from_numpy(batch_obs.astype(np.float32)).to(device)
        m_t = torch.from_numpy(batch_mask.astype(np.float32)).to(device)

        with torch.inference_mode():
            logits_batch, values_batch = main_net(o_t)
            inf_mask = torch.clamp(torch.log(m_t + 1e-10), min=-1e10)
            masked_logits_batch = logits_batch + inf_mask
            dist = torch.distributions.Categorical(logits=masked_logits_batch)
            actions_batch = dist.sample()
            log_probs_batch = dist.log_prob(actions_batch)

        actions_cpu = actions_batch.cpu().numpy()
        log_probs_cpu = log_probs_batch.cpu().numpy()
        values_cpu = values_batch.cpu().numpy()
        masked_logits_cpu = masked_logits_batch.cpu().numpy()

        for idx_in_batch, slot_idx in enumerate(main_indices):
            s = slots[slot_idx]
            a = int(actions_cpu[idx_in_batch])
            lp = float(log_probs_cpu[idx_in_batch])
            v = float(values_cpu[idx_in_batch])
            ml = masked_logits_cpu[idx_in_batch]

            s.ep_obs.append(s.pending_obs)
            s.ep_mask.append(s.pending_mask)
            s.ep_actions.append(a)
            s.ep_log_probs.append(lp)
            s.ep_values.append(v)
            s.ep_logits.append(ml)
            s.ep_rewards.append(0.0)
            s.waiting_for_main = False
            s.pending_obs = None
            s.pending_mask = None

            s.env.step(a)
            main_steps_count += 1
            if s.env._last_step_forfeit_injected:
                forfeit_injected_count += 1
            if s.env._last_step_blocked:
                blocked_count += 1
            if s.env._last_step_main_succeeded:
                main_success_count += 1
            if s.env._last_step_persist:
                persist_count += 1
            if s.env._last_step_partial_block:
                partial_block_count += 1
            if s.env._last_step_waste:
                waste_count += 1
            s.ep_rewards[-1] += s.env.rewards.get(s.main_agent, 0.0)

            done_any = any(
                s.env.terminations.get(ag, False) or s.env.truncations.get(ag, False)
                for ag in s.env.possible_agents
            )
            if done_any:
                _finalise_episode(s)
            else:
                _advance_slot_until_pause_or_done(s)

    denom = max(main_steps_count, 1)
    denom_succ = max(main_success_count, 1)

    def _side_rate(wins: int, losses: int, draws: int) -> float:
        total = wins + losses + draws
        return wins / total if total > 0 else float("nan")

    first_n = side_stats["first_n"]
    second_n = side_stats["second_n"]
    batch = {
        "obs": torch.from_numpy(np.array(obs_list, dtype=np.float32)).to(device),
        "action_mask": torch.from_numpy(np.array(action_mask_list, dtype=np.float32)).to(device),
        "actions": torch.tensor(action_list, dtype=torch.long, device=device),
        "old_log_probs": torch.tensor(log_prob_list, dtype=torch.float32, device=device),
        "values": torch.tensor(value_list, dtype=torch.float32, device=device),
        "rewards": torch.tensor(reward_list, dtype=torch.float32, device=device),
        "dones": torch.tensor(done_list, dtype=torch.bool, device=device),
        "old_logits": torch.from_numpy(np.array(logits_list, dtype=np.float32)).to(device),
        "ff_inj_rate": forfeit_injected_count / denom,
        "blk_rate": blocked_count / denom,
        "persist_rate": persist_count / denom_succ,
        "partial_block_rate": partial_block_count / denom_succ,
        "waste_rate": waste_count / denom_succ,
        "first_win_rate": _side_rate(
            side_stats["first_wins"], side_stats["first_losses"], side_stats["first_draws"]
        ),
        "second_win_rate": _side_rate(
            side_stats["second_wins"], side_stats["second_losses"], side_stats["second_draws"]
        ),
        "first_ep_len": (side_stats["first_len_total"] / first_n) if first_n > 0 else float("nan"),
        "second_ep_len": (side_stats["second_len_total"] / second_n) if second_n > 0 else float("nan"),
        "first_n": first_n,
        "second_n": second_n,
    }
    return batch


def _rollout_forfeit_and_episode_len(batch: dict) -> tuple[float, float]:
    """Share of main-agent steps with forfeit penalty, and mean length of completed episodes in batch."""
    r_np = batch["rewards"].detach().cpu().numpy()
    d_np = batch["dones"].detach().cpu().numpy()
    # Float32 rollout rewards may not match Python FORFEIT_PENALTY exactly
    forfeit_rate = float(
        np.mean(np.isclose(r_np, float(FORFEIT_PENALTY), rtol=1e-4, atol=1e-5))
    )
    lengths: list[int] = []
    acc = 0
    for j in range(len(d_np)):
        acc += 1
        if d_np[j]:
            lengths.append(acc)
            acc = 0
    mean_len = float(np.mean(lengths)) if lengths else float("nan")
    return forfeit_rate, mean_len


def _compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lam: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    state_value = values.unsqueeze(0).unsqueeze(-1)
    next_sv = torch.cat([values[1:], torch.zeros(1, device=device)], dim=0)
    next_sv = next_sv.unsqueeze(0).unsqueeze(-1)
    r = rewards.unsqueeze(0).unsqueeze(-1)
    d = dones.unsqueeze(0).unsqueeze(-1)
    advantages, value_target = torchrl_gae(
        gamma=gamma, lmbda=lam,
        state_value=state_value,
        next_state_value=next_sv,
        reward=r, done=d, terminated=d,
        time_dim=-2,
    )
    return advantages.squeeze(0).squeeze(-1), value_target.squeeze(0).squeeze(-1)


def _ppo_update(
    main_net: SuperTTTNet,
    optimizer: torch.optim.Optimizer,
    batch: dict,
    *,
    clip_param: float,
    vf_coeff: float,
    entropy_coeff: float,
    num_epochs: int,
    minibatch_size: int,
    grad_clip: float,
    gamma: float,
    lam: float,
    device: torch.device,
    augment: bool = False,
) -> dict:
    with torch.no_grad():
        advantages, returns = _compute_gae(
            batch["rewards"], batch["values"], batch["dones"],
            gamma, lam, device,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    n = batch["obs"].shape[0]
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy_loss = 0.0
    total_clip_frac = 0.0
    total_approx_kl = 0.0
    total_grad_norm = 0.0
    num_updates = 0

    for _ in range(num_epochs):
        indices = torch.randperm(n, device=device)
        for start in range(0, n, minibatch_size):
            end = min(start + minibatch_size, n)
            idx = indices[start:end]

            mb = {
                "obs": batch["obs"][idx],
                "action_mask": batch["action_mask"][idx],
                "actions": batch["actions"][idx],
                "old_log_probs": batch["old_log_probs"][idx],
                "old_logits": batch["old_logits"][idx],
            }
            mb_advantages = advantages[idx]
            mb_returns = returns[idx]

            if augment and torch.rand(1).item() < 0.5:
                from augment_learner import flip_batch_inplace
                flip_batch_inplace(mb)

            logits, values = main_net(mb["obs"])
            inf_mask = torch.clamp(torch.log(mb["action_mask"] + 1e-10), min=-1e10)
            masked_logits = logits + inf_mask

            dist = torch.distributions.Categorical(logits=masked_logits)
            new_log_probs = dist.log_prob(mb["actions"])
            entropy = dist.entropy()

            ratio = torch.exp(new_log_probs - mb["old_log_probs"])
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values.view(-1), mb_returns.view(-1))

            entropy_loss = -entropy.mean()

            loss = policy_loss + vf_coeff * value_loss + entropy_coeff * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(main_net.parameters(), grad_clip)
            optimizer.step()

            with torch.no_grad():
                clip_frac = ((ratio - 1.0).abs() > clip_param).float().mean().item()
                approx_kl = (mb["old_log_probs"] - new_log_probs).mean().item()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy_loss.item()
            total_clip_frac += clip_frac
            total_approx_kl += approx_kl
            total_grad_norm += gn.item() if isinstance(gn, torch.Tensor) else float(gn)
            num_updates += 1

    nu = max(num_updates, 1)
    v_pred = batch["values"]
    with torch.no_grad():
        v_targ = returns
        v_diff = v_targ - v_pred[:len(v_targ)]
        ev = 1.0 - (v_diff.var() / (v_targ.var() + 1e-8)).item() if v_targ.numel() > 1 else 0.0

    return {
        "policy_loss": total_policy_loss / nu,
        "value_loss": total_value_loss / nu,
        "entropy": -total_entropy_loss / nu,
        "clip_fraction": total_clip_frac / nu,
        "approx_kl": total_approx_kl / nu,
        "grad_norm": total_grad_norm / nu,
        "explained_variance": ev,
    }


def _csv_float(x) -> float | str:
    if x is None or x == "":
        return ""
    try:
        return float(x)
    except (TypeError, ValueError):
        return ""


def train(
    num_iterations: int | None = None,
    checkpoint_dir: str = "checkpoints",
    restore_path: str | None = None,
    results_root: str | None = None,
    plot_every: int = 5,
    checkpoint_freq: int | None = None,
    seed: int | None = None,
    early_stop: bool = True,
    augment: bool = False,
    fast: bool = False,
    eval_interval_override: int | None = None,
    eval_num_episodes_override: int | None = None,
):
    if num_iterations is None:
        num_iterations = int(TRAINING_CONFIG["num_iterations"])
    checkpoint_dir = os.path.abspath(os.path.expanduser(checkpoint_dir))

    results_root = results_root or os.path.join(_ASSIGNMENT2_ROOT, "results")
    os.makedirs(results_root, exist_ok=True)
    run_dir, log_f, csv_path = open_run_directory(results_root)
    write_latest_pointer(results_root, run_dir)
    plots_dir = os.path.join(run_dir, "plots")

    eff_seed = seed if seed is not None else TRAINING_CONFIG.get("default_seed")
    if eff_seed is not None:
        apply_seed(int(eff_seed))
    meta = collect_run_meta(sys.argv, int(eff_seed) if eff_seed is not None else None)
    meta["framework"] = "torchrl"
    write_run_meta_json(run_dir, meta)

    device = torch.device(DEVICE)
    log_print(log_f, f"Device: {device} (MPS={'mps' in str(device)})")

    if fast:
        model_cfg = {"num_filters": 64, "num_res_blocks": 2, "value_fc_hidden": 256}
        ppo = dict(PPO_CONFIG)
        tb = 2048
        ppo["train_batch_size"] = tb
        ppo["sgd_minibatch_size"] = 128
        ppo["num_epochs"] = 2
        ppo["lr_schedule"] = [(0, 2e-4), (400 * tb, 8e-5), (700 * tb, 2e-5)]
        ppo["entropy_schedule"] = [(0, 0.03), (300 * tb, 0.015), (600 * tb, 0.005)]
        plot_every = min(max(1, plot_every), 2)
        log_print(log_f, "FAST mode: smaller CNN, batch=2048")
    else:
        model_cfg = dict(MODEL_CONFIG)
        ppo = dict(PPO_CONFIG)

    main_net = _build_net(model_cfg, device)
    opp_net = _build_net(model_cfg, device)
    opp_net.eval()

    param_count = sum(p.numel() for p in main_net.parameters())
    log_print(log_f, f"Network: {param_count:,} parameters")

    optimizer = torch.optim.Adam(main_net.parameters(), lr=ppo["lr_schedule"][0][1])

    opponent_pool = OpponentPool()
    opponent_pool.initialize(main_net.state_dict())

    start_iter = 0
    total_env_steps = 0

    if restore_path:
        restore_path = os.path.abspath(os.path.expanduser(restore_path))
        log_print(log_f, f"Restoring from: {restore_path}")
        ckpt = torch.load(restore_path, map_location="cpu", weights_only=False)
        main_net.load_state_dict(ckpt["main_net"])
        main_net.to(device)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iter = ckpt.get("iteration", 0)
        total_env_steps = ckpt.get("total_env_steps", 0)
        if "opponent_pool" in ckpt:
            pool_data = ckpt["opponent_pool"]
            for i, sd in enumerate(pool_data.get("slots", [])):
                if sd is not None and i < len(opponent_pool._slots):
                    opponent_pool._slots[i] = sd
            opponent_pool._snapshot_rr = pool_data.get("snapshot_rr", 0)
        log_print(log_f, f"Restored at iteration {start_iter}, env_steps={total_env_steps}")

    ckpt_every = checkpoint_freq if checkpoint_freq is not None else TRAINING_CONFIG["checkpoint_freq"]
    ckpt_every = max(1, int(ckpt_every))
    eval_iv = int(
        eval_interval_override
        if eval_interval_override is not None
        else (TRAINING_CONFIG.get("eval_interval", 0) or 0)
    )
    eval_n = int(
        eval_num_episodes_override
        if eval_num_episodes_override is not None
        else (TRAINING_CONFIG.get("eval_num_episodes", 50) or 50)
    )
    stop_thr = float(TRAINING_CONFIG.get("stop_reward", 1.0))

    if fast:
        ckpt_every = min(ckpt_every, 2)
        if eval_iv > 0:
            eval_iv = min(eval_iv, 2)

    remaining = max(0, num_iterations - start_iter) if start_iter > 0 else num_iterations
    if remaining <= 0:
        log_print(log_f, f"Already at iter {start_iter} >= {num_iterations}, nothing to do.")
        return None

    if start_iter > 0:
        log_print(
            log_f,
            f"Training: {remaining} iters to run (global {start_iter + 1}..{num_iterations}), "
            f"batch={ppo['train_batch_size']}",
        )
    else:
        log_print(log_f, f"Training: {num_iterations} iterations, batch={ppo['train_batch_size']}")
    log_print(log_f, f"Checkpoint every {ckpt_every} iter -> {checkpoint_dir}/")
    if augment:
        log_print(log_f, "Symmetry augmentation: ON (50% minibatch flip)")

    rng = np.random.default_rng(int(eff_seed) if eff_seed else 42)
    os.makedirs(checkpoint_dir, exist_ok=True)
    t0 = time.time()
    final_path = None

    try:
        for i in range(1, remaining + 1):
            iter_t0 = time.time()
            global_iter = start_iter + i

            lr = _piecewise_schedule(ppo["lr_schedule"], total_env_steps)
            ent_coeff = _piecewise_schedule(ppo["entropy_schedule"], total_env_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            shaping_mult = _piecewise_schedule(SHAPING_SCHEDULE, total_env_steps)

            main_net.train()
            batch = _collect_rollouts(
                main_net, opp_net, opponent_pool,
                num_steps=ppo["train_batch_size"],
                device=device,
                rng=rng,
                curriculum_cfg=CURRICULUM_CONFIG,
                num_collectors=ppo.get("num_collectors", 8),
                shaping_multiplier=shaping_mult,
            )
            total_env_steps += batch["obs"].shape[0]
            forfeit_rate, mean_episode_len = _rollout_forfeit_and_episode_len(batch)
            ff_inj_rate = float(batch.get("ff_inj_rate", 0.0))
            blk_rate = float(batch.get("blk_rate", 0.0))
            persist_rate = float(batch.get("persist_rate", 0.0))
            partial_block_rate = float(batch.get("partial_block_rate", 0.0))
            waste_rate = float(batch.get("waste_rate", 0.0))
            first_wr = float(batch.get("first_win_rate", float("nan")))
            second_wr = float(batch.get("second_win_rate", float("nan")))
            first_ep_len = float(batch.get("first_ep_len", float("nan")))
            second_ep_len = float(batch.get("second_ep_len", float("nan")))

            losses = _ppo_update(
                main_net, optimizer, batch,
                clip_param=ppo["clip_param"],
                vf_coeff=ppo["vf_loss_coeff"],
                entropy_coeff=ent_coeff,
                num_epochs=ppo["num_epochs"],
                minibatch_size=ppo["sgd_minibatch_size"],
                grad_clip=ppo["grad_clip"],
                gamma=ppo["gamma"],
                lam=ppo["lambda_"],
                device=device,
                augment=augment,
            )

            if opponent_pool.should_snapshot():
                idx = opponent_pool.save_snapshot(main_net.state_dict())
                log_print(log_f, f"  Snapshot -> slot {idx}")

            iter_dt = time.time() - iter_t0
            win_rate = opponent_pool.win_rate()
            mean_reward = batch["rewards"].mean().item()
            batch_steps = batch["obs"].shape[0]
            steps_sec = batch_steps / max(iter_dt, 0.01)
            elapsed_h = (time.time() - t0) / 3600.0
            remain_h = elapsed_h / max(i, 1) * (remaining - i)

            wr_str = f"{win_rate:.3f}" if win_rate is not None else "N/A"
            fwr_str = f"{first_wr:.3f}" if math.isfinite(first_wr) else "N/A"
            swr_str = f"{second_wr:.3f}" if math.isfinite(second_wr) else "N/A"
            log_print(
                log_f,
                f"Iter {global_iter:4d}/{num_iterations} | reward={mean_reward:+.3f} | "
                f"win_rate={wr_str} | 1st={fwr_str} | 2nd={swr_str} | "
                f"lr={lr:.1e} | ent={ent_coeff:.3f} | "
                f"ploss={losses['policy_loss']:.4f} | vloss={losses['value_loss']:.4f} | "
                f"clip={losses['clip_fraction']:.3f} | kl={losses['approx_kl']:.4f} | "
                f"gnorm={losses['grad_norm']:.3f} | ev={losses['explained_variance']:.3f} | "
                f"shaping={shaping_mult:.2f} | snaps={opponent_pool.snapshot_count} | "
                f"ff={forfeit_rate:.3f} | ff_inj={ff_inj_rate:.3f} | blk={blk_rate:.3f} | "
                f"persist={persist_rate:.3f} | pblk={partial_block_rate:.3f} | "
                f"waste={waste_rate:.3f} | "
                f"ep_len={mean_episode_len:.1f} | "
                f"{steps_sec:.0f} stp/s | {iter_dt:.1f}s | "
                f"ETA {remain_h:.1f}h",
            )

            ev_w, ev_d, ev_rand, ev_heur = "", "", "", ""
            ev_lr, ev_rr, ev_cr, ev_br = "", "", "", ""
            ev_gt, ev_la = "", ""
            ev_heur_first = ev_heur_second = ""
            ev_line_first = ev_line_second = ""
            ev_row_first = ev_row_second = ""
            ev_col_first = ev_col_second = ""
            ev_gt_first = ev_gt_second = ""
            ev_la_first = ev_la_second = ""
            ev_pd_first = ev_pd_second = ""
            ev_ro_heur = ev_ro_col = ev_ro_row = ""
            ev_recov_heur = ""
            stop_now = False
            if eval_iv > 0 and i % eval_iv == 0:
                main_net.eval()
                bseed = int(eff_seed or 0) + i * 9973

                opp_eval_net = _build_net(model_cfg, device)
                slot0 = opponent_pool._slots[0]
                if slot0 is not None:
                    opp_eval_net.load_state_dict(slot0)
                opp_eval_net.eval()

                ev = evaluate_main_vs_snapshot(
                    main_net, opp_eval_net,
                    num_episodes=eval_n, base_seed=bseed, device=device,
                )
                ev_w = ev.get("win_rate", float("nan"))
                ev_d = ev.get("draw_rate", float("nan"))
                log_print(log_f, f"  [eval] vs snapshot: win={float(ev_w):.3f} draw={float(ev_d):.3f}")

                ev_r = evaluate_main_vs_random(
                    main_net, num_episodes=eval_n, base_seed=bseed, device=device,
                )
                ev_rand = ev_r.get("win_rate", float("nan"))
                log_print(log_f, f"  [eval] vs random:   win={float(ev_rand):.3f}")

                ev_h_first = evaluate_main_vs_heuristic_by_position(
                    main_net, num_episodes=eval_n, base_seed=bseed,
                    device=device, main_agent_position="first",
                )
                ev_h_second = evaluate_main_vs_heuristic_by_position(
                    main_net, num_episodes=eval_n, base_seed=bseed,
                    device=device, main_agent_position="second",
                )
                ev_heur_first = ev_h_first.get("win_rate", float("nan"))
                ev_heur_second = ev_h_second.get("win_rate", float("nan"))
                ev_heur = (float(ev_heur_first) + float(ev_heur_second)) / 2.0
                log_print(
                    log_f,
                    f"  [eval] heur:        first={float(ev_heur_first):.3f}  "
                    f"second={float(ev_heur_second):.3f}",
                )

                ev_l_first = evaluate_main_vs_scripted(
                    main_net, line_rusher_action,
                    num_episodes=eval_n, base_seed=bseed, device=device,
                    main_agent_position="first",
                )
                ev_l_second = evaluate_main_vs_scripted(
                    main_net, line_rusher_action,
                    num_episodes=eval_n, base_seed=bseed, device=device,
                    main_agent_position="second",
                )
                ev_line_first = ev_l_first.get("win_rate", float("nan"))
                ev_line_second = ev_l_second.get("win_rate", float("nan"))
                ev_lr = (float(ev_line_first) + float(ev_line_second)) / 2.0
                log_print(
                    log_f,
                    f"  [eval] line_rusher: first={float(ev_line_first):.3f}  "
                    f"second={float(ev_line_second):.3f}",
                )

                ev_rw_first = evaluate_main_vs_scripted(
                    main_net, row_rusher_action,
                    num_episodes=eval_n, base_seed=bseed, device=device,
                    main_agent_position="first",
                )
                ev_rw_second = evaluate_main_vs_scripted(
                    main_net, row_rusher_action,
                    num_episodes=eval_n, base_seed=bseed, device=device,
                    main_agent_position="second",
                )
                ev_row_first = ev_rw_first.get("win_rate", float("nan"))
                ev_row_second = ev_rw_second.get("win_rate", float("nan"))
                ev_rr = (float(ev_row_first) + float(ev_row_second)) / 2.0
                log_print(
                    log_f,
                    f"  [eval] row_rusher:  first={float(ev_row_first):.3f}  "
                    f"second={float(ev_row_second):.3f}",
                )

                ev_cl_first = evaluate_main_vs_scripted(
                    main_net, col_rusher_action,
                    num_episodes=eval_n, base_seed=bseed, device=device,
                    main_agent_position="first",
                )
                ev_cl_second = evaluate_main_vs_scripted(
                    main_net, col_rusher_action,
                    num_episodes=eval_n, base_seed=bseed, device=device,
                    main_agent_position="second",
                )
                ev_col_first = ev_cl_first.get("win_rate", float("nan"))
                ev_col_second = ev_cl_second.get("win_rate", float("nan"))
                ev_cr = (float(ev_col_first) + float(ev_col_second)) / 2.0
                log_print(
                    log_f,
                    f"  [eval] col_rusher:  first={float(ev_col_first):.3f}  "
                    f"second={float(ev_col_second):.3f}",
                )

                # Tactical scripted opponents are CPU-heavy per step; use fewer episodes than generic eval.
                tac_eval_n = max(6, min(eval_n, 24))
                ev_gtf = evaluate_main_vs_scripted(
                    main_net, greedy_tactical_action,
                    num_episodes=tac_eval_n, base_seed=bseed, device=device,
                    main_agent_position="first",
                )
                ev_gts = evaluate_main_vs_scripted(
                    main_net, greedy_tactical_action,
                    num_episodes=tac_eval_n, base_seed=bseed, device=device,
                    main_agent_position="second",
                )
                ev_gt_first = ev_gtf.get("win_rate", float("nan"))
                ev_gt_second = ev_gts.get("win_rate", float("nan"))
                ev_gt = (float(ev_gt_first) + float(ev_gt_second)) / 2.0
                log_print(
                    log_f,
                    f"  [eval] greedy_tactical: first={float(ev_gt_first):.3f}  "
                    f"second={float(ev_gt_second):.3f}",
                )

                ev_laf = evaluate_main_vs_scripted(
                    main_net, lookahead_scripted_action,
                    num_episodes=tac_eval_n, base_seed=bseed + 91337, device=device,
                    main_agent_position="first",
                )
                ev_las = evaluate_main_vs_scripted(
                    main_net, lookahead_scripted_action,
                    num_episodes=tac_eval_n, base_seed=bseed + 91337, device=device,
                    main_agent_position="second",
                )
                ev_la_first = ev_laf.get("win_rate", float("nan"))
                ev_la_second = ev_las.get("win_rate", float("nan"))
                ev_la = (float(ev_la_first) + float(ev_la_second)) / 2.0
                log_print(
                    log_f,
                    f"  [eval] lookahead_scripted: first={float(ev_la_first):.3f}  "
                    f"second={float(ev_la_second):.3f}",
                )

                ev_pdf = evaluate_main_vs_scripted(
                    main_net, pure_defender_action,
                    num_episodes=tac_eval_n, base_seed=bseed + 31337, device=device,
                    main_agent_position="first",
                )
                ev_pds = evaluate_main_vs_scripted(
                    main_net, pure_defender_action,
                    num_episodes=tac_eval_n, base_seed=bseed + 31337, device=device,
                    main_agent_position="second",
                )
                ev_pd_first = ev_pdf.get("win_rate", float("nan"))
                ev_pd_second = ev_pds.get("win_rate", float("nan"))
                log_print(
                    log_f,
                    f"  [eval] pure_defender: first={float(ev_pd_first):.3f}  "
                    f"second={float(ev_pd_second):.3f}",
                )

                # Random-opening eval: diverse initial states, agent alternates first/second
                ev_ro_h = evaluate_random_opening(
                    main_net, heuristic_action,
                    num_episodes=eval_n, base_seed=bseed, device=device,
                )
                ev_ro_c = evaluate_random_opening(
                    main_net, col_rusher_action,
                    num_episodes=eval_n, base_seed=bseed, device=device,
                )
                ev_ro_r = evaluate_random_opening(
                    main_net, row_rusher_action,
                    num_episodes=eval_n, base_seed=bseed, device=device,
                )
                ev_ro_heur = ev_ro_h.get("win_rate", float("nan"))
                ev_ro_col = ev_ro_c.get("win_rate", float("nan"))
                ev_ro_row = ev_ro_r.get("win_rate", float("nan"))
                log_print(
                    log_f,
                    f"  [eval] random_open: vs_heur={float(ev_ro_heur):.3f}  "
                    f"vs_col={float(ev_ro_col):.3f}  vs_row={float(ev_ro_row):.3f}",
                )

                # Recovery eval: one forced FORFEIT on main, does it give up?
                ev_rec = evaluate_forfeit_recovery(
                    main_net, heuristic_action,
                    num_episodes=eval_n, base_seed=bseed, device=device,
                )
                ev_recov_heur = ev_rec.get("win_rate", float("nan"))
                log_print(
                    log_f,
                    f"  [eval] recovery_vs_heur={float(ev_recov_heur):.3f} "
                    f"(baseline heur_first={float(ev_heur_first):.3f})",
                )

                br = measure_block_rate(
                    main_net, num_episodes=max(10, eval_n // 2),
                    base_seed=bseed, device=device,
                )
                ev_br = br
                log_print(log_f, f"  [eval] block_rate_vs_line={float(ev_br):.3f}")

                try:
                    evf_h = float(ev_heur)
                except (TypeError, ValueError):
                    evf_h = float("nan")
                if early_stop and math.isfinite(evf_h) and evf_h >= stop_thr:
                    log_print(log_f, f"  Early stop: vs heuristic win rate {evf_h:.3f} >= {stop_thr}")
                    stop_now = True

                del opp_eval_net

            append_metrics_csv(csv_path, {
                "iteration": i,
                "global_iteration": global_iter,
                "mean_reward": round(mean_reward, 6),
                "win_rate": round(win_rate, 6) if win_rate is not None else "",
                "num_episodes_lifetime": total_env_steps,
                "iter_seconds": round(iter_dt, 4),
                "policy_loss": round(losses["policy_loss"], 6),
                "value_loss": round(losses["value_loss"], 6),
                "entropy": round(losses["entropy"], 6),
                "clip_fraction": round(losses["clip_fraction"], 6),
                "approx_kl": round(losses["approx_kl"], 6),
                "explained_variance": round(losses["explained_variance"], 6),
                "grad_norm": round(losses["grad_norm"], 6),
                "learning_rate": lr,
                "entropy_coeff": ent_coeff,
                "shaping_mult": round(shaping_mult, 4),
                "steps_per_sec": round(steps_sec, 1),
                "total_env_steps": total_env_steps,
                "snapshot_count": opponent_pool.snapshot_count,
                "forfeit_rate": round(forfeit_rate, 6),
                "forfeit_injected_rate": round(ff_inj_rate, 6),
                "blocked_rate": round(blk_rate, 6),
                "persist_rate": round(persist_rate, 6),
                "partial_block_rate": round(partial_block_rate, 6),
                "waste_rate": round(waste_rate, 6),
                "first_win_rate": _csv_float(first_wr),
                "second_win_rate": _csv_float(second_wr),
                "first_ep_len": _csv_float(first_ep_len),
                "second_ep_len": _csv_float(second_ep_len),
                "mean_episode_length": round(mean_episode_len, 4)
                if math.isfinite(mean_episode_len)
                else "",
                "eval_win_rate": _csv_float(ev_w),
                "eval_draw_rate": _csv_float(ev_d),
                "eval_vs_random_win": _csv_float(ev_rand),
                "eval_vs_heuristic_win": _csv_float(ev_heur),
                "eval_vs_line_rusher": _csv_float(ev_lr),
                "eval_vs_row_rusher": _csv_float(ev_rr),
                "eval_vs_col_rusher": _csv_float(ev_cr),
                "eval_vs_greedy_tactical": _csv_float(ev_gt),
                "eval_vs_lookahead_scripted": _csv_float(ev_la),
                "eval_block_rate": _csv_float(ev_br),
                "eval_heur_first": _csv_float(ev_heur_first),
                "eval_heur_second": _csv_float(ev_heur_second),
                "eval_line_first": _csv_float(ev_line_first),
                "eval_line_second": _csv_float(ev_line_second),
                "eval_row_first": _csv_float(ev_row_first),
                "eval_row_second": _csv_float(ev_row_second),
                "eval_col_first": _csv_float(ev_col_first),
                "eval_col_second": _csv_float(ev_col_second),
                "eval_random_open_heur": _csv_float(ev_ro_heur),
                "eval_random_open_col": _csv_float(ev_ro_col),
                "eval_random_open_row": _csv_float(ev_ro_row),
                "eval_recovery_heur": _csv_float(ev_recov_heur),
                "eval_gt_first": _csv_float(ev_gt_first),
                "eval_gt_second": _csv_float(ev_gt_second),
                "eval_la_first": _csv_float(ev_la_first),
                "eval_la_second": _csv_float(ev_la_second),
                "eval_pd_first": _csv_float(ev_pd_first),
                "eval_pd_second": _csv_float(ev_pd_second),
            })

            if stop_now:
                final_path = _save_checkpoint(
                    checkpoint_dir, main_net, optimizer, global_iter, total_env_steps, opponent_pool
                )
                try:
                    plot_metrics(csv_path, plots_dir, results_root)
                except Exception as ex:
                    log_print(log_f, f"  [plot] warning: {ex}")
                log_print(log_f, f"\nEarly stop at iter {i}. Checkpoint: {final_path}")
                break

            if i % plot_every == 0 or i == remaining:
                try:
                    plot_metrics(csv_path, plots_dir, results_root)
                except Exception as ex:
                    log_print(log_f, f"  [plot] warning: {ex}")

            if i % ckpt_every == 0:
                path = _save_checkpoint(
                    checkpoint_dir, main_net, optimizer, global_iter, total_env_steps, opponent_pool
                )
                log_print(log_f, f"  Checkpoint: {path}")

        else:
            final_path = _save_checkpoint(
                checkpoint_dir, main_net, optimizer, num_iterations,
                total_env_steps, opponent_pool,
            )
            total_time = time.time() - t0
            try:
                plot_metrics(csv_path, plots_dir, results_root)
            except Exception as ex:
                log_print(log_f, f"[plot] warning: {ex}")
            log_print(log_f, f"\nTraining complete in {total_time:.0f}s. Checkpoint: {final_path}")

    except KeyboardInterrupt:
        log_print(log_f, "\nKeyboardInterrupt — saving checkpoint...")
        try:
            final_path = _save_checkpoint(
                checkpoint_dir, main_net, optimizer, global_iter,
                total_env_steps, opponent_pool,
            )
            log_print(log_f, f"Saved: {final_path}")
        except Exception as ex:
            log_print(log_f, f"Save failed: {ex}")
        try:
            plot_metrics(csv_path, plots_dir, results_root)
        except Exception:
            pass
    finally:
        try:
            log_f.flush()
            log_f.close()
        except Exception:
            pass

    return final_path


def _save_checkpoint(
    checkpoint_dir: str,
    main_net: SuperTTTNet,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    total_env_steps: int,
    opponent_pool: OpponentPool,
) -> str:
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"checkpoint_{iteration:06d}.pt")
    torch.save({
        "main_net": main_net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
        "total_env_steps": total_env_steps,
        "opponent_pool": {
            "slots": opponent_pool._slots,
            "snapshot_rr": opponent_pool._snapshot_rr,
        },
    }, path)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Super TTT agent (TorchRL)")
    parser.add_argument("--iterations", type=int, default=TRAINING_CONFIG["num_iterations"])
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--restore", type=str, default=None, metavar="PATH")
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--plot-every", type=int, default=5)
    parser.add_argument("--checkpoint-freq", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-early-stop", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    args = parser.parse_args()
    train(
        args.iterations,
        args.checkpoint_dir,
        restore_path=args.restore,
        results_root=args.results_dir,
        plot_every=max(1, args.plot_every),
        checkpoint_freq=args.checkpoint_freq,
        seed=args.seed,
        early_stop=not args.no_early_stop,
        augment=args.augment,
        fast=args.fast,
        eval_interval_override=args.eval_interval,
        eval_num_episodes_override=args.eval_episodes,
    )
