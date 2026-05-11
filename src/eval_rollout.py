"""
eval_rollout.py — Offline eval: greedy main vs snapshot, random-legal, or heuristic.

Pure PyTorch: uses SuperTTTNet directly (no RLlib RLModule).
"""

from __future__ import annotations

import numpy as np
import torch

from board import NUM_VALID, ROWS, COLS, VALID_MASK, VALID_POSITIONS, compact_to_rc
from env import (
    MAX_STEPS, PLAYER_MAP, OBS_CHANNELS, SuperTicTacToeEnv, heuristic_action,
    line_rusher_action, row_rusher_action, col_rusher_action,
    greedy_tactical_action, lookahead_scripted_action,
)
from rules import check_win, count_threats, threat_heatmap
from stochastic import placement_distribution, forfeit_probability_map

EVAL_MAX_AGENT_STEPS = max(600, MAX_STEPS * 4)


def _episode_outcome_from_infos(infos: dict, main_agent: str = "player_0") -> str | None:
    opp_agent = "player_1" if main_agent == "player_0" else "player_0"
    i0 = infos.get("player_0") or {}
    i1 = infos.get("player_1") or {}
    if i0.get("winner") == main_agent or i1.get("winner") == main_agent:
        return "main_win"
    if i0.get("winner") == opp_agent or i1.get("winner") == opp_agent:
        return "opp_win"
    if i0.get("draw") or i1.get("draw"):
        return "draw"
    return None


@torch.no_grad()
def greedy_action(net: torch.nn.Module, obs: dict, device: torch.device) -> int:
    o = obs["observations"]
    m = obs["action_mask"]
    if isinstance(o, np.ndarray):
        o_t = torch.from_numpy(o.astype(np.float32)).unsqueeze(0).to(device)
    else:
        o_t = o.unsqueeze(0).to(device) if o.dim() == 3 else o.to(device)
    if isinstance(m, np.ndarray):
        m_t = torch.from_numpy(m.astype(np.float32)).unsqueeze(0).to(device)
    else:
        m_t = m.unsqueeze(0).to(device) if m.dim() == 1 else m.to(device)

    logits, _ = net(o_t)
    inf_mask = torch.clamp(torch.log(m_t + 1e-10), min=-1e10)
    masked = logits + inf_mask
    return int(torch.argmax(masked, dim=-1).item())


def _build_obs_for_board(
    board: np.ndarray, pid: int, last_move_plane: np.ndarray,
) -> np.ndarray:
    opp = 3 - pid
    obs = np.zeros((OBS_CHANNELS, ROWS, COLS), dtype=np.float32)
    obs[0] = (board == pid).astype(np.float32)
    obs[1] = (board == opp).astype(np.float32)
    obs[2] = VALID_MASK.astype(np.float32)
    obs[3] = last_move_plane
    my_hmap = threat_heatmap(board, pid)
    opp_hmap = threat_heatmap(board, opp)
    max_val = max(my_hmap.max(), opp_hmap.max(), 1.0)
    obs[4] = my_hmap / max_val
    obs[5] = opp_hmap / max_val
    obs[6] = forfeit_probability_map(board)
    return obs


@torch.no_grad()
def lookahead_action(
    net: torch.nn.Module, obs: dict,
    board: np.ndarray, player: int,
    device: torch.device,
    top_k: int = 10,
) -> int:
    o = obs["observations"]
    m = obs["action_mask"]
    if isinstance(o, np.ndarray):
        o_t = torch.from_numpy(o.astype(np.float32)).unsqueeze(0).to(device)
    else:
        o_t = o.unsqueeze(0).to(device) if o.dim() == 3 else o.to(device)
    if isinstance(m, np.ndarray):
        m_t = torch.from_numpy(m.astype(np.float32)).unsqueeze(0).to(device)
    else:
        m_t = m.unsqueeze(0).to(device) if m.dim() == 1 else m.to(device)

    logits, current_v = net(o_t)
    inf_mask = torch.clamp(torch.log(m_t + 1e-10), min=-1e10)
    masked = (logits + inf_mask).squeeze(0)

    legal_indices = np.flatnonzero(m if isinstance(m, np.ndarray) else m_t.cpu().numpy().squeeze())
    if len(legal_indices) == 0:
        return int(torch.argmax(masked).item())
    if len(legal_indices) == 1:
        return int(legal_indices[0])

    probs = torch.softmax(masked, dim=-1)
    k = min(top_k, len(legal_indices))
    _, top_indices = torch.topk(probs, k)
    candidate_actions = top_indices.cpu().numpy()

    opp_pid = 3 - player

    TERMINAL_WIN = 1.0
    TERMINAL_FORFEIT = -0.05
    successor_entries: list[list[tuple[np.ndarray | None, float, float | None]]] = []

    for ca in candidate_actions:
        r, c = compact_to_rc(int(ca))
        dist = placement_distribution(board, r, c)
        entries: list[tuple[np.ndarray | None, float, float | None]] = []
        for outcome, p in dist.items():
            if outcome is None:
                entries.append((None, p, TERMINAL_FORFEIT))
            else:
                pr, pc = outcome
                new_board = board.copy()
                new_board[pr, pc] = player
                if check_win(new_board, player) is not None:
                    entries.append((None, p, TERMINAL_WIN))
                else:
                    lmp = np.zeros((ROWS, COLS), dtype=np.float32)
                    lmp[pr, pc] = 1.0
                    new_obs = _build_obs_for_board(new_board, opp_pid, lmp)
                    entries.append((new_obs, p, None))
        successor_entries.append(entries)

    all_obs = []
    for entries in successor_entries:
        for ob, _, terminal_v in entries:
            if ob is not None and terminal_v is None:
                all_obs.append(ob)

    if all_obs:
        batch = torch.from_numpy(np.stack(all_obs)).to(device)
        _, values = net(batch)
        values = values.cpu().numpy()
    else:
        values = np.array([])

    best_action = int(candidate_actions[0])
    best_ev = -999.0
    val_idx = 0

    for i, ca in enumerate(candidate_actions):
        ev = 0.0
        for ob, p, terminal_v in successor_entries[i]:
            if terminal_v is not None:
                ev += p * terminal_v
            else:
                ev += p * (-values[val_idx])
                val_idx += 1
        if ev > best_ev:
            best_ev = ev
            best_action = int(ca)

    return best_action


def _random_legal_action(obs: dict, rng: np.random.Generator) -> int:
    mask = obs["action_mask"]
    legal = np.flatnonzero(mask)
    if len(legal) == 0:
        return rng.integers(0, NUM_VALID)
    return int(rng.choice(legal))


def _resolve_outcome(env, recorded, main_agent: str = "player_0"):
    if recorded is not None:
        return recorded
    opp_agent = "player_1" if main_agent == "player_0" else "player_0"
    i0 = env.infos.get("player_0", {})
    i1 = env.infos.get("player_1", {})
    if i0.get("winner") == main_agent or i1.get("winner") == main_agent:
        return "main_win"
    if i0.get("winner") == opp_agent or i1.get("winner") == opp_agent:
        return "opp_win"
    if i0.get("draw") or i1.get("draw"):
        return "draw"
    t0 = env.terminations.get("player_0") or env.truncations.get("player_0")
    t1 = env.terminations.get("player_1") or env.truncations.get("player_1")
    if t0 or t1:
        return "draw"
    return "incomplete"


def play_one_episode(main_net, opp_net, *, seed: int, device: torch.device,
                     main_agent: str = "player_0") -> str:
    opp_agent = "player_1" if main_agent == "player_0" else "player_0"
    env = SuperTicTacToeEnv(seed=seed)
    env.reset(seed=seed)
    recorded = None
    for agent in env.agent_iter(max_iter=EVAL_MAX_AGENT_STEPS):
        if env.terminations.get(agent) or env.truncations.get(agent):
            env.step(None)
        else:
            obs = env.observe(agent)
            if agent == main_agent:
                a = greedy_action(main_net, obs, device)
            else:
                a = greedy_action(opp_net, obs, device)
            env.step(a)
        o = _episode_outcome_from_infos(env.infos, main_agent)
        if o is not None:
            recorded = o
    return _resolve_outcome(env, recorded, main_agent)


def play_one_episode_vs_random(main_net, *, seed: int, device: torch.device,
                               main_agent: str = "player_0") -> str:
    env = SuperTicTacToeEnv(seed=seed)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed + 777)
    recorded = None
    for agent in env.agent_iter(max_iter=EVAL_MAX_AGENT_STEPS):
        if env.terminations.get(agent) or env.truncations.get(agent):
            env.step(None)
        else:
            obs = env.observe(agent)
            if agent == main_agent:
                a = greedy_action(main_net, obs, device)
            else:
                a = _random_legal_action(obs, rng)
            env.step(a)
        o = _episode_outcome_from_infos(env.infos, main_agent)
        if o is not None:
            recorded = o
    return _resolve_outcome(env, recorded, main_agent)


def play_one_episode_vs_heuristic(main_net, *, seed: int, device: torch.device,
                                  main_agent: str = "player_0") -> str:
    env = SuperTicTacToeEnv(seed=seed)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed + 888)
    recorded = None
    for agent in env.agent_iter(max_iter=EVAL_MAX_AGENT_STEPS):
        if env.terminations.get(agent) or env.truncations.get(agent):
            env.step(None)
        else:
            obs = env.observe(agent)
            if agent == main_agent:
                a = greedy_action(main_net, obs, device)
            else:
                pid = PLAYER_MAP[agent]
                a = heuristic_action(env.board, pid, rng)
            env.step(a)
        o = _episode_outcome_from_infos(env.infos, main_agent)
        if o is not None:
            recorded = o
    return _resolve_outcome(env, recorded, main_agent)


def evaluate_main_vs_snapshot(
    main_net,
    opp_net,
    *,
    num_episodes: int,
    base_seed: int,
    device: torch.device,
) -> dict[str, float]:
    nan3 = {"win_rate": float("nan"), "draw_rate": float("nan"),
            "loss_rate": float("nan"), "incomplete_episodes": float("nan")}
    if num_episodes < 1:
        return nan3
    if main_net is None or opp_net is None:
        return nan3
    wins = losses = draws = incomplete = 0
    for i in range(num_episodes):
        ma = "player_0" if i % 2 == 0 else "player_1"
        out = play_one_episode(main_net, opp_net, seed=base_seed + 10_000 + i,
                               device=device, main_agent=ma)
        if out == "main_win":
            wins += 1
        elif out == "opp_win":
            losses += 1
        elif out == "draw":
            draws += 1
        else:
            incomplete += 1
    n = float(num_episodes)
    return {"win_rate": wins / n, "draw_rate": (draws + incomplete) / n,
            "loss_rate": losses / n, "incomplete_episodes": float(incomplete)}


def evaluate_main_vs_random(
    main_net,
    *,
    num_episodes: int,
    base_seed: int,
    device: torch.device,
) -> dict[str, float]:
    nan3 = {"win_rate": float("nan"), "draw_rate": float("nan"), "loss_rate": float("nan")}
    if num_episodes < 1:
        return nan3
    if main_net is None:
        return nan3
    wins = losses = draws = 0
    for i in range(num_episodes):
        ma = "player_0" if i % 2 == 0 else "player_1"
        out = play_one_episode_vs_random(main_net, seed=base_seed + 20_000 + i,
                                         device=device, main_agent=ma)
        if out == "main_win":
            wins += 1
        elif out == "opp_win":
            losses += 1
        else:
            draws += 1
    n = float(num_episodes)
    return {"win_rate": wins / n, "draw_rate": draws / n, "loss_rate": losses / n}


def evaluate_main_vs_heuristic(
    main_net,
    *,
    num_episodes: int,
    base_seed: int,
    device: torch.device,
) -> dict[str, float]:
    nan3 = {"win_rate": float("nan"), "draw_rate": float("nan"), "loss_rate": float("nan")}
    if num_episodes < 1:
        return nan3
    if main_net is None:
        return nan3
    wins = losses = draws = 0
    for i in range(num_episodes):
        ma = "player_0" if i % 2 == 0 else "player_1"
        out = play_one_episode_vs_heuristic(main_net, seed=base_seed + 30_000 + i,
                                            device=device, main_agent=ma)
        if out == "main_win":
            wins += 1
        elif out == "opp_win":
            losses += 1
        else:
            draws += 1
    n = float(num_episodes)
    return {"win_rate": wins / n, "draw_rate": draws / n, "loss_rate": losses / n}


def _play_one_episode_vs_scripted(
    main_net, scripted_fn, *, seed: int, device: torch.device,
    main_agent: str = "player_0",
) -> str:
    env = SuperTicTacToeEnv(seed=seed)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed + 555)
    recorded = None
    for agent in env.agent_iter(max_iter=EVAL_MAX_AGENT_STEPS):
        if env.terminations.get(agent) or env.truncations.get(agent):
            env.step(None)
        else:
            obs = env.observe(agent)
            if agent == main_agent:
                a = greedy_action(main_net, obs, device)
            else:
                pid = PLAYER_MAP[agent]
                a = scripted_fn(env.board, pid, rng)
            env.step(a)
        o = _episode_outcome_from_infos(env.infos, main_agent)
        if o is not None:
            recorded = o
    return _resolve_outcome(env, recorded, main_agent)


def evaluate_main_vs_scripted(
    main_net,
    scripted_fn,
    *,
    num_episodes: int,
    base_seed: int,
    device: torch.device,
    main_agent_position: str = "both",
) -> dict[str, float]:
    """``main_agent_position``: "first" (always player_0), "second" (always player_1), or "both" (alternating)."""
    nan3 = {"win_rate": float("nan"), "draw_rate": float("nan"), "loss_rate": float("nan")}
    if num_episodes < 1 or main_net is None:
        return nan3
    wins = losses = draws = 0
    for i in range(num_episodes):
        if main_agent_position == "first":
            ma = "player_0"
        elif main_agent_position == "second":
            ma = "player_1"
        else:
            ma = "player_0" if i % 2 == 0 else "player_1"
        out = _play_one_episode_vs_scripted(
            main_net, scripted_fn, seed=base_seed + 40_000 + i, device=device,
            main_agent=ma,
        )
        if out == "main_win":
            wins += 1
        elif out == "opp_win":
            losses += 1
        else:
            draws += 1
    n = float(num_episodes)
    return {"win_rate": wins / n, "draw_rate": draws / n, "loss_rate": losses / n}


def evaluate_main_vs_heuristic_by_position(
    main_net, *, num_episodes: int, base_seed: int, device: torch.device,
    main_agent_position: str = "both",
) -> dict[str, float]:
    nan3 = {"win_rate": float("nan"), "draw_rate": float("nan"), "loss_rate": float("nan")}
    if num_episodes < 1 or main_net is None:
        return nan3
    wins = losses = draws = 0
    for i in range(num_episodes):
        if main_agent_position == "first":
            ma = "player_0"
        elif main_agent_position == "second":
            ma = "player_1"
        else:
            ma = "player_0" if i % 2 == 0 else "player_1"
        out = play_one_episode_vs_heuristic(
            main_net, seed=base_seed + 30_000 + i, device=device, main_agent=ma,
        )
        if out == "main_win":
            wins += 1
        elif out == "opp_win":
            losses += 1
        else:
            draws += 1
    n = float(num_episodes)
    return {"win_rate": wins / n, "draw_rate": draws / n, "loss_rate": losses / n}


def evaluate_random_opening(
    main_net, scripted_fn, *,
    num_episodes: int, base_seed: int, device: torch.device,
    opening_steps: int = 4,
) -> dict[str, float]:
    """Both agents play random legal for first ``opening_steps`` turns, then policy/scripted take over.

    Simulates 'any-position' / diverse starting state. Alternates first/second across episodes.
    """
    nan3 = {"win_rate": float("nan"), "draw_rate": float("nan"), "loss_rate": float("nan")}
    if num_episodes < 1 or main_net is None:
        return nan3
    wins = losses = draws = 0
    for i in range(num_episodes):
        seed = base_seed + 60_000 + i
        ma = "player_0" if i % 2 == 0 else "player_1"
        env = SuperTicTacToeEnv(
            seed=seed,
            random_opening_prob=1.0,
            random_opening_steps=opening_steps,
        )
        env.reset(seed=seed)
        rng = np.random.default_rng(seed + 555)
        recorded = None
        for agent in env.agent_iter(max_iter=EVAL_MAX_AGENT_STEPS):
            if env.terminations.get(agent) or env.truncations.get(agent):
                env.step(None)
            else:
                obs = env.observe(agent)
                if agent == ma:
                    a = greedy_action(main_net, obs, device)
                else:
                    pid = PLAYER_MAP[agent]
                    a = scripted_fn(env.board, pid, rng)
                env.step(a)
            o = _episode_outcome_from_infos(env.infos, ma)
            if o is not None:
                recorded = o
        out = _resolve_outcome(env, recorded, ma)
        if out == "main_win":
            wins += 1
        elif out == "opp_win":
            losses += 1
        else:
            draws += 1
    n = float(num_episodes)
    return {"win_rate": wins / n, "draw_rate": draws / n, "loss_rate": losses / n}


def evaluate_forfeit_recovery(
    main_net, scripted_fn, *,
    num_episodes: int, base_seed: int, device: torch.device,
    forfeit_step_range: tuple[int, int] = (1, 6),
) -> dict[str, float]:
    """Force main agent to FORFEIT once at a random step within ``forfeit_step_range``, then
    measure win rate vs scripted opponent. Directly tests 'does agent give up after FORFEIT'."""
    nan3 = {"win_rate": float("nan"), "draw_rate": float("nan"), "loss_rate": float("nan")}
    if num_episodes < 1 or main_net is None:
        return nan3
    wins = losses = draws = 0
    for i in range(num_episodes):
        seed = base_seed + 70_000 + i
        seed_rng = np.random.default_rng(seed)
        lo, hi = forfeit_step_range
        forced_step = int(seed_rng.integers(lo, max(lo + 1, hi)))
        ma = "player_0" if i % 2 == 0 else "player_1"
        env = SuperTicTacToeEnv(seed=seed)
        env.reset(seed=seed)
        rng = np.random.default_rng(seed + 333)
        recorded = None
        main_step_counter = 0
        for agent in env.agent_iter(max_iter=EVAL_MAX_AGENT_STEPS):
            if env.terminations.get(agent) or env.truncations.get(agent):
                env.step(None)
            else:
                obs = env.observe(agent)
                if agent == ma:
                    main_step_counter += 1
                    if main_step_counter == forced_step:
                        mask = obs["action_mask"]
                        illegal = np.flatnonzero(mask == 0)
                        if len(illegal) > 0:
                            a = int(illegal[0])
                        else:
                            a = 0
                    else:
                        a = greedy_action(main_net, obs, device)
                else:
                    pid = PLAYER_MAP[agent]
                    a = scripted_fn(env.board, pid, rng)
                env.step(a)
            o = _episode_outcome_from_infos(env.infos, ma)
            if o is not None:
                recorded = o
        out = _resolve_outcome(env, recorded, ma)
        if out == "main_win":
            wins += 1
        elif out == "opp_win":
            losses += 1
        else:
            draws += 1
    n = float(num_episodes)
    return {"win_rate": wins / n, "draw_rate": draws / n, "loss_rate": losses / n}


def measure_block_rate(
    main_net,
    *,
    num_episodes: int,
    base_seed: int,
    device: torch.device,
) -> float:
    if num_episodes < 1 or main_net is None:
        return float("nan")
    total_threats = 0
    total_blocked = 0
    for i in range(num_episodes):
        seed = base_seed + 50_000 + i
        main_agent = "player_0" if i % 2 == 0 else "player_1"
        opp_agent = "player_1" if main_agent == "player_0" else "player_0"
        env = SuperTicTacToeEnv(seed=seed)
        env.reset(seed=seed)
        rng = np.random.default_rng(seed + 666)
        for agent in env.agent_iter(max_iter=EVAL_MAX_AGENT_STEPS):
            if env.terminations.get(agent) or env.truncations.get(agent):
                env.step(None)
                continue
            obs = env.observe(agent)
            if agent == main_agent:
                opp_pid = PLAYER_MAP[opp_agent]
                threats_before = count_threats(env.board, opp_pid)
                a = greedy_action(main_net, obs, device)
                env.step(a)
                if threats_before > 0:
                    threats_after = count_threats(env.board, opp_pid)
                    total_threats += threats_before
                    total_blocked += max(0, threats_before - threats_after)
            else:
                pid = PLAYER_MAP[agent]
                a = line_rusher_action(env.board, pid, rng)
                env.step(a)
    if total_threats == 0:
        return float("nan")
    return total_blocked / total_threats
