"""
compare_models.py — Cross-version model battle and comparison.

Usage:
    python src/compare_models.py models/v1.0 models/v2.0 --games 100
    python src/compare_models.py models/v1.0 --vs heuristic --games 200
    python src/compare_models.py models/v1.0 --vs random --games 200
    python src/compare_models.py models/v1.0 models/v2.0 --games 100 --output-dir models/comparisons/

Supports old-architecture (3ch/144-action) and new-architecture (7ch/96-action) models.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

from board import (
    ROWS, COLS, NUM_VALID, VALID_MASK, VALID_POSITIONS,
    POS_TO_INDEX, INDEX_TO_POS, empty_board, compact_to_rc,
    COMPACT_TO_FLAT, FLAT_TO_COMPACT, get_valid_mask_flat,
)
from rules import check_win, is_draw, threat_heatmap
from stochastic import resolve_placement, forfeit_probability_map
from env import heuristic_action, random_legal_action, OBS_CHANNELS
from network import SuperTTTNet, get_device


ARCH_OLD = "old"
ARCH_NEW = "new"


def detect_architecture(version_dir: str) -> str:
    info_path = os.path.join(version_dir, "version_info.json")
    if os.path.isfile(info_path):
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)
        arch = info.get("architecture", {})
        if isinstance(arch, dict):
            if arch.get("obs_channels", 7) == 3:
                return ARCH_OLD
            return ARCH_NEW
        if arch == "old":
            return ARCH_OLD
        return ARCH_NEW

    reg_path = os.path.join(os.path.dirname(version_dir), "registry.json")
    if os.path.isfile(reg_path):
        with open(reg_path, encoding="utf-8") as f:
            reg = json.load(f)
        ver_name = os.path.basename(version_dir).lstrip("v")
        for v in reg.get("versions", []):
            if v.get("version") == ver_name and v.get("architecture") == "old":
                return ARCH_OLD

    return ARCH_NEW


def load_model(version_dir: str, device: torch.device) -> tuple[SuperTTTNet, str]:
    arch = detect_architecture(version_dir)
    weights_path = os.path.join(version_dir, "model_weights.pt")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"No model_weights.pt in {version_dir}")

    if arch == ARCH_OLD:
        net = SuperTTTNet(
            in_channels=3, num_filters=128, num_res_blocks=6,
            num_actions=144, value_fc_hidden=512,
        )
    else:
        net = SuperTTTNet(
            in_channels=7, num_filters=192, num_res_blocks=8,
            num_actions=96, value_fc_hidden=768,
        )

    sd = torch.load(weights_path, map_location=device, weights_only=True)
    net.load_state_dict(sd, strict=False)
    net.to(device)
    net.eval()
    return net, arch


def build_obs_old(board: np.ndarray, player: int) -> np.ndarray:
    pid = player
    opp = 3 - pid
    obs = np.zeros((3, ROWS, COLS), dtype=np.float32)
    obs[0] = (board == pid).astype(np.float32)
    obs[1] = (board == opp).astype(np.float32)
    valid = np.zeros((ROWS, COLS), dtype=np.float32)
    for r in range(ROWS):
        for c in range(COLS):
            if VALID_MASK[r, c] and board[r, c] == 0:
                valid[r, c] = 1.0
    obs[2] = valid
    return obs


def build_obs_new(board: np.ndarray, player: int,
                  last_move_plane: np.ndarray) -> np.ndarray:
    pid = player
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


def get_action_mask_compact(board: np.ndarray) -> np.ndarray:
    mask = np.zeros(NUM_VALID, dtype=np.float32)
    for idx, (r, c) in enumerate(VALID_POSITIONS):
        if board[r, c] == 0:
            mask[idx] = 1.0
    return mask


def get_action_mask_flat(board: np.ndarray) -> np.ndarray:
    mask = np.zeros(ROWS * COLS, dtype=np.float32)
    for r in range(ROWS):
        for c in range(COLS):
            if VALID_MASK[r, c] and board[r, c] == 0:
                mask[r * COLS + c] = 1.0
    return mask


def choose_action_nn(net: SuperTTTNet, arch: str, board: np.ndarray,
                     player: int, last_move_plane: np.ndarray,
                     device: torch.device) -> int:
    if arch == ARCH_OLD:
        obs = build_obs_old(board, player)
        mask = get_action_mask_flat(board)
    else:
        obs = build_obs_new(board, player, last_move_plane)
        mask = get_action_mask_compact(board)

    with torch.no_grad():
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
        logits, _ = net(obs_t)
        mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)
        inf_mask = torch.clamp(torch.log(mask_t + 1e-10), min=-1e10)
        masked = logits + inf_mask
        action_idx = int(torch.argmax(masked, dim=-1).item())

    if arch == ARCH_OLD:
        r, c = divmod(action_idx, COLS)
        if (r, c) in POS_TO_INDEX:
            return POS_TO_INDEX[(r, c)]
        return random_legal_action(board, np.random.default_rng())
    else:
        return action_idx


def play_one_game(
    agent_a, agent_b,
    rng: np.random.Generator,
    device: torch.device,
    max_steps: int = 200,
) -> dict:
    board = empty_board()
    last_move_plane = np.zeros((ROWS, COLS), dtype=np.float32)
    agents = [agent_a, agent_b]
    pids = [1, 2]
    step = 0

    while step < max_steps:
        for side in range(2):
            if step >= max_steps:
                break
            pid = pids[side]
            agent = agents[side]

            action = agent["choose"](board, pid, last_move_plane, rng)
            r, c = compact_to_rc(action)

            placed = False
            if not VALID_MASK[r, c] or board[r, c] != 0:
                last_move_plane = np.zeros((ROWS, COLS), dtype=np.float32)
            else:
                result = resolve_placement(board, r, c, rng)
                if result is not None:
                    pr, pc = result
                    board[pr, pc] = pid
                    last_move_plane = np.zeros((ROWS, COLS), dtype=np.float32)
                    last_move_plane[pr, pc] = 1.0
                    placed = True
                else:
                    last_move_plane = np.zeros((ROWS, COLS), dtype=np.float32)

            step += 1

            if placed:
                win = check_win(board, pid)
                if win is not None:
                    return {"winner": side, "pid": pid, "steps": step, "draw": False}
            if is_draw(board):
                return {"winner": None, "pid": None, "steps": step, "draw": True}

    return {"winner": None, "pid": None, "steps": step, "draw": True}


def make_nn_agent(net, arch, device):
    def choose(board, player, last_move_plane, rng):
        return choose_action_nn(net, arch, board, player, last_move_plane, device)
    return {"name": f"NN({arch})", "choose": choose}


def make_heuristic_agent():
    def choose(board, player, last_move_plane, rng):
        return heuristic_action(board, player, rng)
    return {"name": "Heuristic", "choose": choose}


def make_random_agent():
    def choose(board, player, last_move_plane, rng):
        return random_legal_action(board, rng)
    return {"name": "Random", "choose": choose}


def run_comparison(agent_a, agent_b, num_games: int = 100,
                   seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    device = get_device()

    a_wins = 0
    b_wins = 0
    draws = 0
    a_wins_first = 0
    a_wins_second = 0
    b_wins_first = 0
    b_wins_second = 0

    t0 = time.time()
    for i in range(num_games):
        if i % 2 == 0:
            first_agent, second_agent = agent_a, agent_b
            mapping = {0: "a", 1: "b"}
        else:
            first_agent, second_agent = agent_b, agent_a
            mapping = {0: "b", 1: "a"}

        result = play_one_game(first_agent, second_agent, rng, device)

        if result["draw"]:
            draws += 1
        elif mapping[result["winner"]] == "a":
            a_wins += 1
            if i % 2 == 0:
                a_wins_first += 1
            else:
                a_wins_second += 1
        else:
            b_wins += 1
            if i % 2 == 0:
                b_wins_first += 1
            else:
                b_wins_second += 1

        if (i + 1) % 20 == 0 or i == num_games - 1:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{num_games}] A={a_wins} B={b_wins} D={draws} "
                  f"({elapsed:.1f}s)")

    elapsed = time.time() - t0
    return {
        "agent_a": agent_a["name"],
        "agent_b": agent_b["name"],
        "num_games": num_games,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "draws": draws,
        "a_win_rate": a_wins / num_games,
        "b_win_rate": b_wins / num_games,
        "draw_rate": draws / num_games,
        "a_wins_as_first": a_wins_first,
        "a_wins_as_second": a_wins_second,
        "b_wins_as_first": b_wins_first,
        "b_wins_as_second": b_wins_second,
        "elapsed_seconds": round(elapsed, 2),
        "seed": seed,
    }


def print_results(results: dict) -> None:
    print("\n" + "=" * 60)
    print(f"  {results['agent_a']}  vs  {results['agent_b']}")
    print("=" * 60)
    print(f"  Games:    {results['num_games']}")
    print(f"  A wins:   {results['a_wins']}  ({results['a_win_rate']:.1%})")
    print(f"  B wins:   {results['b_wins']}  ({results['b_win_rate']:.1%})")
    print(f"  Draws:    {results['draws']}  ({results['draw_rate']:.1%})")
    print(f"  ----")
    print(f"  A wins as first:  {results['a_wins_as_first']}")
    print(f"  A wins as second: {results['a_wins_as_second']}")
    print(f"  B wins as first:  {results['b_wins_as_first']}")
    print(f"  B wins as second: {results['b_wins_as_second']}")
    print(f"  Time:     {results['elapsed_seconds']:.1f}s")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Cross-version model comparison")
    parser.add_argument("model_a", help="Path to first model version directory (e.g. models/v1.0)")
    parser.add_argument("model_b", nargs="?", default=None,
                        help="Path to second model version directory")
    parser.add_argument("--vs", choices=["random", "heuristic"],
                        help="Compare model_a against a scripted opponent")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None,
                        help="Save results JSON to this directory")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    net_a, arch_a = load_model(args.model_a, device)
    agent_a = make_nn_agent(net_a, arch_a, device)
    agent_a["name"] = os.path.basename(args.model_a)
    print(f"Loaded A: {args.model_a} (arch={arch_a})")

    if args.model_b:
        net_b, arch_b = load_model(args.model_b, device)
        agent_b = make_nn_agent(net_b, arch_b, device)
        agent_b["name"] = os.path.basename(args.model_b)
        print(f"Loaded B: {args.model_b} (arch={arch_b})")
    elif args.vs == "heuristic":
        agent_b = make_heuristic_agent()
    elif args.vs == "random":
        agent_b = make_random_agent()
    else:
        parser.error("Provide model_b or --vs {random,heuristic}")
        return

    print(f"\nRunning {args.games} games: {agent_a['name']} vs {agent_b['name']}")
    results = run_comparison(agent_a, agent_b, args.games, args.seed)
    print_results(results)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(
            args.output_dir,
            f"compare_{agent_a['name']}_vs_{agent_b['name']}_{args.games}g.json"
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
