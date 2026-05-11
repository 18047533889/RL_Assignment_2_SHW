"""
analyze_move_distribution.py — Empirical move-frequency over 96 compact actions.

Runs offline self-play games (greedy policy vs random / heuristic / line_rusher) and
counts how often the **main** agent selects each compact action index. Does **not**
read metrics.csv (which has no per-move data). Does **not** touch the training loop.

Usage (from assignment2new/):
  PYTHONPATH=src python src/analyze_move_distribution.py checkpoints/checkpoint_000700.pt \\
      --games 800 --opponent heuristic --out-dir results/move_dist_700

Uses random_opening_prob=0 so counts reflect the policy, not curriculum random moves.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

from board import COLS, NUM_VALID, ROWS, VALID_POSITIONS
from config import DEVICE, MODEL_CONFIG
from env import (
    SuperTicTacToeEnv,
    heuristic_action,
    line_rusher_action,
    PLAYER_MAP,
)
from eval_rollout import EVAL_MAX_AGENT_STEPS, greedy_action
from network import SuperTTTNet
from env import OBS_CHANNELS


def _build_net(device: torch.device) -> SuperTTTNet:
    return SuperTTTNet(
        in_channels=OBS_CHANNELS,
        num_filters=int(MODEL_CONFIG.get("num_filters", 192)),
        num_res_blocks=int(MODEL_CONFIG.get("num_res_blocks", 8)),
        num_actions=NUM_VALID,
        value_fc_hidden=int(MODEL_CONFIG.get("value_fc_hidden", 768)),
    ).to(device)


def _random_legal_action(obs: dict, rng: np.random.Generator) -> int:
    m = obs["action_mask"]
    legal = np.flatnonzero(m)
    if len(legal) == 0:
        return int(rng.integers(0, NUM_VALID))
    return int(rng.choice(legal))


def _opponent_action(
    opponent: str,
    obs: dict,
    board: np.ndarray,
    pid: int,
    rng: np.random.Generator,
) -> int:
    if opponent == "random":
        return _random_legal_action(obs, rng)
    if opponent == "heuristic":
        return heuristic_action(board, pid, rng)
    if opponent == "line_rusher":
        return line_rusher_action(board, pid, rng)
    raise ValueError(f"unknown opponent: {opponent}")


def collect_main_moves_one_episode(
    net: torch.nn.Module,
    *,
    seed: int,
    device: torch.device,
    main_agent: str,
    opponent: str,
) -> list[int]:
    """Return list of compact actions **chosen for main** (greedy), one per main turn."""
    env = SuperTicTacToeEnv(
        seed=seed,
        random_opening_prob=0.0,
        random_opening_steps=0,
    )
    env.reset(seed=seed)
    rng = np.random.default_rng(seed + 4242)
    moves: list[int] = []

    for agent in env.agent_iter(max_iter=EVAL_MAX_AGENT_STEPS):
        if env.terminations.get(agent) or env.truncations.get(agent):
            env.step(None)
        else:
            obs = env.observe(agent)
            if agent == main_agent:
                a = greedy_action(net, obs, device)
                moves.append(int(a))
                env.step(a)
            else:
                pid = PLAYER_MAP[agent]
                a = _opponent_action(opponent, obs, env.board, pid, rng)
                env.step(a)
    return moves


def run_distribution(
    checkpoint: str,
    *,
    num_games: int,
    base_seed: int,
    device: torch.device,
    opponent: str,
) -> np.ndarray:
    net = _build_net(device)
    ckpt = torch.load(
        os.path.abspath(os.path.expanduser(checkpoint)),
        map_location="cpu",
        weights_only=False,
    )
    sd = ckpt["main_net"] if isinstance(ckpt, dict) and "main_net" in ckpt else ckpt
    net.load_state_dict(sd, strict=False)
    net.eval()

    counts = np.zeros(NUM_VALID, dtype=np.int64)
    for i in range(num_games):
        ma = "player_0" if i % 2 == 0 else "player_1"
        seed = base_seed + 10_000 + i
        for a in collect_main_moves_one_episode(
            net, seed=seed, device=device, main_agent=ma, opponent=opponent,
        ):
            if 0 <= a < NUM_VALID:
                counts[a] += 1
    return counts


def write_csv(counts: np.ndarray, path: str) -> None:
    total = int(counts.sum())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["compact_idx", "row", "col", "count", "freq_if_total_moves"],
        )
        for idx in range(NUM_VALID):
            r, c = VALID_POSITIONS[idx]
            freq = counts[idx] / total if total > 0 else 0.0
            w.writerow([idx, r, c, int(counts[idx]), f"{freq:.8f}"])


def write_heatmap_png(counts: np.ndarray, path: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise RuntimeError("matplotlib required for --plot") from e

    grid = np.full((ROWS, COLS), np.nan, dtype=np.float64)
    for idx in range(NUM_VALID):
        r, c = VALID_POSITIONS[idx]
        grid[r, c] = float(counts[idx])

    total = float(counts.sum())
    if total > 0:
        grid = grid / total

    fig, ax = plt.subplots(figsize=(7, 6), dpi=120)
    im = ax.imshow(grid, cmap="YlOrRd", interpolation="nearest")
    ax.set_title("Main move frequency (per cell, normalized)")
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Histogram of greedy main moves over 96 compact actions (offline).",
    )
    p.add_argument(
        "checkpoint",
        help="Path to checkpoint .pt (uses main_net state_dict)",
    )
    p.add_argument("--games", type=int, default=500, help="Number of eval games")
    p.add_argument(
        "--opponent",
        choices=("random", "heuristic", "line_rusher"),
        default="heuristic",
        help="Scripted opponent type for the non-main seat",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-dir",
        default="results/move_distribution",
        help="Directory for move_counts.csv and optional heatmap",
    )
    p.add_argument("--no-plot", action="store_true", help="Skip PNG heatmap")
    args = p.parse_args()

    device = torch.device(DEVICE)
    os.makedirs(args.out_dir, exist_ok=True)

    counts = run_distribution(
        args.checkpoint,
        num_games=args.games,
        base_seed=args.seed,
        device=device,
        opponent=args.opponent,
    )

    csv_path = os.path.join(args.out_dir, "move_counts.csv")
    write_csv(counts, csv_path)
    print(f"Wrote {csv_path} (total main moves: {int(counts.sum())})")

    if not args.no_plot:
        png_path = os.path.join(args.out_dir, "move_frequency_heatmap.png")
        write_heatmap_png(counts, png_path)
        print(f"Wrote {png_path}")

    # Simple summary: top-5 cells + entropy of distribution
    total = float(counts.sum())
    if total > 0:
        pvec = counts.astype(np.float64) / total
        pnz = pvec[pvec > 0]
        ent = -np.sum(pnz * np.log(pnz + 1e-20))
        top5 = np.argsort(-counts)[:5]
        print("Top 5 compact indices (idx, r, c, count):")
        for idx in top5:
            r, c = VALID_POSITIONS[int(idx)]
            print(f"  {int(idx):3d}  ({r:2d},{c:2d})  {int(counts[idx])}")
        print(f"Entropy (nats) over occupied actions: {ent:.4f}  (max ~log(96)≈4.56 if uniform)")


if __name__ == "__main__":
    main()
