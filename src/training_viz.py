"""
training_viz.py — Logs, CSV metrics, and PNG plots for each training run.

**Quick read:** ``open_run_directory`` creates ``run_*``; each iter ``append_metrics_csv``;
``plot_metrics`` refreshes PNGs and mirrors to ``results/latest/plots`` for stable report links.

Writes under a run directory:
  training.log   (mirrored console lines)
  metrics.csv    (one row per iteration)
  plots/*.png    (updated periodically during training)
  run_meta.json  (seed, versions, optional git hash; written once at start)
"""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import subprocess
import sys
from typing import IO, Any

# Non-interactive backend for headless / CI
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Column order for metrics.csv (and plots). Training loop fills eval_* when eval runs.
METRIC_FIELDS = [
    "iteration",
    "global_iteration",
    "mean_reward",
    "win_rate",
    "num_episodes_lifetime",
    "iter_seconds",
    "policy_loss",
    "value_loss",
    "entropy",
    "clip_fraction",
    "approx_kl",
    "explained_variance",
    "grad_norm",
    "learning_rate",
    "entropy_coeff",
    "shaping_mult",
    "steps_per_sec",
    "total_env_steps",
    "snapshot_count",
    "forfeit_rate",
    "forfeit_injected_rate",
    "blocked_rate",
    "mean_episode_length",
    "eval_win_rate",
    "eval_draw_rate",
    "eval_vs_random_win",
    "eval_vs_heuristic_win",
    "eval_vs_line_rusher",
    "eval_vs_row_rusher",
    "eval_vs_col_rusher",
    "eval_vs_greedy_tactical",
    "eval_vs_lookahead_scripted",
    "eval_block_rate",
    "eval_heur_first",
    "eval_heur_second",
    "eval_line_first",
    "eval_line_second",
    "eval_row_first",
    "eval_row_second",
    "eval_col_first",
    "eval_col_second",
    "eval_random_open_heur",
    "eval_random_open_col",
    "eval_random_open_row",
    "eval_recovery_heur",
]


def open_run_directory(results_root: str) -> tuple[str, IO[str], str]:
    """
    Create results_root/run_YYYYMMDD_HHMMSS/ with plots/ subdir.
    Returns (run_dir, log_file_handle, csv_path).
    """
    import time

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_root, f"run_{ts}")
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    log_path = os.path.join(run_dir, "training.log")
    csv_path = os.path.join(run_dir, "metrics.csv")
    log_f = open(log_path, "w", encoding="utf-8")
    log_f.write(f"# Super Tic-Tac-Toe PPO training log\n# run_dir={run_dir}\n\n")
    log_f.flush()

    # Stabilize report links: mirror final plots here when updated
    latest = os.path.join(results_root, "latest")
    os.makedirs(os.path.join(latest, "plots"), exist_ok=True)

    return run_dir, log_f, csv_path


def collect_run_meta(argv: list[str], seed: int | None) -> dict[str, Any]:
    """
    Build metadata for ``run_meta.json``: argv, seed, python/platform, torch/torchrl versions, git short hash.
    """
    meta: dict[str, Any] = {
        "argv": argv,
        "seed": seed,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import torch

        meta["torch"] = torch.__version__
    except Exception:
        meta["torch"] = None
    try:
        import torchrl

        meta["torchrl"] = torchrl.__version__
    except Exception:
        meta["torchrl"] = None
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if r.returncode == 0:
            meta["git_commit"] = r.stdout.strip()
    except Exception:
        pass
    return meta


def write_run_meta_json(run_dir: str, meta: dict[str, Any]) -> None:
    """Persist reproducibility metadata next to ``training.log``."""
    path = os.path.join(run_dir, "run_meta.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def write_latest_pointer(results_root: str, run_dir: str) -> None:
    """Write ``results/LATEST.txt`` with the absolute path to this run directory."""
    p = os.path.join(results_root, "LATEST.txt")
    with open(p, "w", encoding="utf-8") as f:
        f.write(run_dir + "\n")


def append_metrics_csv(csv_path: str, row: dict[str, Any]) -> None:
    """Append one training-iteration row to ``metrics.csv`` (header on first write)."""
    exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=METRIC_FIELDS, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(row)


def _parse_float_cell(v: Any) -> float:
    """Parse one CSV cell to float; empty or invalid -> NaN."""
    if v is None or v == "":
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _win_rate_half_reference(ax) -> None:
    """Subtle y=0.5 line (chance / parity) on win-rate axes; drawn under series."""
    ax.axhline(
        0.5,
        color="#94a3b8",
        linestyle=(0, (5, 4)),
        linewidth=0.75,
        alpha=0.42,
        zorder=0,
    )


def _read_metrics(csv_path: str) -> dict[str, list]:
    """Load metrics.csv into column lists aligned with METRIC_FIELDS."""
    if not os.path.isfile(csv_path):
        return {k: [] for k in METRIC_FIELDS}
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)
    out: dict[str, list] = {k: [] for k in METRIC_FIELDS}
    float_cols = {
        "mean_reward",
        "win_rate",
        "iter_seconds",
        "policy_loss",
        "value_loss",
        "entropy",
        "clip_fraction",
        "approx_kl",
        "explained_variance",
        "grad_norm",
        "learning_rate",
        "entropy_coeff",
        "shaping_mult",
        "steps_per_sec",
        "forfeit_rate",
        "forfeit_injected_rate",
        "blocked_rate",
        "mean_episode_length",
        "eval_win_rate",
        "eval_draw_rate",
        "eval_vs_random_win",
        "eval_vs_heuristic_win",
        "eval_vs_line_rusher",
        "eval_vs_row_rusher",
        "eval_vs_col_rusher",
        "eval_vs_greedy_tactical",
        "eval_vs_lookahead_scripted",
        "eval_block_rate",
        "eval_heur_first",
        "eval_heur_second",
        "eval_line_first",
        "eval_line_second",
        "eval_row_first",
        "eval_row_second",
        "eval_col_first",
        "eval_col_second",
        "eval_random_open_heur",
        "eval_random_open_col",
        "eval_random_open_row",
        "eval_recovery_heur",
    }
    for row in rows:
        for k in METRIC_FIELDS:
            v = row.get(k, "")
            if k in ("iteration", "global_iteration", "snapshot_count"):
                out[k].append(int(float(v)) if v not in ("", None) else 0)
            elif k in ("num_episodes_lifetime", "total_env_steps"):
                if v in ("", None):
                    out[k].append(0)
                else:
                    try:
                        out[k].append(int(float(v)))
                    except (TypeError, ValueError):
                        out[k].append(0)
            elif k in float_cols:
                out[k].append(_parse_float_cell(v))
    return out


def plot_metrics(csv_path: str, plots_dir: str, results_root: str) -> None:
    """
    Regenerate reward / win-rate / time / combined PNGs from ``metrics.csv``.

    Copies the same files into ``results/latest/plots`` for stable ``report.html`` links.
    """
    data = _read_metrics(csv_path)
    it = data["iteration"]
    if not it:
        return

    mr = data["mean_reward"]
    wr = data["win_rate"]
    ts = data["iter_seconds"]

    def _style(ax, title: str, ylabel: str):
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Training iteration (this run)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    # 1) Reward
    fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
    ax.plot(it, mr, color="#2563eb", linewidth=1.5)
    _style(ax, "Episode reward (mean)", "mean reward")
    fig.tight_layout()
    p1 = os.path.join(plots_dir, "reward_mean.png")
    fig.savefig(p1)
    plt.close(fig)

    # 2) Win rate (matplotlib skips NaN segments) + optional eval overlay
    fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
    _win_rate_half_reference(ax)
    ax.plot(it, wr, color="#059669", linewidth=1.5, marker=".", markersize=3, label="train rollout", zorder=2)
    ewr = data.get("eval_win_rate", [])
    if ewr and any(math.isfinite(x) for x in ewr):
        ax.plot(
            it,
            ewr,
            color="#d97706",
            linewidth=1.3,
            marker="s",
            markersize=3,
            label="eval vs main_v0",
            zorder=2,
        )
    erw = data.get("eval_vs_random_win", [])
    if erw and any(math.isfinite(x) for x in erw):
        ax.plot(
            it,
            erw,
            color="#dc2626",
            linewidth=1.3,
            marker="^",
            markersize=3,
            label="eval vs random",
            zorder=2,
        )
    ehw = data.get("eval_vs_heuristic_win", [])
    if ehw and any(math.isfinite(x) for x in ehw):
        ax.plot(
            it,
            ehw,
            color="#7c3aed",
            linewidth=1.3,
            marker="D",
            markersize=3,
            label="eval vs heuristic",
            zorder=2,
        )
    has_eval = (
        (ewr and any(math.isfinite(x) for x in ewr))
        or (erw and any(math.isfinite(x) for x in erw))
        or (ehw and any(math.isfinite(x) for x in ehw))
    )
    if has_eval:
        ax.legend(loc="lower right", fontsize=8)
    _style(ax, "Main win-rate (custom metric)", "win rate")
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    p2 = os.path.join(plots_dir, "win_rate.png")
    fig.savefig(p2)
    plt.close(fig)

    # 3) Iteration wall time
    fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
    ax.plot(it, ts, color="#7c3aed", linewidth=1.2)
    _style(ax, "Wall time per training iteration", "seconds")
    fig.tight_layout()
    p3 = os.path.join(plots_dir, "iter_seconds.png")
    fig.savefig(p3)
    plt.close(fig)

    # 4) Combined 3x3
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), dpi=120)

    axes[0, 0].plot(it, mr, color="#2563eb", linewidth=1.2)
    axes[0, 0].set_title("Mean episode reward")
    axes[0, 0].grid(True, alpha=0.3)

    _win_rate_half_reference(axes[0, 1])
    axes[0, 1].plot(it, wr, color="#059669", marker=".", markersize=2, label="train", zorder=2)
    ewr = data.get("eval_win_rate", [])
    if ewr and any(math.isfinite(x) for x in ewr):
        axes[0, 1].plot(it, ewr, color="#d97706", marker="s", markersize=2, label="eval v0", zorder=2)
    erw = data.get("eval_vs_random_win", [])
    if erw and any(math.isfinite(x) for x in erw):
        axes[0, 1].plot(it, erw, color="#dc2626", marker="^", markersize=2, label="vs random", zorder=2)
    ehw = data.get("eval_vs_heuristic_win", [])
    if ehw and any(math.isfinite(x) for x in ehw):
        axes[0, 1].plot(it, ehw, color="#7c3aed", marker="D", markersize=2, label="vs heuristic", zorder=2)
    has_eval_combined = (
        (ewr and any(math.isfinite(x) for x in ewr))
        or (erw and any(math.isfinite(x) for x in erw))
        or (ehw and any(math.isfinite(x) for x in ehw))
    )
    if has_eval_combined:
        axes[0, 1].legend(loc="lower right", fontsize=7)
    axes[0, 1].set_title("Main win-rate")
    axes[0, 1].set_ylim(-0.05, 1.05)
    axes[0, 1].grid(True, alpha=0.3)

    _win_rate_half_reference(axes[0, 2])
    elr = data.get("eval_vs_line_rusher", [])
    err_ = data.get("eval_vs_row_rusher", [])
    ecr = data.get("eval_vs_col_rusher", [])
    if elr and any(math.isfinite(x) for x in elr):
        axes[0, 2].plot(it, elr, color="#e11d48", marker="o", markersize=2, label="vs line_rusher", zorder=2)
    if err_ and any(math.isfinite(x) for x in err_):
        axes[0, 2].plot(it, err_, color="#f59e0b", marker="v", markersize=2, label="vs row_rusher", zorder=2)
    if ecr and any(math.isfinite(x) for x in ecr):
        axes[0, 2].plot(it, ecr, color="#06b6d4", marker="x", markersize=2, label="vs col_rusher", zorder=2)
    has_rusher = (
        (elr and any(math.isfinite(x) for x in elr))
        or (err_ and any(math.isfinite(x) for x in err_))
        or (ecr and any(math.isfinite(x) for x in ecr))
    )
    if has_rusher:
        axes[0, 2].legend(loc="lower right", fontsize=7)
    axes[0, 2].set_title("Win-rate vs rushers")
    axes[0, 2].set_ylim(-0.05, 1.05)
    axes[0, 2].grid(True, alpha=0.3)

    pl = data.get("policy_loss", [])
    vl = data.get("value_loss", [])
    if pl and any(math.isfinite(x) for x in pl):
        axes[1, 0].plot(it, pl, color="#dc2626", linewidth=1.0, label="policy", zorder=2)
    if vl and any(math.isfinite(x) for x in vl):
        ax_vl = axes[1, 0].twinx()
        ax_vl.plot(it, vl, color="#2563eb", linewidth=1.0, label="value", zorder=2)
        ax_vl.tick_params(axis="y", colors="#2563eb")
    axes[1, 0].set_title("Losses (policy=red, value=blue)")
    axes[1, 0].grid(True, alpha=0.3)

    ent = data.get("entropy", [])
    cf = data.get("clip_fraction", [])
    if ent and any(math.isfinite(x) for x in ent):
        axes[1, 1].plot(it, ent, color="#059669", linewidth=1.0, label="entropy")
    if cf and any(math.isfinite(x) for x in cf):
        ax_cf = axes[1, 1].twinx()
        ax_cf.plot(it, cf, color="#f59e0b", linewidth=1.0, label="clip frac")
        ax_cf.tick_params(axis="y", colors="#f59e0b")
    axes[1, 1].set_title("Entropy (green) + Clip frac (orange)")
    axes[1, 1].grid(True, alpha=0.3)

    evx = data.get("explained_variance", [])
    if evx and any(math.isfinite(x) for x in evx):
        axes[1, 2].plot(it, evx, color="#7c3aed", linewidth=1.0)
        axes[1, 2].axhline(0, color="#94a3b8", linestyle="--", linewidth=0.5, alpha=0.5)
    axes[1, 2].set_title("Explained variance")
    axes[1, 2].grid(True, alpha=0.3)

    axes[2, 0].plot(it, ts, color="#7c3aed", linewidth=1.0)
    axes[2, 0].set_title("Seconds per iteration")
    axes[2, 0].grid(True, alpha=0.3)

    sps = data.get("steps_per_sec", [])
    if sps and any(math.isfinite(x) for x in sps):
        axes[2, 1].plot(it, sps, color="#06b6d4", linewidth=1.0)
        axes[2, 1].set_title("Steps / second (throughput)")
        axes[2, 1].grid(True, alpha=0.3)
    else:
        ne = data["num_episodes_lifetime"]
        if any(ne):
            axes[2, 1].plot(it, ne, color="#c026d3")
            axes[2, 1].set_title("Episodes (lifetime)")
            axes[2, 1].grid(True, alpha=0.3)
        else:
            axes[2, 1].axis("off")

    ff = data.get("forfeit_rate", [])
    ebr = data.get("eval_block_rate", [])
    gn = data.get("grad_norm", [])
    mel = data.get("mean_episode_length", [])
    if ff and any(math.isfinite(x) for x in ff):
        axes[2, 2].plot(it, ff, color="#ea580c", marker=".", markersize=2, label="forfeit (train)", zorder=2)
        axes[2, 2].set_ylim(-0.05, 1.05)
        axes[2, 2].set_title("Forfeit rate (batch) + block (eval)")
        if ebr and any(math.isfinite(x) for x in ebr):
            ax_b = axes[2, 2].twinx()
            ax_b.plot(it, ebr, color="#059669", marker="s", markersize=2, alpha=0.75, label="block eval")
            ax_b.set_ylim(-0.05, 1.05)
        elif mel and any(math.isfinite(x) for x in mel):
            ax_m = axes[2, 2].twinx()
            ax_m.plot(it, mel, color="#2563eb", linewidth=1.0, alpha=0.8, label="mean ep len")
    elif ebr and any(math.isfinite(x) for x in ebr):
        axes[2, 2].plot(it, ebr, color="#059669", marker="s", markersize=2, label="block rate", zorder=2)
        axes[2, 2].set_title("Block rate (eval)")
        axes[2, 2].set_ylim(-0.05, 1.05)
    elif gn and any(math.isfinite(x) for x in gn):
        axes[2, 2].plot(it, gn, color="#e11d48", linewidth=1.0)
        axes[2, 2].set_title("Gradient norm")
    else:
        axes[2, 2].axis("off")
    axes[2, 2].grid(True, alpha=0.3)

    fig.suptitle("Super Tic-Tac-Toe — PPO training curves (3×3)", fontsize=12, y=1.01)
    fig.tight_layout()
    p4 = os.path.join(plots_dir, "combined.png")
    fig.savefig(p4)
    plt.close(fig)

    # Mirror to results/latest/plots for stable report links
    latest_plots = os.path.join(results_root, "latest", "plots")
    os.makedirs(latest_plots, exist_ok=True)
    import shutil

    for name in ("reward_mean.png", "win_rate.png", "iter_seconds.png", "combined.png"):
        src = os.path.join(plots_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(latest_plots, name))


def log_print(log_f: IO[str], msg: str) -> None:
    """Print to stdout and append the same line to the run's ``training.log``."""
    print(msg)
    log_f.write(msg + "\n")
    log_f.flush()
