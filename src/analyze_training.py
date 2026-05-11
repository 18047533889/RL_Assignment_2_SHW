"""
analyze_training.py — Generate rich analysis charts from metrics.csv.

Usage:
    python src/analyze_training.py results/run_YYYYMMDD_HHMMSS/metrics.csv [--output-dir plots/]
    python src/analyze_training.py models/v1.0/metrics.csv --output-dir models/v1.0/plots/
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def _parse(v):
    if v is None or v == "":
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def load_metrics(csv_path: str) -> dict[str, np.ndarray]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    keys = rows[0].keys()
    data: dict[str, list[float]] = {k: [] for k in keys}
    for row in rows:
        for k in keys:
            data[k].append(_parse(row.get(k, "")))
    return {k: np.array(v) for k, v in data.items()}


def moving_average(arr: np.ndarray, window: int = 20) -> np.ndarray:
    if len(arr) < window:
        return arr
    cumsum = np.nancumsum(arr)
    cumsum = np.insert(cumsum, 0, 0)
    counts = np.nancumsum(~np.isnan(arr)).astype(float)
    counts = np.insert(counts, 0, 0)
    ma = np.full_like(arr, np.nan)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        s = cumsum[i + 1] - cumsum[start]
        c = counts[i + 1] - counts[start]
        if c > 0:
            ma[i] = s / c
    return ma


def plot_reward_detailed(iters, reward, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    axes[0].plot(iters, reward, color="#2563eb", alpha=0.3, linewidth=0.8, label="Raw")
    ma = moving_average(reward, 20)
    axes[0].plot(iters, ma, color="#2563eb", linewidth=2.0, label="MA(20)")
    axes[0].set_title("Episode Reward — Full Training", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Mean Reward")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color="gray", linestyle="--", alpha=0.5)

    hist_data = reward[~np.isnan(reward)]
    axes[1].hist(hist_data, bins=50, color="#2563eb", alpha=0.7, edgecolor="white")
    axes[1].axvline(np.nanmean(reward), color="#dc2626", linestyle="--", linewidth=2, label=f"Mean={np.nanmean(reward):.4f}")
    axes[1].axvline(np.nanmedian(reward), color="#059669", linestyle="--", linewidth=2, label=f"Median={np.nanmedian(reward):.4f}")
    axes[1].set_title("Reward Distribution", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Mean Reward")
    axes[1].set_ylabel("Frequency")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "reward_detailed.png"), bbox_inches="tight")
    plt.close(fig)


def plot_win_rate_detailed(iters, wr, eval_wr, eval_random, eval_heur, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)

    axes[0, 0].plot(iters, wr, color="#059669", alpha=0.3, linewidth=0.8, label="Raw")
    ma = moving_average(wr, 20)
    axes[0, 0].plot(iters, ma, color="#059669", linewidth=2.0, label="MA(20)")
    axes[0, 0].axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    axes[0, 0].set_title("Self-Play Win Rate (Training)", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Win Rate")
    axes[0, 0].set_ylim(-0.05, 1.05)
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    valid_eval = ~np.isnan(eval_wr)
    if valid_eval.any():
        axes[0, 1].plot(iters[valid_eval], eval_wr[valid_eval], "o-", color="#d97706", markersize=4, linewidth=1.5, label="vs Snapshot (main_v0)")
    valid_rand = ~np.isnan(eval_random)
    if valid_rand.any():
        axes[0, 1].plot(iters[valid_rand], eval_random[valid_rand], "s-", color="#dc2626", markersize=4, linewidth=1.5, label="vs Random Legal")
    valid_heur = ~np.isnan(eval_heur)
    if valid_heur.any():
        axes[0, 1].plot(iters[valid_heur], eval_heur[valid_heur], "D-", color="#7c3aed", markersize=4, linewidth=1.5, label="vs Heuristic")
    axes[0, 1].axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    axes[0, 1].set_title("Eval Win Rates (Greedy)", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Win Rate")
    axes[0, 1].set_ylim(-0.05, 1.05)
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    wr_clean = wr[~np.isnan(wr)]
    if len(wr_clean) >= 100:
        early = wr_clean[:100]
        late = wr_clean[-100:]
        axes[1, 0].hist(early, bins=30, alpha=0.6, color="#94a3b8", label=f"First 100 (μ={np.mean(early):.3f})", edgecolor="white")
        axes[1, 0].hist(late, bins=30, alpha=0.6, color="#059669", label=f"Last 100 (μ={np.mean(late):.3f})", edgecolor="white")
        axes[1, 0].set_title("Win Rate: Early vs Late", fontsize=11, fontweight="bold")
        axes[1, 0].legend(fontsize=9)
    else:
        axes[1, 0].hist(wr_clean, bins=30, color="#059669", alpha=0.7, edgecolor="white")
        axes[1, 0].set_title("Win Rate Distribution", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Win Rate")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].grid(True, alpha=0.3)

    window = 50
    if len(wr_clean) > window:
        rolling_std = np.array([np.std(wr_clean[max(0, i-window):i+1]) for i in range(len(wr_clean))])
        axes[1, 1].plot(range(len(rolling_std)), rolling_std, color="#7c3aed", linewidth=1.5)
        axes[1, 1].set_title(f"Win Rate Volatility (σ, window={window})", fontsize=11, fontweight="bold")
        axes[1, 1].set_xlabel("Iteration")
        axes[1, 1].set_ylabel("Rolling Std Dev")
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].axis("off")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "win_rate_detailed.png"), bbox_inches="tight")
    plt.close(fig)


def plot_training_phases(iters, reward, wr, out_dir):
    n = len(iters)
    if n < 30:
        return
    third = n // 3
    phases = [
        ("Early (0-33%)", 0, third, "#ef4444"),
        ("Mid (33-66%)", third, 2*third, "#f59e0b"),
        ("Late (66-100%)", 2*third, n, "#22c55e"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)

    for ax_idx, (label, s, e, color) in enumerate(phases):
        ax = axes[ax_idx]
        r_slice = reward[s:e]
        w_slice = wr[s:e]
        it_slice = iters[s:e]

        ax2 = ax.twinx()
        ax.plot(it_slice, r_slice, color="#2563eb", alpha=0.5, linewidth=0.8)
        ma_r = moving_average(r_slice, 10)
        ax.plot(it_slice, ma_r, color="#2563eb", linewidth=2, label="Reward MA(10)")

        ax2.plot(it_slice, w_slice, color="#059669", alpha=0.5, linewidth=0.8)
        ma_w = moving_average(w_slice, 10)
        ax2.plot(it_slice, ma_w, color="#059669", linewidth=2, label="WinRate MA(10)")

        ax.set_title(f"{label}", fontsize=11, fontweight="bold", color=color)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Reward", color="#2563eb")
        ax2.set_ylabel("Win Rate", color="#059669")
        ax2.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.2)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")

    fig.suptitle("Training Phases Analysis", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_phases.png"), bbox_inches="tight")
    plt.close(fig)


def plot_throughput(iters, seconds, episodes, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)

    axes[0].plot(iters, seconds, color="#7c3aed", alpha=0.4, linewidth=0.8)
    ma = moving_average(seconds, 20)
    axes[0].plot(iters, ma, color="#7c3aed", linewidth=2)
    axes[0].set_title("Seconds per Iteration", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Seconds")
    axes[0].grid(True, alpha=0.3)

    cumtime = np.nancumsum(seconds) / 3600
    axes[1].plot(iters, cumtime, color="#0ea5e9", linewidth=2)
    axes[1].set_title("Cumulative Training Time", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Hours")
    axes[1].grid(True, alpha=0.3)

    eps = np.diff(episodes, prepend=0)
    eps[eps < 0] = 0
    if np.any(seconds > 0):
        eps_per_sec = eps / np.maximum(seconds, 0.01)
        axes[2].plot(iters, eps_per_sec, color="#f59e0b", alpha=0.4, linewidth=0.8)
        ma_eps = moving_average(eps_per_sec, 20)
        axes[2].plot(iters, ma_eps, color="#f59e0b", linewidth=2)
    axes[2].set_title("Episodes / Second", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Eps/s")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "throughput_analysis.png"), bbox_inches="tight")
    plt.close(fig)


def plot_summary_dashboard(iters, reward, wr, eval_wr, eval_random, eval_heur, seconds, out_dir, version_label=""):
    fig = plt.figure(figsize=(16, 12), dpi=150)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

    title = f"Training Summary Dashboard"
    if version_label:
        title += f" — {version_label}"
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(iters, reward, alpha=0.25, color="#2563eb", linewidth=0.7)
    ax1.plot(iters, moving_average(reward, 30), color="#2563eb", linewidth=2)
    ax1.set_title("Mean Episode Reward", fontweight="bold")
    ax1.set_xlabel("Iteration")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.4)

    ax2 = fig.add_subplot(gs[0, 2])
    stats = {
        "Final Reward": f"{np.nanmean(reward[-20:]):.4f}",
        "Peak Reward": f"{np.nanmax(reward):.4f}",
        "Train WR (final)": f"{np.nanmean(wr[-20:]):.3f}",
        "Iterations": f"{len(iters)}",
        "Total Hours": f"{np.nansum(seconds)/3600:.1f}",
    }
    valid_eval = ~np.isnan(eval_wr)
    if valid_eval.any():
        stats["Eval vs v0 (last)"] = f"{eval_wr[valid_eval][-1]:.3f}"
    valid_rand = ~np.isnan(eval_random)
    if valid_rand.any():
        stats["Eval vs Random (last)"] = f"{eval_random[valid_rand][-1]:.3f}"
    valid_heur = ~np.isnan(eval_heur)
    if valid_heur.any():
        stats["Eval vs Heuristic (last)"] = f"{eval_heur[valid_heur][-1]:.3f}"

    ax2.axis("off")
    y = 0.95
    for k, v in stats.items():
        ax2.text(0.1, y, f"{k}:", fontsize=9, fontweight="bold", transform=ax2.transAxes, va="top")
        ax2.text(0.95, y, v, fontsize=9, transform=ax2.transAxes, va="top", ha="right", color="#2563eb")
        y -= 0.11
    ax2.set_title("Key Metrics", fontweight="bold")

    ax3 = fig.add_subplot(gs[1, :2])
    ax3.plot(iters, wr, alpha=0.25, color="#059669", linewidth=0.7)
    ax3.plot(iters, moving_average(wr, 30), color="#059669", linewidth=2, label="Self-play WR")
    if valid_eval.any():
        ax3.plot(iters[valid_eval], eval_wr[valid_eval], "o-", color="#d97706", markersize=3, linewidth=1.2, label="vs Snapshot")
    if valid_rand.any():
        ax3.plot(iters[valid_rand], eval_random[valid_rand], "s-", color="#dc2626", markersize=3, linewidth=1.2, label="vs Random")
    if valid_heur.any():
        ax3.plot(iters[valid_heur], eval_heur[valid_heur], "D-", color="#7c3aed", markersize=3, linewidth=1.2, label="vs Heuristic")
    ax3.axhline(0.5, color="gray", linestyle="--", alpha=0.4)
    ax3.set_title("Win Rates Over Training", fontweight="bold")
    ax3.set_xlabel("Iteration")
    ax3.set_ylim(-0.05, 1.05)
    ax3.legend(fontsize=8, loc="lower right")
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 2])
    wr_clean = wr[~np.isnan(wr)]
    if len(wr_clean) >= 100:
        labels_bp = ["First\n100", "Mid\n100", "Last\n100"]
        bp_data = [wr_clean[:100], wr_clean[len(wr_clean)//2-50:len(wr_clean)//2+50], wr_clean[-100:]]
        ax4.boxplot(bp_data, tick_labels=labels_bp, patch_artist=True,
                    boxprops=dict(facecolor="#e0f2fe"), medianprops=dict(color="#dc2626", linewidth=2))
    elif len(wr_clean) > 0:
        ax4.hist(wr_clean, bins=min(30, len(wr_clean)), color="#059669", alpha=0.7, edgecolor="white")
    else:
        ax4.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax4.transAxes, color="#94a3b8")
    ax4.set_title("Win Rate Progression", fontweight="bold")
    ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(iters, seconds, color="#7c3aed", alpha=0.3, linewidth=0.7)
    ax5.plot(iters, moving_average(seconds, 20), color="#7c3aed", linewidth=2)
    ax5.set_title("Iteration Time", fontweight="bold")
    ax5.set_xlabel("Iteration")
    ax5.set_ylabel("Seconds")
    ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(gs[2, 1])
    cumtime = np.nancumsum(seconds) / 3600
    ax6.fill_between(iters, 0, cumtime, alpha=0.3, color="#0ea5e9")
    ax6.plot(iters, cumtime, color="#0ea5e9", linewidth=2)
    ax6.set_title("Cumulative Time (hours)", fontweight="bold")
    ax6.set_xlabel("Iteration")
    ax6.grid(True, alpha=0.3)

    ax7 = fig.add_subplot(gs[2, 2])
    reward_clean = reward[~np.isnan(reward)]
    ax7.hist(reward_clean, bins=40, color="#2563eb", alpha=0.6, edgecolor="white")
    ax7.axvline(np.mean(reward_clean), color="#dc2626", linestyle="--", linewidth=2)
    ax7.set_title("Reward Distribution", fontweight="bold")
    ax7.set_xlabel("Reward")
    ax7.grid(True, alpha=0.3)

    fig.savefig(os.path.join(out_dir, "summary_dashboard.png"), bbox_inches="tight")
    plt.close(fig)


def plot_learning_efficiency(iters, reward, wr, seconds, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    cumtime = np.nancumsum(seconds) / 3600
    axes[0].plot(cumtime, moving_average(reward, 20), color="#2563eb", linewidth=2)
    axes[0].set_title("Reward vs Wall Time", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Training Time (hours)")
    axes[0].set_ylabel("Reward MA(20)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(cumtime, moving_average(wr, 20), color="#059669", linewidth=2)
    axes[1].axhline(0.5, color="gray", linestyle="--", alpha=0.4)
    axes[1].set_title("Win Rate vs Wall Time", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Training Time (hours)")
    axes[1].set_ylabel("Win Rate MA(20)")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "learning_efficiency.png"), bbox_inches="tight")
    plt.close(fig)


def generate_all(csv_path: str, out_dir: str, version_label: str = ""):
    os.makedirs(out_dir, exist_ok=True)
    data = load_metrics(csv_path)
    if not data or "iteration" not in data:
        print(f"No data in {csv_path}")
        return

    iters = data["iteration"]
    reward = data.get("mean_reward", np.full_like(iters, np.nan))
    wr = data.get("win_rate", np.full_like(iters, np.nan))
    eval_wr = data.get("eval_win_rate", np.full_like(iters, np.nan))
    eval_rand = data.get("eval_vs_random_win", np.full_like(iters, np.nan))
    eval_heur = data.get("eval_vs_heuristic_win", np.full_like(iters, np.nan))
    seconds = data.get("iter_seconds", np.full_like(iters, np.nan))
    episodes = data.get("num_episodes_lifetime", np.full_like(iters, np.nan))

    print(f"Generating analysis charts from {csv_path} ({len(iters)} iterations)")

    plot_reward_detailed(iters, reward, out_dir)
    print("  -> reward_detailed.png")

    plot_win_rate_detailed(iters, wr, eval_wr, eval_rand, eval_heur, out_dir)
    print("  -> win_rate_detailed.png")

    plot_training_phases(iters, reward, wr, out_dir)
    print("  -> training_phases.png")

    plot_throughput(iters, seconds, episodes, out_dir)
    print("  -> throughput_analysis.png")

    plot_summary_dashboard(iters, reward, wr, eval_wr, eval_rand, eval_heur, seconds, out_dir, version_label)
    print("  -> summary_dashboard.png")

    plot_learning_efficiency(iters, reward, wr, seconds, out_dir)
    print("  -> learning_efficiency.png")

    print(f"All charts saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate rich training analysis charts")
    parser.add_argument("csv_path", help="Path to metrics.csv")
    parser.add_argument("--output-dir", default=None, help="Output directory for charts")
    parser.add_argument("--version-label", default="", help="Version label for dashboard title")
    args = parser.parse_args()

    out_dir = args.output_dir or os.path.join(os.path.dirname(args.csv_path), "analysis_plots")
    generate_all(args.csv_path, out_dir, args.version_label)


if __name__ == "__main__":
    main()
