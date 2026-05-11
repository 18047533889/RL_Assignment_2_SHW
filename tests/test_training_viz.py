"""Tests for training_viz.py: CSV schema, append/read, run_meta.json, PNG plots, logging."""
import os
import tempfile

import math
import pytest

from training_viz import (
    METRIC_FIELDS,
    append_metrics_csv,
    collect_run_meta,
    log_print,
    open_run_directory,
    plot_metrics,
    write_latest_pointer,
    write_run_meta_json,
    _parse_float_cell,
    _read_metrics,
)


def test_metric_fields_includes_eval_vs_random_win():
    """CSV columns include eval win rate vs random baseline (for reports)."""
    assert "eval_vs_random_win" in METRIC_FIELDS


def test_parse_float_cell_empty_and_invalid_becomes_nan():
    """``_parse_float_cell``: bad cells must not silently become 0.0 in plots."""
    assert math.isnan(_parse_float_cell(""))
    assert math.isnan(_parse_float_cell(None))
    assert math.isnan(_parse_float_cell("not_a_number"))
    assert _parse_float_cell("0.25") == pytest.approx(0.25)
    assert _parse_float_cell("3") == pytest.approx(3.0)


def test_append_and_read_roundtrip():
    """Write one row and read back: int/float fields; empty win_rate parses as NaN."""
    d = tempfile.mkdtemp()
    csv_path = os.path.join(d, "m.csv")
    append_metrics_csv(
        csv_path,
        {
            "iteration": 1,
            "global_iteration": 1,
            "mean_reward": 0.1,
            "win_rate": "",
            "num_episodes_lifetime": 10,
            "iter_seconds": 2.5,
            "eval_win_rate": "",
            "eval_draw_rate": "",
            "eval_vs_random_win": "",
        },
    )
    data = _read_metrics(csv_path)
    assert data["iteration"] == [1]
    assert abs(data["mean_reward"][0] - 0.1) < 1e-9
    assert data["win_rate"][0] != data["win_rate"][0]  # NaN for empty win_rate
    assert data["num_episodes_lifetime"] == [10]


def test_eval_vs_random_win_numeric_roundtrip():
    """``eval_vs_random_win`` round-trips when numeric."""
    d = tempfile.mkdtemp()
    csv_path = os.path.join(d, "m.csv")
    append_metrics_csv(
        csv_path,
        {
            "iteration": 2,
            "global_iteration": 2,
            "mean_reward": 0.0,
            "win_rate": 0.5,
            "num_episodes_lifetime": 20,
            "iter_seconds": 1.0,
            "eval_win_rate": "",
            "eval_draw_rate": "",
            "eval_vs_random_win": 0.33,
        },
    )
    data = _read_metrics(csv_path)
    assert abs(data["eval_vs_random_win"][0] - 0.33) < 1e-9


def test_write_run_meta_json():
    """``run_meta.json`` should include seed, argv, etc., for reproducibility."""
    d = tempfile.mkdtemp()
    meta = collect_run_meta(["train.py", "--seed", "7"], 7)
    write_run_meta_json(d, meta)
    import json
    import os

    p = os.path.join(d, "run_meta.json")
    assert os.path.isfile(p)
    with open(p, encoding="utf-8") as f:
        j = json.load(f)
    assert j.get("seed") == 7


def test_plot_metrics_writes_pngs():
    """With enough rows, ``plot_metrics`` writes ``combined.png`` etc. for the report."""
    d = tempfile.mkdtemp()
    csv_path = os.path.join(d, "m.csv")
    for i in range(1, 5):
        append_metrics_csv(
            csv_path,
            {
                "iteration": i,
                "global_iteration": i,
                "mean_reward": 0.1 * i,
                "win_rate": 0.5,
                "num_episodes_lifetime": 100,
                "iter_seconds": 1.0,
                "eval_win_rate": "",
                "eval_draw_rate": "",
                "eval_vs_random_win": "",
            },
        )
    plots = os.path.join(d, "plots")
    os.makedirs(plots, exist_ok=True)
    plot_metrics(csv_path, plots, d)
    assert os.path.isfile(os.path.join(plots, "combined.png"))
    assert os.path.isfile(os.path.join(plots, "reward_mean.png"))


def test_plot_metrics_with_eval_overlays():
    d = tempfile.mkdtemp()
    csv_path = os.path.join(d, "m.csv")
    for i in range(1, 5):
        append_metrics_csv(
            csv_path,
            {
                "iteration": i,
                "global_iteration": i,
                "mean_reward": 0.1 * i,
                "win_rate": 0.5,
                "num_episodes_lifetime": 100,
                "iter_seconds": 1.0,
                "eval_win_rate": 0.6,
                "eval_draw_rate": 0.2,
                "eval_vs_random_win": 0.8,
                "eval_vs_heuristic_win": 0.3,
            },
        )
    plots = os.path.join(d, "plots")
    os.makedirs(plots, exist_ok=True)
    plot_metrics(csv_path, plots, d)
    assert os.path.isfile(os.path.join(plots, "win_rate.png"))


def test_metric_fields_includes_rusher_and_block_columns():
    assert "eval_vs_line_rusher" in METRIC_FIELDS
    assert "eval_vs_row_rusher" in METRIC_FIELDS
    assert "eval_vs_col_rusher" in METRIC_FIELDS
    assert "eval_block_rate" in METRIC_FIELDS


def test_metric_fields_includes_rich_ppo_columns():
    for col in ("policy_loss", "value_loss", "entropy", "clip_fraction",
                "approx_kl", "explained_variance", "grad_norm",
                "learning_rate", "entropy_coeff", "shaping_mult",
                "steps_per_sec", "total_env_steps", "snapshot_count",
                "forfeit_rate", "forfeit_injected_rate", "blocked_rate",
                "mean_episode_length"):
        assert col in METRIC_FIELDS, f"missing {col}"


def test_plot_metrics_with_rusher_overlays():
    d = tempfile.mkdtemp()
    csv_path = os.path.join(d, "m.csv")
    for i in range(1, 5):
        append_metrics_csv(
            csv_path,
            {
                "iteration": i,
                "global_iteration": i,
                "mean_reward": 0.1 * i,
                "win_rate": 0.5,
                "num_episodes_lifetime": 100,
                "iter_seconds": 1.0,
                "eval_win_rate": 0.6,
                "eval_draw_rate": 0.2,
                "eval_vs_random_win": 0.8,
                "eval_vs_heuristic_win": 0.3,
                "eval_vs_line_rusher": 0.7,
                "eval_vs_row_rusher": 0.5,
                "eval_vs_col_rusher": 0.6,
                "eval_block_rate": 0.4,
            },
        )
    plots = os.path.join(d, "plots")
    os.makedirs(plots, exist_ok=True)
    plot_metrics(csv_path, plots, d)
    assert os.path.isfile(os.path.join(plots, "combined.png"))
    assert os.path.isfile(os.path.join(plots, "win_rate.png"))


def test_plot_metrics_with_rich_ppo_data():
    d = tempfile.mkdtemp()
    csv_path = os.path.join(d, "m.csv")
    for i in range(1, 6):
        append_metrics_csv(
            csv_path,
            {
                "iteration": i,
                "global_iteration": i,
                "mean_reward": 0.1 * i,
                "win_rate": 0.5,
                "num_episodes_lifetime": 100 * i,
                "iter_seconds": 60.0,
                "policy_loss": 0.05 - 0.001 * i,
                "value_loss": 0.3 - 0.01 * i,
                "entropy": 3.5 - 0.1 * i,
                "clip_fraction": 0.08,
                "approx_kl": 0.01,
                "explained_variance": 0.3 + 0.05 * i,
                "grad_norm": 0.4,
                "learning_rate": 3e-4,
                "entropy_coeff": 0.05,
                "shaping_mult": 1.0,
                "steps_per_sec": 100.0,
                "total_env_steps": 12288 * i,
                "snapshot_count": 0,
            },
        )
    plots = os.path.join(d, "plots")
    os.makedirs(plots, exist_ok=True)
    plot_metrics(csv_path, plots, d)
    assert os.path.isfile(os.path.join(plots, "combined.png"))


def test_read_metrics_missing_file():
    data = _read_metrics("/tmp/nonexistent_metrics_file.csv")
    for k in METRIC_FIELDS:
        assert data[k] == []


def test_log_print_writes_to_file():
    import io
    f = io.StringIO()
    log_print(f, "hello test")
    assert "hello test" in f.getvalue()


def test_open_run_directory_creates_structure():
    d = tempfile.mkdtemp()
    run_dir, log_f, csv_path = open_run_directory(d)
    assert os.path.isdir(run_dir)
    assert os.path.isdir(os.path.join(run_dir, "plots"))
    assert csv_path.endswith("metrics.csv")
    log_f.close()


def test_write_latest_pointer():
    d = tempfile.mkdtemp()
    write_latest_pointer(d, "/some/run/dir")
    p = os.path.join(d, "LATEST.txt")
    assert os.path.isfile(p)
    with open(p) as f:
        assert "/some/run/dir" in f.read()
