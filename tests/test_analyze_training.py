"""
Tests for ``analyze_training.py``: CSV loading, moving average, chart generation.
"""

import csv
import os
import tempfile

import numpy as np
import pytest

from analyze_training import load_metrics, moving_average, _parse, generate_all


class TestParse:

    def test_valid_float(self):
        assert _parse("3.14") == pytest.approx(3.14)

    def test_integer_string(self):
        assert _parse("42") == pytest.approx(42.0)

    def test_empty_string(self):
        assert np.isnan(_parse(""))

    def test_none(self):
        assert np.isnan(_parse(None))

    def test_non_numeric(self):
        assert np.isnan(_parse("abc"))


class TestMovingAverage:

    def test_basic(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ma = moving_average(arr, window=3)
        assert len(ma) == 5
        assert ma[2] == pytest.approx(2.0)
        assert ma[4] == pytest.approx(4.0)

    def test_shorter_than_window(self):
        arr = np.array([1.0, 2.0])
        ma = moving_average(arr, window=5)
        np.testing.assert_array_equal(ma, arr)

    def test_with_nan(self):
        arr = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        ma = moving_average(arr, window=3)
        assert len(ma) == 5
        assert not np.isnan(ma[2])

    def test_all_nan(self):
        arr = np.full(10, np.nan)
        ma = moving_average(arr, window=3)
        assert all(np.isnan(ma))

    def test_single_element(self):
        arr = np.array([5.0])
        ma = moving_average(arr, window=3)
        np.testing.assert_array_equal(ma, arr)


class TestLoadMetrics:

    def _write_csv(self, path, rows):
        with open(path, "w", newline="", encoding="utf-8") as f:
            if rows:
                w = csv.DictWriter(f, fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(rows)

    def test_load_valid_csv(self, tmp_path):
        csv_path = str(tmp_path / "test.csv")
        rows = [
            {"iteration": "1", "mean_reward": "0.5", "win_rate": "0.6"},
            {"iteration": "2", "mean_reward": "0.7", "win_rate": "0.8"},
        ]
        self._write_csv(csv_path, rows)
        data = load_metrics(csv_path)
        assert "iteration" in data
        assert len(data["iteration"]) == 2
        assert data["iteration"][0] == pytest.approx(1.0)
        assert data["mean_reward"][1] == pytest.approx(0.7)

    def test_load_empty_csv(self, tmp_path):
        csv_path = str(tmp_path / "empty.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("")
        data = load_metrics(csv_path)
        assert data == {}

    def test_load_csv_with_missing_values(self, tmp_path):
        csv_path = str(tmp_path / "gaps.csv")
        rows = [
            {"iteration": "1", "mean_reward": "", "win_rate": "0.5"},
            {"iteration": "2", "mean_reward": "0.3", "win_rate": ""},
        ]
        self._write_csv(csv_path, rows)
        data = load_metrics(csv_path)
        assert np.isnan(data["mean_reward"][0])
        assert np.isnan(data["win_rate"][1])


class TestGenerateAll:

    def _make_metrics_csv(self, path, n=50):
        fields = ["iteration", "mean_reward", "win_rate", "eval_win_rate",
                   "eval_vs_random_win", "eval_vs_heuristic_win",
                   "iter_seconds", "num_episodes_lifetime"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for i in range(1, n + 1):
                row = {
                    "iteration": str(i),
                    "mean_reward": str(0.01 * i),
                    "win_rate": str(min(0.5 + 0.005 * i, 1.0)),
                    "eval_win_rate": str(0.3 + 0.005 * i) if i % 10 == 0 else "",
                    "eval_vs_random_win": str(0.8 + 0.002 * i) if i % 10 == 0 else "",
                    "eval_vs_heuristic_win": str(0.4 + 0.005 * i) if i % 10 == 0 else "",
                    "iter_seconds": str(10.0 + 0.1 * i),
                    "num_episodes_lifetime": str(100 * i),
                }
                w.writerow(row)

    def test_generate_all_creates_files(self, tmp_path):
        csv_path = str(tmp_path / "metrics.csv")
        out_dir = str(tmp_path / "plots")
        self._make_metrics_csv(csv_path, n=50)
        generate_all(csv_path, out_dir)
        expected = [
            "reward_detailed.png",
            "win_rate_detailed.png",
            "training_phases.png",
            "throughput_analysis.png",
            "summary_dashboard.png",
            "learning_efficiency.png",
        ]
        for name in expected:
            assert os.path.isfile(os.path.join(out_dir, name)), f"Missing {name}"

    def test_generate_all_empty_csv(self, tmp_path):
        csv_path = str(tmp_path / "empty.csv")
        out_dir = str(tmp_path / "plots")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("")
        generate_all(csv_path, out_dir)

    def test_generate_all_short_csv(self, tmp_path):
        csv_path = str(tmp_path / "short.csv")
        out_dir = str(tmp_path / "plots")
        self._make_metrics_csv(csv_path, n=5)
        generate_all(csv_path, out_dir)
        assert os.path.isfile(os.path.join(out_dir, "reward_detailed.png"))
