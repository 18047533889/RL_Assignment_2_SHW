"""
Tests for self_play.py OpponentPool: initialization, snapshotting, selection, win tracking.
"""

import numpy as np
import pytest

from self_play import OpponentPool
from config import SELF_PLAY_CONFIG, NUM_OPPONENT_SLOTS


@pytest.fixture
def pool():
    p = OpponentPool(num_slots=NUM_OPPONENT_SLOTS)
    fake_state = {"layer.weight": np.zeros(10)}
    p.initialize(fake_state)
    return p


def test_initialize_fills_all_slots(pool):
    for s in pool._slots:
        assert s is not None


def test_save_snapshot_round_robin(pool):
    state_a = {"w": np.ones(5)}
    state_b = {"w": np.ones(5) * 2}
    idx_a = pool.save_snapshot(state_a)
    idx_b = pool.save_snapshot(state_b)
    assert idx_a == 0
    assert idx_b == 1
    assert np.array_equal(pool._slots[0]["w"], np.ones(5))
    assert np.array_equal(pool._slots[1]["w"], np.ones(5) * 2)


def test_win_rate_initially_none(pool):
    assert pool.win_rate() is None


def test_record_and_win_rate(pool):
    for _ in range(7):
        pool.record_outcome("main_win")
    for _ in range(3):
        pool.record_outcome("main_loss")
    assert pool.win_rate() == pytest.approx(0.7)


def test_should_snapshot_above_threshold(pool):
    for _ in range(80):
        pool.record_outcome("main_win")
    for _ in range(20):
        pool.record_outcome("main_loss")
    assert pool.should_snapshot()


def test_should_not_snapshot_below_threshold(pool):
    for _ in range(50):
        pool.record_outcome("main_win")
    for _ in range(50):
        pool.record_outcome("main_loss")
    assert not pool.should_snapshot()


def test_get_opponent_state_returns_valid(pool):
    rng = np.random.default_rng(42)
    for _ in range(50):
        state, scripted = pool.get_opponent_state(rng)
        if scripted is not None:
            assert scripted in [
                "random_legal", "heuristic", "line_rusher", "center_biased",
                "edge_explorer", "row_rusher", "col_rusher",
                "greedy_tactical", "lookahead_scripted", "pure_defender",
            ]
            assert state is None
        else:
            assert state is not None


def test_branch_probabilities_coverage(pool):
    rng = np.random.default_rng(0)
    got_scripted = False
    got_state = False
    for _ in range(200):
        state, scripted = pool.get_opponent_state(rng)
        if scripted is not None:
            got_scripted = True
        if state is not None:
            got_state = True
    assert got_scripted
    assert got_state


def test_snapshot_count(pool):
    assert pool.snapshot_count == 0
    pool.save_snapshot({"w": np.zeros(1)})
    pool.save_snapshot({"w": np.zeros(1)})
    assert pool.snapshot_count == 2


def test_record_draw_counted(pool):
    pool.record_outcome("draw")
    pool.record_outcome("draw")
    pool.record_outcome("main_win")
    assert pool.win_rate() == pytest.approx(1 / 3)


def test_get_opponent_single_slot():
    p = OpponentPool(num_slots=1)
    p.initialize({"w": np.zeros(1)})
    rng = np.random.default_rng(42)
    for _ in range(50):
        state, scripted = p.get_opponent_state(rng)
        if scripted is None:
            assert state is not None


def test_save_snapshot_wraps_around():
    p = OpponentPool(num_slots=2)
    p.initialize({"w": np.zeros(1)})
    p.save_snapshot({"w": np.ones(1)})
    p.save_snapshot({"w": np.ones(1) * 2})
    idx = p.save_snapshot({"w": np.ones(1) * 3})
    assert idx == 0
    assert np.array_equal(p._slots[0]["w"], np.ones(1) * 3)


def test_per_slot_tracking(pool):
    rng = np.random.default_rng(42)
    pool.get_opponent_state(rng)
    pool.record_outcome("main_win")
    has_data = any(sum(g) > 0 for g in pool._per_slot_games)
    scripted_last = pool._last_opponent_slot is None
    assert has_data or scripted_last


def test_pfsp_sampling_biases_towards_hard_opponents():
    p = OpponentPool(num_slots=4)
    p.initialize({"w": np.zeros(1)})
    p._pfsp_min_games = 3
    for _ in range(20):
        p._per_slot_wins[0].append(1)
        p._per_slot_games[0].append(1)
    for _ in range(20):
        p._per_slot_wins[1].append(0)
        p._per_slot_games[1].append(1)
    for _ in range(20):
        p._per_slot_wins[2].append(1)
        p._per_slot_games[2].append(1)
    for _ in range(20):
        p._per_slot_wins[3].append(0)
        p._per_slot_games[3].append(1)

    rng = np.random.default_rng(0)
    counts = [0] * 4
    for _ in range(1000):
        idx = p._pfsp_sample_history_slot(rng)
        counts[idx] += 1
    assert counts[1] > counts[0]
    assert counts[3] > counts[2]


def test_pfsp_cold_start_uniform():
    p = OpponentPool(num_slots=4)
    p.initialize({"w": np.zeros(1)})
    rng = np.random.default_rng(42)
    counts = [0] * 4
    for _ in range(400):
        idx = p._pfsp_sample_history_slot(rng)
        counts[idx] += 1
    for c in counts:
        assert 50 < c < 150


def test_should_snapshot_at_threshold():
    from config import SELF_PLAY_CONFIG
    threshold = float(SELF_PLAY_CONFIG["win_rate_threshold"])
    # Pick a win-rate slightly above threshold so the test tracks config changes.
    wins = int(threshold * 100) + 3
    losses = 100 - wins
    p = OpponentPool(num_slots=2)
    p.initialize({"w": np.zeros(1)})
    for _ in range(wins):
        p.record_outcome("main_win")
    for _ in range(losses):
        p.record_outcome("main_loss")
    assert p.win_rate() == pytest.approx(wins / 100.0)
    assert p.win_rate() > threshold
    assert p.should_snapshot() is True


def test_record_outcome_unrecognized_counts_as_draw():
    p = OpponentPool(num_slots=2)
    p.initialize({"w": np.zeros(1)})
    p.record_outcome("something_unexpected")
    assert sum(p._draw_history) == 1
    assert sum(p._win_history) == 0
    assert sum(p._loss_history) == 0


def test_save_snapshot_deep_copies():
    p = OpponentPool(num_slots=2)
    sd = {"w": np.zeros(3)}
    p.initialize(sd)
    p.save_snapshot(sd)
    sd["w"][0] = 999.0
    assert p._slots[0]["w"][0] != 999.0
