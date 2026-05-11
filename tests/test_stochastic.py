"""
``stochastic.py``: placement resolution (half direct, uniform neighbours, off-board counts as bad),
forfeit probability, exact distribution, Monte Carlo checks.
"""

import numpy as np
import pytest
from board import empty_board, is_valid, VALID_MASK
from stochastic import (
    resolve_placement, forfeit_probability, placement_distribution, NEIGHBOURS,
    forfeit_probability_map,
)

P1, P2 = 1, 2


class TestResolvePlacement:
    """``resolve_placement`` returns on-board coords or None (forfeit)."""

    def test_always_returns_within_board_or_none(self):
        """Many random calls never return out-of-bounds coordinates."""
        rng = np.random.default_rng(42)
        b = empty_board()
        for _ in range(1000):
            result = resolve_placement(b, 0, 4, rng)
            if result is not None:
                r, c = result
                assert is_valid(r, c)

    def test_deterministic_seed_reproducible(self):
        """Same seed -> same sequence."""
        b = empty_board()
        rng1 = np.random.default_rng(0)
        rng2 = np.random.default_rng(0)
        results1 = [resolve_placement(b, 5, 5, rng1) for _ in range(50)]
        results2 = [resolve_placement(b, 5, 5, rng2) for _ in range(50)]
        assert results1 == results2

    def test_empty_board_center_never_forfeits_on_valid_neighbours(self):
        """Centre of empty board: no forfeit (neighbours empty)."""
        rng = np.random.default_rng(123)
        b = empty_board()
        results = [resolve_placement(b, 9, 5, rng) for _ in range(2000)]
        none_count = results.count(None)
        assert none_count == 0

    def test_occupied_neighbour_causes_forfeit(self):
        """Occupied neighbours -> positive forfeit rate; MC sees None."""
        b = empty_board()
        for dr, dc in NEIGHBOURS:
            nr, nc = 9 + dr, 5 + dc
            if is_valid(nr, nc):
                b[nr, nc] = P1
        rng = np.random.default_rng(42)
        forfeits = sum(
            1 for _ in range(2000)
            if resolve_placement(b, 9, 5, rng) is None
        )
        assert forfeits > 0


class TestForfeitProbability:
    """Closed form ``forfeit_probability``: corners and extra pieces."""

    def test_center_of_level3_empty_board(self):
        """L3 centre empty: no bad neighbours -> forfeit prob 0."""
        b = empty_board()
        p = forfeit_probability(b, 9, 5)
        assert p == pytest.approx(0.0, abs=1e-10)

    def test_corner_top_left_of_level1(self):
        """L1 corner: bad_neighbours * 0.5/8."""
        b = empty_board()
        p = forfeit_probability(b, 0, 4)
        bad_neighbours = 0
        for dr, dc in NEIGHBOURS:
            nr, nc = 0 + dr, 4 + dc
            if not is_valid(nr, nc):
                bad_neighbours += 1
        expected = 0.5 * bad_neighbours / 8.0
        assert p == pytest.approx(expected)

    def test_corner_example_from_spec(self):
        """Spec example: 5 bad neighbours -> 5/16."""
        b = empty_board()
        p = forfeit_probability(b, 0, 4)
        bad = 0
        for dr, dc in NEIGHBOURS:
            nr, nc = dr, 4 + dc
            if not is_valid(nr, nc):
                bad += 1
        assert bad == 5
        assert p == pytest.approx(5.0 / 16.0)

    def test_occupied_neighbours_increase_forfeit(self):
        """Extra occupied neighbour increases forfeit probability."""
        b = empty_board()
        p0 = forfeit_probability(b, 9, 5)
        b[8, 4] = P1
        p1 = forfeit_probability(b, 9, 5)
        assert p1 > p0


class TestPlacementDistribution:
    """``placement_distribution``: normalized, 0.5 direct, corner None mass."""

    def test_probabilities_sum_to_one(self):
        """Outcomes sum to 1."""
        b = empty_board()
        dist = placement_distribution(b, 5, 5)
        total = sum(dist.values())
        assert total == pytest.approx(1.0)

    def test_direct_placement_is_half(self):
        """Aim cell probability 0.5."""
        b = empty_board()
        dist = placement_distribution(b, 5, 5)
        assert dist[(5, 5)] == pytest.approx(0.5)

    def test_corner_forfeit_matches_formula(self):
        """Corner includes None with mass 5/16."""
        b = empty_board()
        dist = placement_distribution(b, 0, 4)
        assert None in dist
        assert dist[None] == pytest.approx(5.0 / 16.0)

    def test_all_outcomes_are_valid_or_none(self):
        """Non-None keys are valid coordinates."""
        b = empty_board()
        for r in range(12):
            for c in range(12):
                if VALID_MASK[r, c]:
                    dist = placement_distribution(b, r, c)
                    for key in dist:
                        if key is not None:
                            assert is_valid(*key)

    def test_occupied_neighbour_mass_moves_to_forfeit(self):
        """A filled neighbour loses ``1/16`` landing mass; that mass becomes ``None``."""
        b = empty_board()
        r_aim, c_aim = 9, 5
        dist_open = placement_distribution(b, r_aim, c_aim)
        nr, nc = 8, 4
        assert dist_open.get((nr, nc), 0) == pytest.approx(1.0 / 16.0)
        b[nr, nc] = P1
        dist_blocked = placement_distribution(b, r_aim, c_aim)
        assert (nr, nc) not in dist_blocked
        assert dist_blocked[None] == pytest.approx(dist_open.get(None, 0.0) + 1.0 / 16.0)
        assert sum(dist_blocked.values()) == pytest.approx(1.0)

    def test_two_occupied_neighbours_forfeit_mass(self):
        """Each occupied neighbour adds another ``1/16`` to ``None``."""
        b = empty_board()
        r_aim, c_aim = 9, 5
        base = placement_distribution(empty_board(), r_aim, c_aim)
        b[8, 4] = P1
        one = placement_distribution(b, r_aim, c_aim)
        b[8, 6] = P2
        two = placement_distribution(b, r_aim, c_aim)
        assert two[None] == pytest.approx(one[None] + 1.0 / 16.0)
        assert two[None] == pytest.approx(base.get(None, 0.0) + 2.0 / 16.0)


class TestForfeitProbabilityMatchesPlacementDistribution:
    """Closed-form ``forfeit_probability`` must match ``None`` mass in ``placement_distribution`` (silent bug guard)."""

    def test_forfeit_mass_equals_none_key_on_empty_board(self):
        """Every valid aim: analytic forfeit == dist[None]; distribution sums to 1."""
        b = empty_board()
        for r in range(12):
            for c in range(12):
                if not VALID_MASK[r, c]:
                    continue
                dist = placement_distribution(b, r, c)
                fp = forfeit_probability(b, r, c)
                assert fp == pytest.approx(dist.get(None, 0.0), abs=1e-12)
                assert sum(dist.values()) == pytest.approx(1.0, abs=1e-10)

    def test_forfeit_mass_equals_none_key_with_occupied_cells(self):
        """Mixed board: same identity (neighbour branch only; direct half unchanged)."""
        b = empty_board()
        b[8, 4] = P1
        b[9, 6] = P2
        for r, c in [(9, 5), (0, 4), (11, 6)]:
            dist = placement_distribution(b, r, c)
            fp = forfeit_probability(b, r, c)
            assert fp == pytest.approx(dist.get(None, 0.0), abs=1e-12)
            assert sum(dist.values()) == pytest.approx(1.0, abs=1e-10)


class TestMonteCarloConvergence:
    """Large-sample frequencies near theory (loose tolerance)."""

    def test_direct_placement_rate_approx_half(self):
        """Direct placement rate ~ 0.5."""
        rng = np.random.default_rng(99)
        b = empty_board()
        r, c = 9, 5
        N = 20000
        direct = sum(1 for _ in range(N) if resolve_placement(b, r, c, rng) == (r, c))
        assert direct / N == pytest.approx(0.5, abs=0.02)

    def test_corner_forfeit_rate_mc(self):
        """Corner forfeit rate ~ 5/16."""
        rng = np.random.default_rng(77)
        b = empty_board()
        r, c = 0, 4
        N = 20000
        forfeits = sum(1 for _ in range(N) if resolve_placement(b, r, c, rng) is None)
        assert forfeits / N == pytest.approx(5.0 / 16.0, abs=0.02)

    def test_neighbour_distribution_uniform(self):
        """Each valid neighbour gets ~1/16 total mass (1/8 of the 0.5 branch)."""
        rng = np.random.default_rng(55)
        b = empty_board()
        r, c = 9, 5
        N = 80000
        counts: dict = {}
        for _ in range(N):
            result = resolve_placement(b, r, c, rng)
            counts[result] = counts.get(result, 0) + 1
        for dr, dc in NEIGHBOURS:
            nr, nc = r + dr, c + dc
            if is_valid(nr, nc):
                freq = counts.get((nr, nc), 0) / N
                assert freq == pytest.approx(1.0 / 16.0, abs=0.005)


class TestForfeitProbabilityMap:
    """``forfeit_probability_map`` returns a (12,12) float32 array consistent with scalar ``forfeit_probability``."""

    def test_shape_and_dtype(self):
        b = empty_board()
        fp = forfeit_probability_map(b)
        assert fp.shape == (12, 12)
        assert fp.dtype == np.float32

    def test_matches_scalar_on_empty_board(self):
        b = empty_board()
        fp = forfeit_probability_map(b)
        for r in range(12):
            for c in range(12):
                if VALID_MASK[r, c]:
                    assert fp[r, c] == pytest.approx(forfeit_probability(b, r, c), abs=1e-7)
                else:
                    assert fp[r, c] == pytest.approx(0.0, abs=1e-10)

    def test_matches_scalar_with_occupied_cells(self):
        b = empty_board()
        b[8, 4] = 1
        b[9, 6] = 2
        fp = forfeit_probability_map(b)
        for r, c in [(9, 5), (0, 4), (11, 6)]:
            assert fp[r, c] == pytest.approx(forfeit_probability(b, r, c), abs=1e-7)
