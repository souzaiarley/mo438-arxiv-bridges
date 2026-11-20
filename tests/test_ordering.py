"""Objective 7: linear ordering of the categories (objective, heuristic, bounds, controls)."""

from __future__ import annotations

import numpy as np
import pytest

from mo438 import ordering as O


def _random_w(k: int, seed: int, density: float = 0.8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    W = rng.gamma(0.7, 50, size=(k, k)) * (rng.random((k, k)) < density)
    np.fill_diagonal(W, 0)
    return W


def test_objective_matches_definition_and_reverse_identity():
    W = _random_w(9, 0)
    order = np.random.default_rng(1).permutation(9)
    pos = O.positions(order)
    explicit = sum(W[i, j] for i in range(9) for j in range(9) if i != j and pos[j] < pos[i])
    assert O.compatible_weight(W, order) == pytest.approx(explicit)
    assert O.compatible_fraction(W, order) + O.compatible_fraction(W, order[::-1]) == \
        pytest.approx(1.0)


@pytest.mark.parametrize("seed", range(4))
def test_move_deltas_match_recomputation(seed):
    W = _random_w(10, seed)
    order = np.random.default_rng(seed + 10).permutation(10)
    base = O.compatible_weight(W, order)
    delta = O.move_deltas(W, order)
    for p in range(10):
        for q in range(10):
            moved = O._apply(order, p, q)
            assert delta[p, q] == pytest.approx(O.compatible_weight(W, moved) - base, abs=1e-9)


def test_local_search_reaches_a_local_optimum():
    W = _random_w(15, 3)
    order = O.local_search(W, np.random.default_rng(0).permutation(15))
    assert O.move_deltas(W, order).max() <= 1e-9


@pytest.mark.parametrize("seed", range(6))
def test_heuristic_finds_the_exact_optimum_small(seed):
    W = _random_w(7, seed)
    exact, _ = O.exact_brute_force(W)
    res = O.optimise(W, np.random.default_rng(seed), random_starts=20, kicks=5,
                     initial=[O.net_flow_order(W)])
    assert res["fraction"] == pytest.approx(exact, abs=1e-12)


@pytest.mark.parametrize("seed", range(3))
def test_heuristic_matches_dynamic_programming(seed):
    W = _random_w(11, 100 + seed)
    exact = O.exact_dp(W)
    res = O.optimise(W, np.random.default_rng(seed), random_starts=30, kicks=10)
    assert res["fraction"] == pytest.approx(exact, abs=1e-12)


def test_dp_agrees_with_brute_force():
    W = _random_w(7, 42)
    assert O.exact_dp(W) == pytest.approx(O.exact_brute_force(W)[0])


@pytest.mark.parametrize("seed", range(6))
def test_bounds_are_valid(seed):
    W = _random_w(7, 200 + seed)
    exact, _ = O.exact_brute_force(W)
    b = O.cycle_packing_bound(W)
    assert exact <= b["cycle_packing_bound"] + 1e-12
    assert b["cycle_packing_bound"] <= b["pairwise_bound"] + 1e-12


def test_transitive_matrix_is_fully_orderable():
    k = 8
    rng = np.random.default_rng(5)
    W = np.tril(rng.random((k, k)) * 10, -1)        # i cites j only for j < i
    b = O.cycle_packing_bound(W)
    assert b["dominant_3_cycles"] == 0
    assert b["pairwise_bound"] == pytest.approx(1.0)
    res = O.optimise(W, rng, random_starts=5, kicks=2)
    assert res["fraction"] == pytest.approx(1.0)
    assert O.compatible_fraction(W, np.arange(k)) == pytest.approx(1.0)


def test_cycle_bound_on_a_pure_cycle():
    # a -> b -> c -> a with equal weights: any order keeps 2 of 3 arcs
    W = np.zeros((3, 3))
    W[0, 1] = W[1, 2] = W[2, 0] = 1.0
    b = O.cycle_packing_bound(W)
    assert b["pairwise_bound"] == pytest.approx(1.0)
    assert b["cycle_packing_bound"] == pytest.approx(2 / 3)
    assert O.exact_brute_force(W)[0] == pytest.approx(2 / 3)


def test_random_permutations_average_one_half():
    W = _random_w(12, 7)
    fr = O.random_permutation_fractions(W, np.random.default_rng(0), 4000)
    assert fr.mean() == pytest.approx(0.5, abs=0.01)


def test_direction_randomisation_preserves_pairs():
    W = _random_w(12, 8)
    W2 = O.randomise_directions(W, np.random.default_rng(1))
    assert np.allclose(W + W.T, W2 + W2.T)
    assert O.pairwise_upper_bound(W) == pytest.approx(O.pairwise_upper_bound(W2))
    assert W.sum() == pytest.approx(W2.sum())
    assert not np.allclose(W.sum(axis=1), W2.sum(axis=1))


def test_rank_correlations():
    order = np.array([3, 1, 0, 2])
    assert O.rank_correlations(order, order)["kendall_tau"] == pytest.approx(1.0)
    assert O.rank_correlations(order, order[::-1])["spearman_rho"] == pytest.approx(-1.0)
