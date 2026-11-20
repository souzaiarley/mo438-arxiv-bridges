"""Cross-checks of the scientific review (mo438.review)."""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
import pytest

from mo438 import review as RV


def test_prob_all_drawn_matches_hypergeometric():
    pool, removed = 9, 4
    a = np.array([0, 1, 2, 3, 4, 5])
    expected = [comb(pool - k, removed - k) / comb(pool, removed) if k <= removed else 0.0
                for k in a]
    assert RV.prob_all_drawn(a, removed, pool) == pytest.approx(expected)


def test_expected_isolated_by_enumeration():
    # path 0-1-2-3 plus pendant 4 on 2; pool = edges of class "x" (0-1, 2-4, 2-3)
    edges = [(0, 1), (1, 2), (2, 3), (2, 4)]
    in_pool = [True, False, True, True]
    n = 5
    pool_edges = [e for e, p in zip(edges, in_pool) if p]
    deg_t = np.zeros(n, dtype=int)
    deg_o = np.zeros(n, dtype=int)
    for (u, v), p in zip(edges, in_pool):
        (deg_t if p else deg_o)[[u, v]] += 1
    for removed in range(len(pool_edges) + 1):
        total = 0
        subsets = list(combinations(range(len(pool_edges)), removed))
        for sub in subsets:
            kept = [e for i, e in enumerate(edges) if not (in_pool[i] and
                                                           pool_edges.index(e) in sub)]
            deg = np.zeros(n, dtype=int)
            for u, v in kept:
                deg[[u, v]] += 1
            total += int((deg == 0).sum())
        brute = total / len(subsets)
        assert RV.expected_isolated(deg_t, deg_o, removed, len(pool_edges)) == \
            pytest.approx(brute)


def _summary(q, s_a, s_b, sd=0.0, reps=50):
    rows = []
    for name, s in (("a", s_a), ("b", s_b)):
        for k, (qq, ss) in enumerate(zip(q, s)):
            rows.append({"strategy": name, "k": k, "q": qq, "S_mean": ss, "S_sd": sd,
                         "replicates": reps})
    return pd.DataFrame(rows)


def test_curve_difference_and_crossing():
    q = np.linspace(0, 0.3, 31)
    s_b = np.ones_like(q)
    s_a = 1 + 0.001 * np.sin(np.pi * q / 0.3) - 0.002 * (q > 0.2) * (q - 0.2) / 0.1
    # a above b, then below after the crossing near q = 0.2..0.3
    d = RV.curve_difference(_summary(q, s_a, s_b, sd=1e-5), "a", "b", num_nodes=1000)
    assert d["diff"].iloc[5] == pytest.approx(1000 * (s_a[5] - 1))
    cross = RV.sign_changes(d)
    assert len(cross) == 1
    assert 0.2 < cross[0]["q_cross"] < 0.3
    assert cross[0]["extreme_before"]["diff"] > 0 > cross[0]["extreme_after"]["diff"]


def test_sign_changes_ignores_noise_around_zero():
    q = np.linspace(0, 0.3, 31)
    s_b = np.ones_like(q)
    s_a = 1 + np.where(np.arange(31) % 2 == 0, 1e-6, -1e-6)  # flips sign but never significant
    d = RV.curve_difference(_summary(q, s_a, s_b, sd=1e-3), "a", "b", num_nodes=1000)
    assert RV.sign_changes(d) == []


def test_bridge_decomposition_adds_up():
    table = pd.DataFrame({
        "class": ["inter", "all"], "edges": [1000, 4000],
        "pendant_bridges": [30, 80], "non_pendant_bridges": [5, 10],
        "vertices_detachable_by_non_pendant": [20, 30],
        "vertices_detachable_total": [50, 110]})
    r = RV.bridge_decomposition(table, "inter", "all")
    assert r["difference"] == pytest.approx(50 / 1000 - 110 / 4000)
    assert r["pendant_part"] + r["non_pendant_part"] == pytest.approx(r["difference"])
    assert r["non_pendant_frequency_effect"] + r["non_pendant_size_effect"] == \
        pytest.approx(r["non_pendant_part"])


def test_holm_and_bh_known_example():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    # Holm: sorted 0.005<=0.0125, 0.01<=0.0167, 0.03>0.025 -> stop
    assert RV.holm(p).tolist() == [True, False, False, True]
    # BH: thresholds 0.0125, 0.025, 0.0375, 0.05 -> largest k with p(k)<=t(k) is k=4
    assert RV.benjamini_hochberg(p).tolist() == [True, True, True, True]
    assert RV.benjamini_hochberg(np.array([0.2, 0.5])).tolist() == [False, False]


def test_wald_from_percentile_ci():
    z, p = RV.wald_from_percentile_ci(np.array([1.96, 0.0]), np.array([0.0, -1.96]),
                                      np.array([3.92, 1.96]))
    assert z == pytest.approx([1.96, 0.0], abs=1e-4)
    assert p == pytest.approx([0.05, 1.0], abs=1e-4)


def test_partition_scores():
    labels = np.array([0, 0, 1, 1, 2, 2])
    same = RV.partition_scores(labels, np.array([5, 5, 7, 7, 9, 9]))
    assert same["homogeneity"] == pytest.approx(1) and same["completeness"] == pytest.approx(1)
    split = RV.partition_scores(labels, np.array([0, 1, 2, 3, 4, 5]))
    assert split["homogeneity"] == pytest.approx(1) and split["completeness"] < 1
    fused = RV.partition_scores(labels, np.zeros(6, dtype=int))
    assert fused["homogeneity"] == pytest.approx(0)
    for s in (same, split, fused):
        assert s["v_measure"] == pytest.approx(s["nmi_arithmetic"])


def test_internal_fraction_by_class():
    edges = np.array([(0, 1), (1, 2), (2, 3), (3, 0)])
    membership = np.array([0, 0, 1, 1])
    inter = np.array([False, True, False, True])
    r = RV.internal_fraction_by_class(edges, membership, inter)
    assert r == {"all": 0.5, "intra": 1.0, "inter": 0.0}


def test_spearman_partial_removes_common_driver():
    rng = np.random.default_rng(1)
    z = rng.normal(size=400)
    frame = pd.DataFrame({"z": z, "x": z + 0.3 * rng.normal(size=400),
                          "y": z + 0.3 * rng.normal(size=400)})
    t = RV.spearman_table(frame, [("x", "y")], control="z")
    assert t.loc[0, "rho"] > 0.8
    assert abs(t.loc[0, "partial_rho_given_z"]) < 0.15


def test_category_text_similarity():
    feats = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    labels = np.array([0, 0, 1, 1, 2, 2])
    sim = RV.category_text_similarity(feats, labels, 3, center=False)
    assert np.allclose(np.diag(sim), 1)
    assert sim[0, 1] == pytest.approx(0)
    assert sim[0, 2] == pytest.approx(1 / np.sqrt(2))


def test_expected_isolated_splits_by_vertex_profile():
    rng = np.random.default_rng(3)
    a = rng.integers(0, 4, size=200)       # edges of the removed class
    b = rng.integers(0, 3, size=200)       # edges of the other class
    total = RV.expected_isolated(a, b, 50, 300)
    parts = sum(RV.expected_isolated(a[m], b[m], 50, 300)
                for m in ((b == 0) & (a > 0), (a == 0) & (b > 0), (a > 0) & (b > 0)))
    assert parts == pytest.approx(total)
    # vertices with an edge outside the removed pool are never isolated
    assert RV.expected_isolated(a[b > 0], b[b > 0], 50, 300) == 0
