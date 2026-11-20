"""Objective 2: citation flows, null models and bootstrap."""

from __future__ import annotations

import numpy as np
import pytest

from mo438 import flows as F
from mo438.data import generate_synthetic_citations
from mo438.metrics import category_citation_matrix


@pytest.fixture
def synthetic():
    data = generate_synthetic_citations(num_nodes=900, seed=4)
    k = len(data.category_names)
    return data, k, F.year_pair_stratum(data.edges, data.years)


def test_matrix_matches_f1_code(synthetic):
    data, k, _ = synthetic
    assert np.array_equal(F.citation_matrix(data.edges, data.labels, k),
                          category_citation_matrix(data.edges, data.labels, k))


def test_expected_matrices_preserve_margins(synthetic):
    data, k, stratum = synthetic
    M = F.citation_matrix(data.edges, data.labels, k)
    for s in (np.zeros(len(data.edges), dtype=np.int64), stratum):
        E = F.stratified_expected(data.edges, data.labels, k, s)
        assert np.allclose(E.sum(axis=1), M.sum(axis=1))
        assert np.allclose(E.sum(axis=0), M.sum(axis=0))
        assert E.sum() == pytest.approx(M.sum())


def test_single_stratum_is_outer_product(synthetic):
    data, k, _ = synthetic
    M = F.citation_matrix(data.edges, data.labels, k)
    E = F.stratified_expected(data.edges, data.labels, k, np.zeros(len(data.edges), dtype=int))
    assert np.allclose(E, np.outer(M.sum(axis=1), M.sum(axis=0)) / M.sum())


def test_category_and_pair_identities(synthetic):
    data, k, stratum = synthetic
    st = F.flow_stats(data.edges, data.labels, k, stratum, data.years)
    c, p = st.category(), st.pairs()
    assert c["inter_out"].sum() == c["inter_in"].sum()
    assert np.all((c["nnf"] >= -1) & (c["nnf"] <= 1))
    assert np.nansum(p["net_ij"]) == pytest.approx(np.nansum(p["M_ij"] - p["M_ji"]))
    ok = ~np.isnan(p["asymmetry_ij"])
    assert np.all(np.abs(p["asymmetry_ij"][ok]) <= 1)
    # expected NNF: given pair totals, expected in + out = observed in + out
    for name in ("strength", "time"):
        assert np.all(np.abs(c[f"nnf_expected_{name}"][~np.isnan(c["nnf"])]) <= 1 + 1e-12)


def test_unit_weights_reproduce_point_estimates(synthetic):
    data, k, stratum = synthetic
    a = F.flow_stats(data.edges, data.labels, k, stratum, data.years)
    b = F.flow_stats(data.edges, data.labels, k, stratum, data.years,
                     weights=np.ones(len(data.edges)))
    assert np.allclose(a.M, b.M) and np.allclose(a.E_time, b.E_time)


def test_bootstrap_is_reproducible_and_centred(synthetic):
    data, k, stratum = synthetic
    b1 = F.bootstrap(data.edges, data.labels, k, stratum, data.years, data.num_nodes, 30, seed=1)
    b2 = F.bootstrap(data.edges, data.labels, k, stratum, data.years, data.num_nodes, 30, seed=1)
    assert np.array_equal(b1["cat_nnf"], b2["cat_nnf"], equal_nan=True)
    point = F.flow_stats(data.edges, data.labels, k, stratum, data.years).category()["nnf"]
    low, high = np.nanpercentile(b1["cat_nnf"], [0, 100], axis=0)
    ok = ~np.isnan(point)
    assert np.mean((low[ok] <= point[ok]) & (point[ok] <= high[ok])) > 0.9


def test_temporal_null_removes_a_pure_age_effect():
    """Two areas, A old and B recent; every paper cites random *older* papers regardless
    of area. The raw asymmetry is total (B -> A only); the temporal null explains it."""
    rng = np.random.default_rng(0)
    years = np.r_[rng.integers(2000, 2006, 400), rng.integers(2006, 2012, 400)]
    labels = np.r_[np.zeros(400, int), np.ones(400, int)]
    edges = []
    for u in range(800):
        older = np.flatnonzero(years < years[u])
        if older.size:
            for v in rng.choice(older, size=min(5, older.size), replace=False):
                edges.append((u, v))
    edges = np.asarray(edges)
    st = F.flow_stats(edges, labels, 2, F.year_pair_stratum(edges, years), years)
    p = st.pairs()
    assert p["asymmetry_ij"][0] == pytest.approx(-1.0)          # only B (1) -> A (0)
    assert abs(p["lor_time"][0]) < 0.6                           # explained by time
    assert abs(p["lor_strength"][0]) > 3                         # not by volumes alone
    c = st.category()
    assert c["nnf_age"] == pytest.approx(c["nnf"])               # the age null explains it


def test_oriented_top_pairs(synthetic):
    data, k, stratum = synthetic
    st = F.flow_stats(data.edges, data.labels, k, stratum, data.years)
    boot = F.bootstrap(data.edges, data.labels, k, stratum, data.years, data.num_nodes, 20, seed=2)
    pairs = F.pair_table(st, boot, data.category_names)
    top = F.oriented_top_pairs(pairs, "net_ij", 5)
    assert (top["A_to_B"] >= top["B_to_A"]).all()
    assert (top["asymmetry_ci95_low"] <= top["asymmetry_ci95_high"]).all()


def test_spearman_permutation():
    x = np.arange(40.0)
    out = F.spearman(x, x + np.random.default_rng(0).normal(0, 1, 40), permutations=500)
    assert out["rho"] > 0.9 and out["p_permutation"] < 0.01


def test_margin_preserving_nulls_preserve_net_flows(synthetic):
    """made_i - received_i = inter_out_i - inter_in_i, and the nulls keep made/received."""
    data, k, stratum = synthetic
    st = F.flow_stats(data.edges, data.labels, k, stratum, data.years)
    net = st.M.sum(axis=0) - st.M.sum(axis=1)
    for E in (st.E_strength, st.E_time):
        assert np.allclose(E.sum(axis=0) - E.sum(axis=1), net)
    # the age null keeps each area's inter-area citing volume (row sums), no diagonal,
    # and redistributes what areas receive
    inter_out = st.M.sum(axis=1) - np.diag(st.M)
    assert np.allclose(st.E_age.sum(axis=1), inter_out)
    assert np.allclose(np.diag(st.E_age), 0)
    assert st.E_age.sum() == pytest.approx(inter_out.sum())
