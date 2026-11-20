"""Objective 3: neighbourhood overlap, bridges and effect sizes."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from mo438 import overlap as O


def _edges(graph: nx.Graph) -> np.ndarray:
    return np.array(sorted((min(u, v), max(u, v)) for u, v in graph.edges()), dtype=np.int64)


def _brute_overlap(graph: nx.Graph, u: int, v: int) -> float:
    a = set(graph[u]) - {v}
    b = set(graph[v]) - {u}
    union = a | b
    return float("nan") if not union else len(a & b) / len(union)


GRAPHS = {
    "karate": nx.karate_club_graph(),
    "powerlaw_cluster": nx.powerlaw_cluster_graph(300, 3, 0.5, seed=3),
    "with_isolated_edge_and_tree": nx.disjoint_union_all([
        nx.powerlaw_cluster_graph(80, 2, 0.4, seed=1), nx.path_graph(2),
        nx.random_labeled_tree(30, seed=2) if hasattr(nx, "random_labeled_tree")
        else nx.random_tree(30, seed=2)]),
}


@pytest.mark.parametrize("name", list(GRAPHS))
def test_overlap_matches_definition(name):
    g = GRAPHS[name]
    edges = _edges(g)
    out = O.neighbourhood_overlap(edges, g.number_of_nodes())
    expected = np.array([_brute_overlap(g, u, v) for u, v in edges.tolist()])
    assert np.array_equal(np.isnan(out["overlap"]), np.isnan(expected))
    ok = ~np.isnan(expected)
    assert np.allclose(out["overlap"][ok], expected[ok])
    common = [len(set(g[u]) & set(g[v])) for u, v in edges.tolist()]
    assert out["common"].tolist() == common


def test_undefined_overlap_only_for_isolated_edges():
    g = GRAPHS["with_isolated_edge_and_tree"]
    edges = _edges(g)
    out = O.neighbourhood_overlap(edges, g.number_of_nodes())
    undefined = np.isnan(out["overlap"])
    assert undefined.sum() == 1                                  # the 2-vertex component
    assert (out["k_u"][undefined] == 1).all() and (out["k_v"][undefined] == 1).all()
    assert (out["denominator"][undefined] == 0).all()


@pytest.mark.parametrize("name", list(GRAPHS))
def test_bridges_match_networkx_and_brute_force(name):
    g = GRAPHS[name]
    edges = _edges(g)
    n = g.number_of_nodes()
    is_bridge, detached = O.bridges(edges, n)
    ref = {(min(u, v), max(u, v)) for u, v in nx.bridges(g)}
    assert {tuple(e) for e in edges[is_bridge].tolist()} == ref
    for idx in np.flatnonzero(is_bridge)[:25]:
        h = g.copy()
        u, v = edges[idx]
        h.remove_edge(u, v)
        a = len(nx.node_connected_component(h, int(u)))
        b = len(nx.node_connected_component(h, int(v)))
        assert detached[idx] == min(a, b)
    assert (detached[~is_bridge] == 0).all()


def test_value_index_delta_and_quantile():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 6, 200) / 5.0                          # many ties
    y = rng.integers(0, 8, 300) / 7.0
    vi = O.ValueIndex(np.r_[x, y])
    lab = np.r_[np.ones(x.size, bool), np.zeros(y.size, bool)]
    delta = O.ValueIndex.delta(vi.hist(lab), vi.hist(~lab))
    brute = np.mean(np.sign(x[:, None] - y[None, :]))
    assert delta == pytest.approx(brute)
    for q in (0.1, 0.5, 0.9):
        assert vi.quantile(vi.hist(lab), q) == np.quantile(x, q, method="inverted_cdf")


def test_standardised_mean():
    values = np.array([1.0, 2.0, 3.0, 4.0, 10.0, 20.0])
    strata = np.array([0, 0, 1, 1, 0, 1])
    target = np.array([True, True, False, False, False, False])  # only stratum 0
    source = ~target
    mean, cov = O.standardised_mean(values, strata, target, source)
    assert mean == pytest.approx(10.0) and cov == pytest.approx(1.0)
    same_mix = O.standardised_mean(values, strata, np.ones(6, bool), np.ones(6, bool))
    assert same_mix[0] == pytest.approx(values.mean())


def test_class_effects_and_bootstrap():
    g = GRAPHS["powerlaw_cluster"]
    edges = _edges(g)
    n = g.number_of_nodes()
    values = O.neighbourhood_overlap(edges, n)["overlap"]
    inter = np.random.default_rng(1).random(edges.shape[0]) < 0.35
    deg = np.bincount(edges.ravel(), minlength=n)
    kmin = np.minimum(deg[edges[:, 0]], deg[edges[:, 1]])
    strata = np.floor(np.log2(kmin)).astype(int)
    vi = O.ValueIndex(values)
    eff = O.class_effects(vi, values, inter, strata)
    assert eff["mean_inter"] == pytest.approx(values[inter].mean())
    assert eff["median_intra"] == np.quantile(values[~inter], 0.5, method="inverted_cdf")
    assert eff["frac_zero_inter"] == pytest.approx((values[inter] == 0).mean())
    unit = O.class_effects(vi, values, inter, strata, np.ones(values.size))
    assert unit == pytest.approx(eff)
    b1 = O.bootstrap_effects(values, inter, strata, None, 20, seed=3)
    b2 = O.bootstrap_effects(values, inter, strata, None, 20, seed=3)
    assert b1.equals(b2) and len(b1) == 20
    clusters = edges[:, 0]
    b3 = O.bootstrap_effects(values, inter, strata, clusters, 20, seed=3)
    assert b3["diff_mean"].between(-1, 1).all()


def test_tables_run():
    g = GRAPHS["powerlaw_cluster"]
    edges = _edges(g)
    n = g.number_of_nodes()
    values = O.neighbourhood_overlap(edges, n)["overlap"]
    labels = np.random.default_rng(2).integers(4, size=n)
    inter = labels[edges[:, 0]] != labels[edges[:, 1]]
    deg = np.bincount(edges.ravel(), minlength=n)
    kmin = np.minimum(deg[edges[:, 0]], deg[edges[:, 1]])
    kmax = np.maximum(deg[edges[:, 0]], deg[edges[:, 1]])
    pairs, strata = np.unique(np.column_stack([np.floor(np.log2(kmin)), np.floor(np.log2(kmax))])
                              .astype(int), axis=0, return_inverse=True)
    strata = strata.ravel()
    table = O.stratum_table(values, inter, strata, pairs, min_edges=5)
    assert table["n_inter"].sum() == inter.sum()
    pooled = O.pooled_stratified_delta(table)
    assert -1 <= pooled["pooled_delta"] <= 1
    is_bridge, detached = O.bridges(edges, n)
    cats = O.by_category(values, inter, edges, labels, ["a", "b", "c", "d"], is_bridge)
    assert len(cats) == 4
    bins = O.by_degree_bin(values, inter, kmin, "k_min")
    assert set(bins["class"]) <= {"inter", "intra"}
    summ = O.bridge_summary(is_bridge, detached, inter, kmin)
    assert summ.loc[summ["class"] == "all", "bridges"].item() == is_bridge.sum()
