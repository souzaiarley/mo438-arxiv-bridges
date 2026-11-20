"""Sampled LCC distance estimator (complement of objectives 4-5)."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from mo438 import removal as R
from mo438.edges import is_inter_area
from mo438.lcc_distances import lcc_distance_estimate, run_states


def _graph():
    g = nx.powerlaw_cluster_graph(250, 2, 0.3, seed=11)
    g.add_edges_from([(250, 251), (251, 252)])          # a detached small component
    edges = np.array(sorted((min(u, v), max(u, v)) for u, v in g.edges()), dtype=np.int64)
    return g, edges, g.number_of_nodes()


def test_all_sources_equals_exact_lcc_average():
    g, edges, n = _graph()
    keep = np.ones(len(edges), dtype=bool)
    out = lcc_distance_estimate(edges, n, keep, num_sources=10_000,
                                rng=np.random.default_rng(0), chunk=7)
    lcc = g.subgraph(max(nx.connected_components(g), key=len))
    assert out["lcc_size"] == lcc.number_of_nodes()
    assert out["num_sources"] == lcc.number_of_nodes()
    assert out["mean_distance"] == pytest.approx(nx.average_shortest_path_length(lcc))
    assert out["se"] == pytest.approx(0.0, abs=1e-12)     # whole population: fpc = 0
    assert out["max_sampled_eccentricity"] == nx.diameter(lcc)


def test_after_removal_matches_networkx():
    _, edges, n = _graph()
    keep = np.ones(len(edges), dtype=bool)
    keep[np.random.default_rng(1).choice(len(edges), 120, replace=False)] = False
    out = lcc_distance_estimate(edges, n, keep, 10_000, np.random.default_rng(0))
    h = nx.Graph()
    h.add_nodes_from(range(n))
    h.add_edges_from(edges[keep].tolist())
    lcc = h.subgraph(max(nx.connected_components(h), key=len))
    assert out["mean_distance"] == pytest.approx(nx.average_shortest_path_length(lcc))


def test_sampled_estimate_is_close_and_reproducible():
    g, edges, n = _graph()
    keep = np.ones(len(edges), dtype=bool)
    a = lcc_distance_estimate(edges, n, keep, 60, np.random.default_rng(5))
    b = lcc_distance_estimate(edges, n, keep, 60, np.random.default_rng(5))
    assert a == b
    exact = nx.average_shortest_path_length(g.subgraph(max(nx.connected_components(g), key=len)))
    assert abs(a["mean_distance"] - exact) < 5 * a["se"] + 1e-9


def test_run_states_uses_the_removal_orders():
    _, edges, n = _graph()
    labels = np.random.default_rng(2).integers(3, size=n)
    strata, _ = R.edge_strata(edges, n)
    classes = R.EdgeClasses(edges, n, is_inter_area(edges, labels), strata)
    k = classes.k_max // 2
    rows = run_states(classes, [("inter", 0, k, classes.k_max, 10_000, 9)], n_jobs=1)
    order, _ = R.removal_order("inter", 0, classes, classes.k_max, 9)
    keep = np.ones(len(edges), dtype=bool)
    keep[order[:k]] = False
    h = nx.Graph()
    h.add_nodes_from(range(n))
    h.add_edges_from(edges[keep].tolist())
    lcc = h.subgraph(max(nx.connected_components(h), key=len))
    assert rows[0]["lcc_size"] == lcc.number_of_nodes()
    assert rows[0]["mean_distance"] == pytest.approx(nx.average_shortest_path_length(lcc))
