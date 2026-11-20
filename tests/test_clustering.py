"""Local clustering vs transitivity decomposition (mo438.clustering)."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from mo438 import clustering as CL
from mo438.verification import undirected_adjacency, undirected_summary


def _arrays(graph: nx.Graph) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    graph = nx.convert_node_labels_to_integers(graph)
    edges = np.array(sorted(tuple(sorted(e)) for e in graph.edges()), dtype=np.int64)
    s = undirected_summary(edges, graph.number_of_nodes())
    return edges, s["arrays"]["degree"], s["arrays"]["triangles"]


@pytest.fixture
def graph() -> nx.Graph:
    rng = np.random.default_rng(7)
    g = nx.powerlaw_cluster_graph(300, 3, 0.4, seed=7)
    hubs = [0, 1]
    for h in hubs:  # extra hub-like vertices with low clustering
        for v in rng.choice(np.arange(10, 300), size=60, replace=False):
            g.add_edge(h, int(v))
    return g


def test_global_measures_match_networkx(graph):
    _, degree, triangles = _arrays(graph)
    m = CL.global_measures(degree, triangles)
    g = nx.convert_node_labels_to_integers(graph)
    assert m["transitivity_3T_over_triples"] == pytest.approx(nx.transitivity(g), abs=1e-12)
    assert m["transitivity_weighted_mean"] == pytest.approx(nx.transitivity(g), abs=1e-12)
    local = nx.clustering(g)
    eligible = [local[v] for v in g if g.degree(v) >= 2]
    assert m["average_local_clustering"] == pytest.approx(np.mean(eligible), abs=1e-12)
    assert m["num_triangles"] == sum(nx.triangles(g).values()) // 3


def test_top_split_reconstructs_transitivity(graph):
    _, degree, triangles = _arrays(graph)
    t = CL.global_measures(degree, triangles)["transitivity_3T_over_triples"]
    for top in (1, 2, 5):
        s = CL.top_degree_split(degree, triangles, top)
        assert s["reconstructed_T"] == pytest.approx(t, abs=1e-12)
        assert s["upper_bound_rest_if_hubs_zero"] >= s["weighted_clustering_rest"] - 1e-12
        assert sorted(s["hub_degrees"], reverse=True) == s["hub_degrees"]


def test_bins_partition_both_averages(graph):
    _, degree, triangles = _arrays(graph)
    m = CL.global_measures(degree, triangles)
    bins = CL.by_degree_bin(degree, triangles)
    assert bins["contribution_to_average"].sum() == pytest.approx(
        m["average_local_clustering"], abs=1e-12)
    assert bins["contribution_to_transitivity"].sum() == pytest.approx(
        m["transitivity_3T_over_triples"], abs=1e-12)
    assert bins["vertex_share"].sum() == pytest.approx(1.0)
    assert bins["triple_share"].sum() == pytest.approx(1.0)
    assert bins["num_vertices"].sum() == m["num_vertices_degree_ge_2"]
    assert all(bins["k_low"] == 2 ** bins["bin"])


def test_degree_bins_edges():
    assert CL.degree_bins(np.array([1, 2, 3, 4, 7, 8, 1023, 1024])).tolist() == \
        [0, 1, 1, 2, 2, 3, 9, 10]


def test_neighbour_bound_exact_cases():
    # complete graph: every neighbour can close all pairs -> bound = observed = tau
    edges, degree, triangles = _arrays(nx.complete_graph(6))
    bound = CL.neighbour_degree_bound(edges, degree)
    assert np.allclose(bound, degree * (degree - 1) / 2)
    assert np.array_equal(triangles, bound.astype(int))
    # star: leaves have degree 1, so the centre cannot have any triangle
    edges, degree, triangles = _arrays(nx.star_graph(8))
    bound = CL.neighbour_degree_bound(edges, degree)
    assert bound[0] == 0 and triangles[0] == 0


def test_neighbour_bound_never_exceeded(graph):
    edges, degree, triangles = _arrays(graph)
    bound = CL.neighbour_degree_bound(edges, degree)
    assert np.all(triangles <= bound + 1e-12)
    table = CL.bound_by_degree_bin(degree, triangles, bound)
    assert (table["realised_fraction_weighted"] <= 1 + 1e-12).all()
    assert (table["mean_bound"] >= table["mean_local_clustering"] - 1e-12).all()


def test_bound_rejects_inconsistent_input():
    with pytest.raises(ValueError):
        CL.bound_by_degree_bin(np.array([2, 2, 2]), np.array([1, 1, 1]),
                               np.array([0.5, 1.0, 1.0]))
    with pytest.raises(ValueError):
        CL.local_quantities(np.array([2]), np.array([2]))


def test_by_degree_exact_values():
    # triangle 0-1-2 plus pendant 3 on vertex 0: C_0 = 1/3, C_1 = C_2 = 1
    _, degree, triangles = _arrays(nx.Graph([(0, 1), (1, 2), (0, 2), (0, 3)]))
    table = CL.by_degree(degree, triangles).set_index("degree")
    assert table.loc[2, "num_vertices"] == 2 and table.loc[2, "mean_local_clustering"] == 1
    assert table.loc[3, "mean_local_clustering"] == pytest.approx(1 / 3)
    assert table.loc[2, "fraction_clustering_one"] == 1


def test_trend_recovers_known_slope():
    k_low = 2 ** np.arange(1, 10)
    bins = pd.DataFrame({"bin": np.arange(1, 10), "k_low": k_low, "k_high": 2 * k_low - 1,
                         "num_vertices": 100})
    centre = np.sqrt(bins["k_low"] * bins["k_high"])
    bins["mean_local_clustering"] = 3.0 * centre ** -0.8
    t = CL.trend(bins, min_vertices=30, k_from=2)
    assert t["loglog_slope"] == pytest.approx(-0.8, abs=1e-9)
    assert t["spearman_bin_vs_mean"] == pytest.approx(-1.0)
    assert t["ratio_per_doubling"] == pytest.approx(2 ** -0.8)


def test_degree_distribution_ccdf():
    d = CL.degree_distribution(np.array([0, 1, 1, 2, 5]))
    assert d["degree"].tolist() == [0, 1, 2, 5]
    assert d["ccdf"].tolist() == pytest.approx([1.0, 0.8, 0.4, 0.2])
    assert d["fraction"].sum() == pytest.approx(1.0)


def test_same_community_neighbour_pairs():
    # vertex 0 has neighbours 1, 2 (community 0) and 3, 4 (community 1): 4 of 12 ordered pairs
    edges = np.array([(0, 1), (0, 2), (0, 3), (0, 4)])
    adj = undirected_adjacency(edges, 5)
    membership = np.array([0, 0, 0, 1, 1])
    value = CL.same_community_neighbour_pairs(adj.indptr, adj.indices, membership,
                                              np.array([0, 1]))
    assert value[0] == pytest.approx(4 / 12)
    assert np.isnan(value[1])  # degree 1


def test_closed_triple_share_splits_transitivity(graph):
    _, degree, triangles = _arrays(graph)
    t = CL.global_measures(degree, triangles)["transitivity_3T_over_triples"]
    bins = CL.by_degree_bin(degree, triangles)
    assert bins["closed_triple_share"].sum() == pytest.approx(1.0)
    assert np.allclose(bins["contribution_to_transitivity"], bins["closed_triple_share"] * t)
    assert np.allclose(bins["contribution_to_transitivity"],
                       bins["triple_share"] * bins["weighted_local_clustering"])
