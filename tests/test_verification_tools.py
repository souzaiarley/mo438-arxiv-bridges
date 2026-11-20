"""The independent verifiers (mo438.verification) checked against networkx / igraph."""

from __future__ import annotations

import igraph as ig
import networkx as nx
import numpy as np
import pytest

from mo438.verification import (
    all_pairs_distance_histogram,
    check_distance_histogram,
    directed_summary,
    double_sweep_lower_bound,
    edge_common_neighbours,
    exact_diameter_ifub,
    orientation_by_year,
    undirected_adjacency,
    undirected_summary,
)


def _edges(graph: nx.Graph) -> np.ndarray:
    return np.array(sorted((min(u, v), max(u, v)) for u, v in graph.edges()), dtype=np.int64)


GRAPHS = {
    "karate": nx.karate_club_graph(),
    "gnp_disconnected": nx.gnp_random_graph(120, 0.03, seed=7),
    "powerlaw_cluster": nx.powerlaw_cluster_graph(300, 3, 0.4, seed=1),
}


@pytest.mark.parametrize("name", list(GRAPHS))
def test_undirected_summary_matches_networkx(name):
    graph = GRAPHS[name]
    n = graph.number_of_nodes()
    s = undirected_summary(_edges(graph), n)
    tri = nx.triangles(graph)
    assert s["num_triangles"] == sum(tri.values()) // 3
    assert np.array_equal(s["arrays"]["triangles"], [tri[i] for i in range(n)])
    clust = nx.clustering(graph)
    assert np.allclose(s["arrays"]["clustering"], [clust[i] for i in range(n)], atol=1e-12)
    assert s["transitivity"] == pytest.approx(nx.transitivity(graph), abs=1e-12)
    assert s["average_local_clustering_all_nodes"] == pytest.approx(
        nx.average_clustering(graph), abs=1e-12)
    eligible = [clust[i] for i in range(n) if graph.degree(i) >= 2]
    assert s["average_local_clustering"] == pytest.approx(np.mean(eligible), abs=1e-12)
    assert s["num_connected_components"] == nx.number_connected_components(graph)
    assert s["density"] == pytest.approx(nx.density(graph), abs=1e-15)


def test_edge_common_neighbours_brute_force():
    graph = GRAPHS["powerlaw_cluster"]
    edges = _edges(graph)
    adj = undirected_adjacency(edges, graph.number_of_nodes())
    common = edge_common_neighbours(adj, edges, block_rows=37)  # odd block size on purpose
    expected = [len(set(graph[u]) & set(graph[v])) for u, v in edges.tolist()]
    assert common.tolist() == expected


def test_directed_summary_matches_networkx_and_igraph():
    graph = nx.gnp_random_graph(200, 0.02, seed=3, directed=True)
    base = sorted(graph.edges())
    present = set(base)
    extra = [(v, u) for u, v in base[:8] if (v, u) not in present]  # new reciprocal pairs
    assert extra
    edges = np.array(base + extra + [(0, 0)], dtype=np.int64)  # ... and one self-loop
    n = 200
    s = directed_summary(edges, n)
    g = ig.Graph(n=n, edges=edges.tolist(), directed=True)
    assert s["reciprocity"] == pytest.approx(g.reciprocity(ignore_loops=True, mode="default"))
    ref = nx.MultiDiGraph()
    ref.add_nodes_from(range(n))
    ref.add_edges_from(edges.tolist())
    assert s["num_strongly_connected_components"] == nx.number_strongly_connected_components(ref)
    assert s["num_weakly_connected_components"] == nx.number_weakly_connected_components(ref)
    assert s["largest_scc_size"] == max(len(c) for c in nx.strongly_connected_components(ref))
    assert s["num_self_loops"] == 1


CONNECTED = {
    "path_40": nx.path_graph(40),
    "cycle_31": nx.cycle_graph(31),
    "tree": nx.random_labeled_tree(200, seed=5) if hasattr(nx, "random_labeled_tree")
    else nx.random_tree(200, seed=5),
    "watts_strogatz": nx.connected_watts_strogatz_graph(400, 4, 0.05, seed=2),
    "barabasi_albert": nx.barabasi_albert_graph(500, 2, seed=4),
    "lollipop": nx.lollipop_graph(15, 25),
}


@pytest.mark.parametrize("name", list(CONNECTED))
def test_ifub_diameter_is_exact(name):
    graph = CONNECTED[name]
    adj = undirected_adjacency(_edges(graph), graph.number_of_nodes())
    result = exact_diameter_ifub(adj, chunk=8)
    assert result["diameter"] == nx.diameter(graph)
    a, b = result["witness_pair"]
    assert nx.shortest_path_length(graph, a, b) == result["diameter"]


def test_double_sweep_is_a_lower_bound():
    graph = CONNECTED["watts_strogatz"]
    adj = undirected_adjacency(_edges(graph), graph.number_of_nodes())
    assert double_sweep_lower_bound(adj, [0, 5])["lower_bound"] <= nx.diameter(graph)


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_all_pairs_histogram_matches_igraph(n_jobs):
    graph = CONNECTED["barabasi_albert"]
    n = graph.number_of_nodes()
    adj = undirected_adjacency(_edges(graph), n)
    result = all_pairs_distance_histogram(adj, n_jobs=n_jobs, block=50)
    g = ig.Graph(n=n, edges=_edges(graph).tolist())
    ref = {int(start): count for start, _, count in g.path_length_hist(directed=False).bins()}
    observed = {d: int(c) for d, c in enumerate(result["histogram"]) if d > 0}
    assert observed == {d: 2 * c for d, c in ref.items()}  # igraph counts unordered pairs
    assert result["num_ordered_pairs"] == n * (n - 1)
    assert result["average_distance"] == pytest.approx(nx.average_shortest_path_length(graph))
    assert result["diameter"] == nx.diameter(graph)


def test_histogram_consistency_checks():
    graph = nx.path_graph(4)  # ordered pairs: d=1:6, d=2:4, d=3:2
    checks = check_distance_histogram(np.array([1, 2, 3]), np.array([6, 4, 2]), 4, 3)
    assert checks["sum_equals_n(n-1)"] and checks["distance_1_equals_2m"]
    assert checks["average_distance"] == pytest.approx(nx.average_shortest_path_length(graph))
    bad = check_distance_histogram(np.array([1, 2, 3]), np.array([6, 4, 1]), 4, 3)
    assert not bad["sum_equals_n(n-1)"]


def test_orientation_by_year():
    years = np.array([2010, 2011, 2012])
    edges = np.array([(2, 0), (1, 0), (0, 1), (1, 1)])
    out = orientation_by_year(edges, years)
    assert out["frac_citing_newer_than_cited"] == 0.5
    assert out["frac_citing_older_than_cited"] == 0.25
    assert out["frac_same_year"] == 0.25
