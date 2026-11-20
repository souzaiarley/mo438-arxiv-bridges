"""Construction of G_d and G_u (mo438.graph_build) on controlled inputs."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import TOY_EXPECTED

from mo438.data import generate_synthetic_citations
from mo438.graph_build import (
    build_directed_graph,
    build_undirected_graph,
    csr_from_undirected_edges,
    undirected_edge_array,
)


def _brute_force_conversion(edges: np.ndarray) -> dict:
    """Reference implementation with Python sets."""
    directed = [tuple(e) for e in edges.tolist()]
    unique_directed = set(directed)
    pairs = {(min(u, v), max(u, v)) for u, v in unique_directed if u != v}
    reciprocal = {(min(u, v), max(u, v)) for u, v in unique_directed
                  if u != v and (v, u) in unique_directed}
    return {"loops": sum(u == v for u, v in directed),
            "duplicates": len(directed) - len(unique_directed),
            "pairs": sorted(pairs), "reciprocal": len(reciprocal)}


def test_toy_conversion_counts(toy):
    uedges, stats = undirected_edge_array(toy.edges, toy.num_nodes)
    assert stats.num_directed_edges == TOY_EXPECTED["m_d"]
    assert stats.num_directed_self_loops == TOY_EXPECTED["self_loops"]
    assert stats.num_duplicate_directed_edges == TOY_EXPECTED["duplicates"]
    assert stats.num_reciprocal_pairs == TOY_EXPECTED["reciprocal_pairs"]
    assert stats.num_undirected_edges == TOY_EXPECTED["m_u"]
    assert [tuple(e) for e in uedges.tolist()] == TOY_EXPECTED["gu_edges"]


def test_self_loops_can_be_kept(toy):
    uedges, stats = undirected_edge_array(toy.edges, toy.num_nodes, remove_self_loops=False)
    assert (6, 6) in {tuple(e) for e in uedges.tolist()}
    assert stats.num_undirected_edges == TOY_EXPECTED["m_u"] + 1


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_conversion_matches_brute_force_on_synthetic(seed):
    data = generate_synthetic_citations(num_nodes=600, seed=seed)
    uedges, stats = undirected_edge_array(data.edges, data.num_nodes)
    ref = _brute_force_conversion(data.edges)
    assert [tuple(e) for e in uedges.tolist()] == ref["pairs"]
    assert stats.num_directed_self_loops == ref["loops"]
    assert stats.num_duplicate_directed_edges == ref["duplicates"]
    assert stats.num_reciprocal_pairs == ref["reciprocal"]
    assert np.all(uedges[:, 0] < uedges[:, 1])


def test_igraph_graphs_preserve_nodes_and_attributes(toy):
    g_d = build_directed_graph(toy)
    g_u, stats, _ = build_undirected_graph(toy)
    assert g_d.is_directed() and not g_u.is_directed()
    assert g_d.vcount() == g_u.vcount() == toy.num_nodes
    assert g_d.ecount() == TOY_EXPECTED["m_d"]
    assert g_u.ecount() == TOY_EXPECTED["m_u"] == stats.num_undirected_edges
    assert g_u.is_simple()
    assert g_u.vs["category"] == toy.labels.tolist()
    assert g_u.vs["category_name"] == [toy.category_names[k] for k in toy.labels]
    # orientation of G_d is the input orientation (u cites v)
    assert [tuple(e) for e in g_d.get_edgelist()] == [tuple(e) for e in toy.edges.tolist()]


def test_csr_adjacency_is_symmetric_and_unweighted(toy):
    uedges, _ = undirected_edge_array(toy.edges, toy.num_nodes)
    adj = csr_from_undirected_edges(uedges, toy.num_nodes)
    assert (adj != adj.T).nnz == 0
    assert adj.max() == 1
    assert adj.nnz == 2 * TOY_EXPECTED["m_u"]
