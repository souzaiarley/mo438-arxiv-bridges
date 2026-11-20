"""Intra-/inter-area classification (mo438.edges) on controlled inputs."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import TOY_EXPECTED, TOY_LABELS

from mo438.data import generate_synthetic_citations
from mo438.edges import (
    class_counts_table,
    classify_directed,
    classify_undirected,
    is_inter_area,
    node_composition,
    per_category_table,
    reciprocal_decomposition,
    validate_classification,
)
from mo438.graph_build import undirected_edge_array
from mo438.metrics import category_citation_matrix


@pytest.fixture
def toy_tables(toy):
    uedges, _ = undirected_edge_array(toy.edges, toy.num_nodes)
    gd = classify_directed(toy.edges, toy.labels, toy.num_nodes)
    gu = classify_undirected(uedges, toy.edges, toy.labels, toy.num_nodes)
    return gd, gu


def test_definition_on_individual_edges():
    labels = np.array([0, 0, 1])
    edges = np.array([(0, 1), (1, 0), (0, 2), (2, 1), (2, 2)])
    assert is_inter_area(edges, labels).tolist() == [False, False, True, True, False]


def test_toy_counts(toy_tables):
    gd, gu = toy_tables
    assert gd.counts()["intra"] == TOY_EXPECTED["gd_intra"]
    assert gd.counts()["inter"] == TOY_EXPECTED["gd_inter"]
    assert gu.counts()["intra"] == TOY_EXPECTED["gu_intra"]
    assert gu.counts()["inter"] == TOY_EXPECTED["gu_inter"]
    inter_pairs = {(int(u), int(v)) for u, v, f in zip(gu.src, gu.dst, gu.is_inter) if f}
    assert inter_pairs == TOY_EXPECTED["gu_inter_pairs"]


def test_toy_gd_order_and_flags_are_kept(toy, toy_tables):
    gd, _ = toy_tables
    assert np.array_equal(gd.src, toy.edges[:, 0]) and np.array_equal(gd.dst, toy.edges[:, 1])
    expected_inter = [TOY_LABELS[u] != TOY_LABELS[v] for u, v in toy.edges.tolist()]
    assert gd.is_inter.tolist() == expected_inter
    # reciprocal flags: (4,3)/(3,4) and (5,3)/(3,5) only; duplicates are not reciprocal
    recip = {tuple(e) for e, r in zip(toy.edges.tolist(), gd.reciprocal) if r}
    assert recip == {(4, 3), (3, 4), (5, 3), (3, 5)}


def test_toy_undirected_multiplicity_and_reciprocity(toy_tables):
    _, gu = toy_tables
    merged = {(int(u), int(v)): int(m) for u, v, m in zip(gu.src, gu.dst, gu.multiplicity)}
    assert merged[(1, 2)] == 2           # duplicated citation
    assert merged[(3, 4)] == 2           # reciprocal intra
    assert merged[(3, 5)] == 2           # reciprocal inter
    assert sum(merged.values()) == TOY_EXPECTED["m_d"] - TOY_EXPECTED["self_loops"]
    recip = {(int(u), int(v)) for u, v, r in zip(gu.src, gu.dst, gu.reciprocal) if r}
    assert recip == {(3, 4), (3, 5)}


def test_toy_reciprocal_decomposition(toy_tables):
    dec = reciprocal_decomposition(*toy_tables)
    for cls, expected in TOY_EXPECTED["decomposition"].items():
        for key, value in expected.items():
            assert dec[cls][key] == value, (cls, key)
    assert dec["total_reciprocal_pairs"] == TOY_EXPECTED["reciprocal_pairs"]


def test_toy_per_category_and_nodes(toy, toy_tables):
    gd, gu = toy_tables
    table = per_category_table(gd, gu, toy.labels, toy.category_names)
    for column, values in TOY_EXPECTED["per_category"].items():
        assert table[column].tolist() == values, column
    comp = node_composition(gu, toy.num_nodes)
    for key, value in TOY_EXPECTED["node_composition"].items():
        assert comp[key] == value, key


def test_toy_validation_passes(toy, toy_tables):
    gd, gu = toy_tables
    matrix = category_citation_matrix(toy.edges, toy.labels, len(toy.category_names))
    checks = validate_classification(gd, gu, toy.labels, toy.num_nodes, area_matrix=matrix)
    assert all(c["passed"] for c in checks), [c for c in checks if not c["passed"]]


def test_validation_detects_a_wrong_reference_matrix(toy, toy_tables):
    gd, gu = toy_tables
    matrix = category_citation_matrix(toy.edges, toy.labels, len(toy.category_names))
    matrix[0, 1] += 1  # corrupt the external reference
    checks = validate_classification(gd, gu, toy.labels, toy.num_nodes, area_matrix=matrix)
    failed = {c["check"] for c in checks if not c["passed"]}
    assert "G_d inter = off-diagonal sum of F1 area_citation_matrix.csv" in failed


def test_undirected_classification_rejects_inconsistent_inputs(toy):
    uedges, _ = undirected_edge_array(toy.edges, toy.num_nodes)
    with pytest.raises(ValueError):
        classify_undirected(uedges[1:], toy.edges, toy.labels, toy.num_nodes)
    with pytest.raises(ValueError):
        classify_undirected(uedges[:, ::-1], toy.edges, toy.labels, toy.num_nodes)


def test_classification_is_deterministic(toy, tmp_path):
    uedges, _ = undirected_edge_array(toy.edges, toy.num_nodes)
    first = classify_undirected(uedges, toy.edges, toy.labels, toy.num_nodes)
    second = classify_undirected(uedges.copy(), toy.edges.copy(), toy.labels, toy.num_nodes)
    assert first.content_sha256() == second.content_sha256()
    first.save(tmp_path / "a.npz")
    with np.load(tmp_path / "a.npz") as npz:
        assert np.array_equal(npz["is_inter"], first.is_inter)
        assert np.array_equal(npz["reciprocal"], first.reciprocal)
    second.is_inter = second.is_inter.copy()
    second.is_inter[0] = ~second.is_inter[0]
    assert first.content_sha256() != second.content_sha256()


@pytest.mark.parametrize("seed", [0, 3])
def test_invariants_on_synthetic(seed):
    data = generate_synthetic_citations(num_nodes=800, seed=seed)
    uedges, stats = undirected_edge_array(data.edges, data.num_nodes)
    gd = classify_directed(data.edges, data.labels, data.num_nodes)
    gu = classify_undirected(uedges, data.edges, data.labels, data.num_nodes)
    matrix = category_citation_matrix(data.edges, data.labels, len(data.category_names))
    checks = validate_classification(gd, gu, data.labels, data.num_nodes, area_matrix=matrix)
    assert all(c["passed"] for c in checks), [c for c in checks if not c["passed"]]
    dec = reciprocal_decomposition(gd, gu)
    assert dec["total_reciprocal_pairs"] == stats.num_reciprocal_pairs
    table = class_counts_table(gd, gu)
    assert set(table["graph"]) == {"G_d", "G_u"}
    per_cat = per_category_table(gd, gu, data.labels, data.category_names)
    assert per_cat["G_d_inter_out"].sum() == per_cat["G_d_inter_in"].sum() == gd.counts()["inter"]
    assert per_cat["G_u_inter_incident"].sum() == 2 * gu.counts()["inter"]
    assert per_cat["G_u_intra"].sum() == gu.counts()["intra"]
