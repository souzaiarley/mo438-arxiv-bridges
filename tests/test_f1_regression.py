"""Regression tests of the F1 metrics (real data).

Kinds of evidence are kept separate, as approved in the audit:

* ``TestIndependentRecomputation`` — values recomputed by mo438.verification
  (numpy/scipy; no igraph, no mo438.metrics code);
* ``TestF1CodeRerun`` — the F1 code itself re-executed in this environment (regression
  of the implementation/environment, NOT independent evidence);
* ``TestF1OutputConsistency`` — committed F1 outputs checked against each other and
  against the report; values are not recomputed;
* ``test_all_pairs_distances_independent`` (``slow``) — exact all-pairs distances
  recomputed with one scipy BFS per node (minutes; run with ``pytest -m slow``).
"""

from __future__ import annotations

import os

import igraph as ig
import numpy as np
import pandas as pd
import pytest
from conftest import AREA_GRAPHML, CHARACTERIZATION_DIR

from mo438.reference import F1_REPORTED, matches_report
from mo438.verification import (
    all_pairs_distance_histogram,
    check_distance_histogram,
    exact_diameter_ifub,
)

pytestmark = pytest.mark.realdata

UNDIRECTED_KEYS = ("num_nodes", "num_edges", "mean_degree", "density",
                   "average_local_clustering", "average_local_clustering_all_nodes",
                   "transitivity", "num_triangles", "num_connected_triples",
                   "num_connected_components", "num_nodes_degree_lt_2", "num_isolated_nodes",
                   "degree_max")
DIRECTED_KEYS = ("num_edges", "reciprocity", "in_degree_max", "out_degree_max",
                 "num_nodes_in_degree_0", "num_nodes_out_degree_0", "density",
                 "num_weakly_connected_components", "num_strongly_connected_components",
                 "num_nontrivial_sccs", "num_nodes_in_nontrivial_sccs", "largest_scc_size")


def _same(value, reference) -> bool:
    if isinstance(reference, (int, np.integer)) and not isinstance(reference, bool):
        return int(value) == int(reference)
    return bool(np.isclose(float(value), float(reference), rtol=1e-12, atol=0))


class TestIndependentRecomputation:
    @pytest.mark.parametrize("metric", [m for m in F1_REPORTED
                                        if m not in ("average_distance", "diameter")])
    def test_matches_f1_report(self, independent_undirected, metric):
        assert matches_report(metric, independent_undirected[metric])

    @pytest.mark.parametrize("metric", UNDIRECTED_KEYS)
    def test_matches_f1_summary_undirected(self, independent_undirected, f1_summary, metric):
        assert _same(independent_undirected[metric], f1_summary["undirected"][metric])

    @pytest.mark.parametrize("metric", DIRECTED_KEYS)
    def test_matches_f1_summary_directed(self, independent_directed, f1_summary, metric):
        assert _same(independent_directed[metric], f1_summary["directed"][metric])

    def test_conversion_counts(self, independent_directed, f1_summary):
        conv = f1_summary["conversion"]
        assert independent_directed["num_reciprocal_pairs"] == conv["num_reciprocal_pairs"]
        assert independent_directed["num_self_loops"] == conv["num_directed_self_loops"]
        assert independent_directed["num_duplicate_edges"] == conv["num_duplicate_directed_edges"]

    def test_exact_diameter_ifub(self, real_adjacency, f1_summary):
        result = exact_diameter_ifub(real_adjacency)
        assert result["diameter"] == F1_REPORTED["diameter"][0]
        assert result["diameter"] == f1_summary["undirected"]["distances"]["diameter"]

    def test_per_node_tables(self, independent_undirected):
        arrays = independent_undirected["arrays"]
        local = pd.read_csv(CHARACTERIZATION_DIR / "local_clustering.csv")
        assert np.array_equal(local["degree"].to_numpy(), arrays["degree"])
        assert np.array_equal(local["triangles"].to_numpy(), arrays["triangles"])
        # the CSV stores C_i with 6 significant digits
        assert np.abs(local["clustering"].to_numpy() - arrays["clustering"]).max() < 5e-6
        deg = pd.read_csv(CHARACTERIZATION_DIR / "degree_distribution_undirected.csv")
        k, c = np.unique(arrays["degree"], return_counts=True)
        assert np.array_equal(deg["degree"].to_numpy(), k)
        assert np.array_equal(deg["count"].to_numpy(), c)

    def test_directed_degree_and_scc_tables(self, real_instances):
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components

        edges, n = real_instances.gd_edges, real_instances.num_nodes
        for name, column in (("in_degree", edges[:, 1]), ("out_degree", edges[:, 0])):
            table = pd.read_csv(CHARACTERIZATION_DIR / f"{name}_distribution_directed.csv")
            k, c = np.unique(np.bincount(column, minlength=n), return_counts=True)
            assert np.array_equal(table[name].to_numpy(), k)
            assert np.array_equal(table["count"].to_numpy(), c)
        adj = csr_matrix((np.ones(len(edges), dtype=np.int8), (edges[:, 0], edges[:, 1])),
                         shape=(n, n))
        _, lab = connected_components(adj, directed=True, connection="strong")
        size, num = np.unique(np.bincount(lab), return_counts=True)
        table = pd.read_csv(CHARACTERIZATION_DIR / "strong_component_size_distribution_directed.csv")
        assert np.array_equal(table["size"].to_numpy(), size)
        assert np.array_equal(table["num_components"].to_numpy(), num)

    def test_area_matrix(self, real_instances):
        k = real_instances.num_categories
        labels, edges = real_instances.labels, real_instances.gd_edges
        recount = np.bincount(labels[edges[:, 0]] * k + labels[edges[:, 1]],
                              minlength=k * k).reshape(k, k)
        published = pd.read_csv(CHARACTERIZATION_DIR / "area_citation_matrix.csv", index_col=0)
        assert np.array_equal(published.to_numpy(), recount)
        assert published.columns.tolist() == real_instances.category_names


@pytest.fixture(scope="module")
def f1_metrics(real_instances):
    """The F1 metric code (igraph) re-executed on the instances."""
    from mo438.graph_build import build_directed_graph, build_undirected_graph
    from mo438.metrics import analyze_directed_graph, analyze_undirected_graph

    data = real_instances.to_citation_data()
    g_u, _, _ = build_undirected_graph(data)
    return (analyze_undirected_graph(g_u, {}).summary,
            analyze_directed_graph(build_directed_graph(data), {}).summary)


class TestF1CodeRerun:
    @pytest.mark.parametrize("metric", ("num_edges", "mean_degree", "density",
                                        "average_local_clustering", "transitivity",
                                        "num_triangles", "num_connected_components"))
    def test_undirected(self, f1_metrics, f1_summary, metric):
        assert _same(f1_metrics[0][metric], f1_summary["undirected"][metric])

    @pytest.mark.parametrize("metric", ("reciprocity", "num_strongly_connected_components",
                                        "largest_scc_size", "in_degree_max", "out_degree_max"))
    def test_directed(self, f1_metrics, f1_summary, metric):
        assert _same(f1_metrics[1][metric], f1_summary["directed"][metric])


class TestF1OutputConsistency:
    @pytest.mark.parametrize("metric", [m for m in F1_REPORTED if m != "diameter"
                                        and m != "average_distance"])
    def test_report_matches_summary_json(self, f1_summary, metric):
        assert matches_report(metric, f1_summary["undirected"][metric])

    def test_report_distances_match_summary_json(self, f1_summary):
        dist = f1_summary["undirected"]["distances"]
        assert dist["is_exact"] is True
        assert matches_report("average_distance", dist["average_distance"])
        assert matches_report("diameter", dist["diameter"])

    def test_distance_histogram_is_internally_consistent(self, f1_summary):
        hist = pd.read_csv(CHARACTERIZATION_DIR / "distance_distribution.csv")
        n = f1_summary["undirected"]["num_nodes"]
        m = f1_summary["undirected"]["num_edges"]
        checks = check_distance_histogram(hist["distance"].to_numpy(),
                                          hist["num_ordered_pairs"].to_numpy(), n, m)
        assert checks["sum_equals_n(n-1)"]
        assert checks["distance_1_equals_2m"]
        assert checks["distances_contiguous_from_1"]
        assert checks["average_distance"] == pytest.approx(
            f1_summary["undirected"]["distances"]["average_distance"], rel=1e-12)
        assert checks["max_distance"] == f1_summary["undirected"]["distances"]["diameter"]

    def test_area_graphml_weights_equal_matrix(self):
        if not AREA_GRAPHML.is_file() or \
                AREA_GRAPHML.read_bytes()[:64].startswith(b"version https://git-lfs"):
            pytest.skip("area GraphML not available (run `git lfs pull`)")
        graph = ig.Graph.Read_GraphML(str(AREA_GRAPHML))
        k = graph.vcount()
        weights = np.zeros((k, k), dtype=np.int64)
        for edge in graph.es:
            weights[edge.source, edge.target] = int(edge["weight"])
        published = pd.read_csv(CHARACTERIZATION_DIR / "area_citation_matrix.csv", index_col=0)
        assert np.array_equal(weights, published.to_numpy())


@pytest.mark.slow
def test_all_pairs_distances_independent(real_adjacency, f1_summary):
    """Exact all-pairs recomputation (one scipy BFS per node). Minutes; opt-in."""
    n_jobs = int(os.environ.get("MO438_TEST_JOBS", os.cpu_count() or 1))
    result = all_pairs_distance_histogram(real_adjacency, n_jobs=n_jobs)
    f1_hist = pd.read_csv(CHARACTERIZATION_DIR / "distance_distribution.csv")
    assert np.array_equal(result["histogram"][1:], f1_hist["num_ordered_pairs"].to_numpy())
    assert matches_report("average_distance", result["average_distance"])
    assert result["diameter"] == F1_REPORTED["diameter"][0]
