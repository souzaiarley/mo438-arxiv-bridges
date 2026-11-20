"""Objective 6: directed modularity (validation of leidenalg) and community tables."""

from __future__ import annotations

import igraph as ig
import leidenalg as la
import numpy as np
import pytest

from mo438 import communities as C


def _random_digraph(n: int, p: float, seed: int, reciprocal: float = 0.1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.random((n, n)) < p
    np.fill_diagonal(a, False)
    edges = np.argwhere(a)
    extra = edges[rng.random(len(edges)) < reciprocal][:, ::-1]   # add reverse arcs
    edges = np.unique(np.vstack([edges, extra]), axis=0)
    return edges


@pytest.mark.parametrize("seed", range(5))
def test_independent_formula_matches_igraph_and_leidenalg(seed):
    edges = _random_digraph(60, 0.06, seed)
    n = 60
    g = ig.Graph(n=n, edges=edges.tolist(), directed=True)
    rng = np.random.default_rng(seed)
    for k in (1, 3, 7):
        memb = rng.integers(k, size=n)
        q = C.directed_modularity(edges, n, memb)
        assert q == pytest.approx(g.modularity(memb.tolist(), directed=True), abs=1e-12)
        part = la.ModularityVertexPartition(g, initial_membership=memb.tolist())
        assert q == pytest.approx(part.quality(), abs=1e-12)
        assert C.undirected_modularity(edges, n, memb) == pytest.approx(
            g.modularity(memb.tolist(), directed=False), abs=1e-12)


def test_leidenalg_uses_the_directed_function():
    # partition where directed and undirected modularity differ clearly
    edges = np.array([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (4, 1), (2, 5),
                      (0, 4), (0, 5)])
    memb = np.array([0, 0, 0, 1, 1, 1])
    g = ig.Graph(n=6, edges=edges.tolist(), directed=True)
    q_dir = C.directed_modularity(edges, 6, memb)
    q_und = C.undirected_modularity(edges, 6, memb)
    assert abs(q_dir - q_und) > 1e-3
    q_leiden = la.ModularityVertexPartition(g, initial_membership=memb.tolist()).quality()
    assert q_leiden == pytest.approx(q_dir, abs=1e-12)
    assert q_leiden != pytest.approx(q_und, abs=1e-6)


def test_optimised_partition_quality_equals_formula_and_beats_trivial():
    edges = _random_digraph(120, 0.04, 9)
    runs = C.run_leiden(edges, 120, [1, 2], n_iterations=-1, n_jobs=1)
    for run in runs:
        q = C.directed_modularity(edges, 120, run["membership"])
        assert run["quality_leidenalg"] == pytest.approx(q, abs=1e-12)
        assert q > C.directed_modularity(edges, 120, np.zeros(120, int))
    again = C.run_leiden(edges, 120, [1], n_iterations=-1, n_jobs=1)
    assert np.array_equal(again[0]["membership"], runs[0]["membership"])   # reproducible


def test_planted_directed_communities_are_recovered():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(4), 50)
    same = groups[:, None] == groups[None, :]
    a = rng.random((200, 200)) < np.where(same, 0.15, 0.004)
    np.fill_diagonal(a, False)
    edges = np.argwhere(a)
    run = C.run_leiden(edges, 200, [3], n_iterations=-1, n_jobs=1)[0]
    agree = C.agreement(groups, run["membership"])
    assert agree["nmi"] > 0.95 and agree["ari"] > 0.95
    # the planted partition has high directed modularity too
    assert C.directed_modularity(edges, 200, groups) > 0.6


def test_agreement_extremes():
    a = np.array([0, 0, 1, 1, 2, 2])
    assert C.agreement(a, a) == {"nmi": pytest.approx(1.0), "ari": pytest.approx(1.0)}
    assert C.agreement(a, np.array([5, 5, 7, 7, 9, 9]))["ari"] == pytest.approx(1.0)


def test_tables_on_toy():
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
    memb = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 0])
    N = C.contingency(labels, memb, 3)
    assert N.tolist() == [[3, 1, 0], [0, 3, 0], [1, 0, 2]]
    cats = C.category_table(N, ["a", "b", "c"])
    assert cats["share_in_dominant_community"].tolist() == pytest.approx([0.75, 1.0, 2 / 3])
    assert cats["mutual_best_match"].tolist() == [True, True, True]
    assert cats.loc[1, "effective_num_communities"] == pytest.approx(1.0)
    comm = C.community_table(N, ["a", "b", "c"]).set_index("community")
    assert comm.loc[0, "purity"] == pytest.approx(0.75)
    assert comm.loc[0, "dominant_category"] == "a"
    P = C.comembership(N)
    assert np.allclose(P, P.T)
    assert P[1, 1] == pytest.approx(1.0)


def test_inter_edges_inside_communities():
    labels = np.array([0, 0, 1, 1])
    memb = np.array([0, 0, 0, 1])
    edges = np.array([(0, 1), (1, 2), (2, 3), (3, 0)])
    out = C.inter_edges_inside_communities(edges, labels, memb)
    # arcs: (0,1) intra/inside, (1,2) inter/inside, (2,3) intra/across, (3,0) inter/across
    assert out["arcs_inside_communities"] == pytest.approx(0.5)
    assert out["inter_area_arcs_inside_communities"] == pytest.approx(0.5)
    assert out["intra_area_arcs_inside_communities"] == pytest.approx(0.5)
    assert out["inside_community_arcs_that_are_inter_area"] == pytest.approx(0.5)


def test_size_summary():
    s = C.size_summary(np.array([0, 0, 0, 1, 1, 2]))
    assert s["num_communities"] == 3 and s["largest"] == 3 and s["singletons"] == 1
