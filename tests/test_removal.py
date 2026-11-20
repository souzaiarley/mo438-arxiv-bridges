"""Objectives 4-5: removal orders, union-find sweep and matched controls."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from mo438 import removal as R
from mo438 import removal_experiment as X
from mo438.data import generate_synthetic_citations
from mo438.edges import is_inter_area
from mo438.graph_build import undirected_edge_array


def _classes_from_nx(graph: nx.Graph, num_labels: int, seed: int) -> tuple[R.EdgeClasses,
                                                                          np.ndarray]:
    graph = nx.convert_node_labels_to_integers(graph)
    n = graph.number_of_nodes()
    edges = np.array(sorted((min(u, v), max(u, v)) for u, v in graph.edges()), dtype=np.int64)
    labels = np.random.default_rng(seed).integers(num_labels, size=n)
    strata, _ = R.edge_strata(edges, n)
    return R.EdgeClasses(edges, n, is_inter_area(edges, labels), strata), labels


def _synthetic_classes(seed: int) -> tuple[R.EdgeClasses, np.ndarray]:
    # includes isolated vertices and small detached components (not part of any edge stratum)
    data = generate_synthetic_citations(num_nodes=500, seed=seed)
    edges, _ = undirected_edge_array(data.edges, data.num_nodes)
    strata, _ = R.edge_strata(edges, data.num_nodes)
    return R.EdgeClasses(edges, data.num_nodes, is_inter_area(edges, data.labels), strata), \
        data.labels


GRAPHS = {
    "gnp": lambda: _classes_from_nx(nx.gnp_random_graph(150, 0.03, seed=1), 3, 0),
    "powerlaw_cluster": lambda: _classes_from_nx(nx.powerlaw_cluster_graph(300, 2, 0.3, seed=2),
                                                 4, 1),
    "tree": lambda: _classes_from_nx(nx.random_labeled_tree(120, seed=3)
                                     if hasattr(nx, "random_labeled_tree")
                                     else nx.random_tree(120, seed=3), 2, 2),
    "synthetic_citations": lambda: _synthetic_classes(0),
}


@pytest.mark.parametrize("graph", list(GRAPHS))
@pytest.mark.parametrize("strategy", list(R.STRATEGIES))
def test_union_find_matches_forward_at_every_k(graph, strategy):
    classes, _ = GRAPHS[graph]()
    k_max = classes.k_max
    assert k_max > 0
    ks = np.arange(k_max + 1)
    for replicate in (0, 1):
        order, _ = R.removal_order(strategy, replicate, classes, k_max, base_seed=7)
        fast = R.sweep_union_find(classes.edges, classes.num_nodes, order, ks)
        ref = R.metrics_forward(classes.edges, classes.num_nodes, order, ks)
        for name in R.METRICS:
            assert np.array_equal(fast[name], ref[name]), (graph, strategy, replicate, name)


def test_union_find_matches_networkx():
    classes, _ = GRAPHS["powerlaw_cluster"]()
    order, _ = R.removal_order("uniform", 3, classes, classes.k_max, base_seed=1)
    ks = np.array([0, 1, 7, 50, classes.k_max])
    fast = R.sweep_union_find(classes.edges, classes.num_nodes, order, ks)
    for i, k in enumerate(ks):
        g = nx.Graph()
        g.add_nodes_from(range(classes.num_nodes))
        keep = np.ones(classes.num_edges, dtype=bool)
        keep[order[:k]] = False
        g.add_edges_from(classes.edges[keep].tolist())
        sizes = sorted((len(c) for c in nx.connected_components(g)), reverse=True)
        assert fast["lcc_size"][i] == sizes[0]
        assert fast["second_lcc_size"][i] == (sizes[1] if len(sizes) > 1 else 0)
        assert fast["num_components"][i] == len(sizes)
        assert fast["num_isolated"][i] == sum(s == 1 for s in sizes)


def test_unsorted_and_partial_record_points():
    classes, _ = GRAPHS["gnp"]()
    order, _ = R.removal_order("intra", 0, classes, classes.k_max)
    ks = np.array([classes.k_max, 3, 0, 10])
    fast = R.sweep_union_find(classes.edges, classes.num_nodes, order, ks)
    ref = R.metrics_forward(classes.edges, classes.num_nodes, order, ks)
    for name in R.METRICS:
        assert np.array_equal(fast[name], ref[name])


def test_orders_are_reproducible_eligible_and_nested():
    classes, _ = GRAPHS["synthetic_citations"]()
    k = classes.k_max
    a, info_a = R.removal_order("inter", 4, classes, k)
    b, info_b = R.removal_order("inter", 4, classes, k)
    c, _ = R.removal_order("inter", 5, classes, k)
    assert np.array_equal(a, b) and info_a["order_sha256"] == info_b["order_sha256"]
    assert not np.array_equal(a, c)
    assert np.unique(a).size == a.size and classes.is_inter[a].all()
    intra, _ = R.removal_order("intra", 4, classes, k)
    assert (~classes.is_inter[intra]).all()
    uni, _ = R.removal_order("uniform", 4, classes, k)
    assert np.unique(uni).size == k
    # nested: for the same (strategy, replicate), a shorter order is a prefix of a longer one
    for strategy in R.STRATEGIES:
        full, _ = R.removal_order(strategy, 4, classes, k)
        half, _ = R.removal_order(strategy, 4, classes, k // 2)
        assert np.array_equal(half, full[: k // 2]), strategy


def test_strategy_streams_are_independent():
    seeds = {s: R.rng_for(s, 0).integers(1 << 62) for s in R.STRATEGIES}
    assert len(set(seeds.values())) == len(seeds)
    assert R.rng_for("inter", 0).integers(1 << 62) == seeds["inter"]


def test_grid():
    q = R.default_q_values(0.3457802261)
    ks = R.k_grid(q, 1_157_799, 400_344)
    assert ks[0] == 0 and ks[-1] == 400_344
    assert np.all(np.diff(ks) > 0)
    assert np.sum(ks <= 0.02 * 1_157_799) >= 20          # dense for small q
    tail = np.rint(np.asarray(R.TAIL_FRACTIONS) * 400_344)
    assert np.all(np.isin(tail.astype(int), ks) | (np.abs(tail[:, None] - ks).min(1) <= 1))


def test_within_stratum_rank_brute_force():
    seq = np.random.default_rng(0).integers(5, size=200)
    rank = R._within_stratum_rank(seq)
    expected = [int((seq[:i] == seq[i]).sum()) for i in range(seq.size)]
    assert rank.tolist() == expected


def test_matched_order_reproduces_stratum_composition_of_every_prefix():
    classes, _ = GRAPHS["synthetic_citations"]()
    rng = np.random.default_rng(3)
    target_idx = np.flatnonzero(classes.is_inter)
    target = target_idx[rng.permutation(target_idx.size)]
    control, feasible = R.matched_order(target, np.ones(classes.num_edges, bool),
                                        classes.strata, rng)
    assert feasible == target.size                       # variant A is always feasible
    assert np.unique(control).size == control.size       # no edge drawn twice
    s = int(classes.strata.max()) + 1
    for k in (1, 5, 37, target.size // 2, target.size):
        assert np.array_equal(np.bincount(classes.strata[target[:k]], minlength=s),
                              np.bincount(classes.strata[control[:k]], minlength=s))


def test_matched_order_variant_b_reports_infeasibility():
    strata = np.array([0, 0, 0, 1, 1, 1])
    target_mask = np.array([True, True, False, True, False, False])
    target = np.flatnonzero(target_mask)                 # strata 0, 0, 1
    control, feasible = R.matched_order(target, ~target_mask, strata, np.random.default_rng(0))
    # complement has one edge in stratum 0: the second stratum-0 target cannot be matched
    assert feasible == 1 and control.size == 1 and strata[control[0]] == 0
    assert not target_mask[control].any()


def test_degree_bins():
    assert R.degree_bin(np.array([1, 2, 3, 4, 7, 8, 13161])).tolist() == [0, 1, 1, 2, 2, 3, 13]
    with pytest.raises(ValueError):
        R.degree_bin(np.array([0]))


def test_summaries_and_comparisons():
    classes, _ = GRAPHS["powerlaw_cluster"]()
    ks = R.k_grid(np.array([0, 0.05, 0.1]), classes.num_edges, classes.k_max)
    runs, meta, checks = X.run_all(classes, ks, ("inter", "uniform"), 4, 5, 2)
    assert checks["all_match"] and checks["runs_checked"] == 8
    summary = X.summarize(runs)
    assert set(summary["replicates"]) == {4}
    assert (summary["S_p025"] <= summary["S_mean"] + 1e-12).all()
    assert len(meta) == 8 and meta["order_sha256"].nunique() == 8
    comp = X.compare(runs, ks, classes.num_edges)
    assert {"inter"} <= set(comp["A"])


def test_prob_less():
    assert X._prob_less(np.array([0.0, 0.0]), np.array([1.0, 1.0])) == 1.0
    assert X._prob_less(np.array([1.0]), np.array([1.0])) == 0.5


def test_endpoint_components_are_single_category():
    classes, labels = GRAPHS["synthetic_citations"]()
    names = [f"c{i}" for i in range(labels.max() + 1)]
    largest, cohesion = X.endpoint_components(classes, labels, names)
    assert largest.attrs["all_components_single_category"]
    assert (cohesion["largest_intra_component"] <= cohesion["num_papers"]).all()


# --------------------------------------------------------------------------- real data


@pytest.fixture(scope="module")
def real_classes(real_instances):
    edges, n = real_instances.gu_edges, real_instances.num_nodes
    strata, _ = R.edge_strata(edges, n)
    return R.EdgeClasses(edges, n, is_inter_area(edges, real_instances.labels), strata)


@pytest.mark.realdata
def test_real_domain(real_classes):
    assert real_classes.num_edges == 1_157_799
    assert real_classes.k_max == 400_344


@pytest.mark.realdata
def test_real_inter_endpoint_is_deterministic(real_classes):
    """All inter edges removed: the intra-only graph, identical for every seed."""
    ks = np.array([real_classes.k_max])
    values = []
    for r in (0, 17):
        order, _ = R.removal_order("inter", r, real_classes, real_classes.k_max)
        values.append(R.sweep_union_find(real_classes.edges, real_classes.num_nodes, order, ks))
    for name in R.METRICS:
        assert values[0][name][0] == values[1][name][0]
    # audit probe (2026-11-06): LCC 15.51 % of n, 24,577 components, 20,799 isolated
    assert values[0]["lcc_size"][0] / real_classes.num_nodes == pytest.approx(0.155117, abs=1e-6)
    assert values[0]["num_components"][0] == 24_577
    assert values[0]["num_isolated"][0] == 20_799
    assert values[0]["lcc_size"][0] <= 27_321                # bound: largest category (cs.CV)


@pytest.mark.realdata
def test_real_union_find_spot_check(real_classes):
    ks = np.array([0, 1_158, 57_890, 200_000, real_classes.k_max])
    for strategy in ("intra", "matched_inter"):
        order, _ = R.removal_order(strategy, 1, real_classes, real_classes.k_max)
        fast = R.sweep_union_find(real_classes.edges, real_classes.num_nodes, order, ks)
        ref = R.metrics_forward(real_classes.edges, real_classes.num_nodes, order, ks)
        for name in R.METRICS:
            assert np.array_equal(fast[name], ref[name]), (strategy, name)


def test_figures_smoke(tmp_path):
    import json

    from mo438.plots import removal_figures

    classes, labels = GRAPHS["powerlaw_cluster"]()
    ks = R.k_grid(R.default_q_values(classes.k_max / classes.num_edges), classes.num_edges,
                  classes.k_max)
    runs, _, checks = X.run_all(classes, ks, tuple(R.STRATEGIES), 3, 1, 1)
    assert checks["all_match"]
    X.summarize(runs).to_csv(tmp_path / "summary.csv", index=False)
    sizes = np.bincount(labels)
    (tmp_path / "experiment.json").write_text(json.dumps({"design": {
        "num_nodes": classes.num_nodes,
        "theoretical_bound": {"bound": sizes.max() / classes.num_nodes,
                              "largest_category": "c0"}}}))
    paths = removal_figures(tmp_path)
    assert len(paths) == 6 and all(p.stat().st_size > 0 for p in paths)


def test_degree_profile_columns():
    classes, _ = GRAPHS["gnp"]()
    table = X.degree_profile(classes)
    assert isinstance(table, pd.DataFrame)
    assert table.loc[table["class"] == "all", "num_edges"].item() == classes.num_edges
