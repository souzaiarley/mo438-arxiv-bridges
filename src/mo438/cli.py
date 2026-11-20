"""Command-line interface of the project: the ``mo438`` command.

Every command reads ``config/default.yaml`` (``--config`` selects another file) and writes
under ``outputs/`` (one subdirectory per objective)::

    mo438 download                 # raw ogbn-arxiv archive -> data/raw (needs internet)
    mo438 characterize             # objective 1 pipeline: GraphML, metrics, distances, figures
    mo438 verify-f1 [--all-pairs]  # independent re-check of the values published in the F1
    mo438 classify-edges           # intra/inter classification of G_d and G_u + checks
    mo438 clustering | flows | overlap | removal | removal-distances | communities | ordering
    mo438 <analysis>-figures       # figures redrawn from the saved CSV/JSON results
    mo438 integrate | review       # cross-objective discussion and scientific review
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import __version__
from .config import load_config
from .edges import (
    class_counts_table,
    classify_directed,
    classify_undirected,
    node_composition,
    per_category_table,
    reciprocal_decomposition,
    validate_classification,
)
from .instances import Instances, load_instances, sha256_file
from .reference import AUDIT_EDGE_CLASSES, F1_REPORTED, matches_report
from .runtime import LOGGER_NAME, provenance, setup_logging, write_json

LOGGER = logging.getLogger(LOGGER_NAME)


def _load(cfg: dict[str, Any], use_cache: bool = True) -> Instances:
    start = time.perf_counter()
    inst = load_instances(cfg["data"]["directed_graphml"], cfg["data"]["undirected_graphml"],
                          cache_dir=cfg["data"]["processed_dir"], use_cache=use_cache)
    LOGGER.info("Loaded instances: n=%d, m_d=%d, m_u=%d, %d categories (%.1f s)",
                inst.num_nodes, inst.gd_edges.shape[0], inst.gu_edges.shape[0],
                inst.num_categories, time.perf_counter() - start)
    return inst


def _read_area_matrix(metrics_dir: Path) -> np.ndarray | None:
    path = metrics_dir / "area_citation_matrix.csv"
    if not path.is_file():
        return None
    return pd.read_csv(path, index_col=0).to_numpy(dtype=np.int64)


# --------------------------------------------------------------------------- classify


def cmd_classify_edges(cfg: dict[str, Any], use_cache: bool) -> int:
    inst = _load(cfg, use_cache)
    n = inst.num_nodes
    gd = classify_directed(inst.gd_edges, inst.labels, n)
    gu = classify_undirected(inst.gu_edges, inst.gd_edges, inst.labels, n)
    area = _read_area_matrix(Path(cfg["outputs"]["metrics_dir"]))
    checks = validate_classification(gd, gu, inst.labels, n, area_matrix=area)
    dec = reciprocal_decomposition(gd, gu)
    for graph, table in (("G_d", gd), ("G_u", gu)):
        c = table.counts()
        for cls in ("intra", "inter"):
            checks.append({"check": f"{graph} {cls} = audit 2026-11-06", "observed": c[cls],
                           "expected": AUDIT_EDGE_CLASSES[graph][cls],
                           "passed": c[cls] == AUDIT_EDGE_CLASSES[graph][cls],
                           "reference": "mo438.reference.AUDIT_EDGE_CLASSES"})
    for cls in ("intra", "inter"):
        obs = dec[cls]["reciprocal_pairs_merged"]
        exp = AUDIT_EDGE_CLASSES["reciprocal_pairs"][cls]
        checks.append({"check": f"reciprocal {cls} pairs = audit 2026-11-06", "observed": obs,
                       "expected": exp, "passed": obs == exp,
                       "reference": "mo438.reference.AUDIT_EDGE_CLASSES"})

    out_dir = Path(cfg["outputs"]["dir"]) / "edge_classes"
    cache_dir = Path(cfg["data"]["processed_dir"])
    files: dict[str, Any] = {}
    for table in (gd, gu):
        path = cache_dir / f"edges_{table.graph.lower().replace('_', '')}.npz"
        content = table.content_sha256()
        table.save(path, {"inputs": inst.sources, "mo438_version": __version__,
                          "content_sha256": content})
        files[f"edge_table_{table.graph}"] = {
            "path": str(path), "content_sha256": content, "file_sha256": sha256_file(path),
            "versioned": False,
            "note": "content_sha256 hashes the arrays only; file_sha256 also depends on the "
                    "embedded metadata (input paths, versions) and on the npz encoding"}
    counts_csv = out_dir / "edge_classes.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    class_counts_table(gd, gu).to_csv(counts_csv, index=False, float_format="%.10g")
    by_cat_csv = out_dir / "edge_classes_by_category.csv"
    per_category_table(gd, gu, inst.labels, inst.category_names).to_csv(by_cat_csv, index=False)
    files["edge_classes_csv"] = str(counts_csv)
    files["edge_classes_by_category_csv"] = str(by_cat_csv)

    failed = [c for c in checks if not c["passed"]]
    payload = {
        "description": "Intra-/inter-area classification of the edges of G_d and G_u "
                       "(class = same / different primary category of the endpoints).",
        "counts": {"G_d": gd.counts(), "G_u": gu.counts()},
        "reciprocal_decomposition": dec,
        "G_u_node_composition": node_composition(gu, n),
        "largest_category": {
            "name": inst.category_names[int(np.bincount(inst.labels).argmax())],
            "num_papers": int(np.bincount(inst.labels).max()),
            "fraction_of_nodes": float(np.bincount(inst.labels).max() / n),
            "note": "upper bound of the LCC of G_u once every inter-area edge is removed",
        },
        "checks": checks,
        "all_checks_passed": not failed,
        "files": files,
        "inputs": inst.sources,
        "conversion": inst.conversion.to_dict(),
        "provenance": provenance(),
    }
    json_path = write_json(payload, out_dir / "edge_classification.json")
    for c in checks:
        LOGGER.info("[%s] %s: observed %s, expected %s", "ok" if c["passed"] else "FAIL",
                    c["check"], c["observed"], c["expected"])
    LOGGER.info("G_d: %s | G_u: %s", gd.counts(), gu.counts())
    LOGGER.info("Wrote %s, %s, %s", json_path, counts_csv, by_cat_csv)
    return 1 if failed else 0


# --------------------------------------------------------------------------- verify F1


def _row(metric: str, value: Any, kind: str, method: str, f1_summary: Any = None,
         passed: bool | None = None) -> dict[str, Any]:
    reported = F1_REPORTED.get(metric, (None,))[0]
    if passed is None:
        checks = []
        if metric in F1_REPORTED:
            checks.append(matches_report(metric, value))
        if f1_summary is not None:
            if isinstance(f1_summary, (int, np.integer)) and not isinstance(f1_summary, bool):
                checks.append(int(value) == int(f1_summary))
            else:
                checks.append(bool(np.isclose(float(value), float(f1_summary), rtol=1e-9,
                                              atol=0)))
        passed = all(checks) if checks else None
    return {"metric": metric, "value": value, "f1_reported": reported,
            "f1_summary_json": f1_summary, "kind": kind, "method": method, "passed": passed}


def cmd_verify_f1(cfg: dict[str, Any], all_pairs: bool, use_cache: bool) -> int:
    from . import verification as ver
    from .graph_build import build_directed_graph, build_undirected_graph
    from .metrics import analyze_directed_graph, analyze_undirected_graph

    inst = _load(cfg, use_cache)
    n = inst.num_nodes
    metrics_dir = Path(cfg["outputs"]["metrics_dir"])
    summary = json.loads((metrics_dir / "summary.json").read_text(encoding="utf-8"))
    su, sd = summary["undirected"], summary["directed"]
    rows: list[dict[str, Any]] = []
    indep = "recomputed_independently"
    np_method = "numpy/scipy (mo438.verification), no igraph, no mo438.metrics code"

    t0 = time.perf_counter()
    u = ver.undirected_summary(inst.gu_edges, n)
    LOGGER.info("Independent G_u metrics in %.1f s", time.perf_counter() - t0)
    for metric in ("num_nodes", "num_edges", "mean_degree", "density",
                   "average_local_clustering", "average_local_clustering_all_nodes",
                   "transitivity", "num_triangles", "num_connected_triples",
                   "num_connected_components", "num_nodes_degree_lt_2", "num_isolated_nodes"):
        rows.append(_row(metric, u[metric], indep, np_method, su.get(metric)))
    rows.append(_row("degree_max", u["degree_max"], indep, np_method, su["degree_max"]))

    d = ver.directed_summary(inst.gd_edges, n)
    for metric, key in (("G_d.num_edges", "num_edges"), ("G_d.reciprocity", "reciprocity"),
                        ("G_d.in_degree_max", "in_degree_max"),
                        ("G_d.out_degree_max", "out_degree_max"),
                        ("G_d.num_nodes_in_degree_0", "num_nodes_in_degree_0"),
                        ("G_d.num_nodes_out_degree_0", "num_nodes_out_degree_0"),
                        ("G_d.density", "density"),
                        ("G_d.num_weakly_connected_components", "num_weakly_connected_components"),
                        ("G_d.num_strongly_connected_components",
                         "num_strongly_connected_components"),
                        ("G_d.num_nontrivial_sccs", "num_nontrivial_sccs"),
                        ("G_d.num_nodes_in_nontrivial_sccs", "num_nodes_in_nontrivial_sccs"),
                        ("G_d.largest_scc_size", "largest_scc_size")):
        rows.append(_row(metric, d[key], indep, np_method, sd.get(key)))
    conv = summary["conversion"]
    rows.append(_row("conversion.reciprocal_pairs", d["num_reciprocal_pairs"], indep, np_method,
                     conv["num_reciprocal_pairs"]))
    rows.append(_row("conversion.self_loops", d["num_self_loops"], indep, np_method,
                     conv["num_directed_self_loops"]))
    rows.append(_row("conversion.duplicates", d["num_duplicate_edges"], indep, np_method,
                     conv["num_duplicate_directed_edges"]))

    # Per-node arrays vs the published F1 tables (consistency of the F1 outputs).
    arr = u["arrays"]
    local = pd.read_csv(metrics_dir / "local_clustering.csv")
    rows.append(_row("local_clustering.csv: degree per node",
                     bool(np.array_equal(local["degree"].to_numpy(), arr["degree"])), indep,
                     np_method, passed=bool(np.array_equal(local["degree"].to_numpy(),
                                                           arr["degree"]))))
    rows.append(_row("local_clustering.csv: triangles per node",
                     bool(np.array_equal(local["triangles"].to_numpy(), arr["triangles"])), indep,
                     np_method, passed=bool(np.array_equal(local["triangles"].to_numpy(),
                                                           arr["triangles"]))))
    diff = float(np.abs(local["clustering"].to_numpy() - arr["clustering"]).max())
    rows.append(_row("local_clustering.csv: max |C_i difference| (CSV has 6 significant digits)",
                     diff, indep, np_method, passed=diff < 5e-6))
    deg_tab = pd.read_csv(metrics_dir / "degree_distribution_undirected.csv")
    k, c = np.unique(arr["degree"], return_counts=True)
    ok = bool(np.array_equal(deg_tab["degree"].to_numpy(), k)
              and np.array_equal(deg_tab["count"].to_numpy(), c))
    rows.append(_row("degree_distribution_undirected.csv", ok, indep, np_method, passed=ok))
    area = _read_area_matrix(metrics_dir)
    kcat = inst.num_categories
    recount = np.bincount(inst.labels[inst.gd_edges[:, 0]] * kcat + inst.labels[inst.gd_edges[:, 1]],
                          minlength=kcat * kcat).reshape(kcat, kcat)
    ok = area is not None and bool(np.array_equal(area, recount))
    rows.append(_row("area_citation_matrix.csv", ok, indep, np_method, passed=ok))
    ok = summary["dataset"]["category_names"] == inst.category_names
    rows.append(_row("category names / label order", ok, indep, "GraphML vs summary.json",
                     passed=ok))

    # Distances.
    hist = pd.read_csv(metrics_dir / "distance_distribution.csv")
    hc = ver.check_distance_histogram(hist["distance"].to_numpy(),
                                      hist["num_ordered_pairs"].to_numpy(), n,
                                      inst.gu_edges.shape[0])
    ok = hc["sum_equals_n(n-1)"] and hc["distance_1_equals_2m"] and \
        hc["distances_contiguous_from_1"]
    rows.append(_row("distance_distribution.csv internal consistency", ok, "consistency",
                     "sum = n(n-1), bin 1 = 2m, contiguous bins", passed=ok))
    rows.append(_row("average_distance", hc["average_distance"], "consistency",
                     "mean of the F1 histogram (not recomputed)",
                     su["distances"]["average_distance"]))
    adj = ver.undirected_adjacency(inst.gu_edges, n)
    t0 = time.perf_counter()
    ifub = ver.exact_diameter_ifub(adj, progress=LOGGER.info)
    LOGGER.info("iFUB exact diameter %d with %d BFS in %.1f s", ifub["diameter"],
                ifub["num_bfs"], time.perf_counter() - t0)
    rows.append(_row("diameter", ifub["diameter"], indep,
                     f"iFUB, {ifub['num_bfs']} scipy BFS, root {ifub['root']} "
                     f"(ecc {ifub['root_eccentricity']}), certificate pair "
                     f"{ifub['witness_pair']}", su["distances"]["diameter"]))
    apsp = None
    if all_pairs:
        n_jobs = int(cfg["runtime"]["n_jobs_resolved"])
        LOGGER.info("All-pairs BFS with %d processes (this takes minutes)", n_jobs)
        apsp = ver.all_pairs_distance_histogram(adj, n_jobs=n_jobs, progress=LOGGER.info)
        f1_counts = hist["num_ordered_pairs"].to_numpy()
        same = bool(np.array_equal(apsp["histogram"][1:], f1_counts))
        rows.append(_row("average_distance", apsp["average_distance"], indep,
                         apsp["method"], su["distances"]["average_distance"]))
        rows.append(_row("distance_distribution.csv equals independent histogram (all bins)",
                         same, indep, apsp["method"], passed=same))
        rows.append(_row("diameter", apsp["diameter"], indep, apsp["method"],
                         su["distances"]["diameter"]))

    # Re-run of the F1 code itself (igraph) in this environment: regression, not independent.
    data = inst.to_citation_data()
    g_u, _, _ = build_undirected_graph(data)
    und = analyze_undirected_graph(g_u, {})
    for metric in ("mean_degree", "density", "average_local_clustering", "transitivity",
                   "num_triangles", "num_connected_components"):
        rows.append(_row(metric, und.summary[metric], "f1_code_rerun",
                         "mo438.metrics (igraph) re-run in the uv environment",
                         su.get(metric)))
    dirm = analyze_directed_graph(build_directed_graph(data), {})
    for key in ("reciprocity", "num_strongly_connected_components", "largest_scc_size"):
        rows.append(_row(f"G_d.{key}", dirm.summary[key], "f1_code_rerun",
                         "mo438.metrics (igraph) re-run in the uv environment", sd.get(key)))

    orientation = ver.orientation_by_year(inst.gd_edges, inst.years)
    out_dir = Path(cfg["outputs"]["metrics_dir"]) / "verification"
    table = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "f1_verification.csv", index=False)
    payload: dict[str, Any] = {
        "description": "Re-check of the F1 metrics. kind = recomputed_independently "
                       "(numpy/scipy code path), consistency (F1 output checked, value not "
                       "recomputed) or f1_code_rerun (F1 code re-executed; regression only).",
        "rows": rows,
        "all_passed": bool(all(r["passed"] is not False for r in rows)),
        "diameter_ifub": ifub,
        "distance_histogram_checks": hc,
        "orientation_by_year": orientation,
        "inputs": inst.sources,
        "provenance": provenance(),
    }
    if apsp is not None:
        pd.DataFrame({"distance": np.arange(1, apsp["histogram"].size),
                      "num_ordered_pairs": apsp["histogram"][1:]}).to_csv(
            out_dir / "f1_distance_histogram_independent.csv", index=False)
        payload["all_pairs"] = {k: v for k, v in apsp.items() if k != "histogram"}
    write_json(payload, out_dir / "f1_verification.json")
    for r in rows:
        LOGGER.info("[%s] %-60s %s (%s)", {True: "ok", False: "FAIL", None: "--"}[r["passed"]],
                    r["metric"], r["value"], r["kind"])
    LOGGER.info("Orientation by year: %s", orientation)
    return 0 if payload["all_passed"] else 1


# --------------------------------------------------------------------------- removal


def cmd_removal(cfg: dict[str, Any], use_cache: bool, replicates: int | None) -> int:
    """Objectives 4-5: progressive edge removal on G_u (all strategies, all seeds)."""
    from . import removal as R
    from . import removal_experiment as X
    from .edges import is_inter_area

    rcfg = cfg["removal"]
    replicates = int(replicates or rcfg["replicates"])
    strategies = tuple(rcfg["strategies"])
    base_seed = int(cfg["project"]["seed"])
    t_total = time.perf_counter()
    inst = _load(cfg, use_cache)
    edges, n = inst.gu_edges, inst.num_nodes
    strata, pairs = R.edge_strata(edges, n)
    classes = R.EdgeClasses(edges, n, is_inter_area(edges, inst.labels), strata)
    m, k_max = classes.num_edges, classes.k_max
    q_values = R.default_q_values(k_max / m)
    ks = R.k_grid(q_values, m, k_max)
    LOGGER.info("G_u: n=%d m=%d | inter=%d intra=%d | k_max=%d (q_max=%.6f) | %d grid points",
                n, m, int(classes.is_inter.sum()), int((~classes.is_inter).sum()), k_max,
                k_max / m, ks.size)

    t0 = time.perf_counter()
    runs, meta, validation = X.run_all(classes, ks, strategies, replicates, base_seed,
                                       int(rcfg["validate_points"]), progress=LOGGER.info)
    t_runs = time.perf_counter() - t0
    LOGGER.info("Validation: %d grid points compared in %d runs, %d mismatches",
                validation["points_compared"], validation["runs_checked"],
                validation["mismatches"])
    summary = X.summarize(runs)
    comparisons = pd.concat([X.compare(runs, ks, m, metric) for metric in
                             ("S", "num_isolated", "num_components", "second_lcc_size")],
                            ignore_index=True)
    t0 = time.perf_counter()
    feasibility = R.matching_feasibility(classes, k_max, int(rcfg["feasibility_replicates"]),
                                         base_seed)
    t_feas = time.perf_counter() - t0
    largest, cohesion = X.endpoint_components(classes, inst.labels, inst.category_names)

    out = Path(cfg["outputs"]["dir"]) / "obj4_5_removal"
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "runs_csv": out / "runs.csv",
        "summary_csv": out / "summary.csv",
        "comparisons_csv": out / "comparisons.csv",
        "runs_metadata_csv": out / "runs_metadata.csv",
        "degree_profile_csv": out / "edge_class_degree_profile.csv",
        "strata_csv": out / "degree_strata.csv",
        "endpoint_largest_components_csv": out / "endpoint_all_inter_removed_components.csv",
        "endpoint_category_cohesion_csv": out / "endpoint_all_inter_removed_by_category.csv",
    }
    runs.to_csv(files["runs_csv"], index=False, float_format="%.8g")
    summary.to_csv(files["summary_csv"], index=False, float_format="%.8g")
    comparisons.to_csv(files["comparisons_csv"], index=False, float_format="%.8g")
    meta.to_csv(files["runs_metadata_csv"], index=False)
    X.degree_profile(classes).to_csv(files["degree_profile_csv"], index=False,
                                     float_format="%.6g")
    X.strata_table(classes, pairs).to_csv(files["strata_csv"], index=False, float_format="%.6g")
    largest.to_csv(files["endpoint_largest_components_csv"], index=False, float_format="%.6g")
    cohesion.to_csv(files["endpoint_category_cohesion_csv"], index=False, float_format="%.6g")
    write_json(feasibility, out / "matched_control_feasibility.json")
    files["feasibility_json"] = out / "matched_control_feasibility.json"

    payload = {
        "description": "Objectives 4-5: progressive random removal of inter-area, intra-area "
                       "and uniformly chosen edges of G_u, plus degree-matched controls.",
        "design": {
            "graph": "G_u (simple undirected)", "num_nodes": n, "num_edges": m,
            "inter_edges": int(classes.is_inter.sum()),
            "intra_edges": int((~classes.is_inter).sum()),
            "q_definition": "q = k / m, k = absolute number of removed edges (same k for all "
                            "strategies)",
            "k_max": k_max, "q_max": k_max / m,
            "grid_k": ks, "grid_q": ks / m,
            "strategies": {s: R.STRATEGIES[s][1] for s in strategies},
            "replicates_per_strategy": replicates,
            "seeds": f"numpy.random.SeedSequence([{base_seed}, strategy_code, replicate]); "
                     f"codes: { {s: R.STRATEGIES[s][0] for s in strategies} }",
            "nested_removals": "the set removed at k is the first k edges of one random "
                               "permutation of the eligible edges",
            "algorithm": "reverse union-find sweep (Newman-Ziff): components of the graph at "
                         "k_max with scipy, then edges added back in reverse order",
            "degree_strata": "(floor(log2 min(k_u,k_v)), floor(log2 max(k_u,k_v))) with degrees "
                             f"of the intact G_u; {pairs.shape[0]} non-empty strata",
            "matched_control": "variant A: for each target edge, a random unused edge of the "
                               "whole graph from the same stratum (prefix-wise identical "
                               "stratum composition)",
            "theoretical_bound": {
                "statement": "with all inter-area edges removed every component lies inside one "
                             "category, so S(k_max) <= largest category / n for 'inter'",
                "largest_category": inst.category_names[int(np.bincount(inst.labels).argmax())],
                "bound": float(np.bincount(inst.labels).max() / n),
            },
        },
        "validation": validation,
        "endpoint_all_components_single_category":
            largest.attrs.get("all_components_single_category"),
        "matched_control_feasibility_summary": {
            t: {k: v for k, v in feasibility[t].items() if k != "per_stratum"}
            for t in ("inter", "intra")},
        "timings_seconds": {"runs_including_validation": round(t_runs, 2),
                            "sweeps_only": round(float(meta["seconds_sweep"].sum()), 2),
                            "orders_only": round(float(meta["seconds_order"].sum()), 2),
                            "validation_only": round(float(meta["seconds_validation"].sum()), 2),
                            "feasibility": round(t_feas, 2),
                            "total": round(time.perf_counter() - t_total, 2)},
        "files": {k: str(v) for k, v in files.items()},
        "inputs": inst.sources,
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "provenance": provenance(),
    }
    write_json(payload, out / "experiment.json")
    LOGGER.info("Wrote %s (total %.1f s)", out, time.perf_counter() - t_total)
    return 0 if validation["all_match"] else 1


def cmd_removal_distances(cfg: dict[str, Any], use_cache: bool, replicates: int,
                          num_sources: int) -> int:
    """Complementary: sampled average distance in the LCC at selected removal states."""
    from . import lcc_distances as D
    from . import removal as R
    from .edges import is_inter_area

    base_seed = int(cfg["project"]["seed"])
    n_jobs = int(cfg["runtime"]["n_jobs_resolved"])
    inst = _load(cfg, use_cache)
    edges, n = inst.gu_edges, inst.num_nodes
    strata, _ = R.edge_strata(edges, n)
    classes = R.EdgeClasses(edges, n, is_inter_area(edges, inst.labels), strata)
    m, k_max = classes.num_edges, classes.k_max
    q_points = [0.05, 0.10, 0.20, 0.30, 0.33, 0.99 * k_max / m, 0.999 * k_max / m]
    k_points = sorted({round(q * m) for q in q_points})
    tasks = [(s, r, k, k_max, num_sources, base_seed)
             for s in ("inter", "intra", "uniform") for r in range(replicates) for k in k_points]
    # validation of the estimator on the intact graph (exact value known): 20 samples
    validation_tasks = [("uniform", r, 0, k_max, num_sources, base_seed) for r in range(20)]
    t0 = time.perf_counter()
    rows = D.run_states(classes, validation_tasks + tasks, n_jobs)
    elapsed = time.perf_counter() - t0
    table = pd.DataFrame(rows)
    intact = table[table["k"] == 0].copy()
    states = table[table["k"] > 0].copy()
    summary_path = Path(cfg["outputs"]["metrics_dir"]) / "summary.json"
    exact = json.loads(summary_path.read_text())["undirected"]["distances"]["average_distance"]
    covered = ((intact["ci95_low"] <= exact) & (exact <= intact["ci95_high"]))
    validation = {
        "exact_intact_average_distance": exact,
        "samples": len(intact), "num_sources": num_sources,
        "mean_of_estimates": float(intact["mean_distance"].mean()),
        "bias": float(intact["mean_distance"].mean() - exact),
        "sd_of_estimates": float(intact["mean_distance"].std(ddof=1)),
        "mean_reported_se": float(intact["se"].mean()),
        "ci95_coverage": float(covered.mean()),
        "max_abs_error": float((intact["mean_distance"] - exact).abs().max()),
    }
    group = states.groupby(["strategy", "k"], sort=True)
    summary = group.agg(
        q=("q", "first"), replicates=("replicate", "count"),
        lcc_size_mean=("lcc_size", "mean"),
        mean_distance_mean=("mean_distance", "mean"),
        mean_distance_sd=("mean_distance", "std"),
        mean_distance_p025=("mean_distance", lambda x: x.quantile(0.025)),
        mean_distance_p975=("mean_distance", lambda x: x.quantile(0.975)),
        within_run_se_mean=("se", "mean"),
        effective_diameter_90_median=("effective_diameter_90", "median"),
        max_sampled_eccentricity_max=("max_sampled_eccentricity", "max"),
    ).reset_index()
    summary["ci95_of_mean_halfwidth"] = 1.96 * summary["mean_distance_sd"] / np.sqrt(
        summary["replicates"])
    out = Path(cfg["outputs"]["dir"]) / "obj4_5_removal"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "lcc_distances_runs.csv", index=False, float_format="%.6g")
    summary.to_csv(out / "lcc_distances_summary.csv", index=False, float_format="%.6g")
    write_json({
        "description": "Sampled average shortest-path length inside the LCC at selected removal "
                       "states (complementary to S(q)); same removal orders/seeds as the main "
                       "experiment.",
        "estimator": "mean over uniformly sampled LCC sources of the per-source mean distance "
                     "(unbiased for the ordered-pair mean); SE with finite-population "
                     "correction; CI = estimate +/- 1.96 SE",
        "q_points": [k / m for k in k_points], "k_points": k_points,
        "replicates_per_strategy": replicates, "num_sources": num_sources,
        "seeds": f"sources: SeedSequence([{base_seed}, {D.DISTANCE_SEED_CODE}, strategy_code, "
                 "replicate, k]); removal orders: as in the main experiment",
        "validation_on_intact_graph": validation,
        "seconds": round(elapsed, 1), "n_jobs": n_jobs,
        "provenance": provenance(),
    }, out / "lcc_distances.json")
    LOGGER.info("Estimator validation on the intact graph: %s", validation)
    LOGGER.info("Wrote lcc_distances_* (%.0f s)", elapsed)
    return 0


def cmd_flows(cfg: dict[str, Any], use_cache: bool, replicates: int) -> int:
    """Objective 2: directed flows between categories, null models, bootstrap CIs."""
    from . import flows as F

    base_seed = int(cfg["project"]["seed"])
    t0 = time.perf_counter()
    inst = _load(cfg, use_cache)
    edges, labels, years = inst.gd_edges, inst.labels, inst.years
    names, k = inst.category_names, inst.num_categories
    stratum = F.year_pair_stratum(edges, years)
    stats = F.flow_stats(edges, labels, k, stratum, years)
    f1_matrix = _read_area_matrix(Path(cfg["outputs"]["metrics_dir"]))
    matches_f1 = f1_matrix is not None and bool(np.array_equal(stats.M, f1_matrix))
    t1 = time.perf_counter()
    boot = F.bootstrap(edges, labels, k, stratum, years, inst.num_nodes, replicates,
                       base_seed)
    t_boot = time.perf_counter() - t1
    cats = F.category_table(stats, boot, labels, names, years)
    pairs = F.pair_table(stats, boot, names)

    out = Path(cfg["outputs"]["dir"]) / "obj2_flows"
    out.mkdir(parents=True, exist_ok=True)
    cats.to_csv(out / "category_flows.csv", index=False, float_format="%.6g")
    pairs.to_csv(out / "pair_flows.csv", index=False, float_format="%.6g")
    F.oriented_top_pairs(pairs, "net_ij", 20).to_csv(out / "top_pairs_by_net_flow.csv",
                                                     index=False, float_format="%.6g")
    F.oriented_top_pairs(pairs, "asymmetry_ij", 20, min_total=200).to_csv(
        out / "top_pairs_by_asymmetry_min200.csv", index=False, float_format="%.6g")
    strong = pairs[pairs["direction_beyond_time_null"] & (pairs["total"] >= 200)]
    F.oriented_top_pairs(strong, "lor_time", 20).to_csv(
        out / "top_pairs_beyond_time_null_min200.csv", index=False, float_format="%.6g")
    for name, mat in (("observed", stats.M), ("expected_strength", stats.E_strength),
                      ("expected_time", stats.E_time)):
        pd.DataFrame(mat, index=names, columns=names).rename_axis("citing\\cited").to_csv(
            out / f"matrix_{name}.csv", float_format="%.6g")
    for name, E in (("strength", stats.E_strength), ("time", stats.E_time)):
        with np.errstate(divide="ignore", invalid="ignore"):
            oe = np.log2(stats.M / E)
        oe[stats.M == 0] = np.nan
        pd.DataFrame(oe, index=names, columns=names).rename_axis("citing\\cited").to_csv(
            out / f"matrix_log2_OE_{name}.csv", float_format="%.6g")

    sig = {}
    for min_total in (100, 1000):
        d = pairs[pairs["total"] >= min_total]
        raw = (d["asymmetry_ij_ci95_low"] > 0) | (d["asymmetry_ij_ci95_high"] < 0)
        sig[f"pairs_total_ge_{min_total}"] = {
            "pairs": len(d),
            "direction_differs_from_50_50": int(raw.sum()),
            "direction_beyond_strength_null": int(d["direction_beyond_strength_null"].sum()),
            "direction_beyond_time_null": int(d["direction_beyond_time_null"].sum()),
            "citations_in_these_pairs": int(d["total"].sum()),
        }
    inter_total = int(stats.M.sum() - np.trace(stats.M))
    top10_share = float(np.sort(pairs["total"].to_numpy())[::-1][:10].sum() / inter_total)
    correlations = {
        "nnf_vs_year_mean": F.spearman(cats["nnf"].to_numpy(), cats["year_mean"].to_numpy()),
        "nnf_vs_nnf_expected_time": F.spearman(cats["nnf"].to_numpy(),
                                               cats["nnf_expected_time"].to_numpy()),
        "nnf_excess_time_vs_year_mean": F.spearman(cats["nnf_excess_time"].to_numpy(),
                                                   cats["year_mean"].to_numpy()),
        "nnf_vs_num_papers": F.spearman(cats["nnf"].to_numpy(), cats["num_papers"].to_numpy()),
        "nnf_vs_nnf_age": F.spearman(cats["nnf"].to_numpy(), cats["nnf_age"].to_numpy()),
        "nnf_minus_nnf_age_vs_year_mean": F.spearman(cats["nnf_minus_nnf_age"].to_numpy(),
                                                     cats["year_mean"].to_numpy()),
        "nnf_vs_inter_out_per_paper": F.spearman(cats["nnf"].to_numpy(),
                                                 cats["inter_out_per_paper"].to_numpy()),
    }
    write_json({
        "description": "Objective 2: directed citation flows between the 40 primary categories "
                       "of G_d (M[i,j] = citations from area i to area j).",
        "matrix_matches_f1_area_citation_matrix": matches_f1,
        "totals": {"citations": int(stats.M.sum()), "intra": int(np.trace(stats.M)),
                   "inter": inter_total, "pairs_with_citations": len(pairs),
                   "share_of_inter_citations_in_top10_pairs": top10_share},
        "null_models": {
            "strength": "E_ij = out_i in_j / M_tot (area citing and cited volumes kept)",
            "time": "same, stratified by (citing year, cited year) of each citation",
            "age": "inter-area citations only: E_ij = sum_y O[i,y] P[j,y]/(P[.,y]-P[i,y]), "
                   "j != i; each area keeps its inter-area citing volume and cited-age "
                   "profile, the cited area is drawn among the other areas by its share of "
                   "the papers of each year",
            "note": "strength and time nulls preserve each area's made/received volumes and "
                    "hence its net inter flow; they are used for pairwise direction only",
            "pair_test": "LOR = logit((M_ij+.5)/(T+1)) - logit((T e_ij+.5)/(T+1)), "
                         "e_ij = E_ij/(E_ij+E_ji)",
        },
        "bootstrap": {"type": "cluster bootstrap over citing papers (multinomial weights)",
                      "replicates": replicates,
                      "seed": f"SeedSequence([{base_seed}, 2, 0])",
                      "intervals": "percentile 2.5-97.5 %", "seconds": round(t_boot, 1)},
        "pairs_with_significant_direction": sig,
        "spearman": correlations,
        "seconds_total": round(time.perf_counter() - t0, 1),
        "inputs": inst.sources, "provenance": provenance(),
    }, out / "flows.json")
    LOGGER.info("Flows: matrix matches F1 = %s | %s", matches_f1, sig)
    LOGGER.info("Spearman: %s", {k: round(v["rho"], 3) for k, v in correlations.items()})
    return 0 if matches_f1 else 1


def cmd_overlap(cfg: dict[str, Any], use_cache: bool, replicates: int) -> int:
    """Objective 3: neighbourhood overlap and bridges of intra- vs inter-area edges (G_u)."""
    from . import overlap as O
    from . import removal as R
    from .edges import is_inter_area

    base_seed = int(cfg["project"]["seed"])
    t0 = time.perf_counter()
    inst = _load(cfg, use_cache)
    edges, n, labels = inst.gu_edges, inst.num_nodes, inst.labels
    inter = is_inter_area(edges, labels)
    meas = O.neighbourhood_overlap(edges, n)
    values = meas["overlap"]
    undefined = np.isnan(values)
    kmin = np.minimum(meas["k_u"], meas["k_v"])
    kmax = np.maximum(meas["k_u"], meas["k_v"])
    strata, pairs = R.edge_strata(edges, n)
    t_bridges = time.perf_counter()
    is_bridge, detached = O.bridges(edges, n)
    t_bridges = time.perf_counter() - t_bridges
    # citing paper of each G_u edge (first G_d citation of the pair), for the cluster bootstrap
    gd = inst.gd_edges
    gd_keys = np.minimum(gd[:, 0], gd[:, 1]) * n + np.maximum(gd[:, 0], gd[:, 1])
    uniq, first = np.unique(gd_keys, return_index=True)
    if not np.array_equal(uniq, edges[:, 0] * n + edges[:, 1]):
        raise RuntimeError("G_u edges are not aligned with the G_d pairs")
    citing = gd[first, 0]

    ok = ~undefined
    v, x_inter, s_ok = values[ok], inter[ok], strata[ok]
    vi = O.ValueIndex(v)
    point = O.class_effects(vi, v, x_inter, s_ok)
    t_boot = time.perf_counter()
    boot_iid = O.bootstrap_effects(v, x_inter, s_ok, None, replicates, base_seed + 31)
    boot_cl = O.bootstrap_effects(v, x_inter, s_ok, citing[ok], replicates, base_seed + 32)
    t_boot = time.perf_counter() - t_boot
    effects = pd.DataFrame({"statistic": list(point.keys()),
                            "estimate": list(point.values())})
    for tag, b in (("iid_edges", boot_iid), ("cluster_citing_paper", boot_cl)):
        effects[f"ci95_low_{tag}"] = [b[s].quantile(0.025) for s in point]
        effects[f"ci95_high_{tag}"] = [b[s].quantile(0.975) for s in point]
        effects[f"se_{tag}"] = [b[s].std(ddof=1) for s in point]
    # sensitivity: drop edges touching the 100 highest-degree vertices
    deg = np.bincount(edges.ravel(), minlength=n)
    hubs = np.argsort(-deg)[:100]
    no_hub = ok & ~np.isin(edges, hubs).any(axis=1)
    sens = O.class_effects(O.ValueIndex(values[no_hub]), values[no_hub], inter[no_hub],
                           strata[no_hub])

    extra = {"mean_common_neighbours": meas["common"].astype(float),
             "pendant_fraction": (kmin == 1).astype(float),
             "bridge_fraction": is_bridge.astype(float),
             "mean_k_min": kmin.astype(float), "mean_k_max": kmax.astype(float)}
    summary_rows = []
    for cls, mask in (("inter", inter), ("intra", ~inter), ("all", np.ones_like(inter))):
        sel = mask & ok
        row = {"class": cls, **O.describe(values[sel], {k: a[sel] for k, a in extra.items()})}
        row["undefined_overlap_edges"] = int((mask & undefined).sum())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    strat = O.stratum_table(v, x_inter, s_ok, pairs)
    pooled = O.pooled_stratified_delta(strat)
    by_deg = pd.concat([O.by_degree_bin(v, x_inter, kmin[ok], "k_min"),
                        O.by_degree_bin(v, x_inter, kmax[ok], "k_max")], ignore_index=True)
    by_cat = O.by_category(values, inter, edges, labels, inst.category_names, is_bridge)
    bridges_tab = O.bridge_summary(is_bridge, detached, inter, kmin)

    out = Path(cfg["outputs"]["dir"]) / "obj3_overlap"
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "overlap_by_class.csv", index=False, float_format="%.6g")
    effects.to_csv(out / "overlap_effects.csv", index=False, float_format="%.6g")
    strat.to_csv(out / "overlap_by_degree_stratum.csv", index=False, float_format="%.6g")
    by_deg.to_csv(out / "overlap_by_degree_bin.csv", index=False, float_format="%.6g")
    by_cat.to_csv(out / "overlap_by_category.csv", index=False, float_format="%.6g")
    bridges_tab.to_csv(out / "bridges_by_class.csv", index=False, float_format="%.6g")
    boot_iid.to_csv(out / "bootstrap_iid_edges.csv", index=False, float_format="%.6g")
    boot_cl.to_csv(out / "bootstrap_cluster_citing_paper.csv", index=False, float_format="%.6g")
    cache = Path(cfg["data"]["processed_dir"]) / "overlap_edges.npz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, u=edges[:, 0], v=edges[:, 1], is_inter=inter,
                        common=meas["common"], k_u=meas["k_u"], k_v=meas["k_v"],
                        overlap=values, stratum=strata, is_bridge=is_bridge,
                        detached=detached, citing_paper=citing)
    write_json({
        "description": "Objective 3: neighbourhood overlap O(u,v) = |N(u)∩N(v)| / |N(u)∪N(v)| "
                       "(N excluding the other endpoint) of the edges of G_u, by class.",
        "definition": "O = c / (k_u + k_v - 2 - c), c = common neighbours; undefined (NaN, "
                      "excluded) iff k_u = k_v = 1",
        "undefined_overlap_edges": int(undefined.sum()),
        "point_effects": point,
        "pooled_stratified_cliffs_delta": pooled,
        "sensitivity_without_top100_hub_edges": {
            "edges_kept": int(no_hub.sum()), "edges_dropped": int((ok & ~no_hub).sum()),
            "effects": sens},
        "bootstrap": {"replicates": replicates,
                      "schemes": {"iid_edges": f"multinomial over edges, seed {base_seed + 31}",
                                  "cluster_citing_paper":
                                      "multinomial over citing papers (each G_u edge "
                                      "assigned to the citing paper of its first G_d "
                                      f"citation), seed {base_seed + 32}"},
                      "seconds": round(t_boot, 1),
                      "note": "neither scheme captures the dependence among edges sharing a "
                              "cited hub; see the hub sensitivity analysis"},
        "bridges_seconds": round(t_bridges, 1),
        "seconds_total": round(time.perf_counter() - t0, 1),
        "inputs": inst.sources, "provenance": provenance(),
    }, out / "overlap.json")
    LOGGER.info("Overlap: undefined=%d | point effects %s", int(undefined.sum()),
                {k: round(val, 4) for k, val in point.items()})
    LOGGER.info("Pooled stratified delta: %s", pooled)
    return 0


def cmd_integrate(cfg: dict[str, Any]) -> int:
    """Cross-objective analysis from the CSV outputs (no new experiment)."""
    from . import integration as I

    base = Path(cfg["outputs"]["dir"])
    n = 169_343
    design = base / "obj4_5_removal" / "experiment.json"
    if design.is_file():
        n = int(json.loads(design.read_text())["design"]["num_nodes"])
    pred = I.bridge_prediction(base, n)
    cats = I.category_table(base)
    assoc = I.associations(cats)
    partial = {
        "cohesion_vs_inter_share_given_num_papers": I.partial_spearman(
            cats, "largest_intra_component_fraction", "inter_share_of_incident_edges",
            "num_papers"),
        "cohesion_vs_num_papers_given_inter_share": I.partial_spearman(
            cats, "largest_intra_component_fraction", "num_papers",
            "inter_share_of_incident_edges"),
    }
    out = base / "integration"
    out.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out / "bridge_prediction_initial_slope.csv", index=False, float_format="%.6g")
    cats.to_csv(out / "category_cross_objective.csv", index=False, float_format="%.6g")
    write_json({"description": "Cross-objective associations computed from the CSV outputs of "
                               "objectives 2, 3 and 4-5 (descriptive; n = 40 categories).",
                "spearman": assoc, "partial_spearman": partial,
                "provenance": provenance()}, out / "integration.json")
    LOGGER.info("Bridge prediction (first rows):\n%s",
                pred.groupby("strategy").head(2).to_string(index=False))
    for a in assoc:
        LOGGER.info("rho(%s, %s) = %.3f (p=%.4f)", a["x"], a["y"], a["rho"], a["p_permutation"])
    LOGGER.info("partial: %s", partial)
    extra: dict[str, Any] = {}
    if (base / "obj6_communities" / "contingency_reference_run.csv").is_file():
        pairs, cross, res6 = I.communities_vs_flows(base)
        pairs.to_csv(out / "communities_vs_citation_pairs.csv", index=False, float_format="%.6g")
        cross.to_csv(out / "category_cross_objective_with_communities.csv", index=False,
                     float_format="%.6g")
        extra["communities"] = res6
        LOGGER.info("Obj 6 integration: %s", {k: (round(v["rho"], 3), v["p_permutation"])
                                              if isinstance(v, dict) else round(v, 3)
                                              for k, v in res6.items() if k != "pairs"})
    if (base / "obj7_ordering" / "ordering.csv").is_file():
        extra["ordering"] = I.ordering_vs_others(base)
        LOGGER.info("Obj 7 integration: %s", {k: (round(v["rho"], 3), v["p_permutation"])
                                              if isinstance(v, dict) else round(v, 3)
                                              for k, v in extra["ordering"].items()})
    if extra:
        write_json({"description": "Integration of objectives 6-7 with objectives 2-5 "
                                   "(descriptive associations).", **extra,
                    "provenance": provenance()}, out / "integration_obj6_obj7.json")
    return 0


def cmd_communities(cfg: dict[str, Any], use_cache: bool, num_seeds: int, n_iterations: int,
                    n_jobs: int) -> int:
    """Objective 6: Leiden (directed modularity) on G_d vs the 40 primary categories."""
    from itertools import combinations

    import igraph as ig
    import leidenalg as la

    from . import communities as C

    base_seed = int(cfg["project"]["seed"])
    t0 = time.perf_counter()
    inst = _load(cfg, use_cache)
    edges, n, labels = inst.gd_edges, inst.num_nodes, inst.labels
    names, k = inst.category_names, inst.num_categories
    seeds = [int(s) for s in np.random.SeedSequence([base_seed, 6]).generate_state(num_seeds)
             % (2 ** 31 - 1)]
    LOGGER.info("Leiden: %d seeds, n_iterations=%d, %d processes", len(seeds), n_iterations,
                n_jobs)
    runs = C.run_leiden(edges, n, seeds, n_iterations, n_jobs)
    t_leiden = time.perf_counter() - t0

    # quality of the category partition with exactly the same function (3 implementations)
    g = ig.Graph(n=n, edges=edges.tolist(), directed=True)
    q_cat = C.directed_modularity(edges, n, labels)
    q_cat_leiden = float(la.ModularityVertexPartition(g, initial_membership=labels.tolist())
                         .quality())
    q_cat_igraph = float(g.modularity(labels.tolist(), directed=True))
    del g

    rows = []
    for run in runs:
        memb = run["membership"]
        q = C.directed_modularity(edges, n, memb)
        rows.append({"seed": run["seed"], "modularity_directed": q,
                     "modularity_leidenalg": run["quality_leidenalg"],
                     "abs_diff_implementations": abs(q - run["quality_leidenalg"]),
                     "modularity_undirected_reading": C.undirected_modularity(edges, n, memb),
                     **{k_: v for k_, v in C.size_summary(memb).items() if k_ != "top10_sizes"},
                     **{f"{k_}_vs_categories": v
                        for k_, v in C.agreement(labels, memb).items()},
                     **{f"{k_}": v for k_, v in
                        C.inter_edges_inside_communities(edges, labels, memb).items()},
                     "seconds": run["seconds"]})
    per_run = pd.DataFrame(rows)
    pairs = []
    for (a, ra), (b, rb) in combinations(enumerate(runs), 2):
        pairs.append({"run_a": a, "run_b": b,
                      **C.agreement(ra["membership"], rb["membership"])})
    stability = pd.DataFrame(pairs)
    best = int(per_run["modularity_directed"].idxmax())
    ref = runs[best]["membership"]
    rng = np.random.default_rng(np.random.SeedSequence([base_seed, 6, 1]))
    chance = pd.DataFrame([C.agreement(rng.permutation(labels), ref) for _ in range(20)])
    N = C.contingency(labels, ref, k)
    cats = C.category_table(N, names)
    comms = C.community_table(N, names)
    # stability of each category's concentration across runs
    shares = np.array([C.category_table(C.contingency(labels, r["membership"], k), names)
                       ["share_in_dominant_community"].to_numpy() for r in runs])
    matches = np.array([C.category_table(C.contingency(labels, r["membership"], k), names)
                        ["mutual_best_match"].to_numpy() for r in runs])
    cats["share_in_dominant_community_mean_runs"] = shares.mean(axis=0)
    cats["share_in_dominant_community_sd_runs"] = shares.std(axis=0, ddof=1)
    cats["mutual_best_match_fraction_runs"] = matches.mean(axis=0)

    out = Path(cfg["outputs"]["dir"]) / "obj6_communities"
    out.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(out / "runs.csv", index=False, float_format="%.6g")
    stability.to_csv(out / "stability_between_runs.csv", index=False, float_format="%.6g")
    cats.to_csv(out / "categories_vs_communities.csv", index=False, float_format="%.6g")
    comms.to_csv(out / "communities_composition.csv", index=False, float_format="%.6g")
    pd.DataFrame(N, index=names).rename_axis("category").to_csv(
        out / "contingency_reference_run.csv")
    cache = Path(cfg["data"]["processed_dir"]) / "communities_memberships.npz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, seeds=np.array(seeds),
                        memberships=np.vstack([r["membership"] for r in runs]))
    agg = per_run.describe().T[["mean", "std", "min", "max"]]
    summary = {
        "description": "Objective 6: Leiden communities of G_d (directed modularity, "
                       "Leicht-Newman) vs the 40 primary categories.",
        "algorithm": {"implementation": "leidenalg.ModularityVertexPartition on a directed "
                                        "igraph graph", "n_iterations": n_iterations,
                      "seeds": seeds, "seed_derivation": f"SeedSequence([{base_seed}, 6])"
                                                         ".generate_state(n) % (2^31 - 1)"},
        "quality_function": "Q_d = (1/m) sum_c [m_c - K_c^out K_c^in / m]",
        "category_partition": {"modularity_directed_independent": q_cat,
                               "modularity_leidenalg": q_cat_leiden,
                               "modularity_igraph_directed": q_cat_igraph,
                               "modularity_undirected_reading":
                                   C.undirected_modularity(edges, n, labels),
                               "num_communities": k,
                               **C.inter_edges_inside_communities(edges, labels, labels)},
        "runs_summary": {col: {s: float(agg.loc[col, s]) for s in agg.columns}
                         for col in agg.index if col != "seed"},
        "stability_between_runs": {"pairs": len(stability),
                                   "nmi_mean": float(stability["nmi"].mean()),
                                   "nmi_min": float(stability["nmi"].min()),
                                   "ari_mean": float(stability["ari"].mean()),
                                   "ari_min": float(stability["ari"].min())},
        "reference_run": {"index": best, "seed": seeds[best],
                          "size_summary": C.size_summary(ref)},
        "chance_level_label_permutation": {"replicates": 20,
                                           "nmi_mean": float(chance["nmi"].mean()),
                                           "nmi_max": float(chance["nmi"].max()),
                                           "ari_mean": float(chance["ari"].mean())},
        "timings_seconds": {"leiden_all_runs_wall": round(t_leiden, 1),
                            "total": round(time.perf_counter() - t0, 1)},
        "cache": {"memberships": str(cache)},
        "inputs": inst.sources, "provenance": provenance(),
    }
    write_json(summary, out / "communities.json")
    LOGGER.info("Categories Q_d=%.5f (leidenalg %.5f, igraph %.5f)", q_cat, q_cat_leiden,
                q_cat_igraph)
    LOGGER.info("Runs:\n%s", per_run[["seed", "modularity_directed", "num_communities",
                                      "nmi_vs_categories", "ari_vs_categories",
                                      "abs_diff_implementations"]].to_string(index=False))
    LOGGER.info("Stability: %s | chance: %s", summary["stability_between_runs"],
                summary["chance_level_label_permutation"])
    return 0


def cmd_ordering(cfg: dict[str, Any], use_cache: bool, null_replicates: int,
                 boot_replicates: int) -> int:
    """Objective 7: approximate topological ordering of the categories (linear ordering)."""
    from . import flows as F
    from . import ordering as OR

    base_seed = int(cfg["project"]["seed"])
    t0 = time.perf_counter()
    inst = _load(cfg, use_cache)
    edges, labels, years = inst.gd_edges, inst.labels, inst.years
    names, k = inst.category_names, inst.num_categories
    M = F.citation_matrix(edges, labels, k)
    W = OR.inter_weights(M)
    total = float(W.sum())
    rng = lambda tag: np.random.default_rng(np.random.SeedSequence([base_seed, 7, tag]))

    # ---- observed optimum
    t1 = time.perf_counter()
    obs = OR.optimise(W, rng(0), random_starts=5000, kicks=50, initial=[OR.net_flow_order(W)])
    t_obs = time.perf_counter() - t1
    order = obs["order"]
    pos = OR.positions(order)
    bounds = OR.cycle_packing_bound(W)
    u = bounds["pairwise_bound"]
    realizability = (obs["fraction"] - 0.5) / (u - 0.5)
    # ---- random permutations (as in the proposal)
    perm = OR.random_permutation_fractions(W, rng(1), 10_000)
    # ---- direction-randomised control, re-optimised with a fixed budget
    t1 = time.perf_counter()
    null_rows = []
    r_null = rng(2)
    null_mats = []
    for rep in range(null_replicates):
        Wn = OR.randomise_directions(W, r_null)
        res = OR.optimise(Wn, r_null, random_starts=50, kicks=10,
                          initial=[OR.net_flow_order(Wn)])
        un = OR.pairwise_upper_bound(Wn)
        null_rows.append({"replicate": rep, "optimum_fraction": res["fraction"],
                          "pairwise_bound": un,
                          "realizability": (res["fraction"] - 0.5) / (un - 0.5),
                          "dominant_3_cycles": len(OR.majority_three_cycles(Wn))})
        if rep < 20:
            null_mats.append((rep, Wn, res["fraction"]))
    null = pd.DataFrame(null_rows)
    # convergence check of the null budget: 20 replicates re-run with 20x the budget
    check = []
    for rep, Wn, frac in null_mats:
        big = OR.optimise(Wn, rng(100 + rep), random_starts=1000, kicks=50,
                          initial=[OR.net_flow_order(Wn)])
        check.append(big["fraction"] - frac)
    t_null = time.perf_counter() - t1
    # ---- bootstrap over citing papers: positional uncertainty
    t1 = time.perf_counter()
    r_boot = rng(3)
    p = np.full(inst.num_nodes, 1.0 / inst.num_nodes)
    boot_pos, boot_frac = [], []
    for _ in range(boot_replicates):
        w = r_boot.multinomial(inst.num_nodes, p).astype(np.float64)[edges[:, 0]]
        Wb = OR.inter_weights(F.citation_matrix(edges, labels, k, weights=w))
        res = OR.optimise(Wb, r_boot, random_starts=30, kicks=10,
                          initial=[order, OR.net_flow_order(Wb)])
        boot_pos.append(OR.positions(res["order"]))
        boot_frac.append(res["fraction"])
    boot_pos = np.asarray(boot_pos)
    t_boot = time.perf_counter() - t1
    # ---- time: age-null ordering, ordering of the directional excess beyond the time null
    stats = F.flow_stats(edges, labels, k, F.year_pair_stratum(edges, years), years)
    E_age = OR.inter_weights(stats.E_age)
    age = OR.optimise(E_age, rng(4), random_starts=1000, kicks=20,
                      initial=[OR.net_flow_order(E_age)])
    T = M + M.T
    share_t = np.divide(stats.E_time, stats.E_time + stats.E_time.T,
                        out=np.full(M.shape, 0.5), where=(stats.E_time + stats.E_time.T) > 0)
    excess = OR.inter_weights(np.maximum(M - T * share_t, 0.0))
    exc = OR.optimise(excess, rng(5), random_starts=1000, kicks=20,
                      initial=[OR.net_flow_order(excess)])
    nf = OR.net_flow_order(W)
    papers = np.bincount(labels, minlength=k)
    year_mean = np.bincount(labels, weights=years, minlength=k) / papers
    from scipy.stats import spearmanr

    comparisons = {
        "net_flow_order": {"compatible_fraction_observed": OR.compatible_fraction(W, nf),
                           **OR.rank_correlations(order, nf)},
        "age_null_order": {"its_own_optimum_fraction": age["fraction"],
                           "compatible_fraction_observed": OR.compatible_fraction(W, age["order"]),
                           **OR.rank_correlations(order, age["order"])},
        "time_excess_order": {"compatible_fraction_observed":
                                  OR.compatible_fraction(W, exc["order"]),
                              "excess_weight_fraction_of_total": float(excess.sum() / total),
                              "its_own_optimum_fraction": exc["fraction"],
                              **OR.rank_correlations(order, exc["order"])},
        "position_vs_year_mean": {
            "spearman_rho": float(spearmanr(pos, year_mean).statistic),
            "note": "position 0 = first in the order (cited by the areas after it)"},
    }
    # ---- consistency with objective 2 pair results
    obj2 = Path(cfg["outputs"]["dir"]) / "obj2_flows"
    pair_checks = {}
    top = pd.read_csv(obj2 / "top_pairs_by_net_flow.csv")
    idx = {nm: i for i, nm in enumerate(names)}
    comp = [pos[idx[b]] < pos[idx[a]] for a, b in zip(top["A_cites_more"], top["B_cited_more"])]
    pair_checks["top20_net_flow_pairs_compatible"] = float(np.mean(comp))
    pf = pd.read_csv(obj2 / "pair_flows.csv")
    strong = pf[(pf["total"] >= 200) & pf["direction_beyond_time_null"]]
    dom_ok = [(pos[j] < pos[i]) if r > 0 else (pos[i] < pos[j])
              for i, j, r in zip(strong["i"], strong["j"], strong["net_ij"])]
    exc_ok = [(pos[j] < pos[i]) if r > 0 else (pos[i] < pos[j])
              for i, j, r in zip(strong["i"], strong["j"], strong["lor_time"])]
    pair_checks["pairs_T200_beyond_time_null"] = len(strong)
    pair_checks["their_dominant_direction_compatible"] = float(np.mean(dom_ok))
    pair_checks["their_excess_direction_compatible"] = float(np.mean(exc_ok))
    # largest incompatible arcs
    viol = [(W[i, j], names[i], names[j]) for i in range(k) for j in range(k)
            if i != j and pos[j] > pos[i] and W[i, j] > 0]
    viol = sorted(viol, reverse=True)[:15]

    cats = pd.read_csv(obj2 / "category_flows.csv").set_index("category_name")
    table = pd.DataFrame({
        "position": np.arange(k), "category_name": [names[c] for c in order],
        "year_mean": year_mean[order], "num_papers": papers[order],
        "nnf": [cats.loc[names[c], "nnf"] for c in order],
        "bootstrap_position_median": np.median(boot_pos[:, order], axis=0),
        "bootstrap_position_p025": np.percentile(boot_pos[:, order], 2.5, axis=0),
        "bootstrap_position_p975": np.percentile(boot_pos[:, order], 97.5, axis=0),
        "position_in_net_flow_order": OR.positions(nf)[order],
        "position_in_age_null_order": OR.positions(age["order"])[order],
        "position_in_time_excess_order": OR.positions(exc["order"])[order],
    })
    out = Path(cfg["outputs"]["dir"]) / "obj7_ordering"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "ordering.csv", index=False, float_format="%.6g")
    null.to_csv(out / "direction_null.csv", index=False, float_format="%.6g")
    pd.DataFrame({"fraction": perm}).to_csv(out / "random_permutations.csv", index=False,
                                             float_format="%.6g")
    pd.DataFrame(boot_pos, columns=names).to_csv(out / "bootstrap_positions.csv", index=False)
    pd.DataFrame({"start_fraction": obs["start_fractions"]}).to_csv(
        out / "local_optima_observed.csv", index=False, float_format="%.8g")
    pd.DataFrame(viol, columns=["weight", "citing_area", "cited_area"]).to_csv(
        out / "largest_incompatible_arcs.csv", index=False)
    pd.DataFrame(W, index=names, columns=names).rename_axis("citing\\cited").to_csv(
        out / "inter_area_weights.csv")
    payload = {
        "description": "Objective 7: ordering of the 40 categories maximising the weight of "
                       "inter-area citations compatible with the order (linear ordering).",
        "convention": "arc i -> j from the citing area i to the cited area j; compatible iff "
                      "pos(j) < pos(i) (cited area earlier)",
        "objective": "F(pi) = sum_{pos(j)<pos(i)} W_ij / sum_{i!=j} W_ij, maximised",
        "inter_area_weight_total": total,
        "heuristic": {"method": "multi-start best-improvement insertion + iterated local "
                                "search (random 4-insertion kicks)",
                      "observed_budget": {"starts": obs["starts"], "kicks": 50},
                      "starts_reaching_best": obs["starts_reaching_best"],
                      "distinct_local_optima": int(np.unique(
                          np.round(obs["start_fractions"], 12)).size),
                      "seconds": round(t_obs, 1)},
        "result": {"compatible_fraction": obs["fraction"], "order": [names[c] for c in order],
                   "realizability_(F-0.5)/(U-0.5)": realizability},
        "bounds": bounds,
        "gap_to_cycle_bound": bounds["cycle_packing_bound"] - obs["fraction"],
        "random_permutations": {"n": int(perm.size), "mean": float(perm.mean()),
                                "sd": float(perm.std(ddof=1)), "max": float(perm.max()),
                                "fraction_ge_observed": float((perm >= obs["fraction"]).mean()),
                                "z": float((obs["fraction"] - perm.mean()) / perm.std(ddof=1))},
        "direction_randomised_null": {
            "replicates": int(null_replicates), "budget": {"starts": 51, "kicks": 10},
            "preserves": ["pair totals T_ij", "pair margins |W_ij - W_ji|",
                          "pairwise upper bound U", "total weight"],
            "does_not_preserve": ["area made/received volumes and net flows",
                                  "consistency (transitivity) among dominant directions",
                                  "temporal structure"],
            "optimum_mean": float(null["optimum_fraction"].mean()),
            "optimum_sd": float(null["optimum_fraction"].std(ddof=1)),
            "optimum_max": float(null["optimum_fraction"].max()),
            "fraction_ge_observed": float((null["optimum_fraction"] >= obs["fraction"]).mean()),
            "realizability_mean": float(null["realizability"].mean()),
            "realizability_p975": float(null["realizability"].quantile(0.975)),
            "dominant_3_cycles_mean": float(null["dominant_3_cycles"].mean()),
            "budget_check": {"replicates": len(check), "max_improvement_with_20x_budget":
                             float(max(check)), "mean_improvement": float(np.mean(check))},
            "seconds": round(t_null, 1)},
        "bootstrap": {"replicates": int(boot_replicates),
                      "type": "cluster bootstrap over citing papers",
                      "optimum_fraction_mean": float(np.mean(boot_frac)),
                      "optimum_fraction_ci95": [float(np.percentile(boot_frac, 2.5)),
                                                float(np.percentile(boot_frac, 97.5))],
                      "seconds": round(t_boot, 1)},
        "time": comparisons,
        "objective2_consistency": pair_checks,
        "seeds": f"SeedSequence([{base_seed}, 7, tag]); tag 0 observed, 1 permutations, "
                 "2 null, 3 bootstrap, 4 age-null order, 5 time-excess order, 100+r checks",
        "seconds_total": round(time.perf_counter() - t0, 1),
        "inputs": inst.sources, "provenance": provenance(),
    }
    write_json(payload, out / "ordering.json")
    LOGGER.info("F*=%.5f U=%.5f U_cyc=%.5f realizability=%.3f | perms mean %.4f max %.4f | "
                "null mean %.4f max %.4f (>=obs %.3f) | null budget check max +%.2e",
                obs["fraction"], u, bounds["cycle_packing_bound"], realizability, perm.mean(),
                perm.max(), null["optimum_fraction"].mean(), null["optimum_fraction"].max(),
                (null["optimum_fraction"] >= obs["fraction"]).mean(), max(check))
    LOGGER.info("Order: %s", [names[c] for c in order])
    LOGGER.info("Time: %s", comparisons)
    LOGGER.info("Obj2 consistency: %s", pair_checks)
    return 0


def cmd_clustering(cfg: dict[str, Any], use_cache: bool) -> int:
    """Objective 1 complement: why average local clustering and transitivity differ (G_u)."""
    from scipy import stats as st

    from . import clustering as CL
    from . import verification as V

    t0 = time.perf_counter()
    inst = _load(cfg, use_cache)
    n = inst.num_nodes
    summary = V.undirected_summary(inst.gu_edges, n)
    degree, triangles = summary["arrays"]["degree"], summary["arrays"]["triangles"]
    out = Path(cfg["outputs"]["dir"]) / "obj1_clustering"
    out.mkdir(parents=True, exist_ok=True)

    # cross-check with the per-vertex table published by F1
    f1_path = Path(cfg["outputs"]["metrics_dir"]) / "local_clustering.csv"
    f1_check: dict[str, Any] = {"available": f1_path.is_file()}
    if f1_path.is_file():
        f1 = pd.read_csv(f1_path)
        f1_check |= {"degree_equal": bool(np.array_equal(f1["degree"].to_numpy(), degree)),
                     "triangles_equal": bool(np.array_equal(f1["triangles"].to_numpy(),
                                                            triangles))}

    measures = CL.global_measures(degree, triangles)
    splits = [CL.top_degree_split(degree, triangles, top) for top in (1, 3, 10, 100)]
    bins = CL.by_degree_bin(degree, triangles)
    t_max = CL.neighbour_degree_bound(inst.gu_edges, degree)
    bound = CL.bound_by_degree_bin(degree, triangles, t_max)
    bins = bins.merge(bound.drop(columns=["num_vertices", "mean_local_clustering"]), on="bin")

    q = CL.local_quantities(degree, triangles)
    c, tau, elig = q["clustering"], q["tau"], q["eligible"]
    c_max = np.divide(t_max, tau, out=np.zeros(n), where=tau > 0)
    ratio = np.divide(c, c_max, out=np.full(n, np.nan), where=c_max > 0)

    # optional: neighbour pairs inside the same Leiden community (reference run of obj. 6)
    community: dict[str, Any] = {"available": False}
    mem_path = Path(cfg["data"]["processed_dir"]) / "communities_memberships.npz"
    runs_path = Path(cfg["outputs"]["dir"]) / "obj6_communities" / "runs.csv"
    if mem_path.is_file() and runs_path.is_file():
        ref = int(pd.read_csv(runs_path)["modularity_directed"].idxmax())
        with np.load(mem_path) as z:
            membership = z["memberships"][ref]
        adj = V.undirected_adjacency(inst.gu_edges, n)
        idx = np.flatnonzero(elig)
        same = CL.same_community_neighbour_pairs(adj.indptr, adj.indices, membership, idx)
        b = CL.degree_bins(degree[idx])
        bins = bins.merge(pd.DataFrame({"bin": b, "same_community_neighbour_pairs": same})
                          .groupby("bin", as_index=False).mean(), on="bin", how="left")
        community = {"available": True, "reference_run": ref,
                     "spearman_same_community_vs_clustering": float(
                         st.spearmanr(same, c[idx]).statistic)}

    spearman = {f"k_ge_{k}": float(st.spearmanr(degree[degree >= k], c[degree >= k]).statistic)
                for k in (2, 4, 16, 64)}
    rho_spearman = {f"k_ge_{k}": float(st.spearmanr(degree[degree >= k], ratio[degree >= k]
                                                    ).statistic) for k in (16, 64)}
    trends = [CL.trend(bins, 30, k_from) for k_from in (2, 16, 64)]
    low = {}
    for k in (2, 3, 4):
        m = degree == k
        low[str(k)] = {"num_vertices": int(m.sum()), "mean_clustering": float(c[m].mean()),
                       "fraction_clustering_one": float((c[m] == 1).mean()),
                       "contribution_to_average": float(c[m].sum() / elig.sum())}
    ones = degree[elig & (c == 1)]
    hi = degree >= 64
    groups = {"k_ge_64": {"vertex_share": float((hi & elig).sum() / elig.sum()),
                          "triple_share": float(tau[hi].sum() / tau.sum()),
                          "closed_triple_share": float(triangles[hi].sum() / triangles.sum()),
                          "contribution_to_average": float(c[hi].sum() / elig.sum()),
                          "weighted_clustering": float(triangles[hi].sum() / tau[hi].sum()),
                          "mean_clustering": float(c[hi].mean()),
                          "contribution_to_transitivity": float(triangles[hi].sum() / tau.sum())},
              "k_2_to_63": {"vertex_share": float((elig & ~hi).sum() / elig.sum()),
                            "triple_share": float(tau[elig & ~hi].sum() / tau.sum()),
                            "closed_triple_share": float(triangles[elig & ~hi].sum()
                                                         / triangles.sum()),
                            "contribution_to_average": float(c[elig & ~hi].sum()
                                                             / elig.sum()),
                            "weighted_clustering": float(triangles[elig & ~hi].sum()
                                                         / tau[elig & ~hi].sum()),
                            "mean_clustering": float(c[elig & ~hi].mean()),
                            "contribution_to_transitivity": float(triangles[elig & ~hi].sum()
                                                                  / tau.sum())}}
    gd = inst.gd_edges
    k_in, k_out = np.bincount(gd[:, 1], minlength=n), np.bincount(gd[:, 0], minlength=n)
    top3 = splits[1]["hub_vertices"]
    u, v = inst.gu_edges[:, 0], inst.gu_edges[:, 1]
    x = np.concatenate([degree[u], degree[v]]).astype(float)
    y = np.concatenate([degree[v], degree[u]]).astype(float)
    hubs = []
    for i in top3:
        nb = np.concatenate([v[u == i], u[v == i]])
        hubs.append({"vertex": int(i), "category": inst.category_names[inst.labels[i]],
                     "year": int(inst.years[i]), "degree": int(degree[i]),
                     "in_degree": int(k_in[i]), "out_degree": int(k_out[i]),
                     "local_clustering": float(c[i]), "bound": float(c_max[i]),
                     "neighbour_degree_median": float(np.median(degree[nb])),
                     "neighbour_degree_mean": float(degree[nb].mean())})

    bins.to_csv(out / "clustering_by_degree_bin.csv", index=False)
    CL.by_degree(degree, triangles, max_degree=2048).to_csv(out / "clustering_by_degree.csv",
                                                            index=False)
    dd = []
    for name, values in (("in", k_in), ("out", k_out), ("undirected", degree)):
        dd.append(CL.degree_distribution(values).assign(kind=name))
    pd.concat(dd).to_csv(out / "degree_distributions.csv", index=False)
    write_json({
        "description": "Average local clustering (k>=2) vs transitivity of G_u: exact "
                       "decomposition by degree, hub split, neighbour-degree bound.",
        "f1_cross_check": f1_check, "measures": measures,
        "transitivity_bound": {"t_max_over_triples": float(t_max.sum() / tau.sum()),
                               "realised_fraction": float(triangles.sum() / t_max.sum()),
                               "average_bound_degree_ge_2": float(c_max[elig].mean())},
        "top_degree_splits": splits, "degree_groups": groups,
        "low_degree": low,
        "clustering_one": {"num_vertices": int(ones.size),
                           "by_degree": {str(k): int(cnt) for k, cnt in
                                         zip(*np.unique(ones, return_counts=True))}},
        "fraction_clustering_zero_degree_ge_2": float((c[elig] == 0).mean()),
        "spearman_degree_vs_clustering": spearman,
        "spearman_degree_vs_realised_fraction": rho_spearman,
        "binned_trend": trends,
        "degree_assortativity": float(np.corrcoef(x, y)[0, 1]),
        "hubs": hubs,
        "directed_degree": {"in_max": int(k_in.max()), "out_max": int(k_out.max()),
                            "in_zero": int((k_in == 0).sum()), "out_zero": int((k_out == 0).sum()),
                            "in_degree_share_of_undirected_degree_k_ge_64": float(
                                k_in[hi].sum() / (k_in[hi] + k_out[hi]).sum())},
        "community_link": community,
        "seconds": round(time.perf_counter() - t0, 1),
        "inputs": inst.sources, "provenance": provenance(),
    }, out / "clustering.json")
    LOGGER.info("Clustering: %s | top-3 split %s", measures,
                {k: splits[1][k] for k in ("triple_share_w", "weighted_clustering_rest")})
    return 0


def cmd_review(cfg: dict[str, Any], use_cache: bool) -> int:
    """Scientific review: cross-checks on the saved results of objectives 2-7 (no re-runs)."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    from . import review as RV
    from .edges import is_inter_area

    t0 = time.perf_counter()
    base = Path(cfg["outputs"]["dir"])
    out = base / "review"
    out.mkdir(parents=True, exist_ok=True)
    inst = _load(cfg, use_cache)
    n = inst.num_nodes
    result: dict[str, Any] = {}

    # --- robustness: crossings, isolation predicted by vertex composition, LCC loss parts
    summary = pd.read_csv(base / "obj4_5_removal" / "summary.csv")
    crossings = {f"{a}_minus_{b}": RV.sign_changes(RV.curve_difference(summary, a, b, n))
                 for a, b in (("intra", "matched_intra"), ("inter", "matched_inter"),
                              ("inter", "uniform"), ("intra", "uniform"))}
    diff = RV.curve_difference(summary, "intra", "matched_intra", n)
    diff.to_csv(out / "intra_minus_matched_intra.csv", index=False)
    gu = inst.gu_edges
    inter_u = is_inter_area(gu, inst.labels)
    k_inter = np.bincount(gu[inter_u].ravel(), minlength=n)
    k_intra = np.bincount(gu[~inter_u].ravel(), minlength=n)
    pools = {"inter": (k_inter, k_intra, int(inter_u.sum())),
             "intra": (k_intra, k_inter, int((~inter_u).sum())),
             "uniform": (k_inter + k_intra, np.zeros(n, dtype=np.int64), int(gu.shape[0]))}
    rows = []
    for q in (0.01, 0.05, 0.10, 0.13, 0.20, 0.30):
        for strat in ("inter", "intra", "uniform", "matched_inter", "matched_intra"):
            d = summary[summary["strategy"] == strat]
            r = d.iloc[(d["q"] - q).abs().argmin()]
            row = {"q": float(r["q"]), "strategy": strat, "k": int(r["k"]),
                   "class_fraction_removed": float(r["eligible_fraction_removed"]),
                   "lcc_loss": float(n * (1 - r["S_mean"])),
                   "isolated_observed": float(r["num_isolated_mean"]),
                   "outside_lcc_non_isolated": float(r["outside_lcc_non_isolated_mean"])}
            if strat in pools:
                tgt, oth, pool = pools[strat]
                row["isolated_expected_from_composition"] = RV.expected_isolated(
                    tgt, oth, int(r["k"]), pool)
            rows.append(row)
    loss = pd.DataFrame(rows)
    loss.to_csv(out / "lcc_loss_decomposition.csv", index=False)
    profiles = {"only_inter": (k_intra == 0) & (k_inter > 0),
                "only_intra": (k_inter == 0) & (k_intra > 0),
                "both": (k_inter > 0) & (k_intra > 0)}
    gap_rows = []
    for q in sorted(loss["q"].unique()):
        li = loss[(loss["q"] == q) & (loss["strategy"] == "inter")].iloc[0]
        lu = loss[(loss["q"] == q) & (loss["strategy"] == "uniform")].iloc[0]
        k = int(li["k"])
        row = {"q": float(q), "k": k, "gap_lcc_loss": li["lcc_loss"] - lu["lcc_loss"],
               "gap_isolated_observed": li["isolated_observed"] - lu["isolated_observed"],
               "gap_isolated_expected": li["isolated_expected_from_composition"]
               - lu["isolated_expected_from_composition"]}
        row["isolated_share_of_gap"] = row["gap_isolated_observed"] / row["gap_lcc_loss"]
        for name, mask in profiles.items():
            row[f"inter_expected_{name}"] = RV.expected_isolated(
                k_inter[mask], k_intra[mask], k, int(inter_u.sum()))
            row[f"uniform_expected_{name}"] = float(RV.prob_all_drawn(
                (k_inter + k_intra)[mask], k, int(gu.shape[0])).sum())
        gap_rows.append(row)
    gap = pd.DataFrame(gap_rows)
    gap.to_csv(out / "inter_minus_uniform_isolation.csv", index=False)
    result["removal"] = {
        "crossings": crossings,
        "vertex_class_composition": {
            "only_inter": int(((k_intra == 0) & (k_inter > 0)).sum()),
            "only_intra": int(((k_inter == 0) & (k_intra > 0)).sum()),
            "both": int(((k_inter > 0) & (k_intra > 0)).sum())},
        "inter_minus_uniform_isolation": gap.to_dict("records"),
        "note": "isolated_expected_from_composition is the exact expectation for the class and "
                "uniform strategies (uniform draw without replacement from the pool); its "
                "agreement with the simulations validates the computation and the "
                "implementation, it is not independent evidence of a mechanism. The isolated "
                "gap is a net difference between strategies (see the per-profile columns)."}
    bridges = pd.read_csv(base / "obj3_overlap" / "bridges_by_class.csv")
    result["bridge_decomposition"] = {"inter_vs_all": RV.bridge_decomposition(bridges, "inter",
                                                                              "all"),
                                      "intra_vs_all": RV.bridge_decomposition(bridges, "intra",
                                                                              "all")}

    # --- flows: multiple comparisons, homophily and category size
    pf = pd.read_csv(base / "obj2_flows" / "pair_flows.csv")
    mc = {}
    for tmin in (1000, 200):
        d = pf[pf["total"] >= tmin]
        z, p = RV.wald_from_percentile_ci(d["lor_time"].to_numpy(),
                                          d["lor_time_ci95_low"].to_numpy(),
                                          d["lor_time_ci95_high"].to_numpy())
        mc[f"total_ge_{tmin}"] = {
            "pairs": len(d), "ci_excludes_zero": int(d["direction_beyond_time_null"].sum()),
            "p_lt_005": int((p < 0.05).sum()), "holm": int(RV.holm(p).sum()),
            "benjamini_hochberg": int(RV.benjamini_hochberg(p).sum())}
    result["multiple_comparisons"] = mc | {
        "method": "two-sided normal p-values from the 95 % cluster-bootstrap intervals of the "
                  "temporal-null log-odds ratio; alpha = 0.05"}
    cf = pd.read_csv(base / "obj2_flows" / "category_flows.csv")
    result["homophily_vs_size"] = {
        "spearman": RV.spearman_table(cf, [("num_papers", "homophily_OE_strength"),
                                           ("num_papers", "intra_fraction_of_made")]
                                      ).to_dict("records"),
        "intra_fraction_of_made": {
            "min": float(cf["intra_fraction_of_made"].min()),
            "min_category": cf.loc[cf["intra_fraction_of_made"].idxmin(), "category_name"],
            "max": float(cf["intra_fraction_of_made"].max()),
            "max_category": cf.loc[cf["intra_fraction_of_made"].idxmax(), "category_name"],
            "median": float(cf["intra_fraction_of_made"].median()),
            "categories_below_half": int((cf["intra_fraction_of_made"] < 0.5).sum())}}
    cc = pd.read_csv(base / "integration" / "category_cross_objective_with_communities.csv")
    size_pairs = [("num_papers", "inter_share_of_incident_edges"),
                  ("num_papers", "largest_intra_component_fraction"),
                  ("num_papers", "effective_num_communities"), ("num_papers", "spread"),
                  ("inter_share_of_incident_edges", "largest_intra_component_fraction"),
                  ("inter_share_of_incident_edges", "spread")]
    result["category_size"] = {
        "spearman": RV.spearman_table(cc, size_pairs).to_dict("records"),
        "partial_given_size": RV.spearman_table(
            cc, size_pairs[4:], control="num_papers").to_dict("records")}

    # --- communities: partition scores, internal arcs by class, composition of large groups
    mem_path = Path(cfg["data"]["processed_dir"]) / "communities_memberships.npz"
    if mem_path.is_file():
        with np.load(mem_path) as z:
            mem = z["memberships"]
        runs = pd.read_csv(base / "obj6_communities" / "runs.csv")
        ref = int(runs["modularity_directed"].idxmax())
        scores = pd.DataFrame([RV.partition_scores(inst.labels, m) for m in mem])
        inter_d = is_inter_area(inst.gd_edges, inst.labels)
        internal = pd.DataFrame([RV.internal_fraction_by_class(inst.gd_edges, m, inter_d)
                                 for m in mem])
        result["communities"] = {
            "reference_run": ref,
            "partition_scores_mean": scores.mean().to_dict(),
            "partition_scores_sd": scores.std().to_dict(),
            "internal_arcs_mean": internal.mean().to_dict(),
            "internal_arcs_reference": internal.iloc[ref].to_dict()}
    comp = pd.read_csv(base / "obj6_communities" / "communities_composition.csv")
    result["community_labels"] = comp[comp["size"] >= 1000][
        ["community", "size", "dominant_category", "purity", "top_categories"]
    ].to_dict("records")

    # --- cycles: strong components of G_d by citation direction in time; aggregated graph
    e, years = inst.gd_edges, inst.years
    yu, yv = years[e[:, 0]], years[e[:, 1]]

    def largest_scc(mask: np.ndarray) -> int:
        a = sp.csr_matrix((np.ones(int(mask.sum())), (e[mask, 0], e[mask, 1])), shape=(n, n))
        _, lab = connected_components(a, directed=True, connection="strong")
        return int(np.bincount(lab).max())

    w = pd.read_csv(base / "obj7_ordering" / "inter_area_weights.csv", index_col=0).to_numpy()
    iu = np.triu_indices(w.shape[0], 1)
    wa, wb = w[iu], w.T[iu]
    _, dom_lab = connected_components(sp.csr_matrix((w > w.T).astype(int)), directed=True,
                                      connection="strong")
    result["cycles"] = {
        "arc_time_direction": {"older": float((yu > yv).mean()), "same_year": float(
            (yu == yv).mean()), "newer": float((yu < yv).mean())},
        "largest_scc_all_arcs": largest_scc(np.ones(len(e), dtype=bool)),
        "largest_scc_without_arcs_to_newer_papers": largest_scc(yu >= yv),
        "category_pairs_with_citations": int(((wa + wb) > 0).sum()),
        "category_pairs_both_directions": int(((wa > 0) & (wb > 0)).sum()),
        "weight_share_pairs_both_directions": float(
            (wa + wb)[(wa > 0) & (wb > 0)].sum() / (wa + wb).sum()),
        "minority_direction_weight_share": float(np.minimum(wa, wb).sum() / (wa + wb).sum()),
        "largest_scc_dominant_direction_graph": int(np.bincount(dom_lab).max())}

    # --- optional: textual similarity of categories from the OGB node features
    raw = Path("data/raw/arxiv.zip")
    if raw.is_file():
        feats, ogb_labels = RV.read_ogb_node_features(str(raw))
        if not np.array_equal(ogb_labels, inst.labels):
            raise RuntimeError("OGB labels do not match the F1 instances")
        sim = RV.category_text_similarity(feats, inst.labels, inst.num_categories)
        pairs = pd.read_csv(base / "integration" / "communities_vs_citation_pairs.csv")
        idx = {name: i for i, name in enumerate(inst.category_names)}
        pairs["text_similarity"] = [sim[idx[a], idx[b]] for a, b in
                                    zip(pairs["area_i"], pairs["area_j"])]
        pairs.to_csv(out / "category_pairs_text_similarity.csv", index=False)
        ok = pairs[pairs["citations_total"] > 0]
        result["text_similarity"] = {
            "source": "OGB node-feat.csv.gz: mean skip-gram embedding of title+abstract words; "
                      "features centred before the category centroids",
            "pairs": len(ok),
            "spearman": RV.spearman_table(ok, [
                ("comembership", "citation_OE_symmetric"),
                ("text_similarity", "citation_OE_symmetric"),
                ("text_similarity", "comembership")]).to_dict("records"),
            "partial_given_text_similarity": RV.spearman_table(
                ok, [("comembership", "citation_OE_symmetric")],
                control="text_similarity").to_dict("records")}
    else:
        result["text_similarity"] = {"available": False,
                                     "note": "run `mo438 download` to fetch data/raw/arxiv.zip"}

    write_json(result | {"seconds": round(time.perf_counter() - t0, 1),
                         "inputs": inst.sources, "provenance": provenance()},
               out / "review.json")
    LOGGER.info("Review written to %s (%.1f s)", out, time.perf_counter() - t0)
    return 0


def cmd_figures(cfg: dict[str, Any], which: str) -> int:
    from . import plots

    base = Path(cfg["outputs"]["dir"])
    if which == "flows":
        paths = plots.flows_figures(base / "obj2_flows")
    elif which == "communities":
        paths = plots.communities_figures(base / "obj6_communities")
    elif which == "ordering":
        paths = plots.ordering_figures(base / "obj7_ordering")
    elif which == "clustering":
        paths = plots.clustering_figures(base / "obj1_clustering")
    else:
        paths = plots.overlap_figures(base / "obj3_overlap", Path(cfg["data"]["processed_dir"]))
    for path in paths:
        LOGGER.info("Wrote %s", path)
    return 0


def cmd_removal_figures(cfg: dict[str, Any]) -> int:
    from .plots import removal_figures

    out = Path(cfg["outputs"]["dir"]) / "obj4_5_removal"
    paths = removal_figures(out)
    for path in paths:
        LOGGER.info("Wrote %s", path)
    return 0


# --------------------------------------------------------------------------- entry point


def build_parser() -> argparse.ArgumentParser:
    """Argument parser of the ``mo438`` command."""
    parser = argparse.ArgumentParser(
        prog="mo438",
        description="MO438: structural analysis of the ogbn-arxiv citation network.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--config", default="config/default.yaml", help="YAML configuration file")
    parser.add_argument("--no-cache", action="store_true",
                        help="re-parse the GraphML files instead of using data/processed")
    sub = parser.add_subparsers(dest="command", required=True)

    download = sub.add_parser(
        "download", help="only download/cache the raw ogbn-arxiv archive (needs internet)")
    download.add_argument("--force", action="store_true", help="re-download even if cached")
    char = sub.add_parser(
        "characterize",
        help="objective 1: full characterization pipeline (data, graphs, GraphML, metrics, "
             "figures)",
        description="Run the characterization pipeline. Options override the configuration.")
    char.add_argument("--force-download", action="store_true",
                      help="re-download the raw dataset even if cached")
    char.add_argument("--force-rebuild", action="store_true",
                      help="re-parse the dataset and rewrite the GraphML files")
    char.add_argument("--force-recompute", action="store_true",
                      help="ignore cached distance results")
    char.add_argument("--paths-mode", choices=["exact", "approximate", "auto"],
                      help="distance/diameter computation mode (default: from config)")
    char.add_argument("--num-sources", type=int,
                      help="number of sampled BFS sources for the approximate mode")
    char.add_argument("--n-jobs", help="worker processes for distances ('auto' or an integer)")
    char.add_argument("--no-validation", action="store_true",
                      help="skip the exact-vs-approximate validation table")
    sub.add_parser("classify-edges", help="intra/inter classification of G_d and G_u + checks")
    sub.add_parser("clustering", help="objective 1: local clustering vs transitivity by degree")
    sub.add_parser("clustering-figures", help="figure from outputs/obj1_clustering")
    sub.add_parser("review", help="scientific review: cross-checks on saved results")
    verify = sub.add_parser("verify-f1",
                            help="independent re-check of the metrics published in the F1")
    verify.add_argument("--all-pairs", action="store_true",
                        help="also recompute all-pairs distances (one BFS per node; minutes)")
    verify.add_argument("--n-jobs", help="processes for --all-pairs ('auto' or an integer)")
    removal = sub.add_parser("removal", help="objectives 4-5: progressive edge removal")
    removal.add_argument("--replicates", type=int, help="seeds per strategy (default: config)")
    sub.add_parser("removal-figures", help="figures from outputs/obj4_5_removal/*.csv")
    flows = sub.add_parser("flows", help="objective 2: directed flows between categories")
    flows.add_argument("--replicates", type=int, default=1000, help="bootstrap replicates")
    ovl = sub.add_parser("overlap", help="objective 3: neighbourhood overlap and bridges")
    ovl.add_argument("--replicates", type=int, default=1000, help="bootstrap replicates")
    sub.add_parser("flows-figures", help="figures from outputs/obj2_flows/*.csv")
    sub.add_parser("integrate", help="cross-objective analysis of objectives 2, 3 and 4-5")
    orde = sub.add_parser("ordering", help="objective 7: linear ordering of the categories")
    orde.add_argument("--null-replicates", type=int, default=500)
    orde.add_argument("--bootstrap", type=int, default=200)
    com = sub.add_parser("communities", help="objective 6: Leiden directed modularity")
    com.add_argument("--seeds", type=int, default=10)
    com.add_argument("--n-iterations", type=int, default=-1,
                     help="Leiden iterations (-1: until no improvement)")
    com.add_argument("--n-jobs", help="parallel processes ('auto' or an integer)")
    sub.add_parser("overlap-figures", help="figures from outputs/obj3_overlap/*.csv")
    sub.add_parser("communities-figures", help="figures from outputs/obj6_communities")
    sub.add_parser("ordering-figures", help="figures from outputs/obj7_ordering")
    dist = sub.add_parser("removal-distances",
                          help="sampled average distance in the LCC at selected removal states")
    dist.add_argument("--replicates", type=int, default=10)
    dist.add_argument("--sources", type=int, default=256)
    dist.add_argument("--n-jobs", help="worker processes ('auto' or an integer)")
    return parser


def _characterize_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {"runtime": {}, "paths": {"approximation": {}}}
    if args.force_download:
        overrides["runtime"]["force_download"] = True
    if args.force_rebuild:
        overrides["runtime"]["force_rebuild_graphs"] = True
    if args.force_recompute:
        overrides["runtime"]["force_recompute_metrics"] = True
    if args.n_jobs is not None:
        overrides["runtime"]["n_jobs"] = args.n_jobs if args.n_jobs == "auto" else int(args.n_jobs)
    if args.paths_mode:
        overrides["paths"]["mode"] = args.paths_mode
    if args.num_sources is not None:
        overrides["paths"]["approximation"]["num_sources"] = args.num_sources
    if args.no_validation:
        overrides["paths"]["approximation"]["validation_enabled"] = False
    return overrides


def cmd_download(cfg: dict[str, Any], force: bool) -> int:
    from .data import download_ogbn_arxiv

    if cfg["data"]["dataset"] != "ogbn-arxiv":
        print("Nothing to download: the configured dataset is synthetic.")
        return 0
    print(download_ogbn_arxiv(Path(cfg["data"]["raw_dir"]), cfg["data"]["source_url"],
                              force=force))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point of the ``mo438`` console script."""
    args = build_parser().parse_args(argv)
    if args.command == "characterize":
        from .characterization import run_characterization

        # the pipeline configures its own logging (stderr + run.log in its output directory)
        run_characterization(args.config, _characterize_overrides(args))
        return 0
    setup_logging(None)
    overrides: dict[str, Any] = {}
    if getattr(args, "n_jobs", None):
        overrides["runtime"] = {"n_jobs": args.n_jobs if args.n_jobs == "auto"
                                else int(args.n_jobs)}
    cfg = load_config(args.config, overrides)
    if args.command == "download":
        return cmd_download(cfg, force=args.force)
    if args.command == "classify-edges":
        return cmd_classify_edges(cfg, use_cache=not args.no_cache)
    if args.command == "verify-f1":
        return cmd_verify_f1(cfg, all_pairs=args.all_pairs, use_cache=not args.no_cache)
    if args.command == "removal":
        return cmd_removal(cfg, use_cache=not args.no_cache, replicates=args.replicates)
    if args.command == "removal-figures":
        return cmd_removal_figures(cfg)
    if args.command == "flows":
        return cmd_flows(cfg, use_cache=not args.no_cache, replicates=args.replicates)
    if args.command == "overlap":
        return cmd_overlap(cfg, use_cache=not args.no_cache, replicates=args.replicates)
    if args.command == "ordering":
        return cmd_ordering(cfg, use_cache=not args.no_cache,
                            null_replicates=args.null_replicates,
                            boot_replicates=args.bootstrap)
    if args.command == "communities":
        jobs = min(5, int(cfg["runtime"]["n_jobs_resolved"])) if not args.n_jobs \
            else int(cfg["runtime"]["n_jobs_resolved"])
        return cmd_communities(cfg, use_cache=not args.no_cache, num_seeds=args.seeds,
                               n_iterations=args.n_iterations, n_jobs=jobs)
    if args.command == "integrate":
        return cmd_integrate(cfg)
    if args.command == "clustering":
        return cmd_clustering(cfg, use_cache=not args.no_cache)
    if args.command == "review":
        return cmd_review(cfg, use_cache=not args.no_cache)
    if args.command in ("flows-figures", "overlap-figures", "communities-figures",
                        "ordering-figures", "clustering-figures"):
        return cmd_figures(cfg, args.command.split("-")[0])
    if args.command == "removal-distances":
        return cmd_removal_distances(cfg, use_cache=not args.no_cache,
                                     replicates=args.replicates, num_sources=args.sources)
    return 2  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
