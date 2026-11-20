"""Orchestration, validation and summaries of the removal experiments (objectives 4-5)."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from . import removal as R

SUMMARY_METRICS = ("S", "num_components", "num_isolated", "second_lcc_size", "S_non_isolated",
                   "outside_lcc_non_isolated")
COMPARISON_PAIRS = (
    ("inter", "uniform"), ("intra", "uniform"), ("inter", "intra"),
    ("inter", "matched_inter"), ("intra", "matched_intra"),
    ("matched_inter", "uniform"), ("matched_intra", "uniform"),
)
COMPARISON_Q = (0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.33, None)


def eligible_size(strategy: str, classes: R.EdgeClasses) -> int:
    inter = int(classes.is_inter.sum())
    return {"inter": inter, "intra": classes.num_edges - inter}.get(strategy, classes.num_edges)


def _rows(strategy: str, replicate: int, ks: np.ndarray, q: np.ndarray, metrics: dict,
          n: int, pool: int) -> pd.DataFrame:
    lcc = metrics["lcc_size"]
    iso = metrics["num_isolated"]
    return pd.DataFrame({
        "strategy": strategy, "replicate": replicate, "k": ks, "q": q,
        "eligible_fraction_removed": ks / pool,
        "lcc_size": lcc, "S": lcc / n,
        "num_components": metrics["num_components"], "num_isolated": iso,
        "second_lcc_size": metrics["second_lcc_size"],
        "S_non_isolated": lcc / (n - iso),
        "outside_lcc_non_isolated": n - lcc - iso,
    })


def run_all(classes: R.EdgeClasses, ks: np.ndarray, strategies: tuple[str, ...],
            replicates: int, base_seed: int, validate_points: int,
            progress: Callable[[str], None] | None = None
            ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run every (strategy, replicate); spot-check each run against :func:`metrics_forward`.

    Validation: for every run, ``validate_points`` grid points drawn at random (seeded) plus
    the two ends of the grid are recomputed from scratch with scipy; for replicate 0 of
    every strategy, *all* grid points are recomputed.
    """
    n, m = classes.num_nodes, classes.num_edges
    q = ks / m
    frames, meta = [], []
    checks = {"points_compared": 0, "mismatches": 0, "runs_checked": 0, "full_grid_runs": []}
    for strategy in strategies:
        pool = eligible_size(strategy, classes)
        t_strategy = time.perf_counter()
        for r in range(replicates):
            metrics, info = R.run_replicate(strategy, r, classes, ks, base_seed)
            frames.append(_rows(strategy, r, ks, q, metrics, n, pool))
            # --- validation against the independent forward recomputation
            t0 = time.perf_counter()
            vrng = np.random.default_rng(np.random.SeedSequence(
                [base_seed, 99, R.STRATEGIES[strategy][0], r]))
            if r == 0:
                idx = np.arange(ks.size)
                checks["full_grid_runs"].append(strategy)
            else:
                idx = np.unique(np.r_[0, ks.size - 1,
                                      vrng.choice(ks.size, size=validate_points, replace=False)])
            order, _ = R.removal_order(strategy, r, classes, int(ks.max()), base_seed)
            ref = R.metrics_forward(classes.edges, n, order, ks[idx])
            bad = sum(int((metrics[name][idx] != ref[name]).sum()) for name in R.METRICS)
            checks["points_compared"] += int(idx.size)
            checks["mismatches"] += bad
            checks["runs_checked"] += 1
            info["validation_points"] = int(idx.size)
            info["validation_mismatches"] = bad
            info["seconds_validation"] = round(time.perf_counter() - t0, 4)
            meta.append(info)
        if progress:
            progress(f"{strategy}: {replicates} replicates in "
                     f"{time.perf_counter() - t_strategy:.1f} s (incl. validation)")
    runs = pd.concat(frames, ignore_index=True)
    meta_df = pd.DataFrame(meta)
    checks["all_match"] = checks["mismatches"] == 0
    return runs, meta_df, checks


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean, sd and empirical 2.5/50/97.5 percentiles per (strategy, k)."""
    group = runs.groupby(["strategy", "k"], sort=False)
    base = group[["q", "eligible_fraction_removed"]].first()
    parts = [base]
    for metric in SUMMARY_METRICS:
        g = group[metric]
        parts.append(pd.DataFrame({
            f"{metric}_mean": g.mean(), f"{metric}_sd": g.std(ddof=1),
            f"{metric}_p025": g.quantile(0.025), f"{metric}_p50": g.median(),
            f"{metric}_p975": g.quantile(0.975), f"{metric}_min": g.min(),
            f"{metric}_max": g.max(),
        }))
    out = pd.concat(parts, axis=1).reset_index()
    out.insert(2, "replicates", group.size().to_numpy())
    return out


def _prob_less(a: np.ndarray, b: np.ndarray) -> float:
    """P(A < B) + 0.5 P(A = B) over all cross pairs (common-language effect size)."""
    diff = a[:, None] - b[None, :]
    return float(((diff < 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)


def compare(runs: pd.DataFrame, ks: np.ndarray, m: int, metric: str = "S") -> pd.DataFrame:
    """Pairwise strategy comparisons at selected q (nearest grid point)."""
    rows = []
    k_max = int(ks.max())
    for q_target in COMPARISON_Q:
        k = k_max if q_target is None else int(ks[np.argmin(np.abs(ks - q_target * m))])
        at_k = runs[runs["k"] == k]
        for a, b in COMPARISON_PAIRS:
            va = at_k.loc[at_k["strategy"] == a, metric].to_numpy()
            vb = at_k.loc[at_k["strategy"] == b, metric].to_numpy()
            if va.size == 0 or vb.size == 0:
                continue
            diff = va.mean() - vb.mean()
            se = np.sqrt(va.var(ddof=1) / va.size + vb.var(ddof=1) / vb.size)
            pooled = np.sqrt((va.var(ddof=1) + vb.var(ddof=1)) / 2)
            rows.append({
                "q": k / m, "k": k, "metric": metric, "A": a, "B": b,
                "mean_A": va.mean(), "mean_B": vb.mean(), "diff_A_minus_B": diff,
                "se_diff": se, "ci95_low": diff - 1.96 * se, "ci95_high": diff + 1.96 * se,
                "standardized_diff": diff / pooled if pooled > 0 else np.nan,
                "P_A_less_than_B": _prob_less(va, vb),
                "intervals_overlap": bool(np.quantile(va, 0.025) <= np.quantile(vb, 0.975)
                                          and np.quantile(vb, 0.025) <= np.quantile(va, 0.975)),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- descriptives


def degree_profile(classes: R.EdgeClasses) -> pd.DataFrame:
    """Endpoint-degree profile of the edge classes (explains early isolation effects)."""
    deg = np.bincount(classes.edges.ravel(), minlength=classes.num_nodes)
    du, dv = deg[classes.edges[:, 0]], deg[classes.edges[:, 1]]
    low, high = np.minimum(du, dv), np.maximum(du, dv)
    rows = []
    for name, mask in (("inter", classes.is_inter), ("intra", ~classes.is_inter),
                       ("all", np.ones(classes.num_edges, dtype=bool))):
        rows.append({
            "class": name, "num_edges": int(mask.sum()),
            "pendant_edges": int((low[mask] == 1).sum()),
            "pendant_fraction": float((low[mask] == 1).mean()),
            "min_degree_le_3_fraction": float((low[mask] <= 3).mean()),
            "median_min_degree": float(np.median(low[mask])),
            "median_max_degree": float(np.median(high[mask])),
            "mean_log2_min_degree": float(np.log2(low[mask]).mean()),
            "mean_log2_max_degree": float(np.log2(high[mask]).mean()),
            "edges_touching_top100_hubs_fraction": float(
                np.isin(classes.edges[mask], np.argsort(-deg)[:100]).any(axis=1).mean()),
        })
    return pd.DataFrame(rows)


def strata_table(classes: R.EdgeClasses, pairs: np.ndarray) -> pd.DataFrame:
    s = classes.strata
    k = pairs.shape[0]
    inter = np.bincount(s[classes.is_inter], minlength=k)
    total = np.bincount(s, minlength=k)
    return pd.DataFrame({
        "stratum": np.arange(k), "low_degree_bin": pairs[:, 0], "high_degree_bin": pairs[:, 1],
        "low_degree_range": [f"[{2 ** b}, {2 ** (b + 1) - 1}]" for b in pairs[:, 0]],
        "high_degree_range": [f"[{2 ** b}, {2 ** (b + 1) - 1}]" for b in pairs[:, 1]],
        "edges": total, "inter_edges": inter, "intra_edges": total - inter,
        "inter_fraction": inter / total,
    })


def endpoint_components(classes: R.EdgeClasses, labels: np.ndarray,
                        category_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The graph left when *all* inter-area edges are removed (inter strategy at k_max).

    It is the same for every seed. Every component lies inside one category by
    construction; this returns (largest components, per-category cohesion).
    """
    n = classes.num_nodes
    keep = ~classes.is_inter
    from .removal import _components_of_kept

    comp = _components_of_kept(classes.edges, n, keep)
    sizes = np.bincount(comp)
    k = len(category_names)
    # every component must be single-category: one distinct (component, category) pair each
    pure = np.unique(np.column_stack([comp, labels]), axis=0).shape[0] == sizes.size
    top = np.argsort(-sizes)[:10]
    largest_rows = [{"rank": i + 1, "size": int(sizes[c]), "fraction_of_nodes": sizes[c] / n,
                     "category": category_names[int(labels[np.flatnonzero(comp == c)[0]])]}
                    for i, c in enumerate(top)]
    per_cat = []
    papers = np.bincount(labels, minlength=k)
    for c in range(k):
        members = labels == c
        csizes = np.bincount(comp[members])
        csizes = csizes[csizes > 0]
        per_cat.append({
            "category": category_names[c], "num_papers": int(papers[c]),
            "largest_intra_component": int(csizes.max()),
            "largest_intra_component_fraction": float(csizes.max() / papers[c]),
            "num_intra_components": int(csizes.size),
            "isolated_without_inter_edges": int((csizes == 1).sum()),
        })
    largest = pd.DataFrame(largest_rows)
    largest.attrs["all_components_single_category"] = bool(pure)
    return largest, pd.DataFrame(per_cat).sort_values("num_papers", ascending=False)
