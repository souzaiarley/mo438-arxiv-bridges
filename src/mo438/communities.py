"""Objective 6: communities of G_d by directed modularity (Leiden) vs primary categories.

Quality function (Leicht & Newman, 2008), for a directed graph with m arcs:

    Q_d = (1/m) * sum_ij [A_ij - k_i^out k_j^in / m] * delta(c_i, c_j)
        = (1/m) * sum_c [ m_c - K_c^out K_c^in / m ]

where m_c is the number of arcs inside community c and K_c^out (K_c^in) the total out-
(in-)degree of its vertices. ``leidenalg.ModularityVertexPartition`` optimises exactly this
function on directed igraph graphs (documented, and checked numerically against
:func:`directed_modularity` in ``tests/test_communities.py``). The same function is used
for the partition defined by the 40 primary categories.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- quality


def directed_modularity(edges: np.ndarray, n: int, membership: np.ndarray) -> float:
    """Independent implementation of the Leicht-Newman directed modularity."""
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    membership = np.asarray(membership, dtype=np.int64)
    m = edges.shape[0]
    out_deg = np.bincount(edges[:, 0], minlength=n).astype(np.float64)
    in_deg = np.bincount(edges[:, 1], minlength=n).astype(np.float64)
    internal = float((membership[edges[:, 0]] == membership[edges[:, 1]]).sum())
    k = int(membership.max()) + 1
    k_out = np.bincount(membership, weights=out_deg, minlength=k)
    k_in = np.bincount(membership, weights=in_deg, minlength=k)
    return (internal - float((k_out * k_in).sum()) / m) / m


def undirected_modularity(edges: np.ndarray, n: int, membership: np.ndarray) -> float:
    """Newman-Girvan modularity of the same arcs read as undirected (for comparison)."""
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    membership = np.asarray(membership, dtype=np.int64)
    m = edges.shape[0]
    deg = np.bincount(edges.ravel(), minlength=n).astype(np.float64)
    internal = float((membership[edges[:, 0]] == membership[edges[:, 1]]).sum())
    k = int(membership.max()) + 1
    kc = np.bincount(membership, weights=deg, minlength=k)
    return internal / m - float((kc ** 2).sum()) / (4.0 * m * m)


# --------------------------------------------------------------------------- Leiden runs

_GRAPH: dict[str, Any] = {}


def _init(edges: np.ndarray, n: int) -> None:
    import igraph as ig

    _GRAPH["g"] = ig.Graph(n=n, edges=edges.tolist(), directed=True)


def _run(args: tuple[int, int]) -> dict[str, Any]:
    import leidenalg as la

    seed, n_iterations = args
    t0 = time.perf_counter()
    part = la.find_partition(_GRAPH["g"], la.ModularityVertexPartition,
                             n_iterations=n_iterations, seed=seed)
    return {"seed": seed, "membership": np.asarray(part.membership, dtype=np.int64),
            "quality_leidenalg": float(part.quality()),
            "seconds": round(time.perf_counter() - t0, 1)}


def run_leiden(edges: np.ndarray, n: int, seeds: list[int], n_iterations: int,
               n_jobs: int) -> list[dict[str, Any]]:
    """Leiden (directed modularity) for each seed, in parallel processes."""
    tasks = [(int(s), int(n_iterations)) for s in seeds]
    if n_jobs <= 1:
        _init(edges, n)
        return [_run(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=n_jobs, initializer=_init,
                             initargs=(edges, n)) as pool:
        return list(pool.map(_run, tasks))


# --------------------------------------------------------------------------- comparisons


def agreement(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """NMI (arithmetic normalisation) and ARI between two partitions."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    return {"nmi": float(normalized_mutual_info_score(a, b, average_method="arithmetic")),
            "ari": float(adjusted_rand_score(a, b))}


def size_summary(membership: np.ndarray) -> dict[str, Any]:
    sizes = np.sort(np.bincount(membership))[::-1]
    sizes = sizes[sizes > 0]
    n = sizes.sum()
    return {"num_communities": int(sizes.size), "largest": int(sizes[0]),
            "largest_fraction": float(sizes[0] / n),
            "communities_ge_100": int((sizes >= 100).sum()),
            "communities_ge_1000": int((sizes >= 1000).sum()),
            "nodes_in_communities_ge_1000": float(sizes[sizes >= 1000].sum() / n),
            "singletons": int((sizes == 1).sum()), "median_size": float(np.median(sizes)),
            "top10_sizes": sizes[:10].tolist()}


def _entropy_rows(counts: np.ndarray) -> np.ndarray:
    p = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=1)
    return h


def _cover(counts: np.ndarray, frac: float) -> np.ndarray:
    s = np.sort(counts, axis=1)[:, ::-1]
    c = np.cumsum(s, axis=1) / np.maximum(s.sum(axis=1, keepdims=True), 1)
    return (c < frac).sum(axis=1) + 1


def contingency(labels: np.ndarray, membership: np.ndarray, k: int) -> np.ndarray:
    """``N[category, community]`` = number of papers."""
    c = int(membership.max()) + 1
    return np.bincount(labels * c + membership, minlength=k * c).reshape(k, c)


def category_table(N: np.ndarray, names: list[str]) -> pd.DataFrame:
    """How each category is distributed over the communities."""
    papers = N.sum(axis=1)
    sizes = N.sum(axis=0)
    dom = N.argmax(axis=1)
    share = N[np.arange(len(names)), dom] / papers
    dom_cat_of_comm = N.argmax(axis=0)
    h = _entropy_rows(N)
    return pd.DataFrame({
        "category_name": names, "num_papers": papers,
        "dominant_community": dom, "share_in_dominant_community": share,
        "dominant_community_size": sizes[dom],
        "category_share_of_dominant_community": N[np.arange(len(names)), dom] / sizes[dom],
        "mutual_best_match": dom_cat_of_comm[dom] == np.arange(len(names)),
        "effective_num_communities": np.exp(h),
        "communities_for_50pct": _cover(N, 0.5), "communities_for_90pct": _cover(N, 0.9),
    })


def community_table(N: np.ndarray, names: list[str], min_size: int = 1) -> pd.DataFrame:
    """Composition of each community by category."""
    sizes = N.sum(axis=0)
    keep = np.flatnonzero(sizes >= min_size)
    Nt = N[:, keep].T
    dom = Nt.argmax(axis=1)
    order = np.argsort(-Nt, axis=1)
    rows = []
    h = _entropy_rows(Nt)
    cover = _cover(Nt, 0.9)
    for idx, c in enumerate(keep):
        top = [(names[j], Nt[idx, j] / sizes[c]) for j in order[idx, :3] if Nt[idx, j] > 0]
        rows.append({
            "community": int(c), "size": int(sizes[c]), "dominant_category": names[dom[idx]],
            "purity": float(Nt[idx, dom[idx]] / sizes[c]),
            "effective_num_categories": float(np.exp(h[idx])),
            "categories_for_90pct": int(cover[idx]),
            "top_categories": "; ".join(f"{nm} {sh:.2f}" for nm, sh in top),
        })
    return pd.DataFrame(rows).sort_values("size", ascending=False)


def comembership(N: np.ndarray) -> np.ndarray:
    """P[i, j] = probability that a random paper of i and one of j share a community."""
    p = N / N.sum(axis=1, keepdims=True)
    return p @ p.T


def inter_edges_inside_communities(edges: np.ndarray, labels: np.ndarray,
                                   membership: np.ndarray) -> dict[str, float]:
    src, dst = edges[:, 0], edges[:, 1]
    inter = labels[src] != labels[dst]
    same_comm = membership[src] == membership[dst]
    return {
        "arcs_inside_communities": float(same_comm.mean()),
        "inter_area_arcs_inside_communities": float(same_comm[inter].mean()),
        "intra_area_arcs_inside_communities": float(same_comm[~inter].mean()),
        "inside_community_arcs_that_are_inter_area": float(inter[same_comm].mean()),
    }
