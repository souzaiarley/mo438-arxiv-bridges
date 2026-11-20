"""Objective 1 (complement): local clustering versus transitivity in G_u.

For a vertex ``i`` with degree ``k_i >= 2`` let ``t_i`` be the number of triangles through it,
``tau_i = k_i (k_i - 1) / 2`` the number of connected triples centred on it (pairs of
neighbours) and ``C_i = t_i / tau_i`` its local clustering coefficient. Then

* average local clustering (course convention): ``Cbar = (1/N2) sum_{k_i>=2} C_i``;
* transitivity: ``T = 3 * triangles / triples = sum_i t_i / sum_i tau_i
  = sum_i tau_i C_i / sum_i tau_i``.

``Cbar`` weighs every vertex equally; ``T`` weighs each vertex by ``tau_i``, which grows as
``k_i^2``. The functions below decompose both quantities by degree so that the gap between
them can be explained quantitatively (rather than attributed to "the hubs").
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as st


def local_quantities(degree: np.ndarray, triangles: np.ndarray) -> dict[str, np.ndarray]:
    """Per-vertex ``tau_i`` and ``C_i`` (``C_i = 0`` when ``k_i < 2``)."""
    degree = np.asarray(degree, dtype=np.int64)
    triangles = np.asarray(triangles, dtype=np.int64)
    if degree.shape != triangles.shape:
        raise ValueError("degree and triangles must have the same shape")
    tau = degree * (degree - 1) // 2
    if np.any(triangles > tau):
        raise ValueError("a vertex has more triangles than pairs of neighbours")
    clustering = np.divide(triangles, tau, out=np.zeros(degree.size), where=tau > 0)
    return {"tau": tau, "clustering": clustering, "eligible": degree >= 2}


def global_measures(degree: np.ndarray, triangles: np.ndarray) -> dict[str, Any]:
    """Both global measures, computed by their two equivalent formulas each.

    ``triangles`` holds, per vertex, the triangles through that vertex (so its sum is three
    times the number of triangles of the graph).
    """
    q = local_quantities(degree, triangles)
    tau, c, elig = q["tau"], q["clustering"], q["eligible"]
    n2 = int(elig.sum())
    total_tau = int(tau.sum())
    return {
        "num_vertices_degree_ge_2": n2,
        "num_triangles": int(np.asarray(triangles).sum() // 3),
        "num_connected_triples": total_tau,
        "average_local_clustering": float(c[elig].mean()) if n2 else 0.0,
        "transitivity_3T_over_triples": (3.0 * (int(np.asarray(triangles).sum()) // 3)
                                         / total_tau) if total_tau else 0.0,
        "transitivity_weighted_mean": float((tau * c).sum() / total_tau) if total_tau else 0.0,
    }


def top_degree_split(degree: np.ndarray, triangles: np.ndarray, top: int) -> dict[str, Any]:
    """Split ``T`` into the ``top`` highest-degree vertices and the rest.

    ``T = w * C_hubs + (1 - w) * C_rest`` where ``w`` is the hubs' share of the connected
    triples and ``C_hubs``, ``C_rest`` are triple-weighted clustering coefficients
    (``sum t_i / sum tau_i`` inside each group). Ties in degree are broken by vertex index.
    """
    q = local_quantities(degree, triangles)
    tau = q["tau"]
    triangles = np.asarray(triangles, dtype=np.int64)
    order = np.lexsort((np.arange(len(tau)), -np.asarray(degree)))
    hubs = order[:top]
    rest = np.ones(len(tau), dtype=bool)
    rest[hubs] = False
    total = tau.sum()
    w = tau[hubs].sum() / total
    c_hubs = triangles[hubs].sum() / tau[hubs].sum()
    c_rest = triangles[rest].sum() / tau[rest].sum()
    return {
        "top": int(top),
        "hub_vertices": hubs.tolist(),
        "hub_degrees": np.asarray(degree)[hubs].tolist(),
        "hub_local_clustering": q["clustering"][hubs].tolist(),
        "triple_share_w": float(w),
        "weighted_clustering_hubs": float(c_hubs),
        "weighted_clustering_rest": float(c_rest),
        "reconstructed_T": float(w * c_hubs + (1 - w) * c_rest),
        "upper_bound_rest_if_hubs_zero": float(
            (triangles.sum() / total) / (1 - w)) if w < 1 else float("inf"),
        "mean_local_clustering_rest_degree_ge_2": float(
            q["clustering"][rest & q["eligible"]].mean()),
    }


def degree_bins(degree: np.ndarray) -> np.ndarray:
    """Logarithmic bin index ``floor(log2 k)`` (bin ``b`` covers ``[2^b, 2^(b+1) - 1]``)."""
    degree = np.asarray(degree, dtype=np.int64)
    return np.floor(np.log2(np.maximum(degree, 1))).astype(np.int64)


def by_degree_bin(degree: np.ndarray, triangles: np.ndarray) -> pd.DataFrame:
    """Contributions of each logarithmic degree bin to ``Cbar`` and to ``T``.

    Columns ``contribution_to_average`` sum to ``Cbar`` and ``contribution_to_transitivity``
    sum to ``T`` (over vertices with ``k >= 2``). ``triple_share`` is the bin's share of the
    connected triples (denominator of ``T``) and ``closed_triple_share`` its share of the closed
    triples ``sum t_i`` (numerator), so that ``contribution_to_transitivity =
    closed_triple_share * T``.
    """
    q = local_quantities(degree, triangles)
    tau, c, elig = q["tau"], q["clustering"], q["eligible"]
    triangles = np.asarray(triangles, dtype=np.int64)
    degree = np.asarray(degree, dtype=np.int64)
    b = degree_bins(degree)
    n2, total_tau = elig.sum(), tau.sum()
    rows = []
    for bb in np.unique(b[elig]):
        m = elig & (b == bb)
        rows.append({
            "bin": int(bb), "k_low": int(2 ** bb), "k_high": int(2 ** (bb + 1) - 1),
            "num_vertices": int(m.sum()),
            "vertex_share": m.sum() / n2,
            "mean_local_clustering": float(c[m].mean()),
            "median_local_clustering": float(np.median(c[m])),
            "q25_local_clustering": float(np.quantile(c[m], 0.25)),
            "q75_local_clustering": float(np.quantile(c[m], 0.75)),
            "fraction_clustering_one": float((c[m] == 1).mean()),
            "fraction_clustering_zero": float((c[m] == 0).mean()),
            "triples": int(tau[m].sum()),
            "triple_share": tau[m].sum() / total_tau,
            "closed_triple_share": float(triangles[m].sum() / triangles.sum()),
            "weighted_local_clustering": float(triangles[m].sum() / tau[m].sum()),
            "contribution_to_average": float(c[m].sum() / n2),
            "contribution_to_transitivity": float(triangles[m].sum() / total_tau),
        })
    return pd.DataFrame(rows)


def by_degree(degree: np.ndarray, triangles: np.ndarray, max_degree: int | None = None
              ) -> pd.DataFrame:
    """Empirical ``C(k)``: mean local clustering for each exact degree ``k >= 2``."""
    q = local_quantities(degree, triangles)
    degree = np.asarray(degree, dtype=np.int64)
    d = pd.DataFrame({"degree": degree[q["eligible"]],
                      "clustering": q["clustering"][q["eligible"]]})
    if max_degree is not None:
        d = d[d["degree"] <= max_degree]
    g = d.groupby("degree")["clustering"]
    out = pd.DataFrame({"num_vertices": g.size(), "mean_local_clustering": g.mean(),
                        "fraction_clustering_one": g.apply(lambda s: float((s == 1).mean()))})
    return out.reset_index()


def trend(bins: pd.DataFrame, min_vertices: int = 30, k_from: int = 1) -> dict[str, Any]:
    """Descriptive trend of the binned ``C(k)`` (no power law is assumed).

    Reports the Spearman correlation between bin and mean clustering, and the slope of an
    ordinary least-squares line of ``log10 mean C`` on ``log10`` of the bin's geometric centre,
    restricted to bins with at least ``min_vertices`` vertices and ``k_low >= k_from``.
    The slope is a summary of the binned means, not an estimate of a scaling exponent.
    """
    d = bins[(bins["num_vertices"] >= min_vertices) & (bins["k_low"] >= k_from)
             & (bins["mean_local_clustering"] > 0)]
    centre = np.sqrt(d["k_low"] * d["k_high"])
    x, y = np.log10(centre), np.log10(d["mean_local_clustering"])
    fit = st.linregress(x, y)
    return {
        "bins_used": d["bin"].tolist(), "min_vertices": min_vertices, "k_from": k_from,
        "spearman_bin_vs_mean": float(st.spearmanr(d["bin"], d["mean_local_clustering"]
                                                   ).statistic),
        "loglog_slope": float(fit.slope), "loglog_slope_stderr": float(fit.stderr),
        "loglog_r2": float(fit.rvalue ** 2),
        "ratio_per_doubling": float(10 ** (fit.slope * np.log10(2))),
    }


def neighbour_degree_bound(edges: np.ndarray, degree: np.ndarray) -> np.ndarray:
    """Upper bound on the triangles through each vertex given its neighbours' degrees.

    A neighbour ``j`` of ``i`` can be adjacent to at most ``min(k_j - 1, k_i - 1)`` other
    neighbours of ``i``, so ``t_i <= (1/2) sum_{j in N(i)} min(k_j - 1, k_i - 1)``. This is a
    simplified (looser) version of the bound used by Soffer and Vázquez (2005) to define a
    clustering coefficient free of degree-correlation bias. ``edges`` is the simple undirected
    edge list (each pair once).
    """
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    degree = np.asarray(degree, dtype=np.int64)
    u, v = edges[:, 0], edges[:, 1]
    cap = np.minimum(degree[u] - 1, degree[v] - 1).astype(np.float64)
    n = degree.size
    total = np.bincount(u, weights=cap, minlength=n) + np.bincount(v, weights=cap, minlength=n)
    return total / 2.0


def bound_by_degree_bin(degree: np.ndarray, triangles: np.ndarray, max_triangles: np.ndarray
                        ) -> pd.DataFrame:
    """Per degree bin: mean ``C``, mean bound ``C_max = t_max / tau`` and realised fraction.

    ``realised_fraction_mean`` is the mean of ``C_i / C_max_i`` over the bin's vertices;
    ``realised_fraction_weighted`` is ``sum t_i / sum t_max_i``.
    """
    q = local_quantities(degree, triangles)
    tau, c, elig = q["tau"], q["clustering"], q["eligible"]
    triangles = np.asarray(triangles, dtype=np.float64)
    max_triangles = np.asarray(max_triangles, dtype=np.float64)
    if np.any(triangles > max_triangles + 1e-9):
        raise ValueError("observed triangles exceed the neighbour-degree bound")
    c_max = np.divide(max_triangles, tau, out=np.zeros(tau.size), where=tau > 0)
    ratio = np.divide(c, c_max, out=np.full(tau.size, np.nan), where=c_max > 0)
    b = degree_bins(degree)
    rows = []
    for bb in np.unique(b[elig]):
        m = elig & (b == bb)
        rows.append({
            "bin": int(bb), "num_vertices": int(m.sum()),
            "mean_local_clustering": float(c[m].mean()),
            "mean_bound": float(c_max[m].mean()),
            "realised_fraction_mean": float(np.nanmean(ratio[m])),
            "realised_fraction_weighted": float(triangles[m].sum() / max_triangles[m].sum()),
        })
    return pd.DataFrame(rows)


def degree_distribution(values: np.ndarray) -> pd.DataFrame:
    """Counts, probabilities and complementary cumulative distribution ``P(K >= k)``."""
    counts = np.bincount(np.asarray(values, dtype=np.int64))
    k = np.flatnonzero(counts)
    total = counts.sum()
    ccdf = np.cumsum(counts[::-1])[::-1] / total
    return pd.DataFrame({"degree": k, "count": counts[k], "fraction": counts[k] / total,
                         "ccdf": ccdf[k]})


def same_community_neighbour_pairs(adj_indptr: np.ndarray, adj_indices: np.ndarray,
                                   membership: np.ndarray, vertices: np.ndarray
                                   ) -> np.ndarray:
    """For each vertex, the fraction of its pairs of neighbours that share a community.

    ``sum_c n_c (n_c - 1) / (k (k - 1))`` where ``n_c`` counts the neighbours in community
    ``c``. It is the probability that two distinct random neighbours are co-members, an upper
    reference for how often a pair of neighbours *could* be linked inside a community.
    """
    out = np.full(len(vertices), np.nan)
    for j, v in enumerate(vertices):
        nb = adj_indices[adj_indptr[v]:adj_indptr[v + 1]]
        k = nb.size
        if k < 2:
            continue
        counts = np.bincount(membership[nb])
        out[j] = float((counts * (counts - 1)).sum()) / (k * (k - 1))
    return out
