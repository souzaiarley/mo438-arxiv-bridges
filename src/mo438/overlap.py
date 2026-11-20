"""Objective 3: neighbourhood overlap (and bridges) of intra- vs inter-area edges of G_u.

Definition (course notes, bloco 03; Easley & Kleinberg)
--------------------------------------------------------
For an edge (u, v) of the simple undirected graph G_u, with N(u) the neighbours of u
*excluding v* and N(v) the neighbours of v *excluding u*,

    O(u, v) = |N(u) ∩ N(v)| / |N(u) ∪ N(v)| = c / (k_u + k_v - 2 - c),

where c = number of common neighbours (= triangles through the edge) and k = degree.

* The denominator is 0 iff k_u = k_v = 1 (the edge is an isolated component of two
  vertices); O is then **undefined** (NaN), such edges are excluded from every comparison
  and their number is reported. G_u is connected with n > 2, so none is expected.
* O = 0 means no common neighbour: the edge is a **local bridge** (span > 2).

Complementary structural property: **global bridges** (edges whose removal disconnects
their component) and the size of the part they would detach (Tarjan's algorithm with
subtree sizes). Pendant edges (an endpoint of degree 1) are the trivial bridges.

Effect sizes: difference of means / medians / fractions and Cliff's delta
delta = P(O_inter > O_intra) - P(O_inter < O_intra), computed exactly from weighted value
histograms (so that bootstrap replicates are cheap). Degree adjustment: direct
standardisation of the intra-area class to the stratum mix of the inter-area class, with
strata (floor log2 k_min, floor log2 k_max) as in objectives 4-5.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .verification import edge_common_neighbours, undirected_adjacency

QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90, 0.99)


# --------------------------------------------------------------------------- measures


def neighbourhood_overlap(uedges: np.ndarray, n: int) -> dict[str, np.ndarray]:
    """Per-edge common neighbours, degrees, denominator and overlap (NaN if undefined)."""
    uedges = np.asarray(uedges, dtype=np.int64).reshape(-1, 2)
    adj = undirected_adjacency(uedges, n)
    deg = np.diff(adj.indptr).astype(np.int64)
    common = edge_common_neighbours(adj, uedges)
    ku, kv = deg[uedges[:, 0]], deg[uedges[:, 1]]
    denom = ku + kv - 2 - common
    overlap = np.divide(common, denom, out=np.full(common.shape, np.nan), where=denom > 0)
    return {"common": common, "k_u": ku, "k_v": kv, "denominator": denom, "overlap": overlap}


def bridges(uedges: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Global bridges (iterative Tarjan) and the size of the side each one would detach.

    Returns:
        ``(is_bridge, detached)`` where ``detached[e]`` is the number of vertices on the
        smaller side of bridge e (0 for non-bridges).
    """
    uedges = np.asarray(uedges, dtype=np.int64).reshape(-1, 2)
    m = uedges.shape[0]
    src = np.r_[uedges[:, 0], uedges[:, 1]]
    dst = np.r_[uedges[:, 1], uedges[:, 0]]
    eid = np.r_[np.arange(m), np.arange(m)]
    order = np.argsort(src, kind="stable")
    indptr = np.r_[0, np.cumsum(np.bincount(src, minlength=n))].tolist()
    nbr = dst[order].tolist()
    nbr_e = eid[order].tolist()
    disc = [-1] * n
    low = [0] * n
    size = [1] * n
    parent_edge = [-1] * n
    ptr = indptr[:-1].copy()
    comp_root = [-1] * n
    is_bridge = np.zeros(m, dtype=bool)
    child_side = np.zeros(m, dtype=np.int64)
    comp_size: dict[int, int] = {}
    t = 0
    for root in range(n):
        if disc[root] != -1:
            continue
        disc[root] = low[root] = t
        t += 1
        comp_root[root] = root
        stack = [root]
        while stack:
            v = stack[-1]
            i = ptr[v]
            if i < indptr[v + 1]:
                ptr[v] = i + 1
                w, e = nbr[i], nbr_e[i]
                if e == parent_edge[v]:
                    continue
                if disc[w] == -1:
                    disc[w] = low[w] = t
                    t += 1
                    parent_edge[w] = e
                    comp_root[w] = root
                    stack.append(w)
                elif disc[w] < low[v]:
                    low[v] = disc[w]
            else:
                stack.pop()
                if stack:
                    u = stack[-1]
                    low[u] = min(low[u], low[v])
                    size[u] += size[v]
                    if low[v] > disc[u]:
                        is_bridge[parent_edge[v]] = True
                        child_side[parent_edge[v]] = size[v]
        comp_size[root] = size[root]
    roots = np.asarray(comp_root)
    total = np.asarray([comp_size[r] for r in roots[uedges[:, 0]]])
    detached = np.where(is_bridge, np.minimum(child_side, total - child_side), 0)
    return is_bridge, detached


# --------------------------------------------------------------------------- statistics


class ValueIndex:
    """Exact weighted statistics of one variable through its sorted distinct values."""

    def __init__(self, values: np.ndarray):
        self.values, self.index = np.unique(values, return_inverse=True)

    def hist(self, mask: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
        w = None if weights is None else weights[mask]
        return np.bincount(self.index[mask], weights=w, minlength=self.values.size)

    def quantile(self, hist: np.ndarray, q: float) -> float:
        cdf = np.cumsum(hist)
        return float(self.values[np.searchsorted(cdf, q * cdf[-1])])

    @staticmethod
    def delta(hx: np.ndarray, hy: np.ndarray) -> float:
        """Cliff's delta P(X > Y) - P(X < Y) from two histograms over the same values."""
        cy_below = np.cumsum(hy) - hy
        cy_above = hy.sum() - np.cumsum(hy)
        return float((hx * (cy_below - cy_above)).sum() / (hx.sum() * hy.sum()))


def describe(values: np.ndarray, extra: dict[str, np.ndarray] | None = None
             ) -> dict[str, Any]:
    v = values[~np.isnan(values)]
    out = {"edges": int(v.size), "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
           "frac_zero": float((v == 0).mean()), "frac_one": float((v == 1).mean())}
    for q in QUANTILES:
        out[f"q{int(q * 100):02d}"] = float(np.quantile(v, q))
    for name, arr in (extra or {}).items():
        out[name] = float(np.mean(arr))
    return out


def standardised_mean(values: np.ndarray, strata: np.ndarray, target: np.ndarray,
                      source: np.ndarray, weights: np.ndarray | None = None
                      ) -> tuple[float, float]:
    """Mean of ``values`` over ``source`` reweighted to the stratum mix of ``target``.

    Returns ``(standardised_mean, coverage)``; coverage = share of the target weight in
    strata where the source class is present (other strata are dropped).
    """
    s = int(strata.max()) + 1
    w = np.ones(values.size) if weights is None else weights
    tw = np.bincount(strata[target], weights=w[target], minlength=s)
    sw = np.bincount(strata[source], weights=w[source], minlength=s)
    sv = np.bincount(strata[source], weights=(w * values)[source], minlength=s)
    ok = sw > 0
    means = np.divide(sv, sw, out=np.zeros(s), where=ok)
    coverage = tw[ok].sum() / tw.sum()
    return float((tw[ok] * means[ok]).sum() / tw[ok].sum()), float(coverage)


def class_effects(vi: ValueIndex, values: np.ndarray, inter: np.ndarray, strata: np.ndarray,
                  weights: np.ndarray | None = None) -> dict[str, float]:
    """Inter vs intra effects for one (possibly bootstrap-weighted) sample."""
    intra = ~inter
    hx, hy = vi.hist(inter, weights), vi.hist(intra, weights)
    w = np.ones(values.size) if weights is None else weights
    mean_x = float((w * values)[inter].sum() / w[inter].sum())
    mean_y = float((w * values)[intra].sum() / w[intra].sum())
    zero = (values == 0).astype(np.float64)
    zx = float((w * zero)[inter].sum() / w[inter].sum())
    zy = float((w * zero)[intra].sum() / w[intra].sum())
    std_y, cov = standardised_mean(values, strata, inter, intra, weights)
    std_zy, _ = standardised_mean(zero, strata, inter, intra, weights)
    return {
        "mean_inter": mean_x, "mean_intra": mean_y, "diff_mean": mean_x - mean_y,
        "ratio_mean": mean_x / mean_y,
        "median_inter": vi.quantile(hx, 0.5), "median_intra": vi.quantile(hy, 0.5),
        "diff_median": vi.quantile(hx, 0.5) - vi.quantile(hy, 0.5),
        "frac_zero_inter": zx, "frac_zero_intra": zy, "diff_frac_zero": zx - zy,
        "cliffs_delta": ValueIndex.delta(hx, hy),
        "mean_intra_degree_standardised": std_y,
        "diff_mean_degree_standardised": mean_x - std_y,
        "frac_zero_intra_degree_standardised": std_zy,
        "diff_frac_zero_degree_standardised": zx - std_zy,
        "standardisation_coverage": cov,
    }


def bootstrap_effects(values: np.ndarray, inter: np.ndarray, strata: np.ndarray,
                      clusters: np.ndarray | None, replicates: int, seed: int
                      ) -> pd.DataFrame:
    """Bootstrap of :func:`class_effects` (i.i.d. edges if ``clusters`` is None)."""
    rng = np.random.default_rng(seed)
    vi = ValueIndex(values)
    rows = []
    if clusters is None:
        p = np.full(values.size, 1.0 / values.size)
    else:
        uniq, cl = np.unique(clusters, return_inverse=True)
        p = np.full(uniq.size, 1.0 / uniq.size)
    for _ in range(replicates):
        draw = rng.multinomial(p.size, p).astype(np.float64)
        weights = draw if clusters is None else draw[cl]
        keep = weights > 0
        rows.append(class_effects(vi, values, inter, strata,
                                  np.where(keep, weights, 0.0)))
    return pd.DataFrame(rows)


def stratum_table(values: np.ndarray, inter: np.ndarray, strata: np.ndarray,
                  pairs: np.ndarray, min_edges: int = 30) -> pd.DataFrame:
    rows = []
    for s in range(pairs.shape[0]):
        m = strata == s
        x, y = values[m & inter], values[m & ~inter]
        row = {"stratum": s, "low_degree_bin": int(pairs[s, 0]),
               "high_degree_bin": int(pairs[s, 1]), "n_inter": int(x.size),
               "n_intra": int(y.size),
               "mean_inter": float(x.mean()) if x.size else np.nan,
               "mean_intra": float(y.mean()) if y.size else np.nan,
               "median_inter": float(np.median(x)) if x.size else np.nan,
               "median_intra": float(np.median(y)) if y.size else np.nan,
               "frac_zero_inter": float((x == 0).mean()) if x.size else np.nan,
               "frac_zero_intra": float((y == 0).mean()) if y.size else np.nan}
        if x.size >= min_edges and y.size >= min_edges:
            vi = ValueIndex(np.r_[x, y])
            lab = np.r_[np.ones(x.size, bool), np.zeros(y.size, bool)]
            row["cliffs_delta"] = ValueIndex.delta(vi.hist(lab), vi.hist(~lab))
        else:
            row["cliffs_delta"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def pooled_stratified_delta(table: pd.DataFrame) -> dict[str, float]:
    """Weighted mean of per-stratum deltas (weights = inter-area edges of the stratum)."""
    ok = table["cliffs_delta"].notna()
    w = table.loc[ok, "n_inter"]
    return {"pooled_delta": float((w * table.loc[ok, "cliffs_delta"]).sum() / w.sum()),
            "strata_used": int(ok.sum()), "strata_total": len(table),
            "inter_edges_covered": float(w.sum() / table["n_inter"].sum()),
            "strata_with_negative_delta": int((table.loc[ok, "cliffs_delta"] < 0).sum())}


def by_degree_bin(values: np.ndarray, inter: np.ndarray, degree: np.ndarray,
                  name: str) -> pd.DataFrame:
    bins = np.floor(np.log2(degree)).astype(int)
    rows = []
    for b in np.unique(bins):
        for cls, mask in (("inter", inter), ("intra", ~inter)):
            v = values[(bins == b) & mask]
            if v.size == 0:
                continue
            rows.append({"degree_variable": name, "degree_bin": int(b),
                         "degree_range": f"[{2 ** b}, {2 ** (b + 1) - 1}]", "class": cls,
                         "edges": int(v.size), "mean": float(v.mean()),
                         "ci95_halfwidth_iid": float(1.96 * v.std(ddof=1) / np.sqrt(v.size))
                         if v.size > 1 else np.nan,
                         "median": float(np.median(v)), "frac_zero": float((v == 0).mean())})
    return pd.DataFrame(rows)


def by_category(values: np.ndarray, inter: np.ndarray, edges: np.ndarray, labels: np.ndarray,
                names: list[str], is_bridge: np.ndarray) -> pd.DataFrame:
    """Per category: its intra-area edges vs the inter-area edges incident to it."""
    lu, lv = labels[edges[:, 0]], labels[edges[:, 1]]
    rows = []
    for c, name in enumerate(names):
        intra_c = (~inter) & (lu == c)
        inter_c = inter & ((lu == c) | (lv == c))
        x, y = values[inter_c], values[intra_c]
        row = {"category": c, "category_name": name, "intra_edges": int(y.size),
               "inter_edges_incident": int(x.size),
               "inter_share_of_incident_edges": x.size / max(1, x.size + y.size),
               "mean_intra": float(y.mean()) if y.size else np.nan,
               "mean_inter": float(x.mean()) if x.size else np.nan,
               "median_intra": float(np.median(y)) if y.size else np.nan,
               "median_inter": float(np.median(x)) if x.size else np.nan,
               "frac_zero_intra": float((y == 0).mean()) if y.size else np.nan,
               "frac_zero_inter": float((x == 0).mean()) if x.size else np.nan,
               "bridge_frac_intra": float(is_bridge[intra_c].mean()) if y.size else np.nan,
               "bridge_frac_inter": float(is_bridge[inter_c].mean()) if x.size else np.nan}
        if x.size and y.size:
            vi = ValueIndex(np.r_[x, y])
            lab = np.r_[np.ones(x.size, bool), np.zeros(y.size, bool)]
            row["cliffs_delta"] = ValueIndex.delta(vi.hist(lab), vi.hist(~lab))
        else:
            row["cliffs_delta"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def bridge_summary(is_bridge: np.ndarray, detached: np.ndarray, inter: np.ndarray,
                   kmin: np.ndarray) -> pd.DataFrame:
    rows = []
    for cls, mask in (("inter", inter), ("intra", ~inter), ("all", np.ones_like(inter))):
        b = is_bridge & mask
        pend = b & (kmin == 1)
        nonpend = b & (kmin > 1)
        rows.append({
            "class": cls, "edges": int(mask.sum()), "bridges": int(b.sum()),
            "bridge_fraction": float(b.sum() / mask.sum()),
            "pendant_bridges": int(pend.sum()), "non_pendant_bridges": int(nonpend.sum()),
            "non_pendant_bridge_fraction": float(nonpend.sum() / mask.sum()),
            "vertices_detachable_total": int(detached[b].sum()),
            "vertices_detachable_by_non_pendant": int(detached[nonpend].sum()),
            "detached_size_median_non_pendant": float(np.median(detached[nonpend]))
            if nonpend.any() else np.nan,
            "detached_size_max": int(detached[b].max()) if b.any() else 0,
        })
    return pd.DataFrame(rows)
