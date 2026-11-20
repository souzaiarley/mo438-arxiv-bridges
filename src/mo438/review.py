"""Cross-checks used in the scientific review of the saved results (no experiment is re-run).

Every function works on saved results (CSV/JSON) or on cached arrays, so the review adds
evidence without overwriting the validated outputs of objectives 2-7.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- removal


def curve_difference(summary: pd.DataFrame, a: str, b: str, num_nodes: int) -> pd.DataFrame:
    """Difference of mean LCC size between strategies ``a`` and ``b`` at every common ``k``.

    The two strategies use independent seeds, so the standard error of the difference is
    ``sqrt(sd_a^2/r_a + sd_b^2/r_b)`` (in vertices).
    """
    sa = summary[summary["strategy"] == a].set_index("k")
    sb = summary[summary["strategy"] == b].set_index("k")
    k = sa.index.intersection(sb.index)
    diff = (sa.loc[k, "S_mean"] - sb.loc[k, "S_mean"]) * num_nodes
    se = np.sqrt(sa.loc[k, "S_sd"] ** 2 / sa.loc[k, "replicates"]
                 + sb.loc[k, "S_sd"] ** 2 / sb.loc[k, "replicates"]) * num_nodes
    return pd.DataFrame({"k": k, "q": sa.loc[k, "q"].to_numpy(), "diff": diff.to_numpy(),
                         "ci95_low": (diff - 1.96 * se).to_numpy(),
                         "ci95_high": (diff + 1.96 * se).to_numpy()})


def sign_changes(diff: pd.DataFrame) -> list[dict[str, Any]]:
    """Systematic changes of sign of a mean difference along ``q``.

    Only grid points whose 95 % CI excludes zero are used to detect a change of sign, so
    fluctuations around zero (e.g. at very small ``q``) are not reported as crossings. The
    crossing ``q`` is interpolated on the full curve between the last significant point of
    one sign and the first significant point of the other sign.
    """
    d = diff.sort_values("k").reset_index(drop=True)
    sig = d[(d["ci95_low"] > 0) | (d["ci95_high"] < 0)]
    s = np.sign(sig["diff"].to_numpy())
    out = []
    for j in np.flatnonzero(s[:-1] != s[1:]):
        i0, i1 = sig.index[j], sig.index[j + 1]
        seg = d.loc[i0:i1]
        sg = np.sign(seg["diff"].to_numpy())
        t = int(np.flatnonzero(sg[:-1] != sg[1:])[-1])
        r0, r1 = seg.iloc[t], seg.iloc[t + 1]
        q_cross = r0["q"] + (r1["q"] - r0["q"]) * r0["diff"] / (r0["diff"] - r1["diff"])
        before = d.loc[:i0][np.sign(d.loc[:i0, "diff"]) == s[j]]
        after = d.loc[i1:][np.sign(d.loc[i1:, "diff"]) == s[j + 1]]
        b_ext = before.loc[before["diff"].abs().idxmax()]
        a_ext = after.loc[after["diff"].abs().idxmax()]
        out.append({
            "q_cross": float(q_cross), "between_q": [float(r0["q"]), float(r1["q"])],
            "extreme_before": {"q": float(b_ext["q"]), "diff": float(b_ext["diff"]),
                               "ci95": [float(b_ext["ci95_low"]), float(b_ext["ci95_high"])]},
            "extreme_after": {"q": float(a_ext["q"]), "diff": float(a_ext["diff"]),
                              "ci95": [float(a_ext["ci95_low"]), float(a_ext["ci95_high"])]},
        })
    return out


def prob_all_drawn(a: np.ndarray, removed: int, pool: int) -> np.ndarray:
    """``P(all a items drawn)`` when ``removed`` of ``pool`` items are drawn without replacement.

    ``prod_{j<a} (removed - j) / (pool - j)``; zero when ``a > removed``; one when ``a = 0``.
    """
    a = np.asarray(a, dtype=np.int64)
    out = np.ones(a.size)
    amax = int(a.max()) if a.size else 0
    factor = 1.0
    cum = np.ones(amax + 1)
    for j in range(amax):
        factor *= max(removed - j, 0) / (pool - j)
        cum[j + 1] = factor
    out[:] = cum[a]
    return out


def expected_isolated(degree_target: np.ndarray, degree_other: np.ndarray, removed: int,
                      pool: int) -> float:
    """Expected number of vertices isolated by removing ``removed`` edges of a pool.

    A vertex becomes isolated iff it has no edge outside the pool (``degree_other == 0``)
    and all its pool edges are drawn. Exact (linearity of expectation) for removal of a
    uniformly random subset of the pool, which is how the class and uniform strategies draw.
    """
    degree_target = np.asarray(degree_target)
    eligible = (np.asarray(degree_other) == 0) & (degree_target > 0)
    return float(prob_all_drawn(degree_target[eligible], removed, pool).sum())


def bridge_decomposition(bridges: pd.DataFrame, a: str = "inter", b: str = "all"
                         ) -> dict[str, Any]:
    """Split the difference in vertices detachable per edge into pendant and non-pendant parts.

    Uses the per-class bridge table of objective 3 (pendant bridges detach one vertex each;
    non-pendant bridges detach their smaller side). For non-pendant bridges the difference is
    further split into frequency and mean-size effects (midpoint decomposition).
    """
    r = bridges.set_index("class")

    def parts(cls: str) -> dict[str, float]:
        e = float(r.loc[cls, "edges"])
        pend = float(r.loc[cls, "pendant_bridges"])
        nonp = float(r.loc[cls, "non_pendant_bridges"])
        det_nonp = float(r.loc[cls, "vertices_detachable_by_non_pendant"])
        return {"per_edge_total": float(r.loc[cls, "vertices_detachable_total"]) / e,
                "pendant_per_edge": pend / e, "non_pendant_per_edge": det_nonp / e,
                "non_pendant_frequency": nonp / e, "non_pendant_mean_size": det_nonp / nonp}

    pa, pb = parts(a), parts(b)
    total = pa["per_edge_total"] - pb["per_edge_total"]
    pend = pa["pendant_per_edge"] - pb["pendant_per_edge"]
    nonp = pa["non_pendant_per_edge"] - pb["non_pendant_per_edge"]
    freq = (pa["non_pendant_frequency"] - pb["non_pendant_frequency"]) * \
        (pa["non_pendant_mean_size"] + pb["non_pendant_mean_size"]) / 2
    size = (pa["non_pendant_mean_size"] - pb["non_pendant_mean_size"]) * \
        (pa["non_pendant_frequency"] + pb["non_pendant_frequency"]) / 2
    return {a: pa, b: pb, "difference": total, "pendant_part": pend, "non_pendant_part": nonp,
            "pendant_share_of_difference": pend / total,
            "non_pendant_frequency_effect": freq, "non_pendant_size_effect": size}


# --------------------------------------------------------------------------- statistics


def holm(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Holm step-down rejections (family-wise error rate ``alpha``)."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = p.size
    reject = np.zeros(m, dtype=bool)
    for rank, idx in enumerate(order):
        if p[idx] <= alpha / (m - rank):
            reject[idx] = True
        else:
            break
    return reject


def benjamini_hochberg(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg step-up rejections (false discovery rate ``alpha``)."""
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    order = np.argsort(p)
    thresholds = alpha * np.arange(1, m + 1) / m
    below = p[order] <= thresholds
    reject = np.zeros(m, dtype=bool)
    if below.any():
        reject[order[: np.flatnonzero(below).max() + 1]] = True
    return reject


def wald_from_percentile_ci(estimate: np.ndarray, low: np.ndarray, high: np.ndarray
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Approximate two-sided p-values from a 95 % bootstrap interval (normal approximation).

    ``se = (high - low) / (2 * 1.96)``, ``z = estimate / se``. Used only to apply
    multiple-comparison corrections; the percentile intervals remain the primary result.
    """
    from scipy import stats as st

    se = (np.asarray(high) - np.asarray(low)) / (2 * 1.959964)
    z = np.asarray(estimate) / se
    return z, 2 * st.norm.sf(np.abs(z))


# --------------------------------------------------------------------------- communities


def partition_scores(labels: np.ndarray, membership: np.ndarray) -> dict[str, float]:
    """Homogeneity, completeness and V-measure of communities with respect to categories.

    With the arithmetic normalisation, NMI equals the V-measure (beta = 1). Homogeneity is 1
    when every community holds a single category (penalised by *fusion*); completeness is 1
    when every category falls in a single community (penalised by *division*).
    """
    from sklearn import metrics

    h, c, v = metrics.homogeneity_completeness_v_measure(labels, membership)
    nmi = metrics.normalized_mutual_info_score(labels, membership, average_method="arithmetic")
    return {"homogeneity": float(h), "completeness": float(c), "v_measure": float(v),
            "nmi_arithmetic": float(nmi)}


def internal_fraction_by_class(edges: np.ndarray, membership: np.ndarray, is_inter: np.ndarray
                               ) -> dict[str, float]:
    """Fraction of arcs whose endpoints share a community, overall and per class."""
    edges = np.asarray(edges)
    inside = membership[edges[:, 0]] == membership[edges[:, 1]]
    is_inter = np.asarray(is_inter, dtype=bool)
    return {"all": float(inside.mean()), "intra": float(inside[~is_inter].mean()),
            "inter": float(inside[is_inter].mean())}


def spearman_table(frame: pd.DataFrame, pairs: list[tuple[str, str]],
                   control: str | None = None) -> pd.DataFrame:
    """Spearman correlations (and partial correlations given ``control``) for column pairs."""
    from scipy import stats as st

    rows = []
    for x, y in pairs:
        d = frame[[x, y] + ([control] if control else [])].dropna()
        rho = st.spearmanr(d[x], d[y])
        row = {"x": x, "y": y, "n": len(d), "rho": float(rho.statistic),
               "p": float(rho.pvalue)}
        if control:
            r = d.rank()
            res = {}
            for col in (x, y):
                beta = np.polyfit(r[control], r[col], 1)
                res[col] = r[col] - np.polyval(beta, r[control])
            row["partial_rho_given_" + control] = float(np.corrcoef(res[x], res[y])[0, 1])
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- text similarity


def category_text_similarity(features: np.ndarray, labels: np.ndarray, num_categories: int,
                             center: bool = True) -> np.ndarray:
    """Cosine similarity between category centroids of the node features.

    In ogbn-arxiv the 128 features of a paper are the average of skip-gram embeddings of the
    words of its title and abstract. Centering removes the large common component shared by
    all papers (without it every pair of centroids has cosine > 0.89).
    """
    x = np.asarray(features, dtype=np.float64)
    if center:
        x = x - x.mean(axis=0)
    cent = np.vstack([x[labels == c].mean(axis=0) for c in range(num_categories)])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    return cent @ cent.T


def read_ogb_node_features(zip_path: str, members: tuple[str, str] = (
        "arxiv/raw/node-feat.csv.gz", "arxiv/raw/node-label.csv.gz")
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Node features and labels from the official OGB archive (no extraction to disk)."""
    import io
    import zipfile

    with zipfile.ZipFile(zip_path) as z:
        feats = pd.read_csv(io.BytesIO(z.read(members[0])), compression="gzip", header=None,
                            dtype=np.float32).to_numpy()
        labels = pd.read_csv(io.BytesIO(z.read(members[1])), compression="gzip",
                             header=None)[0].to_numpy()
    return feats, labels
