"""Objective 2: directed citation flows between the 40 primary categories (G_d).

Notation: ``M[i, j]`` = number of citations made by papers of area i to papers of area j
(i cites j). Knowledge, in the usual reading, flows the other way (from j to i).

Per category
------------
made_i = sum_j M_ij, received_i = sum_j M_ji, intra_i = M_ii,
inter_out_i = made_i - M_ii, inter_in_i = received_i - M_ii and the **normalised net
inter-area flow** NNF_i = (inter_in_i - inter_out_i) / (inter_in_i + inter_out_i) in [-1, 1]
(> 0: the area is cited by other areas more than it cites them).

Per pair {i, j}
---------------
T_ij = M_ij + M_ji, net flow F_ij = M_ij - M_ji, asymmetry A_ij = F_ij / T_ij in [-1, 1].
A raw asymmetry can simply reflect how active the two areas are, or their age (citations
point backwards in time). Two null models give the *expected* split of each pair:

* **strength null**: E_ij = out_i in_j / M_tot (keeps every area's citing and cited
  volumes, intra included);
* **temporal null**: the same, stratified by the (citing year, cited year) of each citation,
  E_ij = sum_s O_is I_js / T_s, which keeps, for every combination of years, how much each
  area cites and is cited; asymmetries due only to the areas' age profiles vanish.

For a pair, the expected share of i -> j is e_ij = E_ij / (E_ij + E_ji). Directional excess
is the log-odds ratio LOR_ij = logit((M_ij + 0.5) / (T_ij + 1)) - logit((T_ij e_ij + 0.5) /
(T_ij + 1)): observed vs expected counts with the same continuity correction (so a pair
whose observed and expected splits are both 0 : T gets LOR = 0). Conditioning on T_ij
isolates *direction* from *volume*.

Important identity: made_i - received_i = inter_out_i - inter_in_i (intra cancels). Both
nulls above preserve every area's made and received volumes, hence they preserve each
area's net inter-area flow: they are suited to the *pairwise direction* question ("given
the areas' overall roles and ages, is this pair more one-sided than expected?") but cannot
explain an area's net role itself. For that, a third, **age null** (:func:`age_expected`)
keeps each area's inter-area citing volume and the age profile of what it cites, but lets
every area receive what the age composition of its papers predicts (choice among the other
areas' papers of each year in proportion to their numbers); NNF_age is the net flow it
implies.

Uncertainty: cluster bootstrap over citing papers (each replicate resamples the papers'
reference lists with replacement), which respects the dependence between citations of
the same paper. The null models are re-estimated inside every replicate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def citation_matrix(edges: np.ndarray, labels: np.ndarray, k: int,
                    weights: np.ndarray | None = None) -> np.ndarray:
    """``M[i, j]`` = (weighted) citations from area i to area j."""
    keys = labels[edges[:, 0]] * k + labels[edges[:, 1]]
    return np.bincount(keys, weights=weights, minlength=k * k).reshape(k, k)


def stratified_expected(edges: np.ndarray, labels: np.ndarray, k: int, stratum: np.ndarray,
                        weights: np.ndarray | None = None) -> np.ndarray:
    """Expected matrix with citing/cited area volumes preserved within each stratum.

    ``E_ij = sum_s O[i, s] * I[j, s] / T[s]`` where ``O[i, s]`` (``I[j, s]``) counts the
    citations of stratum s made by area i (received by area j). One stratum gives the
    classic ``out_i * in_j / total``. Row and column sums of E equal those of M.
    """
    stratum = np.asarray(stratum, dtype=np.int64)
    s = int(stratum.max()) + 1
    out = np.bincount(labels[edges[:, 0]] * s + stratum, weights=weights,
                      minlength=k * s).reshape(k, s)
    inn = np.bincount(labels[edges[:, 1]] * s + stratum, weights=weights,
                      minlength=k * s).reshape(k, s)
    total = out.sum(axis=0)
    scale = np.divide(1.0, total, out=np.zeros(s), where=total > 0)
    return (out * scale) @ inn.T


def year_pair_stratum(edges: np.ndarray, years: np.ndarray) -> np.ndarray:
    """Stratum id of each citation = (citing year, cited year)."""
    y0 = int(years.min())
    span = int(years.max()) - y0 + 1
    return (years[edges[:, 0]] - y0) * span + (years[edges[:, 1]] - y0)


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(p) - np.log1p(-p)


def age_expected(edges: np.ndarray, labels: np.ndarray, years: np.ndarray, k: int,
                 weights: np.ndarray | None = None) -> np.ndarray:
    """Age null for the *inter-area* citations (diagonal = 0).

    ``E_ij = sum_y O[i, y] * P[j, y] / (P[., y] - P[i, y])`` for j != i, where ``O[i, y]`` =
    inter-area citations made by area i to papers published in year y and ``P[j, y]`` =
    papers of area j published in year y. Every area keeps its inter-area citing volume and
    the age profile of what it cites, but chooses among the *other* areas' papers of each
    year in proportion to their numbers. Hence an area receives what the age composition of
    its papers predicts. Unlike the margin-preserving nulls, this can test whether an area's
    net inter-area flow is explained by the age of its papers (given citing intensities).
    Homophily plays no role: only inter-area citations enter.
    """
    y0 = int(years.min())
    span = int(years.max()) - y0 + 1
    inter = labels[edges[:, 0]] != labels[edges[:, 1]]
    w = None if weights is None else weights[inter]
    e = edges[inter]
    O = np.bincount(labels[e[:, 0]] * span + (years[e[:, 1]] - y0), weights=w,
                    minlength=k * span).reshape(k, span)
    P = np.bincount(labels * span + (years - y0), minlength=k * span).reshape(k, span)
    P = P.astype(np.float64)
    others = P.sum(axis=0, keepdims=True) - P                     # (k, span): papers of other areas
    rate = np.divide(O, others, out=np.zeros_like(O, dtype=np.float64), where=others > 0)
    E = rate @ P.T
    np.fill_diagonal(E, 0.0)
    return E


@dataclass
class FlowStats:
    """Point estimates derived from one (possibly bootstrap-weighted) citation matrix."""

    M: np.ndarray
    E_strength: np.ndarray
    E_time: np.ndarray
    E_age: np.ndarray

    def category(self) -> dict[str, np.ndarray]:
        M = self.M
        intra = np.diag(M)
        made, received = M.sum(axis=1), M.sum(axis=0)
        inter_out, inter_in = made - intra, received - intra
        tot = inter_in + inter_out
        nnf = np.divide(inter_in - inter_out, tot, out=np.full(tot.shape, np.nan),
                        where=tot > 0)
        out = {"made": made, "received": received, "intra": intra, "inter_out": inter_out,
               "inter_in": inter_in, "nnf": nnf}
        T = M + M.T
        np.fill_diagonal(T, 0)
        for name, E in (("strength", self.E_strength), ("time", self.E_time)):
            share = np.divide(E, E + E.T, out=np.full(E.shape, 0.5), where=(E + E.T) > 0)
            exp_out = (T * share).sum(axis=1)        # expected inter_out given pair totals
            exp_in = (T * share.T).sum(axis=1)
            out[f"nnf_expected_{name}"] = np.divide(exp_in - exp_out, tot,
                                                    out=np.full(tot.shape, np.nan),
                                                    where=tot > 0)
            out[f"nnf_excess_{name}"] = nnf - out[f"nnf_expected_{name}"]
            out[f"homophily_OE_{name}"] = np.divide(intra, np.diag(E),
                                                    out=np.full(intra.shape, np.nan),
                                                    where=np.diag(E) > 0)
        # age null for inter-area citations: keeps inter_out, predicts inter_in from the age
        # composition of each area's papers
        ea_in = self.E_age.sum(axis=0)
        out["inter_in_expected_age"] = ea_in
        out["nnf_age"] = np.divide(ea_in - inter_out, ea_in + inter_out,
                                   out=np.full(tot.shape, np.nan),
                                   where=(ea_in + inter_out) > 0)
        out["inter_in_OE_age"] = np.divide(inter_in, ea_in, out=np.full(tot.shape, np.nan),
                                           where=ea_in > 0)
        return out

    def pairs(self) -> dict[str, np.ndarray]:
        """Upper-triangle pair statistics (i < j), oriented as i -> j."""
        M = self.M
        k = M.shape[0]
        iu, ju = np.triu_indices(k, 1)
        mij, mji = M[iu, ju], M[ju, iu]
        T = mij + mji
        s = (mij + 0.5) / (T + 1.0)
        out = {"i": iu, "j": ju, "M_ij": mij, "M_ji": mji, "total": T, "net_ij": mij - mji,
               "asymmetry_ij": np.divide(mij - mji, T, out=np.full(T.shape, np.nan),
                                         where=T > 0)}
        for name, E in (("strength", self.E_strength), ("time", self.E_time)):
            eij, eji = E[iu, ju], E[ju, iu]
            share = np.divide(eij, eij + eji, out=np.full(eij.shape, 0.5),
                              where=(eij + eji) > 0)
            out[f"expected_share_{name}"] = share
            # same continuity correction for the expected count T * e as for the observed one
            expected_s = (T * share + 0.5) / (T + 1.0)
            out[f"lor_{name}"] = _logit(s) - _logit(expected_s)
            out[f"net_excess_{name}"] = (mij - mji) - T * (2 * share - 1)
        return out


def flow_stats(edges: np.ndarray, labels: np.ndarray, k: int, time_stratum: np.ndarray,
               years: np.ndarray, weights: np.ndarray | None = None) -> FlowStats:
    M = citation_matrix(edges, labels, k, weights)
    E_s = stratified_expected(edges, labels, k, np.zeros(len(edges), dtype=np.int64), weights)
    E_t = stratified_expected(edges, labels, k, time_stratum, weights)
    E_a = age_expected(edges, labels, years, k, weights)
    return FlowStats(M, E_s, E_t, E_a)


CAT_BOOT_KEYS = ("nnf", "nnf_excess_strength", "nnf_excess_time", "homophily_OE_strength",
                 "homophily_OE_time", "nnf_age", "inter_in_OE_age")
PAIR_BOOT_KEYS = ("asymmetry_ij", "lor_strength", "lor_time", "net_ij")


def bootstrap(edges: np.ndarray, labels: np.ndarray, k: int, time_stratum: np.ndarray,
              years: np.ndarray, num_nodes: int, replicates: int, seed: int
              ) -> dict[str, np.ndarray]:
    """Cluster bootstrap over citing papers: arrays of shape (replicates, ...)."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 2, 0]))
    src = edges[:, 0]
    cat_keys, pair_keys = CAT_BOOT_KEYS, PAIR_BOOT_KEYS
    res = {f"cat_{key}": [] for key in cat_keys} | {f"pair_{key}": [] for key in pair_keys}
    p = np.full(num_nodes, 1.0 / num_nodes)
    for _ in range(replicates):
        w = rng.multinomial(num_nodes, p).astype(np.float64)[src]
        st = flow_stats(edges, labels, k, time_stratum, years, weights=w)
        cat, pr = st.category(), st.pairs()
        for key in cat_keys:
            res[f"cat_{key}"].append(cat[key])
        for key in pair_keys:
            res[f"pair_{key}"].append(pr[key])
    return {key: np.asarray(v) for key, v in res.items()}


def _ci(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.nanpercentile(samples, 2.5, axis=0), np.nanpercentile(samples, 97.5, axis=0)


# --------------------------------------------------------------------------- tables


def category_table(stats: FlowStats, boot: dict[str, np.ndarray], labels: np.ndarray,
                   names: list[str], years: np.ndarray) -> pd.DataFrame:
    k = len(names)
    c = stats.category()
    papers = np.bincount(labels, minlength=k)
    year_mean = np.bincount(labels, weights=years, minlength=k) / papers
    year_median = np.array([np.median(years[labels == i]) for i in range(k)])
    recent = np.bincount(labels, weights=(years >= 2018).astype(float), minlength=k) / papers
    table = pd.DataFrame({
        "category": np.arange(k), "category_name": names, "num_papers": papers,
        "year_mean": year_mean, "year_median": year_median, "fraction_papers_2018_2020": recent,
        "citations_made": c["made"], "citations_received": c["received"],
        "intra": c["intra"], "inter_out": c["inter_out"], "inter_in": c["inter_in"],
        "inter_out_per_paper": c["inter_out"] / papers,
        "inter_in_per_paper": c["inter_in"] / papers,
        "intra_fraction_of_made": c["intra"] / c["made"],
        "intra_fraction_of_received": c["intra"] / c["received"],
        "net_inter_flow": c["inter_in"] - c["inter_out"],
        "nnf": c["nnf"],
        "nnf_expected_strength": c["nnf_expected_strength"],
        "nnf_expected_time": c["nnf_expected_time"],
        "nnf_excess_strength": c["nnf_excess_strength"],
        "nnf_excess_time": c["nnf_excess_time"],
        "homophily_OE_strength": c["homophily_OE_strength"],
        "homophily_OE_time": c["homophily_OE_time"],
        "nnf_age": c["nnf_age"],
        "nnf_minus_nnf_age": c["nnf"] - c["nnf_age"],
        "inter_in_expected_age": c["inter_in_expected_age"],
        "inter_in_OE_age": c["inter_in_OE_age"],
    })
    for key in CAT_BOOT_KEYS:
        low, high = _ci(boot[f"cat_{key}"])
        table[f"{key}_ci95_low"], table[f"{key}_ci95_high"] = low, high
    return table


def pair_table(stats: FlowStats, boot: dict[str, np.ndarray], names: list[str]
               ) -> pd.DataFrame:
    p = stats.pairs()
    table = pd.DataFrame({key: value for key, value in p.items()})
    table.insert(2, "area_i", [names[i] for i in p["i"]])
    table.insert(3, "area_j", [names[j] for j in p["j"]])
    for key in ("asymmetry_ij", "lor_strength", "lor_time", "net_ij"):
        low, high = _ci(boot[f"pair_{key}"])
        table[f"{key}_ci95_low"], table[f"{key}_ci95_high"] = low, high
    table = table[table["total"] > 0].copy()
    for name in ("strength", "time"):
        table[f"direction_beyond_{name}_null"] = (
            (table[f"lor_{name}_ci95_low"] > 0) | (table[f"lor_{name}_ci95_high"] < 0))
    return table


def oriented_top_pairs(pairs: pd.DataFrame, by: str, top: int, min_total: int = 0
                       ) -> pd.DataFrame:
    """Top pairs re-oriented so that the dominant direction reads 'A cites B more'."""
    d = pairs[pairs["total"] >= min_total].copy()
    d = d.reindex(d[by].abs().sort_values(ascending=False).index).head(top)
    flip = d["net_ij"] < 0
    rows = []
    for _, r in d.iterrows():
        f = bool(flip.loc[r.name])
        sign = -1 if f else 1
        rows.append({
            "A_cites_more": r["area_j"] if f else r["area_i"],
            "B_cited_more": r["area_i"] if f else r["area_j"],
            "A_to_B": int(r["M_ji"] if f else r["M_ij"]),
            "B_to_A": int(r["M_ij"] if f else r["M_ji"]),
            "total": int(r["total"]), "net": int(abs(r["net_ij"])),
            "asymmetry": abs(r["asymmetry_ij"]),
            "asymmetry_ci95": (sign * r["asymmetry_ij_ci95_low"],
                               sign * r["asymmetry_ij_ci95_high"]),
            "expected_share_A_to_B_strength": (1 - r["expected_share_strength"]) if f
            else r["expected_share_strength"],
            "expected_share_A_to_B_time": (1 - r["expected_share_time"]) if f
            else r["expected_share_time"],
            "lor_strength": sign * r["lor_strength"],
            "lor_time": sign * r["lor_time"],
            "lor_time_ci95": tuple(sorted((sign * r["lor_time_ci95_low"],
                                           sign * r["lor_time_ci95_high"]))),
            "direction_beyond_time_null": bool(r["direction_beyond_time_null"]),
        })
    out = pd.DataFrame(rows)
    for col in ("asymmetry_ci95", "lor_time_ci95"):
        out[f"{col}_low"] = [min(v) for v in out[col]]
        out[f"{col}_high"] = [max(v) for v in out[col]]
        del out[col]
    return out


def spearman(x: np.ndarray, y: np.ndarray, permutations: int = 10_000, seed: int = 0
             ) -> dict[str, Any]:
    """Spearman correlation with a two-sided permutation p-value (n is only 40)."""
    from scipy.stats import rankdata

    rx, ry = rankdata(x), rankdata(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    rng = np.random.default_rng(seed)
    perm = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(permutations)])
    return {"rho": rho, "p_permutation": float((np.abs(perm) >= abs(rho)).mean()),
            "n": len(x), "permutations": permutations}
