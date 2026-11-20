"""Cross-objective integration: edge properties (obj. 2-3) vs robustness (obj. 4-5).

Everything here is computed from the CSV outputs of the other commands (no new
experiment). Two analyses:

1. **First-order bridge prediction of the initial slope of S(q).** If k edges of a class
   are removed at random and k is small, removals rarely interact, so the expected number
   of vertices cut off from the giant component per removed edge is approximately the
   class average of "vertices that the edge alone would detach" = (sum over the class's
   bridges of the size of their smaller side) / (edges in the class). This is compared
   with the observed vertices outside the LCC per removed edge at the smallest q.
2. **Category-level associations** (Spearman, n = 40, permutation p-values): internal
   cohesion without inter-area edges (obj. 4-5 endpoint) vs share of inter-area edges,
   net flow and age (obj. 2), and overlap of inter-area edges (obj. 3).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .flows import spearman


def bridge_prediction(base: Path, n: int) -> pd.DataFrame:
    bridges = pd.read_csv(base / "obj3_overlap" / "bridges_by_class.csv").set_index("class")
    summary = pd.read_csv(base / "obj4_5_removal" / "summary.csv")
    rows = []
    mapping = {"inter": "inter", "intra": "intra", "uniform": "all"}
    for strategy, cls in mapping.items():
        b = bridges.loc[cls]
        pred_total = b["vertices_detachable_total"] / b["edges"]
        pred_iso = b["pendant_bridges"] / b["edges"]
        pred_frag = b["vertices_detachable_by_non_pendant"] / b["edges"]
        d = summary[(summary["strategy"] == strategy) & (summary["k"] > 0)].sort_values("k")
        for _, r in d.head(8).iterrows():
            k = r["k"]
            outside = n * (1 - r["S_mean"])
            rows.append({
                "strategy": strategy, "edge_class": cls, "q": r["q"], "k": int(k),
                "observed_outside_lcc_per_edge": outside / k,
                "predicted_outside_lcc_per_edge": pred_total,
                "observed_isolated_per_edge": r["num_isolated_mean"] / k,
                "predicted_isolated_per_edge (pendant bridges)": pred_iso,
                "observed_small_components_per_edge": r["outside_lcc_non_isolated_mean"] / k,
                "predicted_small_components_per_edge (non-pendant bridges)": pred_frag,
            })
    return pd.DataFrame(rows)


def category_table(base: Path) -> pd.DataFrame:
    flows = pd.read_csv(base / "obj2_flows" / "category_flows.csv")
    over = pd.read_csv(base / "obj3_overlap" / "overlap_by_category.csv")
    cohesion = pd.read_csv(base / "obj4_5_removal" / "endpoint_all_inter_removed_by_category.csv")
    table = flows[["category_name", "num_papers", "year_mean", "nnf", "inter_in_OE_age",
                   "intra_fraction_of_made"]].merge(
        over[["category_name", "inter_share_of_incident_edges", "mean_intra", "mean_inter",
              "cliffs_delta", "bridge_frac_inter", "bridge_frac_intra"]], on="category_name"
    ).merge(cohesion[["category", "largest_intra_component_fraction"]].rename(
        columns={"category": "category_name"}), on="category_name")
    return table


ASSOCIATIONS = (
    ("largest_intra_component_fraction", "inter_share_of_incident_edges"),
    ("largest_intra_component_fraction", "num_papers"),
    ("largest_intra_component_fraction", "nnf"),
    ("largest_intra_component_fraction", "year_mean"),
    ("inter_share_of_incident_edges", "num_papers"),
    ("cliffs_delta", "inter_share_of_incident_edges"),
    ("cliffs_delta", "num_papers"),
    ("mean_inter", "nnf"),
    ("bridge_frac_inter", "largest_intra_component_fraction"),
)


def associations(table: pd.DataFrame, permutations: int = 10_000) -> list[dict[str, Any]]:
    out = []
    for a, b in ASSOCIATIONS:
        d = table[[a, b]].dropna()
        res = spearman(d[a].to_numpy(), d[b].to_numpy(), permutations=permutations, seed=7)
        out.append({"x": a, "y": b, **res})
    return out


def partial_spearman(table: pd.DataFrame, x: str, y: str, control: str) -> float:
    """Spearman correlation of x and y after removing the rank-linear effect of control."""
    from scipy.stats import rankdata

    rx, ry, rc = (rankdata(table[c]) for c in (x, y, control))
    res = []
    for r in (rx, ry):
        beta = np.polyfit(rc, r, 1)
        res.append(r - np.polyval(beta, rc))
    return float(np.corrcoef(res[0], res[1])[0, 1])


# --------------------------------------------------------------------------- objectives 6-7


def communities_vs_flows(base: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Do communities co-group categories that cite each other unusually often?

    Pair level (780 category pairs): co-membership probability P_ij (a random paper of i
    and one of j fall in the same community, reference Leiden run) vs the symmetric
    observed/expected citation intensity (M_ij + M_ji) / (E_ij + E_ji) under the strength
    null of objective 2. Category level: spread over communities vs share of inter-area
    edges, internal cohesion, size and net flow.
    """
    from .communities import comembership

    N = pd.read_csv(base / "obj6_communities" / "contingency_reference_run.csv", index_col=0)
    names = N.index.tolist()
    P = comembership(N.to_numpy(dtype=float))
    M = pd.read_csv(base / "obj2_flows" / "matrix_observed.csv", index_col=0).loc[names, names]
    E = pd.read_csv(base / "obj2_flows" / "matrix_expected_strength.csv",
                    index_col=0).loc[names, names]
    M, E = M.to_numpy(dtype=float), E.to_numpy(dtype=float)
    iu, ju = np.triu_indices(len(names), 1)
    T = M[iu, ju] + M[ju, iu]
    oe = T / (E[iu, ju] + E[ju, iu])
    pairs = pd.DataFrame({"area_i": [names[i] for i in iu], "area_j": [names[j] for j in ju],
                          "comembership": P[iu, ju], "citations_total": T,
                          "citation_OE_symmetric": oe})
    ok = pairs["citations_total"] > 0
    res = {
        "pairs": int(ok.sum()),
        "comembership_vs_citation_OE": spearman(pairs.loc[ok, "comembership"].to_numpy(),
                                                pairs.loc[ok, "citation_OE_symmetric"]
                                                .to_numpy(), permutations=2000, seed=11),
        "comembership_vs_citations_total": spearman(pairs.loc[ok, "comembership"].to_numpy(),
                                                    pairs.loc[ok, "citations_total"].to_numpy(),
                                                    permutations=2000, seed=12),
    }
    cats = pd.read_csv(base / "obj6_communities" / "categories_vs_communities.csv")
    cross = category_table(base).merge(
        cats[["category_name", "share_in_dominant_community", "effective_num_communities",
              "mutual_best_match"]], on="category_name")
    cross["spread"] = 1 - cross["share_in_dominant_community"]
    for other in ("inter_share_of_incident_edges", "largest_intra_component_fraction",
                  "num_papers", "nnf", "year_mean"):
        res[f"spread_vs_{other}"] = spearman(cross["spread"].to_numpy(),
                                             cross[other].to_numpy(), seed=13)
    res["spread_vs_inter_share_given_num_papers"] = partial_spearman(
        cross, "spread", "inter_share_of_incident_edges", "num_papers")
    return pairs, cross, res


def ordering_vs_others(base: Path) -> dict[str, Any]:
    """Position in the objective-7 order vs objective-2 and objective-4/5 quantities."""
    order = pd.read_csv(base / "obj7_ordering" / "ordering.csv")
    flows = pd.read_csv(base / "obj2_flows" / "category_flows.csv")
    d = order.merge(flows[["category_name", "nnf_age", "inter_in_OE_age"]], on="category_name")
    d = d.merge(category_table(base)[["category_name", "largest_intra_component_fraction"]],
                on="category_name")
    out = {}
    for other in ("nnf", "nnf_age", "inter_in_OE_age", "year_mean", "num_papers",
                  "largest_intra_component_fraction"):
        out[f"position_vs_{other}"] = spearman(d["position"].to_numpy(), d[other].to_numpy(),
                                               seed=14)
    out["position_vs_year_given_nnf"] = partial_spearman(d, "position", "year_mean", "nnf")
    return out
