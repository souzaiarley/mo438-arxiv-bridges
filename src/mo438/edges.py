"""Intra-area / inter-area classification of the edges of G_d and G_u.

Definition (proposal, section 3.2): a relation is **intra-area** when both endpoints have
the same primary category and **inter-area** otherwise. The class depends only on the two
endpoint labels, so it is symmetric: the two directed citations ``u -> v`` and ``v -> u``
that G_u merges into ``{u, v}`` always have the same class, and the classification of G_u
is unambiguous. This module classifies every edge, decomposes the G_d -> G_u difference
(reciprocal pairs) per class and runs the consistency checks used as Phase-0 validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

INTRA, INTER = "intra", "inter"


def is_inter_area(edges: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Boolean mask: ``True`` where the endpoints of the edge have different categories."""
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    return labels[edges[:, 0]] != labels[edges[:, 1]]


@dataclass
class EdgeTable:
    """Per-edge classification of one graph.

    Attributes:
        graph: ``"G_d"`` or ``"G_u"``.
        src, dst: endpoints (G_d: ``src`` cites ``dst``; G_u: ``src < dst``).
        cat_src, cat_dst: primary-category labels of the endpoints.
        is_inter: inter-area flag.
        multiplicity: G_d -> number of G_d edges on the same *unordered* pair (counts
            duplicates and both directions); G_u -> number of G_d edges merged into it.
        reciprocal: the unordered pair is cited in both directions (``u -> v`` and
            ``v -> u`` both exist in G_d); always ``False`` for self-loops.
    """

    graph: str
    src: np.ndarray
    dst: np.ndarray
    cat_src: np.ndarray
    cat_dst: np.ndarray
    is_inter: np.ndarray
    multiplicity: np.ndarray
    reciprocal: np.ndarray

    @property
    def num_edges(self) -> int:
        return int(self.src.size)

    def counts(self) -> dict[str, Any]:
        inter = int(self.is_inter.sum())
        total = self.num_edges
        return {"graph": self.graph, "num_edges": total, INTRA: total - inter, INTER: inter,
                "inter_fraction": inter / total if total else 0.0}

    def content_sha256(self) -> str:
        """Hash of the table *content* only: unlike the hash of the ``.npz`` file, it does not
        depend on the embedded metadata (paths, versions) or on the npz/zip encoding."""
        import hashlib

        digest = hashlib.sha256(self.graph.encode())
        for array in (self.src, self.dst, self.cat_src, self.cat_dst, self.is_inter,
                      self.multiplicity, self.reciprocal):
            digest.update(np.ascontiguousarray(array).astype(array.dtype.newbyteorder("<"))
                          .tobytes())
        return digest.hexdigest()

    def save(self, path: Path, metadata: dict[str, Any] | None = None) -> None:
        """Write the table as a compressed ``.npz`` (regenerable cache, not versioned)."""
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(
            tmp, src=self.src, dst=self.dst, cat_src=self.cat_src, cat_dst=self.cat_dst,
            is_inter=self.is_inter, multiplicity=self.multiplicity, reciprocal=self.reciprocal,
            meta=np.array(json.dumps({"graph": self.graph, **(metadata or {})})),
        )
        tmp.replace(path)


def _unordered_keys(edges: np.ndarray, n: int) -> np.ndarray:
    return np.minimum(edges[:, 0], edges[:, 1]) * np.int64(n) + np.maximum(edges[:, 0],
                                                                           edges[:, 1])


def _reverse_exists(gd_edges: np.ndarray, n: int) -> np.ndarray:
    """For every directed edge ``u -> v``: does ``v -> u`` exist (``u != v``)?"""
    src, dst = gd_edges[:, 0], gd_edges[:, 1]
    keys = np.unique(src * np.int64(n) + dst)
    reverse = dst * np.int64(n) + src
    pos = np.minimum(np.searchsorted(keys, reverse), keys.size - 1)
    return (keys[pos] == reverse) & (src != dst)


def classify_directed(gd_edges: np.ndarray, labels: np.ndarray, n: int) -> EdgeTable:
    """Classify every citation of G_d (keeps the G_d edge order)."""
    gd_edges = np.asarray(gd_edges, dtype=np.int64).reshape(-1, 2)
    keys = _unordered_keys(gd_edges, n)
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    return EdgeTable(
        graph="G_d", src=gd_edges[:, 0], dst=gd_edges[:, 1],
        cat_src=labels[gd_edges[:, 0]], cat_dst=labels[gd_edges[:, 1]],
        is_inter=is_inter_area(gd_edges, labels),
        multiplicity=np.minimum(counts[inverse], 127).astype(np.int8),
        reciprocal=_reverse_exists(gd_edges, n),
    )


def classify_undirected(gu_edges: np.ndarray, gd_edges: np.ndarray, labels: np.ndarray,
                        n: int) -> EdgeTable:
    """Classify every edge of G_u and record how many G_d edges were merged into it."""
    gu_edges = np.asarray(gu_edges, dtype=np.int64).reshape(-1, 2)
    gd_edges = np.asarray(gd_edges, dtype=np.int64).reshape(-1, 2)
    if np.any(gu_edges[:, 0] >= gu_edges[:, 1]):
        raise ValueError("G_u edges must satisfy u < v (simple graph, no self-loops)")
    gu_keys = gu_edges[:, 0] * np.int64(n) + gu_edges[:, 1]
    if np.any(np.diff(gu_keys) <= 0):
        raise ValueError("G_u edges must be sorted and unique")
    non_loop = gd_edges[gd_edges[:, 0] != gd_edges[:, 1]]
    uniq, counts = np.unique(_unordered_keys(non_loop, n), return_counts=True)
    if not np.array_equal(uniq, gu_keys):
        raise ValueError("G_u is not the set of unordered non-loop pairs of G_d")
    directed_keys = np.unique(non_loop[:, 0] * np.int64(n) + non_loop[:, 1])
    forward = np.isin(gu_edges[:, 0] * np.int64(n) + gu_edges[:, 1], directed_keys)
    backward = np.isin(gu_edges[:, 1] * np.int64(n) + gu_edges[:, 0], directed_keys)
    return EdgeTable(
        graph="G_u", src=gu_edges[:, 0], dst=gu_edges[:, 1],
        cat_src=labels[gu_edges[:, 0]], cat_dst=labels[gu_edges[:, 1]],
        is_inter=is_inter_area(gu_edges, labels),
        multiplicity=np.minimum(counts, 127).astype(np.int8),
        reciprocal=forward & backward,
    )


# --------------------------------------------------------------------------- summaries


def class_counts_table(gd: EdgeTable, gu: EdgeTable) -> pd.DataFrame:
    """Long table: graph x class -> number and fraction of edges."""
    rows = []
    for table in (gd, gu):
        c = table.counts()
        for cls in (INTRA, INTER):
            rows.append({"graph": table.graph, "class": cls, "num_edges": c[cls],
                         "fraction_of_graph": c[cls] / c["num_edges"]})
        rows.append({"graph": table.graph, "class": "total", "num_edges": c["num_edges"],
                     "fraction_of_graph": 1.0})
    return pd.DataFrame(rows)


def reciprocal_decomposition(gd: EdgeTable, gu: EdgeTable) -> dict[str, Any]:
    """Explain ``|G_d class| - |G_u class|`` for each class.

    ``difference = self_loops + extra_merged_edges`` where ``extra_merged_edges`` is the
    sum over G_u edges of ``(G_d edges merged - 1)``; it splits into reciprocal pairs
    (one extra edge each) and duplicated citations. Self-loops are intra-area by
    definition (both endpoints are the same paper).
    """
    out: dict[str, Any] = {}
    loops_d = gd.src == gd.dst
    for cls, mask_d, mask_u in ((INTRA, ~gd.is_inter, ~gu.is_inter),
                                (INTER, gd.is_inter, gu.is_inter)):
        extra = int((gu.multiplicity[mask_u].astype(np.int64) - 1).sum())
        recip_pairs = int(gu.reciprocal[mask_u].sum())
        out[cls] = {
            "G_d_edges": int(mask_d.sum()),
            "G_u_edges": int(mask_u.sum()),
            "difference": int(mask_d.sum() - mask_u.sum()),
            "self_loops_removed": int((loops_d & mask_d).sum()),
            "extra_merged_edges": extra,
            "reciprocal_pairs_merged": recip_pairs,
            "duplicate_citations_merged": extra - recip_pairs,
            "G_d_edges_in_reciprocal_pairs": int(gd.reciprocal[mask_d].sum()),
        }
    out["total_reciprocal_pairs"] = out[INTRA]["reciprocal_pairs_merged"] + \
        out[INTER]["reciprocal_pairs_merged"]
    return out


def per_category_table(gd: EdgeTable, gu: EdgeTable, labels: np.ndarray,
                       category_names: list[str]) -> pd.DataFrame:
    """Per primary category: papers and edge counts by class in G_d and G_u.

    ``G_d_inter_out`` = inter-area citations *made* by papers of the category,
    ``G_d_inter_in`` = inter-area citations *received*; ``G_u_inter_incident`` counts the
    inter-area edges of G_u with one endpoint in the category (each inter edge is
    counted for both of its categories, so the column sums to ``2 * |inter(G_u)|``).
    """
    k = len(category_names)
    intra_d = ~gd.is_inter
    intra_u = ~gu.is_inter
    return pd.DataFrame({
        "category": np.arange(k),
        "category_name": category_names,
        "num_papers": np.bincount(labels, minlength=k),
        "G_d_intra": np.bincount(gd.cat_src[intra_d], minlength=k),
        "G_d_inter_out": np.bincount(gd.cat_src[gd.is_inter], minlength=k),
        "G_d_inter_in": np.bincount(gd.cat_dst[gd.is_inter], minlength=k),
        "G_u_intra": np.bincount(gu.cat_src[intra_u], minlength=k),
        "G_u_inter_incident": np.bincount(gu.cat_src[gu.is_inter], minlength=k)
        + np.bincount(gu.cat_dst[gu.is_inter], minlength=k),
    })


def node_composition(gu: EdgeTable, n: int) -> dict[str, int]:
    """How the edges of each node of G_u split between the two classes.

    Nodes with only inter-area edges become isolated when all inter-area edges are
    removed (and vice versa); this matters for the removal experiments (objective 4).
    """
    inter_deg = np.bincount(gu.src[gu.is_inter], minlength=n) + \
        np.bincount(gu.dst[gu.is_inter], minlength=n)
    intra_deg = np.bincount(gu.src[~gu.is_inter], minlength=n) + \
        np.bincount(gu.dst[~gu.is_inter], minlength=n)
    deg = inter_deg + intra_deg
    return {
        "num_nodes": int(n),
        "isolated": int((deg == 0).sum()),
        "only_intra_edges": int(((intra_deg > 0) & (inter_deg == 0)).sum()),
        "only_inter_edges": int(((inter_deg > 0) & (intra_deg == 0)).sum()),
        "both_classes": int(((inter_deg > 0) & (intra_deg > 0)).sum()),
    }


# --------------------------------------------------------------------------- checks


def _check(name: str, observed: Any, expected: Any, source: str) -> dict[str, Any]:
    return {"check": name, "observed": observed, "expected": expected,
            "passed": bool(observed == expected), "reference": source}


def validate_classification(gd: EdgeTable, gu: EdgeTable, labels: np.ndarray,
                            num_nodes: int, area_matrix: np.ndarray | None = None
                            ) -> list[dict[str, Any]]:
    """Consistency checks of the classification (each returns observed vs expected).

    Args:
        area_matrix: optional citation matrix of the characterization pipeline
            (``outputs/obj1_characterization/area_citation_matrix.csv``)
            used as an external reference for the G_d counts.
    """
    checks = []
    cd, cu = gd.counts(), gu.counts()
    checks.append(_check("G_d intra + inter = m_d", cd[INTRA] + cd[INTER], gd.num_edges,
                         "definition"))
    checks.append(_check("G_u intra + inter = m_u", cu[INTRA] + cu[INTER], gu.num_edges,
                         "definition"))
    # Independent recount through the category-pair histogram (different code path).
    k = int(labels.max()) + 1
    pair_hist = np.bincount(labels[gd.src] * k + labels[gd.dst], minlength=k * k).reshape(k, k)
    checks.append(_check("G_d intra = trace of recomputed category matrix",
                         cd[INTRA], int(np.trace(pair_hist)), "recount (bincount of label pairs)"))
    if area_matrix is not None:
        checks.append(_check("G_d intra = trace(F1 area_citation_matrix.csv)",
                             cd[INTRA], int(np.trace(area_matrix)), "F1 output"))
        checks.append(_check("G_d inter = off-diagonal sum of F1 area_citation_matrix.csv",
                             cd[INTER], int(area_matrix.sum() - np.trace(area_matrix)),
                             "F1 output"))
        checks.append(_check("recomputed category matrix = F1 area_citation_matrix.csv",
                             bool(np.array_equal(pair_hist, area_matrix)), True, "F1 output"))
    dec = reciprocal_decomposition(gd, gu)
    for cls in (INTRA, INTER):
        checks.append(_check(f"{cls}: |G_d| - |G_u| = self-loops + extra merged edges",
                             dec[cls]["difference"],
                             dec[cls]["self_loops_removed"] + dec[cls]["extra_merged_edges"],
                             "G_d -> G_u conversion"))
    reciprocated = np.unique(gd.src[gd.reciprocal] * np.int64(num_nodes) + gd.dst[gd.reciprocal])
    checks.append(_check("reciprocal pairs in G_u = distinct reciprocated G_d citations / 2",
                         dec["total_reciprocal_pairs"], int(reciprocated.size // 2),
                         "G_d reverse-edge lookup"))
    checks.append(_check("sum of G_u multiplicities = non-loop edges of G_d",
                         int(gu.multiplicity.astype(np.int64).sum()),
                         int((gd.src != gd.dst).sum()), "G_d -> G_u conversion"))
    # The class of a G_u edge equals the class of every G_d citation merged into it.
    non_loop = gd.src != gd.dst
    gu_keys = gu.src * np.int64(num_nodes) + gu.dst
    gd_keys = (np.minimum(gd.src, gd.dst) * np.int64(num_nodes)
               + np.maximum(gd.src, gd.dst))[non_loop]
    pos = np.searchsorted(gu_keys, gd_keys)
    found = bool(np.array_equal(gu_keys[np.minimum(pos, gu_keys.size - 1)], gd_keys))
    checks.append(_check("every non-loop G_d citation maps to a G_u edge", found, True,
                         "G_d -> G_u conversion"))
    checks.append(_check("every G_d citation has the class of its G_u edge",
                         found and bool(np.array_equal(gu.is_inter[pos], gd.is_inter[non_loop])),
                         True, "symmetry of the definition"))
    return checks
