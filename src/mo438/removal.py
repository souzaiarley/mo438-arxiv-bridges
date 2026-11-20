"""Objectives 4 and 5: progressive edge removal on G_u and random controls.

Experimental design (approved 2026-11-06, before the experiments were run):

* q = k / m with m = |E(G_u)| = 1,157,799; strategies are compared at the **same absolute
  number k** of removed edges, on the common domain 0 <= k <= k_max = |inter| = 400,344.
* Strategies (random order within the eligible set, 50 reproducible seeds each):
  ``inter`` (inter-area edges), ``intra`` (intra-area edges), ``uniform`` (any edge);
  complementary degree-matched controls ``matched_inter`` / ``matched_intra`` (see
  :func:`matched_order`).
* For one seed, the removed sets are **nested**: the set removed at k is the first k edges of
  a random permutation of the eligible edges. Hence the whole curve can be computed exactly
  with one union-find sweep that starts from the graph at k_max and *adds the edges back* in
  reverse order (Newman & Ziff, PRL 85, 4104, 2000), recording the metrics at the grid
  points.

Metrics at each k: size of the largest component (S = LCC / n), number of components,
number of isolated vertices and size of the second largest component.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

BASE_SEED = 438

# strategy name -> (seed-stream code, description)
STRATEGIES: dict[str, tuple[int, str]] = {
    "inter": (1, "uniformly random order of the inter-area edges"),
    "intra": (2, "uniformly random order of the intra-area edges"),
    "uniform": (3, "uniformly random order of all edges of G_u"),
    "matched_inter": (4, ("degree-matched control: random edges of the whole graph with the "
                          "same endpoint-degree stratum sequence as a random inter-area order")),
    "matched_intra": (5, ("degree-matched control: random edges of the whole graph with the "
                          "same endpoint-degree stratum sequence as a random intra-area order")),
}
MAIN_STRATEGIES = ("inter", "intra", "uniform")
MATCHED_STRATEGIES = ("matched_inter", "matched_intra")


def rng_for(strategy: str, replicate: int, base_seed: int = BASE_SEED) -> np.random.Generator:
    """Independent, reproducible stream for (strategy, replicate).

    ``numpy.random.SeedSequence([base_seed, strategy_code, replicate])``.
    """
    code = STRATEGIES[strategy][0]
    return np.random.default_rng(np.random.SeedSequence([base_seed, code, replicate]))


# --------------------------------------------------------------------------- grid


TAIL_FRACTIONS = (0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.999)


def default_q_values(q_max: float) -> np.ndarray:
    """Grid of q: dense for small and intermediate q, capped at ``q_max``.

    0.0001-0.0005 (very small), 0.001 steps up to 0.02, 0.0025 steps up to 0.10,
    0.01 steps up to 0.34, a fine tail at ``f * q_max`` for f in :data:`TAIL_FRACTIONS`
    (where the inter-area class is almost exhausted), plus ``q_max`` itself.
    """
    q = np.concatenate([
        [0.0, 0.0001, 0.00025, 0.0005],
        np.arange(1, 21) * 0.001,
        0.02 + np.arange(1, 33) * 0.0025,
        0.10 + np.arange(1, 25) * 0.01,
    ])
    q = np.round(q, 6)
    tail = np.asarray(TAIL_FRACTIONS) * q_max
    return np.unique(np.concatenate([q[q < q_max], tail, [q_max]]))


def k_grid(q_values: np.ndarray, m: int, k_max: int) -> np.ndarray:
    """Absolute numbers of removed edges for the q grid (unique, sorted, within [0, k_max])."""
    k = np.rint(np.asarray(q_values) * m).astype(np.int64)
    k = np.unique(np.clip(k, 0, k_max))
    if k[-1] != k_max:
        k = np.append(k, k_max)
    return k


# --------------------------------------------------------------------------- strata


def degree_bin(degree: np.ndarray) -> np.ndarray:
    """Logarithmic degree bin ``floor(log2 k)`` (k >= 1): {1}, {2,3}, {4..7}, ..."""
    degree = np.asarray(degree, dtype=np.int64)
    if np.any(degree < 1):
        raise ValueError("degree bins are defined for k >= 1")
    return np.floor(np.log2(degree)).astype(np.int64)


def edge_strata(edges: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Stratum of every edge: (bin(min(k_u, k_v)), bin(max(k_u, k_v))) in the intact graph.

    Returns:
        ``(stratum_id, pairs)`` where ``pairs[s] = (low_bin, high_bin)`` of stratum ``s``.
    """
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    deg = np.bincount(edges.ravel(), minlength=n)
    bu, bv = degree_bin(deg[edges[:, 0]]), degree_bin(deg[edges[:, 1]])
    low, high = np.minimum(bu, bv), np.maximum(bu, bv)
    pairs, stratum = np.unique(np.column_stack([low, high]), axis=0, return_inverse=True)
    return stratum.ravel().astype(np.int64), pairs


def _within_stratum_rank(strata_sequence: np.ndarray) -> np.ndarray:
    """``rank[i]`` = number of earlier positions j < i with the same stratum."""
    order = np.argsort(strata_sequence, kind="stable")
    sorted_s = strata_sequence[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_s)) + 1]
    run_start = np.repeat(starts, np.diff(np.r_[starts, sorted_s.size]))
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size) - run_start
    return rank


def matched_order(target_order: np.ndarray, pool_mask: np.ndarray, strata: np.ndarray,
                  rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """Degree-matched control sequence for a target removal order.

    Position i of the control is the next unused edge, in a random permutation of the
    *pool* edges of stratum ``strata[target_order[i]]``. Every prefix of the control has the
    same stratum composition as the same prefix of the target.

    Args:
        target_order: edge indices of the target removal order.
        pool_mask: edges eligible for the control (variant A: all edges; variant B: the
            complementary class only).
        strata: stratum id of every edge.

    Returns:
        ``(control_order, feasible_length)``: ``feasible_length`` is the longest prefix that
        could be matched; ``control_order`` has that length.
    """
    num_strata = int(strata.max()) + 1
    pool = np.flatnonzero(pool_mask)
    pool = pool[rng.permutation(pool.size)]
    pool = pool[np.argsort(strata[pool], kind="stable")]       # grouped by stratum, shuffled
    offsets = np.r_[0, np.cumsum(np.bincount(strata[pool], minlength=num_strata))]
    capacity = np.diff(offsets)
    seq = strata[target_order]
    rank = _within_stratum_rank(seq)
    bad = np.flatnonzero(rank >= capacity[seq])
    feasible = int(bad[0]) if bad.size else int(seq.size)
    control = pool[offsets[seq[:feasible]] + rank[:feasible]]
    return control, feasible


# --------------------------------------------------------------------------- orders


@dataclass
class EdgeClasses:
    """What the strategies need to know about the edges of G_u."""

    edges: np.ndarray        # (m, 2)
    num_nodes: int
    is_inter: np.ndarray     # (m,) bool
    strata: np.ndarray       # (m,) int

    @property
    def num_edges(self) -> int:
        return int(self.edges.shape[0])

    @property
    def k_max(self) -> int:
        """Common domain: the smallest eligible set among inter / intra / uniform."""
        inter = int(self.is_inter.sum())
        return min(inter, self.num_edges - inter)


def removal_order(strategy: str, replicate: int, classes: EdgeClasses, k_max: int,
                  base_seed: int = BASE_SEED) -> tuple[np.ndarray, dict[str, Any]]:
    """First ``k_max`` edges (indices into ``classes.edges``) of the removal order."""
    rng = rng_for(strategy, replicate, base_seed)
    info: dict[str, Any] = {"strategy": strategy, "replicate": replicate,
                            "seed_entropy": [base_seed, STRATEGIES[strategy][0], replicate]}
    if strategy in ("inter", "intra", "uniform"):
        eligible = {"inter": classes.is_inter, "intra": ~classes.is_inter,
                    "uniform": np.ones(classes.num_edges, dtype=bool)}[strategy]
        idx = np.flatnonzero(eligible)
        if idx.size < k_max:
            raise ValueError(f"{strategy}: only {idx.size} eligible edges < k_max={k_max}")
        order = idx[rng.permutation(idx.size)[:k_max]]
    elif strategy in MATCHED_STRATEGIES:
        target_mask = classes.is_inter if strategy == "matched_inter" else ~classes.is_inter
        idx = np.flatnonzero(target_mask)
        target = idx[rng.permutation(idx.size)[:k_max]]
        order, feasible = matched_order(target, np.ones(classes.num_edges, dtype=bool),
                                        classes.strata, rng)
        if feasible < k_max:  # cannot happen for variant A (pool contains the target)
            raise RuntimeError(f"{strategy}: matching infeasible after {feasible} edges")
        info["fraction_of_control_in_target_class"] = float(target_mask[order].mean())
        info["overlap_with_own_target_edges"] = float(
            np.isin(order, target, assume_unique=True).mean())
    else:
        raise KeyError(strategy)
    info["order_sha256"] = hashlib.sha256(np.ascontiguousarray(order, dtype="<i8")
                                          .tobytes()).hexdigest()
    return order, info


# --------------------------------------------------------------------------- sweeps

METRICS = ("lcc_size", "num_components", "num_isolated", "second_lcc_size")


def _components_of_kept(edges: np.ndarray, n: int, keep: np.ndarray) -> np.ndarray:
    kept = edges[keep]
    adj = sp.csr_matrix((np.ones(kept.shape[0], dtype=np.int8), (kept[:, 0], kept[:, 1])),
                        shape=(n, n))
    _, labels = connected_components(adj, directed=False)
    return labels


def _metrics_from_sizes(sizes: np.ndarray) -> tuple[int, int, int, int]:
    top = np.sort(sizes)[-2:] if sizes.size >= 2 else np.r_[0, sizes]
    return int(top[-1]), int(sizes.size), int((sizes == 1).sum()), int(top[0])


def sweep_union_find(edges: np.ndarray, n: int, order: np.ndarray,
                     record_k: np.ndarray) -> dict[str, np.ndarray]:
    """Metrics after removing ``order[:k]`` for every k in ``record_k`` (reverse sweep).

    The graph with ``order[:k_max]`` removed is decomposed once with scipy; the removed
    edges are then added back from ``order[k_max-1]`` down to ``order[0]`` with a
    union-find (union by size, path halving), and the metrics are updated incrementally:
    number of components, isolated vertices, largest size and a size histogram (for the
    second largest component).
    """
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    record_k = np.asarray(record_k, dtype=np.int64)
    k_max = int(record_k.max())
    if k_max > order.size:
        raise ValueError("record_k exceeds the length of the removal order")
    keep = np.ones(edges.shape[0], dtype=bool)
    keep[order[:k_max]] = False
    labels = _components_of_kept(edges, n, keep)
    sizes = np.bincount(labels)
    reps = np.unique(labels, return_index=True)[1]
    parent = reps[labels].tolist()
    size_arr = np.zeros(n, dtype=np.int64)
    size_arr[reps] = sizes
    size = size_arr.tolist()
    uniq, counts = np.unique(sizes, return_counts=True)
    hist = dict(zip(uniq.tolist(), counts.tolist()))
    ncomp, iso, largest = int(sizes.size), int(hist.get(1, 0)), int(sizes.max())

    src = edges[order[:k_max], 0].tolist()
    dst = edges[order[:k_max], 1].tolist()
    wanted = set(record_k.tolist())
    out = {name: {} for name in METRICS}

    def snapshot(k: int) -> None:
        if hist[largest] >= 2:
            second = largest
        else:
            second = max((s for s in hist if s != largest), default=0)
        out["lcc_size"][k] = largest
        out["num_components"][k] = ncomp
        out["num_isolated"][k] = iso
        out["second_lcc_size"][k] = second

    for k in range(k_max, 0, -1):
        if k in wanted:
            snapshot(k)
        u, v = src[k - 1], dst[k - 1]
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        if u == v:
            continue
        a, b = size[u], size[v]
        if a < b:
            u, v, a, b = v, u, b, a
        parent[v] = u
        size[u] = a + b
        ncomp -= 1
        for s in (a, b):
            c = hist[s] - 1
            if c:
                hist[s] = c
            else:
                del hist[s]
        hist[a + b] = hist.get(a + b, 0) + 1
        iso -= (a == 1) + (b == 1)
        if a + b > largest:  # noqa: PLR1730 (hot loop: a branch is faster than max())
            largest = a + b
    if 0 in wanted:
        snapshot(0)
    return {name: np.array([out[name][k] for k in record_k], dtype=np.int64)
            for name in METRICS}


def metrics_forward(edges: np.ndarray, n: int, order: np.ndarray,
                    record_k: np.ndarray) -> dict[str, np.ndarray]:
    """Independent reference: recompute components from scratch (scipy) at every k."""
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    rows = {name: [] for name in METRICS}
    for k in np.asarray(record_k, dtype=np.int64):
        keep = np.ones(edges.shape[0], dtype=bool)
        keep[order[:k]] = False
        sizes = np.bincount(_components_of_kept(edges, n, keep))
        for name, value in zip(METRICS, _metrics_from_sizes(sizes)):
            rows[name].append(value)
    return {name: np.asarray(values, dtype=np.int64) for name, values in rows.items()}


# --------------------------------------------------------------------------- one run


def run_replicate(strategy: str, replicate: int, classes: EdgeClasses, record_k: np.ndarray,
                  base_seed: int = BASE_SEED) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Removal order + union-find sweep for one (strategy, replicate)."""
    k_max = int(np.max(record_k))
    t0 = time.perf_counter()
    order, info = removal_order(strategy, replicate, classes, k_max, base_seed)
    t1 = time.perf_counter()
    metrics = sweep_union_find(classes.edges, classes.num_nodes, order, record_k)
    info["seconds_order"] = round(t1 - t0, 4)
    info["seconds_sweep"] = round(time.perf_counter() - t1, 4)
    return metrics, info


# --------------------------------------------------------------------------- feasibility


def matching_feasibility(classes: EdgeClasses, k_max: int, replicates: int = 50,
                         base_seed: int = BASE_SEED) -> dict[str, Any]:
    """Supply/demand of the degree-matched controls, per stratum and per variant.

    Variant A draws from the whole graph (always feasible: the pool contains the target).
    Variant B draws only from the complementary class (isolates the class with the degree
    profile fixed); it is feasible only while every stratum of the complementary class has
    enough edges. For each target class this reports, per stratum, the target count at
    k_max (expected demand) and the supply under both variants, and the realised feasible
    prefix length of variant B over ``replicates`` random target orders.
    """
    report: dict[str, Any] = {"num_strata": int(classes.strata.max()) + 1}
    num_strata = report["num_strata"]
    all_count = np.bincount(classes.strata, minlength=num_strata)
    for target in ("inter", "intra"):
        mask = classes.is_inter if target == "inter" else ~classes.is_inter
        idx = np.flatnonzero(mask)
        target_count = np.bincount(classes.strata[mask], minlength=num_strata)
        complement = all_count - target_count
        expected_demand = target_count * (k_max / idx.size)
        feasible_lengths = []
        for r in range(replicates):
            rng = np.random.default_rng(np.random.SeedSequence([base_seed, 90, r]))
            order = idx[rng.permutation(idx.size)[:k_max]]
            _, feasible = matched_order(order, ~mask, classes.strata, rng)
            feasible_lengths.append(feasible)
        feasible_lengths = np.asarray(feasible_lengths)
        short = expected_demand > complement
        report[target] = {
            "target_edges": int(idx.size),
            "variant_A_always_feasible": bool(np.all(all_count >= target_count)),
            "variant_B_strata_short_at_kmax": int(short.sum()),
            "variant_B_target_edges_in_short_strata_at_kmax":
                int(target_count[short].sum()),
            "variant_B_feasible_k": {
                "min": int(feasible_lengths.min()), "median": float(np.median(feasible_lengths)),
                "max": int(feasible_lengths.max())},
            "variant_B_feasible_q": {
                "min": float(feasible_lengths.min() / classes.num_edges),
                "median": float(np.median(feasible_lengths) / classes.num_edges)},
            "per_stratum": {
                "target_count": target_count.tolist(),
                "complement_count": complement.tolist(),
                "expected_demand_at_kmax": np.round(expected_demand, 1).tolist(),
            },
        }
    return report
