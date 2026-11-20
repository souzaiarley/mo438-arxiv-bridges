"""Independent recomputation of the characterization metrics published in the F1
(numpy/scipy only).

This module intentionally does **not** import igraph nor the characterization code
(:mod:`mo438.metrics`, :mod:`mo438.paths`, :mod:`mo438.graph_build`): it is a second,
independent code path used to audit the F1 numbers. Every function is itself tested
against networkx/igraph on small graphs (``tests/test_verification_tools.py``).

Kinds of evidence (kept explicit in the reports, as approved in the audit):

* ``recomputed_independently`` — value recomputed here from the graph;
* ``consistency`` — an F1 output is checked for internal consistency (e.g. the distance
  histogram sums to ``n(n-1)``), which does not recompute the value itself;
* ``lower_bound`` — e.g. a double sweep only proves ``diameter >= value``.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, shortest_path

# --------------------------------------------------------------------------- structure


def undirected_adjacency(uedges: np.ndarray, n: int) -> sp.csr_matrix:
    """Symmetric 0/1 CSR adjacency of a simple undirected edge list (``u != v``)."""
    uedges = np.asarray(uedges, dtype=np.int64).reshape(-1, 2)
    rows = np.concatenate([uedges[:, 0], uedges[:, 1]])
    cols = np.concatenate([uedges[:, 1], uedges[:, 0]])
    adj = sp.csr_matrix((np.ones(rows.size, dtype=np.int32), (rows, cols)), shape=(n, n))
    adj.sum_duplicates()
    if adj.data.size and adj.data.max() > 1:
        raise ValueError("edge list has repeated pairs: not a simple graph")
    adj.sort_indices()
    return adj


def edge_common_neighbours(adj: sp.csr_matrix, uedges: np.ndarray,
                           block_rows: int = 4000) -> np.ndarray:
    """``|N(u) ∩ N(v)|`` for every edge ``(u, v)`` (= triangles through the edge).

    Computed block-wise as ``(A[rows] @ A)[u, v]`` to bound memory.
    """
    uedges = np.asarray(uedges, dtype=np.int64).reshape(-1, 2)
    n = adj.shape[0]
    order = np.argsort(uedges[:, 0], kind="stable")
    us, vs = uedges[order, 0], uedges[order, 1]
    starts = np.searchsorted(us, np.arange(n + 1))
    common = np.zeros(uedges.shape[0], dtype=np.int64)
    for r0 in range(0, n, block_rows):
        r1 = min(n, r0 + block_rows)
        a, b = starts[r0], starts[r1]
        if a == b:
            continue
        two_hop = (adj[r0:r1] @ adj).tocsr()
        common[order[a:b]] = np.asarray(two_hop[us[a:b] - r0, vs[a:b]]).ravel()
    return common


def undirected_summary(uedges: np.ndarray, n: int) -> dict[str, Any]:
    """Degree, density, clustering (both conventions), transitivity and components of G_u.

    Returns the summary plus the per-node arrays ``degree``, ``triangles`` and
    ``clustering`` (C_i = 0 when d_i < 2) under the key ``"arrays"``.
    """
    uedges = np.asarray(uedges, dtype=np.int64).reshape(-1, 2)
    m = int(uedges.shape[0])
    adj = undirected_adjacency(uedges, n)
    deg = np.diff(adj.indptr).astype(np.int64)
    common = edge_common_neighbours(adj, uedges)
    tri = np.zeros(n, dtype=np.int64)
    np.add.at(tri, uedges[:, 0], common)
    np.add.at(tri, uedges[:, 1], common)
    tri //= 2  # each triangle at a node is seen through its two incident edges
    pairs = deg * (deg - 1) // 2
    clustering = np.divide(tri, pairs, out=np.zeros(n), where=pairs > 0)
    eligible = deg >= 2
    ncomp, membership = connected_components(adj, directed=False)
    sizes = np.bincount(membership)
    triangles = int(common.sum() // 3)
    triples = int(pairs.sum())
    return {
        "num_nodes": int(n),
        "num_edges": m,
        "mean_degree": 2.0 * m / n,
        "density": 2.0 * m / (n * (n - 1)),
        "degree_max": int(deg.max()),
        "degree_min": int(deg.min()),
        "num_isolated_nodes": int((deg == 0).sum()),
        "num_nodes_degree_lt_2": int((deg < 2).sum()),
        "num_triangles": triangles,
        "num_connected_triples": triples,
        "transitivity": 3.0 * triangles / triples if triples else 0.0,
        "average_local_clustering": float(clustering[eligible].mean()) if eligible.any()
        else 0.0,
        "average_local_clustering_all_nodes": float(clustering.mean()),
        "num_connected_components": int(ncomp),
        "largest_component_size": int(sizes.max()),
        "arrays": {"degree": deg, "triangles": tri, "clustering": clustering,
                   "edge_common_neighbours": common},
    }


def directed_summary(edges: np.ndarray, n: int) -> dict[str, Any]:
    """In/out degrees, reciprocity and weak/strong components of G_d."""
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    m = int(edges.shape[0])
    src, dst = edges[:, 0], edges[:, 1]
    indeg = np.bincount(dst, minlength=n)
    outdeg = np.bincount(src, minlength=n)
    non_loop = src != dst
    keys = np.unique(src[non_loop] * np.int64(n) + dst[non_loop])
    reverse = dst[non_loop] * np.int64(n) + src[non_loop]
    reciprocated = np.isin(reverse, keys, assume_unique=False)
    adj = sp.csr_matrix((np.ones(m, dtype=np.int8), (src, dst)), shape=(n, n))
    nweak, _ = connected_components(adj, directed=True, connection="weak")
    nstrong, lab = connected_components(adj, directed=True, connection="strong")
    scc = np.bincount(lab)
    return {
        "num_nodes": int(n),
        "num_edges": m,
        "num_self_loops": int((~non_loop).sum()),
        "num_duplicate_edges": int(m - np.unique(src * np.int64(n) + dst).size),
        "in_degree_max": int(indeg.max()),
        "out_degree_max": int(outdeg.max()),
        "num_nodes_in_degree_0": int((indeg == 0).sum()),
        "num_nodes_out_degree_0": int((outdeg == 0).sum()),
        "density": m / (n * (n - 1)),
        # igraph's reciprocity(mode="default", ignore_loops=True): fraction of the
        # (non-loop) edges whose reverse edge also exists.
        "reciprocity": float(reciprocated.sum() / non_loop.sum()) if non_loop.any() else 0.0,
        "num_reciprocal_pairs": int(reciprocated.sum() // 2),
        "num_weakly_connected_components": int(nweak),
        "num_strongly_connected_components": int(nstrong),
        "num_nontrivial_sccs": int((scc > 1).sum()),
        "num_nodes_in_nontrivial_sccs": int(scc[scc > 1].sum()),
        "largest_scc_size": int(scc.max()),
    }


def orientation_by_year(edges: np.ndarray, years: np.ndarray) -> dict[str, float]:
    """Share of citations ``u -> v`` with year(u) >, = or < year(v).

    If ``u -> v`` means *u cites v*, most citations should point to older or same-year
    papers.
    """
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    yu, yv = years[edges[:, 0]], years[edges[:, 1]]
    return {"frac_citing_newer_than_cited": float((yu > yv).mean()),
            "frac_same_year": float((yu == yv).mean()),
            "frac_citing_older_than_cited": float((yu < yv).mean())}


# --------------------------------------------------------------------------- distances


def bfs_distances(adj: sp.csr_matrix, sources: np.ndarray | int) -> np.ndarray:
    """Hop distances from ``sources`` (rows) to every node; ``-1`` if unreachable."""
    dist = shortest_path(adj, method="D", unweighted=True, directed=False, indices=sources)
    dist = np.atleast_2d(dist)
    out = np.full(dist.shape, -1, dtype=np.int64)
    finite = np.isfinite(dist)
    out[finite] = dist[finite].astype(np.int64)
    return out


def _sweep(adj: sp.csr_matrix, start: int) -> tuple[int, int, np.ndarray]:
    """BFS from ``start``: (eccentricity, first farthest node, predecessor array)."""
    dist, pred = shortest_path(adj, method="D", unweighted=True, directed=False,
                               indices=start, return_predecessors=True)
    if not np.all(np.isfinite(dist)):
        raise ValueError("graph is not connected")
    ecc = int(dist.max())
    return ecc, int(np.flatnonzero(dist == ecc)[0]), pred


def _walk_back(pred: np.ndarray, node: int, steps: int) -> int:
    for _ in range(steps):
        node = int(pred[node])
    return node


def exact_diameter_ifub(adj: sp.csr_matrix, chunk: int = 64,
                        progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Exact diameter of a connected undirected graph with the iFUB algorithm.

    Crescenzi et al. (2013), "On computing the diameter of real-world undirected graphs".
    A central root ``u`` is chosen by a 4-sweep; nodes are then processed from the
    farthest BFS level of ``u`` inwards. After all nodes at levels ``>= i`` have had their
    eccentricity computed, every remaining pair is within ``2 (i - 1)`` hops (through
    ``u``), so the search stops as soon as the best eccentricity found ``>= 2 (i - 1)``.
    The answer is exact (a certificate pair at that distance is returned).
    """
    n = adj.shape[0]
    deg = np.diff(adj.indptr)
    lower, witness, num_bfs = 0, (0, 0), 0

    def note(ecc: int, a: int, b: int) -> None:
        nonlocal lower, witness
        if ecc > lower:
            lower, witness = ecc, (a, b)

    start = int(np.argmax(deg))
    root = start
    for _ in range(2):  # 4-sweep: r -> a -> b (midpoint) twice
        ecc_r, a, _ = _sweep(adj, root)
        note(ecc_r, root, a)
        ecc_a, b, pred_a = _sweep(adj, a)
        note(ecc_a, a, b)
        num_bfs += 2
        root = _walk_back(pred_a, b, ecc_a // 2)
    dist_root = bfs_distances(adj, root)[0]
    num_bfs += 1
    ecc_root = int(dist_root.max())
    note(ecc_root, root, int(np.argmax(dist_root)))
    for level in range(ecc_root, 0, -1):
        fringe = np.flatnonzero(dist_root == level)
        for i in range(0, fringe.size, chunk):
            block = fringe[i:i + chunk]
            dist = bfs_distances(adj, block)
            ecc = dist.max(axis=1)
            best = int(np.argmax(ecc))
            note(int(ecc[best]), int(block[best]), int(np.argmax(dist[best])))
        num_bfs += fringe.size
        if progress:
            progress(f"iFUB level {level}: {fringe.size} nodes, lower bound {lower}")
        if lower >= 2 * (level - 1):
            return {"diameter": lower, "witness_pair": witness, "root": root,
                    "root_eccentricity": ecc_root, "stopped_at_level": level,
                    "num_bfs": num_bfs, "num_nodes": n,
                    "method": "iFUB (4-sweep root) with scipy BFS; exact"}
    return {"diameter": lower, "witness_pair": witness, "root": root,
            "root_eccentricity": ecc_root, "stopped_at_level": 0, "num_bfs": num_bfs,
            "num_nodes": n, "method": "iFUB (4-sweep root) with scipy BFS; exact"}


def double_sweep_lower_bound(adj: sp.csr_matrix, starts: list[int], sweeps: int = 4
                             ) -> dict[str, Any]:
    """Diameter *lower bound* from repeated double sweeps (not a proof of exactness)."""
    best, witness = 0, (0, 0)
    for s in starts:
        cur = int(s)
        for _ in range(sweeps):
            ecc, far, _ = _sweep(adj, cur)
            if ecc > best:
                best, witness = ecc, (cur, far)
            cur = far
    return {"lower_bound": best, "witness_pair": witness, "num_starts": len(starts),
            "sweeps_per_start": sweeps}


_WORKER_ADJ: sp.csr_matrix | None = None


def _init_worker(indptr: np.ndarray, indices: np.ndarray, n: int) -> None:
    global _WORKER_ADJ
    _WORKER_ADJ = sp.csr_matrix((np.ones(indices.size, dtype=np.int32), indices, indptr),
                                shape=(n, n))


def _histogram_block(sources: np.ndarray) -> np.ndarray:
    assert _WORKER_ADJ is not None
    dist = shortest_path(_WORKER_ADJ, method="D", unweighted=True, directed=False,
                         indices=sources)
    if not np.all(np.isfinite(dist)):
        raise ValueError("graph is not connected")
    return np.bincount(dist.astype(np.int64).ravel())


def all_pairs_distance_histogram(adj: sp.csr_matrix, n_jobs: int = 1, block: int = 64,
                                 progress: Callable[[str], None] | None = None
                                 ) -> dict[str, Any]:
    """Exact histogram of hop distances over all ordered pairs (one BFS per node).

    ``histogram[d]`` = number of ordered pairs (s, t), s != t, at distance d.
    Cost: n BFS runs (scipy C code), distributed over ``n_jobs`` processes.
    """
    n = adj.shape[0]
    blocks = [np.arange(i, min(n, i + block)) for i in range(0, n, block)]
    start = time.perf_counter()
    total = np.zeros(1, dtype=np.int64)

    def add(h: np.ndarray, done: int) -> None:
        nonlocal total
        if h.size > total.size:
            total = np.pad(total, (0, h.size - total.size))
        total[: h.size] += h
        if progress and (done % max(1, len(blocks) // 20) == 0 or done == len(blocks)):
            el = time.perf_counter() - start
            progress(f"all-pairs BFS: {done}/{len(blocks)} blocks, {el:.0f} s elapsed, "
                     f"ETA {el / done * (len(blocks) - done):.0f} s")

    if n_jobs <= 1:
        _init_worker(adj.indptr, adj.indices, n)
        for i, b in enumerate(blocks, 1):
            add(_histogram_block(b), i)
    else:
        with ProcessPoolExecutor(max_workers=n_jobs, initializer=_init_worker,
                                 initargs=(adj.indptr, adj.indices, n)) as pool:
            for i, h in enumerate(pool.map(_histogram_block, blocks, chunksize=4), 1):
                add(h, i)
    total[0] = 0  # pairs (s, s)
    pairs = int(total.sum())
    if pairs != n * (n - 1):
        raise ValueError(f"{pairs} reachable ordered pairs, expected {n * (n - 1)}")
    distance_sum = int((np.arange(total.size) * total).sum())
    return {"histogram": total, "num_ordered_pairs": pairs,
            "average_distance": distance_sum / pairs, "diameter": int(total.size - 1),
            "elapsed_seconds": time.perf_counter() - start, "n_jobs": n_jobs,
            "method": "one scipy BFS per node (independent of the F1 MS-BFS engine)"}


def check_distance_histogram(distances: np.ndarray, counts: np.ndarray, n: int, m: int
                             ) -> dict[str, Any]:
    """Internal-consistency checks of a published distance histogram (F1 output)."""
    distances = np.asarray(distances, dtype=np.int64)
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    return {
        "sum_equals_n(n-1)": total == n * (n - 1),
        "distance_1_equals_2m": bool(counts[distances == 1].sum() == 2 * m),
        "distances_contiguous_from_1": bool(np.array_equal(distances,
                                                           np.arange(1, distances.size + 1))),
        "average_distance": float((distances * counts).sum() / total) if total else math.nan,
        "max_distance": int(distances.max()),
    }
