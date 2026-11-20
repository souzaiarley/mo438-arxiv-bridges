"""Complementary analysis for objectives 4-5: distances inside the largest component.

For selected removal states (same strategies and seeds as the validated removal
experiment), the average shortest-path length of the largest connected component (LCC)
is *estimated* from uniformly sampled BFS sources:

    d_hat = (1/s) * sum_{x in sources} mean_{y in LCC, y != x} d(x, y)

which is unbiased for the all-pairs mean over ordered pairs of the LCC (every source has
the same number of targets). The standard error uses the finite-population correction.
The estimator is validated against the exact value of the intact graph (F1, recomputed
independently in phase 0: 5.750052) before it is used on the removal states.
"""

from __future__ import annotations

import math
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, shortest_path

from . import removal as R

DISTANCE_SEED_CODE = 77


def lcc_distance_estimate(edges: np.ndarray, n: int, keep: np.ndarray, num_sources: int,
                          rng: np.random.Generator, chunk: int = 32) -> dict[str, Any]:
    """Sampled average distance (and summary of the distance distribution) in the LCC."""
    kept = edges[keep]
    adj = sp.csr_matrix((np.ones(2 * kept.shape[0], dtype=np.int8),
                         (np.r_[kept[:, 0], kept[:, 1]], np.r_[kept[:, 1], kept[:, 0]])),
                        shape=(n, n))
    _, labels = connected_components(adj, directed=False)
    sizes = np.bincount(labels)
    lcc = np.flatnonzero(labels == int(np.argmax(sizes)))
    n_lcc = int(lcc.size)
    k = min(int(num_sources), n_lcc)
    sources = np.sort(rng.choice(lcc, size=k, replace=False))
    per_source = np.empty(k)
    ecc = np.empty(k, dtype=np.int64)
    hist = np.zeros(1, dtype=np.int64)
    for i in range(0, k, chunk):
        block = sources[i:i + chunk]
        dist = shortest_path(adj, method="D", unweighted=True, directed=False, indices=block)
        d_lcc = dist[:, lcc]
        if not np.all(np.isfinite(d_lcc)):
            raise RuntimeError("source outside the largest component")
        d_int = d_lcc.astype(np.int64)
        per_source[i:i + block.size] = d_int.sum(axis=1) / (n_lcc - 1)
        ecc[i:i + block.size] = d_int.max(axis=1)
        h = np.bincount(d_int.ravel())
        if h.size > hist.size:
            hist = np.pad(hist, (0, h.size - hist.size))
        hist[: h.size] += h
    hist[0] = 0
    estimate = float(per_source.mean())
    fpc = math.sqrt((n_lcc - k) / (n_lcc - 1)) if n_lcc > 1 else 0.0
    se = float(per_source.std(ddof=1) / math.sqrt(k) * fpc) if k > 1 else float("nan")
    cdf = np.cumsum(hist) / hist.sum()
    return {
        "lcc_size": n_lcc, "num_sources": k, "mean_distance": estimate, "se": se,
        "ci95_low": estimate - 1.96 * se, "ci95_high": estimate + 1.96 * se,
        "effective_diameter_90": int(np.searchsorted(cdf, 0.9)),  # 90th pct of pair distances
        "max_sampled_eccentricity": int(ecc.max()),
    }


# --------------------------------------------------------------------------- workers

_STATE: dict[str, Any] = {}


def _init(edges: np.ndarray, n: int, is_inter: np.ndarray, strata: np.ndarray) -> None:
    _STATE["classes"] = R.EdgeClasses(edges, n, is_inter, strata)


def _task(args: tuple[str, int, int, int, int, int]) -> dict[str, Any]:
    strategy, replicate, k, k_max, num_sources, base_seed = args
    classes: R.EdgeClasses = _STATE["classes"]
    t0 = time.perf_counter()
    keep = np.ones(classes.num_edges, dtype=bool)
    if k > 0:
        order, info = R.removal_order(strategy, replicate, classes, k_max, base_seed)
        keep[order[:k]] = False
        order_sha = info["order_sha256"]
    else:
        order_sha = None
    code = R.STRATEGIES[strategy][0]
    rng = np.random.default_rng(np.random.SeedSequence(
        [base_seed, DISTANCE_SEED_CODE, code, replicate, k]))
    out = lcc_distance_estimate(classes.edges, classes.num_nodes, keep, num_sources, rng)
    out.update({"strategy": strategy, "replicate": replicate, "k": k,
                "q": k / classes.num_edges, "order_sha256": order_sha,
                "seconds": round(time.perf_counter() - t0, 3)})
    return out


def run_states(classes: R.EdgeClasses, tasks: list[tuple[str, int, int, int, int, int]],
               n_jobs: int) -> list[dict[str, Any]]:
    """Evaluate the removal states in parallel (one process per task)."""
    if n_jobs <= 1:
        _init(classes.edges, classes.num_nodes, classes.is_inter, classes.strata)
        return [_task(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=n_jobs, initializer=_init,
                             initargs=(classes.edges, classes.num_nodes, classes.is_inter,
                                       classes.strata)) as pool:
        return list(pool.map(_task, tasks))
