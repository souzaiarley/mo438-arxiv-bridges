"""Objective 7: approximate topological ordering of the categories (linear ordering problem).

Formal problem
--------------
W[i, j] (i != j) = number of citations made by papers of area i to papers of area j: every
arc goes from the **citing** area to the **cited** area. An ordering is a bijection
pos: areas -> {1, ..., K}. Convention: a citation i -> j is **compatible** with the ordering
when the cited area comes first, pos(j) < pos(i) (areas are listed from "cited by those
after them" to "citing those before them"). The objective is

    F(pi) = sum_{i != j, pos(j) < pos(i)} W[i, j] / sum_{i != j} W[i, j]   (maximise),

i.e. the weighted linear ordering problem / minimum weighted feedback-arc-set on the
aggregated graph. Since every ordering and its reverse satisfy F(pi) + F(rev pi) = 1, a
uniformly random ordering has E[F] = 1/2 for any matrix.

Bounds
------
* pairwise bound U = sum_{i<j} max(W_ij, W_ji) / total: attainable only if the dominant
  directions of all pairs are acyclic (transitive);
* cycle-packing bound U_cyc <= U: for each 3-cycle of dominant directions any ordering
  violates at least one of its three pairs, losing at least the smallest margin
  |W_ij - W_ji| of the cycle; losses of pair-disjoint cycles add up. U - U_cyc is a proven
  lower bound on the looseness of U (it is not the exact optimum).

Heuristic
---------
Best-improvement *insertion* local search (all K(K-1) moves evaluated per step from
cumulative sums) started from the net-flow order and many random orders, each followed by
an iterated local search (random multi-insertion kicks, accepted when they improve).
"""

from __future__ import annotations

from itertools import combinations, permutations
from typing import Any

import numpy as np


def inter_weights(M: np.ndarray) -> np.ndarray:
    W = np.asarray(M, dtype=np.float64).copy()
    np.fill_diagonal(W, 0.0)
    return W


def positions(order: np.ndarray) -> np.ndarray:
    pos = np.empty(len(order), dtype=np.int64)
    pos[np.asarray(order)] = np.arange(len(order))
    return pos


def compatible_weight(W: np.ndarray, order: np.ndarray) -> float:
    """Sum of W[i, j] with pos(j) < pos(i) (citing after cited)."""
    pos = positions(order)
    return float(W[pos[:, None] > pos[None, :]].sum())


def compatible_fraction(W: np.ndarray, order: np.ndarray) -> float:
    return compatible_weight(W, order) / float(W.sum())


def move_deltas(W: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Delta[p, q] = change of the compatible weight when the element at position p is
    moved to position q (insertion), for all p, q."""
    k = len(order)
    B = W[np.ix_(order, order)]                  # B[a, b] = W[order[a], order[b]]
    D = B.T - B                                  # D[p, t] = W[o_t, o_p] - W[o_p, o_t]
    S = np.zeros((k, k + 1))
    S[:, 1:] = np.cumsum(D, axis=1)
    P = np.arange(k)[:, None]
    Q = np.arange(k)[None, :]
    left = S[P, P] - S[P, Q]                     # q < p: sum_{t=q}^{p-1} D[p, t]
    right = -(S[P, np.minimum(Q + 1, k)] - S[P, P + 1])   # q > p: -sum_{t=p+1}^{q} D[p, t]
    return np.where(Q < P, left, np.where(Q > P, right, 0.0))


def _apply(order: np.ndarray, p: int, q: int) -> np.ndarray:
    x = order[p]
    rest = np.delete(order, p)
    return np.insert(rest, q, x)


def local_search(W: np.ndarray, order: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Best-improvement insertion until no move improves by more than ``tol``."""
    order = np.asarray(order, dtype=np.int64).copy()
    while True:
        delta = move_deltas(W, order)
        p, q = np.unravel_index(int(np.argmax(delta)), delta.shape)
        if delta[p, q] <= tol:
            return order
        order = _apply(order, int(p), int(q))


def _kick(order: np.ndarray, rng: np.random.Generator, moves: int) -> np.ndarray:
    order = order.copy()
    k = len(order)
    for _ in range(moves):
        p, q = rng.integers(k, size=2)
        order = _apply(order, int(p), int(q))
    return order


def net_flow_order(W: np.ndarray) -> np.ndarray:
    """Areas sorted by (received - made) inter-area citations, largest first."""
    return np.argsort(-(W.sum(axis=0) - W.sum(axis=1)), kind="stable")


def optimise(W: np.ndarray, rng: np.random.Generator, random_starts: int, kicks: int,
             initial: list[np.ndarray] | None = None, kick_moves: int = 4) -> dict[str, Any]:
    """Multi-start iterated local search. Returns the best order and per-start optima."""
    k = W.shape[0]
    tol = 1e-12 * float(W.sum())
    starts = [np.asarray(o, dtype=np.int64) for o in (initial or [])]
    starts += [rng.permutation(k) for _ in range(random_starts)]
    best_order, best = None, -np.inf
    finals = []
    for start in starts:
        order = local_search(W, start)
        value = compatible_weight(W, order)
        for _ in range(kicks):
            cand = local_search(W, _kick(order, rng, kick_moves))
            cand_value = compatible_weight(W, cand)
            if cand_value > value + tol:
                order, value = cand, cand_value
        finals.append(value)
        if value > best + tol:
            best_order, best = order, value
    finals = np.asarray(finals) / float(W.sum())
    best_frac = best / float(W.sum())
    return {"order": best_order, "fraction": best_frac, "start_fractions": finals,
            "starts": len(starts), "kicks_per_start": kicks,
            "starts_reaching_best": int((np.abs(finals - best_frac) <= 1e-12).sum())}


# --------------------------------------------------------------------------- bounds


def pairwise_upper_bound(W: np.ndarray) -> float:
    iu = np.triu_indices(W.shape[0], 1)
    return float(np.maximum(W[iu], W.T[iu]).sum() / W.sum())


def majority_three_cycles(W: np.ndarray) -> list[tuple[float, tuple[int, int, int]]]:
    """3-cycles of dominant directions (i before j if W[j, i] > W[i, j]) with min margin."""
    k = W.shape[0]
    margin = W.T - W                           # margin[i, j] > 0: i should precede j
    cycles = []
    for i, j, l in combinations(range(k), 3):
        a, b, c = margin[i, j], margin[j, l], margin[l, i]
        if (a > 0 and b > 0 and c > 0) or (a < 0 and b < 0 and c < 0):
            cycles.append((float(min(abs(a), abs(b), abs(c))), (i, j, l)))
    return cycles


def cycle_packing_bound(W: np.ndarray) -> dict[str, Any]:
    """Tighter (still valid) upper bound from pair-disjoint dominant 3-cycles."""
    k = W.shape[0]
    cycles = sorted(majority_three_cycles(W), reverse=True)
    used: set[tuple[int, int]] = set()
    loss, packed = 0.0, 0
    for value, (i, j, l) in cycles:
        pairs = {tuple(sorted(p)) for p in ((i, j), (j, l), (l, i))}
        if used & pairs:
            continue
        used |= pairs
        loss += value
        packed += 1
    total = float(W.sum())
    u = pairwise_upper_bound(W)
    triples = k * (k - 1) * (k - 2) // 6
    return {"pairwise_bound": u, "cycle_packing_bound": u - loss / total,
            "guaranteed_loss_fraction": loss / total, "dominant_3_cycles": len(cycles),
            "triples": triples, "fraction_cyclic_triples": len(cycles) / triples,
            "packed_cycles": packed}


# --------------------------------------------------------------------------- controls


def random_permutation_fractions(W: np.ndarray, rng: np.random.Generator, n: int
                                 ) -> np.ndarray:
    k = W.shape[0]
    total = float(W.sum())
    return np.array([compatible_weight(W, rng.permutation(k)) / total for _ in range(n)])


def randomise_directions(W: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Swap W_ij <-> W_ji independently with probability 1/2 for every pair.

    Preserves: the pair totals T_ij, the pair margins |W_ij - W_ji| (hence U and the
    maximum attainable per pair) and the total weight. Does not preserve: each area's
    made/received volumes and net flow, the consistency (transitivity) among the dominant
    directions, nor any temporal structure.
    """
    k = W.shape[0]
    iu, ju = np.triu_indices(k, 1)
    swap = rng.random(iu.size) < 0.5
    W2 = W.copy()
    a, b = W[iu[swap], ju[swap]], W[ju[swap], iu[swap]]
    W2[iu[swap], ju[swap]], W2[ju[swap], iu[swap]] = b, a
    return W2


# --------------------------------------------------------------------------- exact (small K)


def exact_brute_force(W: np.ndarray) -> tuple[float, np.ndarray]:
    """Exhaustive search (K <= 8): maximum compatible fraction and an optimal order."""
    best, best_order = -1.0, None
    for perm in permutations(range(W.shape[0])):
        value = compatible_weight(W, np.array(perm))
        if value > best:
            best, best_order = value, np.array(perm)
    return best / float(W.sum()), best_order


def exact_dp(W: np.ndarray) -> float:
    """Subset dynamic programming (K <= ~14): maximum compatible fraction."""
    k = W.shape[0]
    full = 1 << k
    gain = np.zeros((k, full))
    for s in range(1, full):
        low = (s & -s).bit_length() - 1
        gain[:, s] = gain[:, s & (s - 1)] + W[:, low]
    f = np.full(full, -np.inf)
    f[0] = 0.0
    for s in range(full):
        if f[s] == -np.inf:
            continue
        for x in range(k):
            if not s >> x & 1:
                t = s | (1 << x)
                v = f[s] + gain[x, s]        # x placed after every area of s: cites them
                f[t] = max(f[t], v)
    return float(f[full - 1] / W.sum())


def rank_correlations(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Kendall tau and Spearman rho between two orders (as position vectors)."""
    from scipy.stats import kendalltau, spearmanr

    pa, pb = positions(a), positions(b)
    return {"kendall_tau": float(kendalltau(pa, pb).statistic),
            "spearman_rho": float(spearmanr(pa, pb).statistic)}
