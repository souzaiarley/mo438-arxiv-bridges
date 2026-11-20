"""Static figures of the project (Portuguese labels).

* Objective 1, characterization pipeline (:mod:`characterization`): PNG figures drawn from
  the in-memory results, with :data:`CHARACTERIZATION_STYLE` (one hue per series, blue
  first; category-coloured figures show the 7 largest categories and fold the rest into a
  grey "other" class).
* Analyses of objectives 1-7 (``mo438 *-figures``): PNG + PDF figures redrawn from the saved
  CSV/JSON results, with :data:`STYLE`. Palette: categorical slots 1-3 of the validated
  reference palette (blue, orange, aqua; all-pairs CVD-safe). Aqua is below 3:1 contrast
  on white, so every series is also direct-labelled. Degree-matched controls reuse the hue
  of their target class with a dashed line (secondary encoding). Bands are empirical
  2.5-97.5 % intervals over the replicates (one run) or 95 % intervals of the mean
  difference, as stated per figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm

INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8983", "#e1e0d9"
COLORS = {"inter": "#2a78d6", "intra": "#eb6834", "uniform": "#1baf7a",
          "matched_inter": "#2a78d6", "matched_intra": "#eb6834"}
DASHED = {"matched_inter", "matched_intra"}
LABELS = {"inter": "interárea", "intra": "intra-área", "uniform": "uniforme",
          "matched_inter": "controle pareado por grau (inter)",
          "matched_intra": "controle pareado por grau (intra)"}
SHORT = {"inter": "inter", "intra": "intra", "uniform": "uniforme",
         "matched_inter": "pareado (inter)", "matched_intra": "pareado (intra)"}

STYLE = {
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "axes.titlesize": 10.5,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelsize": 9.5,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "legend.frameon": False, "legend.fontsize": 8.5,
    "font.size": 9, "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
}


def _pt(value: float, digits: int = 3) -> str:
    """Number with a decimal comma (Portuguese report)."""
    return f"{value:.{digits}f}".replace(".", ",")


def _save(fig: plt.Figure, path: Path) -> list[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in (".png", ".pdf"):
        target = path.with_suffix(suffix)
        fig.savefig(target, dpi=200, bbox_inches="tight")
        paths.append(target)
    plt.close(fig)
    return paths


def _series(summary: pd.DataFrame, strategy: str, metric: str, scale: float = 1.0,
            q_below: float | None = None) -> pd.DataFrame:
    d = summary[summary["strategy"] == strategy].sort_values("k")
    if q_below is not None:
        d = d[d["q"] < q_below]
    return pd.DataFrame({"q": d["q"].to_numpy(),
                         "mean": d[f"{metric}_mean"].to_numpy() / scale,
                         "low": d[f"{metric}_p025"].to_numpy() / scale,
                         "high": d[f"{metric}_p975"].to_numpy() / scale})


def _line(ax, x, y, strategy, label=True, **kw):
    ax.plot(x, y, color=COLORS[strategy], linestyle="--" if strategy in DASHED else "-",
            label=LABELS[strategy] if label else None, **kw)


def _band(ax, x, low, high, strategy):
    ax.fill_between(x, low, high, color=COLORS[strategy],
                    alpha=0.10 if strategy in DASHED else 0.16, linewidth=0)


def _end_label(ax, x, y, strategy, text=None, dy=0.0):
    ax.annotate(text or SHORT[strategy], xy=(x, y), xytext=(4, dy), textcoords="offset points",
                color=INK_2, fontsize=8, va="center", ha="left")


# --------------------------------------------------------------------------- objective 1 (PNG)

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OTHER = "#b5b4ad"
SEQUENTIAL = "Blues"

CHARACTERIZATION_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "legend.frameon": False,
    "font.size": 9.5,
}


def _save_png(fig: plt.Figure, path: Path, dpi: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _log_bins(values: np.ndarray, bins: int) -> np.ndarray:
    positive = values[values > 0]
    hi = max(2.0, float(positive.max()) + 1) if positive.size else 2.0
    return np.unique(np.floor(np.logspace(0, np.log10(hi), bins)).astype(int))


def _ccdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.sort(values[values > 0])
    uniq, idx = np.unique(values, return_index=True)
    return uniq, 1.0 - idx / values.size


def plot_degree_distribution(
    degrees: np.ndarray, path: Path, dpi: int, title: str, log_x: bool = True, log_y: bool = True
) -> Path:
    """P(k) (counts per degree) and complementary CDF of the degree of G_u."""
    with plt.rc_context(CHARACTERIZATION_STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        uniq, counts = np.unique(degrees[degrees > 0], return_counts=True)
        ax1.scatter(uniq, counts, s=9, color=SERIES[0], linewidths=0, alpha=0.85)
        ax1.set_xlabel("degree k")
        ax1.set_ylabel("number of nodes with degree k")
        ax1.set_title("Degree distribution")
        if log_x:
            ax1.set_xscale("log")
        if log_y:
            ax1.set_yscale("log")
        isolated = int(np.count_nonzero(degrees == 0))
        ax1.text(0.98, 0.97,
                 f"n = {degrees.size:,}\n<k> = {degrees.mean():.2f}\nmax k = {degrees.max():,}"
                 + (f"\nk = 0 (not shown): {isolated:,}" if isolated else ""),
                 transform=ax1.transAxes, ha="right", va="top", color=INK_2, fontsize=8.5)
        x, y = _ccdf(degrees)
        ax2.step(x, y, where="post", color=SERIES[0], linewidth=1.6)
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel("degree k")
        ax2.set_ylabel("fraction of nodes with degree ≥ k")
        ax2.set_title("Complementary CDF")
        fig.suptitle(title, fontsize=12.5, fontweight="bold", color=INK)
        fig.tight_layout()
        return _save_png(fig, path, dpi)


def plot_in_out_degree(in_deg: np.ndarray, out_deg: np.ndarray, path: Path, dpi: int,
                       title: str) -> Path:
    """In-degree (citations received) and out-degree (references made) of G_d."""
    with plt.rc_context(CHARACTERIZATION_STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        for values, color, label in ((in_deg, SERIES[0], "in-degree (citations received)"),
                                     (out_deg, SERIES[1], "out-degree (references made)")):
            uniq, counts = np.unique(values[values > 0], return_counts=True)
            ax1.scatter(uniq, counts, s=9, color=color, linewidths=0, alpha=0.8, label=label)
            x, y = _ccdf(values)
            ax2.step(x, y, where="post", color=color, linewidth=1.6, label=label)
        for ax in (ax1, ax2):
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("degree k")
            ax.legend(loc="lower left", fontsize=8.5)
        ax1.set_ylabel("number of nodes")
        ax1.set_title("Degree distribution (k ≥ 1)")
        ax2.set_ylabel("fraction of nodes with degree ≥ k")
        ax2.set_title("Complementary CDF")
        fig.suptitle(title, fontsize=12.5, fontweight="bold", color=INK)
        fig.tight_layout()
        return _save_png(fig, path, dpi)


def plot_local_clustering(coeff: np.ndarray, degrees: np.ndarray, path: Path, dpi: int,
                          bins: int, average: float, transitivity: float, title: str) -> Path:
    """Histogram of C_i (all nodes) and mean C(k) as a function of the degree."""
    eligible = degrees >= 2
    with plt.rc_context(CHARACTERIZATION_STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.hist(coeff, bins=np.linspace(0, 1, min(bins, 50) + 1), color=SERIES[0],
                 edgecolor="white", linewidth=0.5)
        ax1.set_yscale("log")
        ax1.set_xlabel("local clustering coefficient $C_i$")
        ax1.set_ylabel("number of nodes")
        ax1.set_title("Distribution of $C_i$")
        ax1.axvline(average, color=INK, linewidth=1, linestyle="--")
        ax1.text(0.98, 0.97,
                 f"mean $C_i$ (k ≥ 2) = {average:.4f}\ntransitivity = {transitivity:.4f}\n"
                 f"nodes with k < 2 (C = 0, included): {int((~eligible).sum()):,}",
                 transform=ax1.transAxes, ha="right", va="top", color=INK_2, fontsize=8.5)
        ks = degrees[eligible]
        uniq, inverse = np.unique(ks, return_inverse=True)
        mean_c = np.bincount(inverse, weights=coeff[eligible]) / np.bincount(inverse)
        positive = mean_c > 0
        ax2.scatter(uniq[positive], mean_c[positive], s=9, color=SERIES[0], linewidths=0)
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel("degree k")
        ax2.set_ylabel("mean $C_i$ of nodes with degree k")
        ax2.set_title("Clustering vs. degree, C(k)")
        fig.suptitle(title, fontsize=12.5, fontweight="bold", color=INK)
        fig.tight_layout()
        return _save_png(fig, path, dpi)


def plot_component_sizes(sizes: np.ndarray, path: Path, dpi: int, title: str) -> Path:
    """Number of connected components of each size (log-log)."""
    uniq, counts = np.unique(sizes, return_counts=True)
    with plt.rc_context(CHARACTERIZATION_STYLE):
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.scatter(uniq, counts, s=22, color=SERIES[0], linewidths=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("component size (number of nodes)")
        ax.set_ylabel("number of components")
        lcc = int(sizes.max())
        ax.annotate(f"largest component\n{lcc:,} nodes ({lcc / sizes.sum():.2%})",
                    xy=(lcc, 1), xytext=(0.62, 0.55), textcoords="axes fraction",
                    color=INK_2, fontsize=8.5,
                    arrowprops={"arrowstyle": "-", "color": INK_2, "linewidth": 0.8})
        ax.set_title(title)
        ax.text(0.02, 0.03, f"{sizes.size:,} components in total", transform=ax.transAxes,
                color=INK_2, fontsize=8.5)
        fig.tight_layout()
        return _save_png(fig, path, dpi)


def plot_distance_distribution(histogram: np.ndarray, path: Path, dpi: int, title: str,
                               is_exact: bool, average: float) -> Path:
    """Fraction of (ordered) node pairs at each distance in the largest component."""
    dist = np.arange(histogram.size)[1:]
    frac = histogram[1:] / max(1, histogram[1:].sum())
    with plt.rc_context(CHARACTERIZATION_STYLE):
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.bar(dist, frac, width=0.8, color=SERIES[0], edgecolor="white", linewidth=0.5)
        ax.axvline(average, color=INK, linewidth=1, linestyle="--")
        ax.text(0.98, 0.97, f"mean distance = {average:.3f} (dashed)\nmax distance = {dist.max()}",
                transform=ax.transAxes, ha="right", va="top", color=INK_2, fontsize=8.5)
        ax.set_xlabel("shortest-path distance (hops)")
        ax.set_ylabel("fraction of node pairs" if is_exact else "fraction of sampled pairs")
        ax.set_title(title + ("" if is_exact else " — APPROXIMATE (sampled sources)"))
        ax.grid(axis="x", visible=False)
        fig.tight_layout()
        return _save_png(fig, path, dpi)


def _category_colors(labels: np.ndarray, names: list[str], top: int = 7
                     ) -> tuple[list[str], list[tuple[str, str]]]:
    counts = np.bincount(labels, minlength=len(names))
    order = np.argsort(-counts)[:top]
    palette = {int(c): SERIES[i] for i, c in enumerate(order)}
    colors = [palette.get(int(lab), OTHER) for lab in labels]
    legend = [(names[int(c)], palette[int(c)]) for c in order] + [("other", OTHER)]
    return colors, legend


def plot_sample_graph(edges: np.ndarray, labels: np.ndarray, names: list[str], path: Path,
                      dpi: int, title: str, seed: int) -> Path:
    """Force-directed drawing of a small connected sample (nodes coloured by category)."""
    import random

    import igraph as ig

    random.seed(seed)
    graph = ig.Graph(n=labels.size, edges=edges.tolist())
    layout = np.asarray(graph.layout_fruchterman_reingold(niter=1000).coords)
    colors, legend = _category_colors(labels, names)
    degrees = np.asarray(graph.degree())
    with plt.rc_context({**CHARACTERIZATION_STYLE, "axes.grid": False}):
        fig, ax = plt.subplots(figsize=(8, 8))
        segments = layout[edges]
        ax.add_collection(LineCollection(segments, colors="#9f9e98", linewidths=0.25, alpha=0.5))
        ax.scatter(layout[:, 0], layout[:, 1], s=4 + 3 * np.sqrt(degrees), c=colors,
                   edgecolors="white", linewidths=0.3, zorder=3)
        ax.set_axis_off()
        handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, label=n, markersize=6)
                   for n, c in legend]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8.5,
                  title="primary category")
        ax.set_title(title)
        fig.tight_layout()
        return _save_png(fig, path, dpi)


def plot_area_heatmap(matrix: np.ndarray, names: list[str], path: Path, dpi: int,
                      title: str) -> Path:
    """Row-normalised citation matrix between areas (share of each area's references)."""
    made = matrix.sum(axis=1, keepdims=True)
    share = np.divide(matrix, made, out=np.zeros(matrix.shape, dtype=float), where=made > 0)
    k = matrix.shape[0]
    with plt.rc_context({**CHARACTERIZATION_STYLE, "axes.grid": False}):
        fig, ax = plt.subplots(figsize=(9.5, 8.5))
        floor = 1e-4
        image = ax.imshow(np.clip(share, floor, 1), cmap=SEQUENTIAL,
                          norm=LogNorm(vmin=floor, vmax=1))
        ax.set_xticks(range(k), names, rotation=90, fontsize=7)
        ax.set_yticks(range(k), names, fontsize=7)
        ax.set_xlabel("cited area")
        ax.set_ylabel("citing area")
        bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
        bar.set_label("share of the citing area's references (log scale)")
        ax.set_title(title)
        fig.tight_layout()
        return _save_png(fig, path, dpi)


def plot_area_graph(matrix: np.ndarray, names: list[str], num_papers: np.ndarray, path: Path,
                    dpi: int, top_edges: int, title: str) -> Path:
    """Aggregated area graph G_A: node size ∝ papers, colour = intra-area share,
    arrows = the ``top_edges`` heaviest inter-area citation flows (width ∝ citations)."""
    k = matrix.shape[0]
    angles = 2 * np.pi * np.arange(k) / k
    pos = np.column_stack([np.cos(angles), np.sin(angles)])
    made = matrix.sum(axis=1)
    intra_share = np.divide(np.diag(matrix), made, out=np.zeros(k), where=made > 0)
    off = matrix.astype(float).copy()
    np.fill_diagonal(off, 0)
    flat = np.argsort(off, axis=None)[::-1][:top_edges]
    flat = flat[off.flat[flat] > 0]
    weights = off.flat[flat]
    with plt.rc_context({**CHARACTERIZATION_STYLE, "axes.grid": False}):
        fig, ax = plt.subplots(figsize=(9, 9))
        wmax = weights.max() if weights.size else 1.0
        for idx, w in zip(flat, weights):
            i, j = divmod(int(idx), k)
            ax.annotate("", xy=pos[j], xytext=pos[i],
                        arrowprops={"arrowstyle": "-|>", "color": "#6f6e69",
                                    "alpha": 0.25 + 0.6 * w / wmax, "linewidth": 0.4 + 4 * w / wmax,
                                    "shrinkA": 8, "shrinkB": 8,
                                    "connectionstyle": "arc3,rad=0.15"})
        sizes = 30 + 900 * num_papers / max(1, num_papers.max())
        points = ax.scatter(pos[:, 0], pos[:, 1], s=sizes, c=intra_share, cmap=SEQUENTIAL,
                            vmin=0, vmax=1, edgecolors="#52514e", linewidths=0.6, zorder=3)
        for i in range(k):
            ax.text(1.17 * pos[i, 0], 1.17 * pos[i, 1], names[i], ha="center", va="center",
                    fontsize=7.5, color=INK)
        bar = fig.colorbar(points, ax=ax, fraction=0.04, pad=0.02, shrink=0.7)
        bar.set_label("intra-area share of references (self-loop weight / out-strength)")
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-1.35, 1.35)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(title)
        fig.tight_layout()
        return _save_png(fig, path, dpi)


# --------------------------------------------------------------------------- objectives 4-5: removal


def removal_figures(out_dir: Path) -> list[Path]:
    """All figures of objectives 4-5 from ``summary.csv`` (written by ``mo438 removal``)."""
    import json

    summary = pd.read_csv(out_dir / "summary.csv")
    design = json.loads((out_dir / "experiment.json").read_text(encoding="utf-8"))["design"]
    fig_dir = out_dir / "figures"
    n = int(design["num_nodes"])
    k_max = int(summary["k"].max())
    q_max = float(summary["q"].max())
    bound = float(design["theoretical_bound"]["bound"])
    largest_cat = design["theoretical_bound"]["largest_category"]
    paths: list[Path] = []
    main = ("inter", "intra", "uniform")

    # ---- Figure 1: S(q), full domain + tail of the inter-area strategy
    with plt.rc_context(STYLE):
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.5, 4.0),
                                     gridspec_kw={"width_ratios": [1.35, 1]})
        for s in main:
            d = _series(summary, s, "S")
            _band(ax, d["q"], d["low"], d["high"], s)
            _line(ax, d["q"], d["mean"], s)
        ax.axhline(bound, color=MUTED, linewidth=1, linestyle=":")
        ax.annotate(f"limite por construção: maior categoria ({largest_cat}) / n = "
                    f"{_pt(bound)}", xy=(0.005, bound), xytext=(0, 5),
                    textcoords="offset points", color=INK_2, fontsize=8)
        end = summary[(summary["strategy"] == "inter") & (summary["k"] == k_max)]
        s_end = float(end["S_mean"].iloc[0])
        s2_end = float(end["second_lcc_size_mean"].iloc[0]) / n
        ax.scatter([q_max], [s_end], s=28, color=COLORS["inter"], zorder=4,
                   edgecolors="white", linewidths=1.2)
        ax.annotate(f"todas as arestas interárea\nremovidas: S = {_pt(s_end)}",
                    xy=(q_max, s_end),
                    xytext=(-118, 26), textcoords="offset points", color=INK_2, fontsize=8,
                    arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8})
        for s, dy in (("intra", 5), ("uniform", -5), ("inter", 0)):
            d = _series(summary, s, "S", q_below=q_max)
            _end_label(ax, d["q"].iloc[-1], d["mean"].iloc[-1], s, dy=dy)
        ax.set_xlim(0, q_max * 1.12)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("q = arestas removidas / m")
        ax.set_ylabel("S = |maior componente| / n")
        ax.set_title("(a) Fração de vértices na maior componente")
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.18))

        tail = summary[(summary["strategy"] == "inter")
                       & (summary["eligible_fraction_removed"] >= 0.899)
                       & (summary["k"] < k_max)].sort_values("k")
        remaining = k_max - tail["k"].to_numpy()
        bx.fill_between(remaining, tail["S_p025"], tail["S_p975"], color=COLORS["inter"],
                        alpha=0.16, linewidth=0)
        bx.plot(remaining, tail["S_mean"], color=COLORS["inter"], marker="o", markersize=4,
                label="S (maior componente)")
        bx.plot(remaining, tail["second_lcc_size_mean"] / n, color=INK_2, marker="o",
                markersize=4, linewidth=1.4, label="segunda maior componente / n")
        for s in ("intra", "uniform"):
            ref = summary[(summary["strategy"] == s) & summary["k"].isin(tail["k"])]
            ref = ref.sort_values("k")
            bx.plot(k_max - ref["k"].to_numpy(), ref["S_mean"], color=COLORS[s], linewidth=1.4,
                    label=f"S, {LABELS[s]} (mesmo k)")
        bx.set_xscale("log")
        bx.invert_xaxis()
        bx.set_ylim(0, 1.02)
        bx.set_xlabel("arestas interárea ainda presentes (escala log)")
        bx.set_ylabel("fração de n")
        bx.set_title("(b) Interárea: últimos 10% da classe")
        bx.annotate(f"com 0 restantes: S = {_pt(s_end)};\n"
                    f"segunda maior componente / n = {_pt(s2_end)}",
                    xy=(0.02, 0.06), xycoords="axes fraction", color=INK_2, fontsize=8)
        bx.legend(loc="center left", bbox_to_anchor=(0.0, 0.42), fontsize=7.8)
        fig.tight_layout()
        paths += _save(fig, fig_dir / "fig1_S_q")

    # ---- Figure 2: difference to the uniform control, with the degree-matched controls
    q_cut = 0.3301
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=False)
        uni = summary[summary["strategy"] == "uniform"].set_index("k")
        for ax, group, title in (
                (axes[0], ("inter", "matched_inter"),
                 "(a) Interárea e seu controle pareado por grau"),
                (axes[1], ("intra", "matched_intra"),
                 "(b) Intra-área e seu controle pareado por grau")):
            ax.axhline(0, color=COLORS["uniform"], linewidth=1.6)
            _end_label(ax, q_cut, 0, "uniform", "uniforme (referência)", dy=6)
            for s in group:
                d = summary[(summary["strategy"] == s) & (summary["q"] <= q_cut)].set_index("k")
                u = uni.loc[d.index]
                diff = (d["S_mean"] - u["S_mean"]) * n
                se = np.sqrt(d["S_sd"] ** 2 / d["replicates"]
                             + u["S_sd"] ** 2 / u["replicates"]) * n
                _band(ax, d["q"], diff - 1.96 * se, diff + 1.96 * se, s)
                _line(ax, d["q"], diff, s)
                _end_label(ax, d["q"].iloc[-1], diff.iloc[-1], s)
            ax.set_xlim(0, q_cut * 1.22)
            ax.set_xlabel("q = arestas removidas / m")
            ax.set_ylabel("Δ vértices na maior componente vs. uniforme")
            ax.set_title(title)
            ax.legend(loc="lower left" if group[0] == "inter" else "upper left")
        fig.tight_layout()
        paths += _save(fig, fig_dir / "fig2_delta_vs_uniform")

    # ---- Figure 3: complementary fragmentation measures
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0))
        panels = (
            (axes[0, 0], "num_isolated", "(a) Vértices isolados", "vértices", False,
             {"inter": 0, "intra": -6, "uniform": 6}),
            (axes[0, 1], "num_components", "(b) Número de componentes", "componentes", False,
             {"inter": 0, "intra": -6, "uniform": 6}),
            (axes[1, 0], "second_lcc_size", "(c) Segunda maior componente (mediana)",
             "vértices (escala log)", True, {"inter": 0, "intra": -6, "uniform": 6}),
            (axes[1, 1], "S_non_isolated",
             "(d) Maior componente entre os vértices não isolados", "|LCC| / (n − isolados)",
             False, {"inter": 0, "intra": 6, "uniform": -6}),
        )
        for ax, metric, title, ylabel, logy, offsets in panels:
            for s in main:
                q_below = q_max if metric == "S_non_isolated" else None
                d = _series(summary, s, metric, q_below=q_below)
                if metric == "second_lcc_size":
                    # right-skewed across runs (a ~128-vertex cluster detaches in some runs):
                    # the median is the representative curve; no band.
                    med = summary[summary["strategy"] == s].sort_values("k")
                    d = d.assign(mean=med["second_lcc_size_p50"].to_numpy())
                    d = d[d["q"] > 0]
                else:
                    _band(ax, d["q"], d["low"], d["high"], s)
                _line(ax, d["q"], d["mean"], s, label=(metric == "num_isolated"))
                _end_label(ax, d["q"].iloc[-1], d["mean"].iloc[-1], s, dy=offsets[s])
            if logy:
                ax.set_yscale("log")
                ax.set_ylim(0.8, 4e4)
            ax.set_xlim(0, q_max * 1.16)
            ax.set_xlabel("q = arestas removidas / m")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
        axes[0, 0].legend(loc="upper left")
        sni_end = float(end["S_non_isolated_mean"].iloc[0])
        axes[1, 1].annotate(f"interárea em q = {_pt(q_max)} (todas removidas): "
                            f"{_pt(sni_end)} (fora da escala)",
                            xy=(0.02, 0.04), xycoords="axes fraction", color=INK_2,
                            fontsize=8)
        fig.tight_layout()
        paths += _save(fig, fig_dir / "fig3_fragmentation")

    # ---- Figure 4 (complementary): sampled average distance inside the LCC
    dist_path = out_dir / "lcc_distances_summary.csv"
    if dist_path.is_file():
        dist = pd.read_csv(dist_path)
        meta = json.loads((out_dir / "lcc_distances.json").read_text(encoding="utf-8"))
        exact = meta["validation_on_intact_graph"]["exact_intact_average_distance"]
        with plt.rc_context(STYLE):
            fig, ax = plt.subplots(figsize=(7.0, 4.2))
            for s, dx in (("inter", 0.0), ("intra", -0.0015), ("uniform", 0.0015)):
                d = dist[dist["strategy"] == s].sort_values("q")
                x = d["q"].to_numpy() + dx
                ax.vlines(x, d["mean_distance_p025"], d["mean_distance_p975"], color=COLORS[s],
                          linewidth=1.4)
                ax.plot(np.r_[0, x], np.r_[exact, d["mean_distance_mean"]], color=COLORS[s],
                        marker="o", markersize=4, label=LABELS[s])
                _end_label(ax, x[-1], d["mean_distance_mean"].iloc[-1], s,
                           dy={"inter": 0, "intra": -6, "uniform": 6}[s])
            ax.scatter([0], [exact], s=40, color=INK, zorder=5)
            ax.annotate(f"exato, rede intacta: {_pt(exact, 3)}", xy=(0, exact), xytext=(6, 8),
                        textcoords="offset points", fontsize=8, color=INK_2)
            ax.set_xlim(-0.01, q_max * 1.15)
            ax.set_xlabel("q = arestas removidas / m")
            ax.set_ylabel("distância média na maior componente (passos)")
            ax.set_title("Distâncias dentro da maior componente (estimativa amostral)")
            ax.legend(loc="upper left")
            ax.annotate(f"{meta['num_sources']} fontes BFS por estado; {meta['replicates_per_strategy']} "
                        "sementes; barras: intervalo empírico 2,5–97,5% entre execuções",
                        xy=(0.0, -0.2), xycoords="axes fraction", fontsize=7.5, color=INK_2)
            fig.tight_layout()
            paths += _save(fig, fig_dir / "fig4_lcc_distances")
    return paths


# --------------------------------------------------------------------------- shared helpers

DIVERGING = ("#1c5cab", "#f0efec", "#c13b3a")   # blue pole, neutral midpoint, red pole


def _diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blue_gray_red", DIVERGING, N=256)
    cmap.set_bad("#ffffff")
    return cmap


# --------------------------------------------------------------------------- objective 2


def flows_figures(out_dir: Path) -> list[Path]:
    """Figures of objective 2 from ``outputs/obj2_flows``."""
    cats = pd.read_csv(out_dir / "category_flows.csv")
    oe = pd.read_csv(out_dir / "matrix_log2_OE_strength.csv", index_col=0)
    observed = pd.read_csv(out_dir / "matrix_observed.csv", index_col=0)
    fig_dir = out_dir / "figures"
    min_cell = 20
    paths: list[Path] = []
    order = cats.sort_values("nnf", ascending=False)["category_name"].tolist()

    # ---- Figure 1: observed / expected (strength null), ordered by net flow
    mat = oe.loc[order, order].to_numpy(dtype=float)
    small = observed.loc[order, order].to_numpy() < min_cell
    np.fill_diagonal(small, False)
    mat[small] = np.nan
    np.fill_diagonal(mat, np.nan)
    with plt.rc_context({**STYLE, "axes.grid": False}):
        fig, ax = plt.subplots(figsize=(8.6, 7.8))
        image = ax.imshow(np.clip(mat, -3, 3), cmap=_diverging_cmap(), vmin=-3, vmax=3)
        rows, cols = np.nonzero(small)
        ax.scatter(cols, rows, s=1.6, color=MUTED, linewidths=0)
        idx = np.arange(len(order))
        ax.scatter(idx, idx, marker="s", s=14, color=INK_2, linewidths=0)
        ax.set_xticks(range(len(order)), order, rotation=90, fontsize=6.8)
        ax.set_yticks(range(len(order)), order, fontsize=6.8)
        ax.set_xlabel("área citada (ordenada por fluxo líquido interárea, maior → menor)")
        ax.set_ylabel("área citante")
        ax.set_title("Citações observadas / esperadas (nulo de intensidade), log₂")
        bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02,
                           ticks=[-3, -2, -1, 0, 1, 2, 3])
        bar.ax.set_yticklabels(["≤ 1/8", "1/4", "1/2", "1", "2", "4", "≥ 8"])
        bar.set_label("observado / esperado")
        ax.annotate(f"diagonal (intra-área, quadrados escuros) omitida; pontos: menos de "
                    f"{min_cell} citações observadas (razão instável)",
                    xy=(0, -0.09), xycoords="axes fraction", color=INK_2, fontsize=7.5,
                    va="top")
        fig.tight_layout()
        paths += _save(fig, fig_dir / "fig1_observed_expected_heatmap")

    # ---- Figure 2: normalised net inter-area flow per category, with the age-null value
    d = cats.sort_values("nnf")
    y = np.arange(len(d))
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 8.2))
        ax.axvline(0, color=MUTED, linewidth=1)
        ax.hlines(y, d["nnf_ci95_low"], d["nnf_ci95_high"], color=COLORS["inter"], linewidth=1.6)
        ax.scatter(d["nnf"], y, s=34, color=COLORS["inter"], zorder=3,
                   label="observado (IC 95%, bootstrap por artigo citante)")
        ax.scatter(d["nnf_age"], y, s=30, facecolors="white", edgecolors=INK_2, linewidths=1.2,
                   zorder=3, label="esperado pelo nulo de idade")
        ax.set_yticks(y, d["category_name"], fontsize=7.5)
        ax.set_xlim(-0.75, 0.75)
        ax.set_xlabel("fluxo líquido interárea normalizado (recebidas − feitas) / (recebidas + feitas)")
        ax.set_title("Papel líquido de cada área nas citações interárea")
        ax.annotate("← cita mais outras áreas", xy=(0.02, -0.075), xycoords="axes fraction",
                    color=INK_2, fontsize=8)
        ax.annotate("é mais citada por outras áreas →", xy=(0.98, -0.075),
                    xycoords="axes fraction", color=INK_2, fontsize=8, ha="right")
        ax.grid(axis="y", visible=False)
        ax.legend(loc="upper left", fontsize=7.8)
        fig.tight_layout()
        paths += _save(fig, fig_dir / "fig2_net_flow_by_category")

    # ---- Figure 3: the 15 pairs with the largest net flow, observed vs expected direction
    top = pd.read_csv(out_dir / "top_pairs_by_net_flow.csv").head(15).iloc[::-1]
    y = np.arange(len(top))
    share = top["A_to_B"] / top["total"]
    lo = (1 + top["asymmetry_ci95_low"]) / 2
    hi = (1 + top["asymmetry_ci95_high"]) / 2
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.6, 5.6))
        ax.axvline(0.5, color=MUTED, linewidth=1)
        ax.hlines(y, lo, hi, color=COLORS["inter"], linewidth=1.6)
        ax.scatter(share, y, s=34, color=COLORS["inter"], zorder=3,
                   label="observado (IC 95%)")
        ax.scatter(top["expected_share_A_to_B_time"], y, marker="|", s=160, color=INK_2,
                   linewidths=2, zorder=4, label="esperado pelo nulo temporal")
        labels = [f"{a} → {b}  (n = {t:,})".replace(",", ".")
                  for a, b, t in zip(top["A_cites_more"], top["B_cited_more"], top["total"])]
        ax.set_yticks(y, labels, fontsize=7.8)
        ax.set_xlim(0.45, 0.85)
        ax.set_xlabel("fração das citações do par que vão de A para B")
        ax.set_title("Pares com maior fluxo líquido: A cita B mais do que B cita A")
        ax.grid(axis="y", visible=False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, fontsize=7.8)
        fig.tight_layout()
        paths += _save(fig, fig_dir / "fig3_top_pairs_net_flow")

    # ---- Figure 4: time — net flow vs mean publication year; received vs age expectation
    rho = {}
    import json

    meta = json.loads((out_dir / "flows.json").read_text(encoding="utf-8"))["spearman"]
    rho["nnf"] = meta["nnf_vs_year_mean"]["rho"]
    with plt.rc_context(STYLE):
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.5, 4.3))
        size = 12 + 120 * np.sqrt(cats["num_papers"] / cats["num_papers"].max())
        ax.axhline(0, color=MUTED, linewidth=1)
        ax.scatter(cats["year_mean"], cats["nnf"], s=size, color=COLORS["inter"], alpha=0.85,
                   edgecolors="white", linewidths=0.8, zorder=3)
        bx.axhline(0, color=MUTED, linewidth=1)
        log_oe = np.log2(cats["inter_in_OE_age"])
        bx.scatter(cats["year_mean"], log_oe, s=size, color=COLORS["inter"], alpha=0.85,
                   edgecolors="white", linewidths=0.8, zorder=3)
        show = set(cats.nlargest(4, "nnf")["category_name"]) | \
            set(cats.nsmallest(4, "nnf")["category_name"]) | \
            set(cats.nlargest(5, "num_papers")["category_name"])
        off_a = {"cs.IT": (4, -10), "cs.AI": (-26, 5), "cs.RO": (4, -11), "cs.SD": (4, 4), "cs.HC": (-14, -12)}
        for _, r in cats.iterrows():
            if r["category_name"] in show:
                ax.annotate(r["category_name"], (r["year_mean"], r["nnf"]),
                            xytext=off_a.get(r["category_name"], (4, 3)),
                            textcoords="offset points", fontsize=7.5, color=INK_2)
        show_b = set(cats.nlargest(5, "inter_in_OE_age")["category_name"]) | \
            set(cats.nsmallest(5, "inter_in_OE_age")["category_name"])
        off_b = {"cs.GR": (-26, -11), "cs.CL": (-24, 5)}
        for _, r in cats.iterrows():
            if r["category_name"] in show_b:
                bx.annotate(r["category_name"], (r["year_mean"], np.log2(r["inter_in_OE_age"])),
                            xytext=off_b.get(r["category_name"], (4, 3)),
                            textcoords="offset points", fontsize=7.5, color=INK_2)
        ax.set_xlabel("ano médio de publicação dos artigos da área")
        ax.set_ylabel("fluxo líquido interárea normalizado")
        ax.set_title(f"(a) Fluxo líquido vs. idade (Spearman ρ = {_pt(rho['nnf'], 2)})")
        bx.set_xlabel("ano médio de publicação dos artigos da área")
        bx.set_ylabel("log₂(citações interárea recebidas / esperadas pela idade)")
        bx.set_title("(b) Citações recebidas além da idade dos artigos")
        fig.tight_layout()
        paths += _save(fig, fig_dir / "fig4_time_effects")
    return paths


# --------------------------------------------------------------------------- objective 3


def overlap_figures(out_dir: Path, cache_dir: Path = Path("data/processed")) -> list[Path]:
    """Figures of objective 3 from ``outputs/obj3_overlap`` (+ the per-edge cache)."""
    import json

    fig_dir = out_dir / "figures"
    paths: list[Path] = []
    cache = Path(cache_dir) / "overlap_edges.npz"
    with np.load(cache) as npz:
        values, inter = npz["overlap"], npz["is_inter"]
    effects = pd.read_csv(out_dir / "overlap_effects.csv").set_index("statistic")
    bridges = pd.read_csv(out_dir / "bridges_by_class.csv").set_index("class")
    by_deg = pd.read_csv(out_dir / "overlap_by_degree_bin.csv")
    strata = pd.read_csv(out_dir / "overlap_by_degree_stratum.csv")
    cats = pd.read_csv(out_dir / "overlap_by_category.csv")
    pooled = json.loads((out_dir / "overlap.json").read_text())["pooled_stratified_cliffs_delta"]

    # ---- Figure 1: distributions and fractions
    with plt.rc_context(STYLE):
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.5, 4.2),
                                     gridspec_kw={"width_ratios": [1.35, 1]})
        for cls, mask in (("inter", inter), ("intra", ~inter)):
            v = np.sort(values[mask & ~np.isnan(values)])
            cdf = np.arange(1, v.size + 1) / v.size
            step = max(1, v.size // 4000)
            ax.step(v[::step], cdf[::step], where="post", color=COLORS[cls],
                    label=f"{LABELS[cls]} (mediana {_pt(np.median(v), 3)})")
        ax.set_xscale("symlog", linthresh=0.01)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.01, 0.03, 0.1, 0.3, 1], ["0", "0.01", "0.03", "0.1", "0.3", "1"])
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("neighborhood overlap O(u, v) (escala simlog)")
        ax.set_ylabel("fração das arestas com overlap ≤ x")
        ax.set_title(f"(a) Distribuição acumulada (Cliff's δ = "
                     f"{_pt(effects.loc['cliffs_delta', 'estimate'], 3)})")
        ax.legend(loc="lower right")
        metrics = [
            ("overlap = 0\n(ponte local)", effects.loc["frac_zero_inter", "estimate"],
             effects.loc["frac_zero_intra", "estimate"]),
            ("ponte global\n(qualquer)", bridges.loc["inter", "bridge_fraction"],
             bridges.loc["intra", "bridge_fraction"]),
            ("ponte global\nnão pendente", bridges.loc["inter", "non_pendant_bridge_fraction"],
             bridges.loc["intra", "non_pendant_bridge_fraction"]),
        ]
        x = np.arange(len(metrics))
        width = 0.34
        for off, cls, idx in ((-width / 2, "inter", 1), (width / 2, "intra", 2)):
            vals = [m[idx] * 100 for m in metrics]
            bars = bx.bar(x + off, vals, width=width * 0.92, color=COLORS[cls],
                          label=LABELS[cls])
            for rect, val in zip(bars, vals):
                bx.annotate(f"{_pt(val, 2)}%", (rect.get_x() + rect.get_width() / 2, val),
                            xytext=(0, 3), textcoords="offset points", ha="center",
                            fontsize=7.5, color=INK_2)
        bx.set_yscale("log")
        bx.set_ylim(0.1, 60)
        bx.set_xticks(x, [m[0] for m in metrics], fontsize=8)
        bx.set_ylabel("% das arestas da classe (escala log)")
        bx.set_title("(b) Pontes locais e globais")
        bx.grid(axis="x", visible=False)
        bx.legend(loc="upper right")
        fig.tight_layout()
        paths += _save(fig, fig_dir / "fig1_overlap_distribution")

    # ---- Figure 2: degree stratification
    with plt.rc_context(STYLE):
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.5, 4.4))
        d = by_deg[(by_deg["degree_variable"] == "k_min") & (by_deg["degree_bin"] >= 1)]
        for cls in ("inter", "intra"):
            c = d[d["class"] == cls].sort_values("degree_bin")
            ax.errorbar(2.0 ** c["degree_bin"], c["mean"], yerr=c["ci95_halfwidth_iid"],
                        color=COLORS[cls], marker="o", markersize=4, linewidth=2,
                        capsize=0, label=LABELS[cls])
        ax.set_xscale("log", base=2)
        ax.set_xlabel("menor grau dos extremos, k_min (faixa log₂; k_min = 1 omitido: O = 0)")
        ax.set_ylabel("overlap médio")
        ax.set_title("(a) Overlap médio por faixa de grau")
        ax.legend(loc="upper left")
        lowb, highb = strata["low_degree_bin"].max() + 1, strata["high_degree_bin"].max() + 1
        grid = np.full((lowb, highb), np.nan)
        for _, r in strata.iterrows():
            grid[int(r["low_degree_bin"]), int(r["high_degree_bin"])] = r["cliffs_delta"]
        image = bx.imshow(grid, origin="lower", cmap=_diverging_cmap(), vmin=-0.6, vmax=0.6,
                          aspect="auto")
        bx.set_xticks(range(highb), [str(2 ** b) for b in range(highb)], fontsize=7,
                      rotation=90)
        bx.set_yticks(range(lowb), [str(2 ** b) for b in range(lowb)], fontsize=7)
        bx.set_xlabel("faixa do maior grau (limite inferior)")
        bx.set_ylabel("faixa do menor grau (limite inferior)")
        bx.set_title(f"(b) Cliff's δ por estrato (combinado: {_pt(pooled['pooled_delta'], 3)})")
        bx.grid(False)
        bar = fig.colorbar(image, ax=bx, fraction=0.046, pad=0.02)
        bar.set_label("δ < 0: interárea com menor overlap")
        bx.annotate(f"{pooled['strata_with_negative_delta']} de {pooled['strata_used']} "
                    "estratos com δ < 0; brancos: < 30 arestas em alguma classe",
                    xy=(0, -0.3), xycoords="axes fraction", fontsize=7.5, color=INK_2)
        fig.tight_layout()
        paths += _save(fig, fig_dir / "fig2_overlap_by_degree")

    # ---- Figure 3: per category
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.4, 5.8))
        c = cats.dropna(subset=["mean_intra", "mean_inter"])
        c = c[c["intra_edges"] >= 30]
        size = 10 + 140 * np.sqrt((c["intra_edges"] + c["inter_edges_incident"])
                                  / (c["intra_edges"] + c["inter_edges_incident"]).max())
        lim = max(c["mean_intra"].max(), c["mean_inter"].max()) * 1.08
        ax.plot([0, lim], [0, lim], color=MUTED, linewidth=1)
        ax.scatter(c["mean_intra"], c["mean_inter"], s=size, color=COLORS["inter"], alpha=0.85,
                   edgecolors="white", linewidths=0.8, zorder=3)
        off_c = {"cs.LG": (-30, 6), "cs.CV": (6, 2), "cs.CL": (6, -9), "cs.LO": (6, -9)}
        for _, r in c.iterrows():
            if r["mean_intra"] > 0.14 or r["intra_edges"] > 50_000:
                ax.annotate(r["category_name"], (r["mean_intra"], r["mean_inter"]),
                            xytext=off_c.get(r["category_name"], (4, 3)),
                            textcoords="offset points", fontsize=7.5, color=INK_2)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("overlap médio das arestas intra-área da categoria")
        ax.set_ylabel("overlap médio das arestas interárea incidentes")
        below = int((c["mean_inter"] < c["mean_intra"]).sum())
        ax.set_title(f"Overlap por categoria: {below} de {len(c)} abaixo da diagonal")
        ax.annotate("diagonal: igualdade; tamanho ∝ arestas da categoria;\n"
                    "categorias com ≥ 30 arestas intra-área", xy=(0.02, 0.92),
                    xycoords="axes fraction", fontsize=7.5, color=INK_2)
        fig.tight_layout()
        paths += _save(fig, fig_dir / "fig3_overlap_by_category")
    return paths


# --------------------------------------------------------------------------- objective 6


def communities_figures(out_dir: Path) -> list[Path]:
    """Figures of objective 6 from ``outputs/obj6_communities``."""
    import json

    runs = pd.read_csv(out_dir / "runs.csv")
    meta = json.loads((out_dir / "communities.json").read_text(encoding="utf-8"))
    cats = pd.read_csv(out_dir / "categories_vs_communities.csv")
    comms = pd.read_csv(out_dir / "communities_composition.csv")
    N = pd.read_csv(out_dir / "contingency_reference_run.csv", index_col=0)
    fig_dir = out_dir / "figures"
    paths: list[Path] = []
    q_cat = meta["category_partition"]["modularity_directed_independent"]
    x = np.arange(len(runs))

    # ---- Figure 1: runs
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
        ax = axes[0]
        ax.scatter(x, runs["modularity_directed"], color=COLORS["inter"], s=26, zorder=3,
                   label="Leiden (cada semente)")
        ax.axhline(q_cat, color=COLORS["intra"], linewidth=1.8, label="40 categorias")
        ax.set_ylim(0.45, 0.75)
        ax.set_xticks(x, [str(i + 1) for i in x], fontsize=7.5)
        ax.set_xlabel("execução (semente)")
        ax.set_ylabel("modularidade dirigida Q_d")
        ax.set_title("(a) Modularidade")
        ax.legend(loc="center right", fontsize=7.8)
        ax = axes[1]
        ax.scatter(x, runs["num_communities"], color=COLORS["inter"], s=26, zorder=3,
                   label="todas")
        ax.scatter(x, runs["communities_ge_1000"], color=INK_2, s=22, marker="s", zorder=3,
                   label="com ≥ 1.000 artigos")
        ax.axhline(40, color=COLORS["intra"], linewidth=1.8, label="40 categorias")
        ax.set_xticks(x, [str(i + 1) for i in x], fontsize=7.5)
        ax.set_xlabel("execução (semente)")
        ax.set_ylabel("número de comunidades")
        ax.set_title("(b) Número de comunidades")
        ax.set_ylim(0, None)
        ax.legend(loc="center right", fontsize=7.8)
        ax = axes[2]
        ax.scatter(x, runs["nmi_vs_categories"], color=COLORS["inter"], s=26, zorder=3,
                   label="NMI vs. categorias")
        ax.scatter(x, runs["ari_vs_categories"], color=COLORS["uniform"], s=26, marker="D",
                   zorder=3, label="ARI vs. categorias")
        stab = meta["stability_between_runs"]
        ax.axhline(stab["nmi_mean"], color=COLORS["inter"], linewidth=1, linestyle="--")
        ax.annotate(f"NMI entre execuções: {_pt(stab['nmi_mean'], 2)}",
                    xy=(0.02, stab["nmi_mean"]), xycoords=("axes fraction", "data"),
                    xytext=(0, 4), textcoords="offset points", fontsize=7.5, color=INK_2)
        chance = meta["chance_level_label_permutation"]["nmi_mean"]
        ax.axhline(chance, color=MUTED, linewidth=1, linestyle=":")
        ax.annotate(f"acaso (rótulos permutados): {_pt(chance, 3)}", xy=(0.02, chance),
                    xycoords=("axes fraction", "data"), xytext=(0, 4),
                    textcoords="offset points", fontsize=7.5, color=INK_2)
        ax.set_ylim(0, 1)
        ax.set_xticks(x, [str(i + 1) for i in x], fontsize=7.5)
        ax.set_xlabel("execução (semente)")
        ax.set_title("(c) Concordância com as categorias")
        ax.legend(loc="center right", fontsize=7.8)
        fig.tight_layout()
        paths += _save(fig, fig_dir / "fig1_runs")

    # ---- Figure 2: contingency (categories x large communities), row-normalised
    sizes = N.sum(axis=0)
    big = sizes[sizes >= 1000].index
    share = N.div(N.sum(axis=1), axis=0)
    cat_order = cats.sort_values("num_papers", ascending=False)["category_name"].tolist()
    dom = N[big].idxmax(axis=0)        # dominant category of each large community
    rank = {c: i for i, c in enumerate(cat_order)}
    comm_order = sorted(big, key=lambda c: (rank[N.index[N.index.get_loc(dom[c])]],
                                            -sizes[c]))
    # rows: categories ordered by the column of their dominant large community (block view)
    col_pos = {c: i for i, c in enumerate(comm_order)}
    best_col = {cat: max(comm_order, key=lambda c: share.loc[cat, c]) for cat in share.index}
    cat_order = sorted(share.index, key=lambda cat: (col_pos[best_col[cat]],
                                                     -share.loc[cat, best_col[cat]]))
    mat = share.loc[cat_order, comm_order].to_numpy()
    with plt.rc_context({**STYLE, "axes.grid": False}):
        fig, ax = plt.subplots(figsize=(10.5, 8.0))
        image = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_yticks(range(len(cat_order)), cat_order, fontsize=7)
        labels = [f"{dom[c]}·{int(sizes[c] / 1000)}k" for c in comm_order]
        ax.set_xticks(range(len(comm_order)), labels, rotation=90, fontsize=6.8)
        ax.set_xlabel("comunidades com ≥ 1.000 artigos (rótulo: categoria dominante · tamanho)")
        ax.set_ylabel("categoria primária (agrupada pela comunidade em que mais se concentra)")
        ax.set_title("Fração dos artigos de cada categoria em cada comunidade (execução de "
                     "maior Q_d)")
        bar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
        bar.set_label("fração dos artigos da categoria")
        covered = sizes[big].sum() / sizes.sum()
        ax.annotate(f"{len(big)} comunidades grandes cobrem {_pt(100 * covered, 1)}% dos "
                    "artigos", xy=(0, -0.2), xycoords="axes fraction", fontsize=7.5,
                    color=INK_2)
        fig.tight_layout()
        paths += _save(fig, fig_dir / "fig2_contingency")

    # ---- Figure 3: heterogeneity of the correspondence
    with plt.rc_context(STYLE):
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.5, 4.6))
        size = 12 + 140 * np.sqrt(cats["num_papers"] / cats["num_papers"].max())
        match = cats["mutual_best_match"].astype(bool)
        ax.scatter(cats.loc[match, "share_in_dominant_community"],
                   cats.loc[match, "effective_num_communities"], s=size[match],
                   color=COLORS["inter"], alpha=0.85, edgecolors="white", linewidths=0.8,
                   label="domina a sua comunidade principal")
        ax.scatter(cats.loc[~match, "share_in_dominant_community"],
                   cats.loc[~match, "effective_num_communities"], s=size[~match],
                   facecolors="white", edgecolors=COLORS["inter"], linewidths=1.2,
                   label="minoria na sua comunidade principal")
        off = {"cs.LG": (-40, -20), "cs.DC": (-32, 6), "cs.MM": (6, 3), "cs.NI": (6, 3),
               "cs.CG": (-30, 6), "cs.SI": (6, 2), "cs.DL": (-28, -4), "cs.LO": (6, -6),
               "cs.FL": (6, 2)}
        for _, r in cats.iterrows():
            if r["num_papers"] > 6000 or r["share_in_dominant_community"] < 0.2 or \
                    r["effective_num_communities"] > 14 or r["share_in_dominant_community"] > 0.75:
                arrow = {"arrowstyle": "-", "color": MUTED, "linewidth": 0.7} \
                    if r["category_name"] == "cs.LG" else None
                ax.annotate(r["category_name"], (r["share_in_dominant_community"],
                                                 r["effective_num_communities"]),
                            xytext=off.get(r["category_name"], (5, 3)),
                            textcoords="offset points", fontsize=7.3, color=INK_2,
                            arrowprops=arrow)
        ax.set_yscale("log")
        ax.set_xlim(0, 1)
        ax.set_xlabel("fração dos artigos na comunidade dominante")
        ax.set_ylabel("número efetivo de comunidades (exp. da entropia)")
        ax.set_title("(a) Dispersão de cada categoria")
        ax.legend(loc="upper right", fontsize=7.8)
        c = comms[comms["size"] >= 100]
        bx.scatter(c["size"], c["purity"], s=18 + 10 * np.log10(c["size"]),
                   color=COLORS["inter"], alpha=0.8, edgecolors="white", linewidths=0.6)
        for (_, r), (dx, dy) in zip(c.nlargest(4, "size").iterrows(),
                                    ((-70, -22), (-75, 14), (-40, -30), (-80, 4))):
            bx.annotate(f"{r['dominant_category']} ({_pt(r['purity'], 2)})",
                        (r["size"], r["purity"]), xytext=(dx, dy),
                        textcoords="offset points", fontsize=7.3, color=INK_2,
                        arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.7})
        bx.annotate("rótulo: categoria dominante da comunidade", xy=(0.02, 0.03),
                    xycoords="axes fraction", fontsize=7.5, color=INK_2)
        bx.set_xscale("log")
        bx.set_ylim(0, 1.02)
        bx.set_xlabel("tamanho da comunidade (artigos, escala log)")
        bx.set_ylabel("pureza (fração da categoria dominante)")
        bx.set_title("(b) Composição das comunidades (≥ 100 artigos)")
        fig.tight_layout()
        paths += _save(fig, fig_dir / "fig3_heterogeneity")
    return paths


# --------------------------------------------------------------------------- objective 7


def ordering_figures(out_dir: Path) -> list[Path]:
    """Figures of objective 7 from ``outputs/obj7_ordering``."""
    import json

    meta = json.loads((out_dir / "ordering.json").read_text(encoding="utf-8"))
    table = pd.read_csv(out_dir / "ordering.csv")
    W = pd.read_csv(out_dir / "inter_area_weights.csv", index_col=0)
    perm = pd.read_csv(out_dir / "random_permutations.csv")["fraction"]
    null = pd.read_csv(out_dir / "direction_null.csv")
    fig_dir = out_dir / "figures"
    paths: list[Path] = []
    order = table["category_name"].tolist()
    Wo = W.loc[order, order].to_numpy()
    T = Wo + Wo.T
    min_total = 50

    # ---- Figure 1: matrix in the optimal order
    with np.errstate(divide="ignore", invalid="ignore"):
        share = Wo / T - 0.5
    small = T < min_total
    share[small] = np.nan
    np.fill_diagonal(share, np.nan)
    with plt.rc_context({**STYLE, "axes.grid": False}):
        fig, ax = plt.subplots(figsize=(8.6, 7.8))
        image = ax.imshow(share, cmap=_diverging_cmap(), vmin=-0.5, vmax=0.5)
        rows, cols = np.nonzero(small & ~np.eye(len(order), dtype=bool))
        ax.scatter(cols, rows, s=1.6, color=MUTED, linewidths=0)
        ax.plot([-0.5, len(order) - 0.5], [-0.5, len(order) - 0.5], color=INK_2, linewidth=0.8)
        ax.set_xticks(range(len(order)), order, rotation=90, fontsize=6.8)
        ax.set_yticks(range(len(order)), order, fontsize=6.8)
        ax.set_xlabel("área citada (ordem encontrada)")
        ax.set_ylabel("área citante (ordem encontrada)")
        ax.set_title(f"Direção das citações na ordem encontrada (F = "
                     f"{_pt(meta['result']['compatible_fraction'], 3)})")
        bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02,
                           ticks=[-0.5, -0.25, 0, 0.25, 0.5])
        bar.ax.set_yticklabels(["0 %", "25 %", "50 %", "75 %", "100 %"])
        bar.set_label("fração das citações do par que vão da linha para a coluna")
        ax.annotate("abaixo da diagonal: citante depois do citado (compatível) — vermelho = "
                    "a direção dominante concorda com a ordem; pontos: par com < "
                    f"{min_total} citações", xy=(0, -0.1), xycoords="axes fraction",
                    fontsize=7.3, color=INK_2, va="top")
        fig.tight_layout()
        paths += _save(fig, fig_dir / "fig1_ordered_matrix")

    # ---- Figure 2: observed optimum vs random permutations and direction-randomised optima
    obs = meta["result"]["compatible_fraction"]
    b = meta["bounds"]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8.0, 4.0))
        bins = np.linspace(0.40, 0.62, 111)
        ax.hist(perm, bins=bins, color=MUTED, alpha=0.75, density=True,
                label="permutações aleatórias (10.000)")
        ax.hist(null["optimum_fraction"], bins=bins, color=COLORS["uniform"], alpha=0.85,
                density=True, label="ótimo com direções sorteadas por par (500)")
        ax.axvline(obs, color=COLORS["inter"], linewidth=2.2, label="ótimo observado")
        ax.axvline(b["pairwise_bound"], color=INK, linewidth=1, linestyle="--",
                   label="limite por pares U")
        ax.axvline(b["cycle_packing_bound"], color=INK_2, linewidth=1, linestyle=":",
                   label="limite com ciclos U_cyc")
        ax.set_xlabel("fração do peso das citações interárea compatível com a ordem")
        ax.set_ylabel("densidade")
        ax.set_title("Ordem observada vs. controles")
        ax.legend(loc="upper left", fontsize=7.8)
        ax.grid(axis="y", visible=False)
        fig.tight_layout()
        paths += _save(fig, fig_dir / "fig2_controls")

    # ---- Figure 3: positional uncertainty and age
    rho = meta["time"]["position_vs_year_mean"]["spearman_rho"]
    with plt.rc_context(STYLE):
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.5, 7.2),
                                     gridspec_kw={"width_ratios": [1.1, 1]})
        y = table["position"].to_numpy()
        ax.hlines(y, table["bootstrap_position_p025"], table["bootstrap_position_p975"],
                  color=COLORS["inter"], linewidth=1.6)
        ax.scatter(table["bootstrap_position_median"], y, color=COLORS["inter"], s=22,
                   zorder=3, label="mediana bootstrap (IC 95%)")
        ax.scatter(y, y, color=INK, marker="|", s=80, zorder=4, label="posição na ordem ótima")
        ax.set_yticks(y, table["category_name"], fontsize=7.3)
        ax.invert_yaxis()
        ax.set_xlabel("posição (0 = primeira; citada pelas seguintes)")
        ax.set_title("(a) Incerteza da posição (200 réplicas bootstrap)")
        ax.legend(loc="lower left", fontsize=7.8)
        ax.grid(axis="y", visible=False)
        bx.scatter(table["year_mean"], table["position"], color=COLORS["inter"],
                   s=12 + 120 * np.sqrt(table["num_papers"] / table["num_papers"].max()),
                   alpha=0.85, edgecolors="white", linewidths=0.8)
        for _, r in table.iterrows():
            if r["position"] < 5 or r["position"] > 34 or r["num_papers"] > 10000:
                bx.annotate(r["category_name"], (r["year_mean"], r["position"]),
                            xytext=(4, 2), textcoords="offset points", fontsize=7.3,
                            color=INK_2)
        bx.invert_yaxis()
        bx.set_xlabel("ano médio de publicação dos artigos da área")
        bx.set_ylabel("posição na ordem")
        bx.set_title(f"(b) Posição vs. idade (Spearman ρ = {_pt(rho, 2)})")
        fig.tight_layout()
        paths += _save(fig, fig_dir / "fig3_positions_and_age")
    return paths


def clustering_figures(out_dir: Path) -> list[Path]:
    """Objective 1 complement: C(k), the neighbour-degree bound, weights and degree tails."""
    bins = pd.read_csv(out_dir / "clustering_by_degree_bin.csv")
    dd = pd.read_csv(out_dir / "degree_distributions.csv")
    fig_dir = out_dir / "figures"
    centre = np.sqrt(bins["k_low"] * bins["k_high"])
    with plt.rc_context(STYLE):
        fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(11.0, 3.9),
                                         gridspec_kw={"width_ratios": [1.15, 1.0, 0.95]})
        # (a) C(k) and the bound imposed by the neighbours' degrees
        ax.plot(centre, bins["mean_bound"], color=MUTED, linestyle="--", marker="s",
                markersize=4, linewidth=1.6, label="limite pelos graus dos vizinhos")
        ax.plot(centre, bins["mean_local_clustering"], color=COLORS["inter"], marker="o",
                markersize=5, label="C(k) observado (média na faixa)")
        if "same_community_neighbour_pairs" in bins:
            ok = bins["num_vertices"] >= 30  # same minimum bin size as the descriptive trend
            ax.plot(centre[ok], bins.loc[ok, "same_community_neighbour_pairs"],
                    color=COLORS["uniform"], marker="^", markersize=4.5, linewidth=1.6,
                    linestyle=":", label="pares de vizinhos na mesma comunidade")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(4e-4, 2.0)
        for x, nv in zip(centre, bins["num_vertices"]):
            label = f"{nv / 1000:.0f}k" if nv >= 1000 else f"{nv}"
            ax.annotate(label, xy=(x, 5.2e-4), ha="center", va="bottom", fontsize=7,
                        color=INK_2, rotation=90)
        ax.set_xlabel("grau k (faixas log₂; rótulos: vértices na faixa)")
        ax.set_ylabel("média na faixa de grau")
        ax.set_title("(a) C(k) e limite imposto pelos vizinhos")
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.13), fontsize=8)
        # (b) share of vertices vs share of connected triples per degree bin
        x = np.arange(len(bins))
        width = 0.29
        bx.bar(x - width, bins["vertex_share"], width, color=COLORS["inter"],
               label="vértices (k ≥ 2)")
        bx.bar(x, bins["triple_share"], width, color=COLORS["intra"],
               label="triplas (possíveis)")
        bx.bar(x + width, bins["closed_triple_share"], width, color=COLORS["uniform"],
               label="triplas fechadas")
        bx.set_yscale("log")
        bx.set_ylim(1e-5, 40.0)
        bx.set_xticks(x, [f"{lo}" for lo in bins["k_low"]], rotation=90, fontsize=7.5)
        bx.set_xlabel("limite inferior da faixa de grau")
        bx.set_ylabel("fração do total")
        bx.set_title("(b) Vértices e triplas por faixa de grau")
        bx.legend(loc="upper center", fontsize=8, ncol=1)
        bx.grid(axis="x", visible=False)
        # (c) in- and out-degree tails of G_d
        for kind, color, label in (("in", COLORS["inter"], "entrada (citações recebidas)"),
                                   ("out", COLORS["intra"], "saída (citações feitas)")):
            d = dd[(dd["kind"] == kind) & (dd["degree"] > 0)]
            cx.plot(d["degree"], d["ccdf"], color=color, linewidth=1.8, label=label)
        cx.set_xscale("log")
        cx.set_yscale("log")
        cx.set_xlabel("grau em $G_d$")
        cx.set_ylabel("P(grau ≥ k)")
        cx.set_title("(c) Caudas dos graus de entrada e saída")
        cx.legend(loc="lower left", fontsize=8)
        fig.tight_layout()
        paths = _save(fig, fig_dir / "fig1_clustering_degree")
    return paths
