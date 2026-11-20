"""Loading of the published GraphML instances into plain arrays for the analyses.

Source of truth
---------------
The project instances were published as GraphML in the F1 (``outputs/graphml``, versioned
with Git LFS). The analyses read them directly instead of re-downloading the raw OGB archive:

* ``G_d`` (edges, node attributes) comes from ``ogbn_arxiv_directed.graphml``;
* ``G_u`` is *rebuilt* from ``G_d`` with the project conversion code
  (:func:`mo438.graph_build.undirected_edge_array`) and then checked to be
  identical — same edge set, same node attributes — to ``ogbn_arxiv_undirected.graphml``.

The reader is deliberately strict: it understands exactly the schema written by
:func:`mo438.graph_io.write_graphml` (nodes ``n0..n{N-1}`` in order, typed ``<data>``
children, empty ``<edge/>`` elements) and raises on anything else instead of guessing.

Parsed arrays are cached in ``data/processed`` keyed by the SHA-256 of both GraphML
files, so a changed input can never be served from a stale cache.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.sax.saxutils import unescape

import numpy as np

from .data import CitationData
from .graph_build import ConversionStats, undirected_edge_array

READER_VERSION = 1

_KEY_RE = re.compile(
    r'<key id="([^"]+)" for="(graph|node|edge)" attr\.name="([^"]+)" attr\.type="([^"]+)"/>'
)
_GRAPH_RE = re.compile(r'<graph id="[^"]*" edgedefault="(directed|undirected)">')
_GRAPH_DATA_RE = re.compile(r'<data key="(g_[^"]+)">(.*?)</data>')
_NODE_RE = re.compile(r'<node id="n(\d+)">(.*?)</node>')
_DATA_RE = re.compile(r'<data key="([^"]+)">(.*?)</data>')
_EDGE_RE = re.compile(r'<edge source="n(\d+)" target="n(\d+)"/>')

_NUMERIC_TYPES = {"int": np.int64, "long": np.int64, "double": np.float64}
REQUIRED_NODE_ATTRIBUTES = ("node_id", "category", "category_name", "year", "paper_id")


class GraphMLFormatError(ValueError):
    """The file does not follow the schema written by ``mo438.graph_io``."""


@dataclass
class GraphMLArrays:
    """Content of one GraphML file as arrays."""

    path: Path
    directed: bool
    num_nodes: int
    edges: np.ndarray
    node_attributes: dict[str, np.ndarray]
    graph_attributes: dict[str, str]


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    """Hex SHA-256 of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _looks_like_lfs_pointer(path: Path) -> bool:
    with Path(path).open("rb") as handle:
        return handle.read(64).startswith(b"version https://git-lfs")


def read_graphml(path: str | Path) -> GraphMLArrays:
    """Read a GraphML file written by :func:`mo438.graph_io.write_graphml`.

    Raises:
        FileNotFoundError: the file does not exist.
        GraphMLFormatError: the file is a Git LFS pointer or deviates from the schema.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if _looks_like_lfs_pointer(path):
        raise GraphMLFormatError(f"{path} is a Git LFS pointer; run `git lfs pull` first")
    text = path.read_text(encoding="utf-8")

    graph_match = _GRAPH_RE.search(text)
    if graph_match is None:
        raise GraphMLFormatError(f"{path}: <graph edgedefault=...> element not found")
    directed = graph_match.group(1) == "directed"

    node_keys: dict[str, tuple[str, str]] = {}
    for key_id, scope, name, kind in _KEY_RE.findall(text):
        if scope == "node":
            node_keys[key_id] = (name, kind)
    graph_attributes = {k[2:]: unescape(v) for k, v in _GRAPH_DATA_RE.findall(text)}

    ids: list[int] = []
    columns: dict[str, list[str]] = {name: [] for name, _ in node_keys.values()}
    for match in _NODE_RE.finditer(text):
        ids.append(int(match.group(1)))
        values = dict(_DATA_RE.findall(match.group(2)))
        if values.keys() != node_keys.keys():
            raise GraphMLFormatError(f"{path}: node n{ids[-1]} has data keys {sorted(values)}")
        for key_id, (name, _) in node_keys.items():
            columns[name].append(values[key_id])
    num_nodes = len(ids)
    if text.count("<node ") != num_nodes:
        raise GraphMLFormatError(f"{path}: some <node> elements do not match the schema")
    if not np.array_equal(np.asarray(ids, dtype=np.int64), np.arange(num_nodes)):
        raise GraphMLFormatError(f"{path}: node ids are not n0..n{num_nodes - 1} in order")

    node_attributes: dict[str, np.ndarray] = {}
    for name, kind in node_keys.values():
        raw = columns[name]
        if kind in _NUMERIC_TYPES:
            node_attributes[name] = np.asarray(raw, dtype=_NUMERIC_TYPES[kind])
        else:
            node_attributes[name] = np.asarray([unescape(v) for v in raw], dtype=object)

    pairs = _EDGE_RE.findall(text)
    if text.count("<edge ") != len(pairs):
        raise GraphMLFormatError(f"{path}: some <edge> elements do not match the schema")
    edges = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if edges.size and (edges.min() < 0 or edges.max() >= num_nodes):
        raise GraphMLFormatError(f"{path}: edge endpoint outside n0..n{num_nodes - 1}")
    return GraphMLArrays(path, directed, num_nodes, edges, node_attributes, graph_attributes)


@dataclass
class Instances:
    """G_d and G_u of the project as arrays (node index = OGB node index).

    Attributes:
        labels: primary-category index of every paper.
        category_names: ``category_names[k]`` is the arXiv name of label ``k``.
        gd_edges: ``(m_d, 2)`` directed edges, row ``(u, v)`` = paper u cites paper v,
            in the order of the directed GraphML (= OGB ``edge.csv`` order).
        gu_edges: ``(m_u, 2)`` undirected edges with ``u < v``, sorted.
        conversion: bookkeeping of the G_d -> G_u conversion (:mod:`graph_build`).
        sources: input files with their SHA-256 (provenance).
    """

    num_nodes: int
    labels: np.ndarray
    category_names: list[str]
    years: np.ndarray
    paper_ids: np.ndarray
    gd_edges: np.ndarray
    gu_edges: np.ndarray
    conversion: ConversionStats
    sources: dict[str, Any] = field(default_factory=dict)

    @property
    def num_categories(self) -> int:
        return len(self.category_names)

    def to_citation_data(self) -> CitationData:
        """View as a :class:`~mo438.data.CitationData` (to reuse the graph-building code)."""
        return CitationData(
            name="ogbn-arxiv",
            num_nodes=self.num_nodes,
            edges=self.gd_edges,
            labels=self.labels,
            category_names=self.category_names,
            years=self.years,
            paper_ids=self.paper_ids,
            source={"dataset": "ogbn-arxiv", "loaded_from": self.sources},
        )


def _category_names(labels: np.ndarray, names: np.ndarray) -> list[str]:
    """Label -> name mapping, checking that it is a bijection onto 0..K-1."""
    k = int(labels.max()) + 1
    mapping: dict[int, str] = {}
    for label, name in zip(labels.tolist(), names.tolist()):
        if mapping.setdefault(label, name) != name:
            raise GraphMLFormatError(f"label {label} has two names: {mapping[label]!r}, {name!r}")
    if sorted(mapping) != list(range(k)):
        raise GraphMLFormatError("category labels are not contiguous 0..K-1")
    result = [mapping[i] for i in range(k)]
    if len(set(result)) != k:
        raise GraphMLFormatError("two labels share the same category name")
    return result


def build_instances(directed: GraphMLArrays, undirected: GraphMLArrays) -> Instances:
    """Assemble :class:`Instances` and check G_u (rebuilt) against the undirected GraphML."""
    if not directed.directed or undirected.directed:
        raise GraphMLFormatError("expected one directed and one undirected GraphML")
    missing = set(REQUIRED_NODE_ATTRIBUTES) - set(directed.node_attributes)
    if missing:
        raise GraphMLFormatError(f"directed GraphML lacks node attributes {sorted(missing)}")
    n = directed.num_nodes
    attrs = directed.node_attributes
    if not np.array_equal(attrs["node_id"], np.arange(n)):
        raise GraphMLFormatError("node_id attribute differs from the node index")
    labels = attrs["category"].astype(np.int64)
    category_names = _category_names(labels, attrs["category_name"])

    gu_edges, conversion = undirected_edge_array(directed.edges, n, remove_self_loops=True)

    # G_u rebuilt from G_d must be exactly the published undirected instance.
    if undirected.num_nodes != n:
        raise GraphMLFormatError("directed/undirected GraphML have different node counts")
    for name in REQUIRED_NODE_ATTRIBUTES:
        if not np.array_equal(undirected.node_attributes[name], attrs[name]):
            raise GraphMLFormatError(f"node attribute {name!r} differs between the GraphML files")
    published = np.sort(np.minimum(undirected.edges[:, 0], undirected.edges[:, 1]) * n
                        + np.maximum(undirected.edges[:, 0], undirected.edges[:, 1]))
    rebuilt = gu_edges[:, 0] * n + gu_edges[:, 1]
    if published.size != rebuilt.size or not np.array_equal(published, rebuilt):
        raise GraphMLFormatError("G_u rebuilt from G_d differs from the undirected GraphML")

    return Instances(
        num_nodes=n,
        labels=labels,
        category_names=category_names,
        years=attrs["year"].astype(np.int64),
        paper_ids=attrs["paper_id"].astype(np.int64),
        gd_edges=directed.edges,
        gu_edges=gu_edges,
        conversion=conversion,
    )


# --------------------------------------------------------------------------- cache


def _cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"instances_{key[:16]}.npz"


def _save_cache(inst: Instances, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "reader_version": READER_VERSION,
        "num_nodes": inst.num_nodes,
        "category_names": inst.category_names,
        "conversion": inst.conversion.to_dict(),
        "sources": inst.sources,
    }
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        tmp, labels=inst.labels, years=inst.years, paper_ids=inst.paper_ids,
        gd_edges=inst.gd_edges, gu_edges=inst.gu_edges, meta=np.array(json.dumps(meta)),
    )
    tmp.replace(path)


def _load_cache(path: Path) -> Instances | None:
    try:
        with np.load(path, allow_pickle=False) as npz:
            meta = json.loads(str(npz["meta"]))
            if meta.get("reader_version") != READER_VERSION:
                return None
            return Instances(
                num_nodes=int(meta["num_nodes"]),
                labels=npz["labels"],
                category_names=list(meta["category_names"]),
                years=npz["years"],
                paper_ids=npz["paper_ids"],
                gd_edges=npz["gd_edges"],
                gu_edges=npz["gu_edges"],
                conversion=ConversionStats(**meta["conversion"]),
                sources=meta["sources"],
            )
    except (OSError, KeyError, ValueError):
        return None


def load_instances(
    directed_graphml: str | Path,
    undirected_graphml: str | Path,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
) -> Instances:
    """Load G_d and G_u from the published GraphML files (with a content-addressed cache)."""
    directed_graphml, undirected_graphml = Path(directed_graphml), Path(undirected_graphml)
    sources = {
        "directed_graphml": {"path": str(directed_graphml),
                             "sha256": sha256_file(directed_graphml)},
        "undirected_graphml": {"path": str(undirected_graphml),
                               "sha256": sha256_file(undirected_graphml)},
        "reader_version": READER_VERSION,
    }
    key = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    cache = _cache_path(Path(cache_dir), key) if cache_dir is not None else None
    if use_cache and cache is not None and cache.exists():
        cached = _load_cache(cache)
        if cached is not None and cached.sources.get("directed_graphml", {}).get("sha256") == \
                sources["directed_graphml"]["sha256"]:
            cached.sources = {**cached.sources, "cache_file": str(cache)}
            return cached
    inst = build_instances(read_graphml(directed_graphml), read_graphml(undirected_graphml))
    inst.sources = sources
    if cache is not None:
        _save_cache(inst, cache)
        inst.sources = {**sources, "cache_file": str(cache)}
    return inst
