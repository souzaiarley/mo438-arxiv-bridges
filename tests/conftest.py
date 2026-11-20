"""Shared fixtures.

* ``toy``: a hand-made 8-node citation graph whose expected values were derived by hand
  (see ``TOY_EXPECTED``); it contains a duplicated citation, a self-loop and one
  reciprocal pair in each class, so every branch of G_d -> G_u is exercised.
* ``real_instances`` & co.: the ogbn-arxiv instances published in the F1
  (``outputs/graphml``, Git LFS).
  Tests using them are marked ``realdata`` and are skipped if the files are missing or are
  LFS pointers (run ``git lfs pull``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
GRAPHML_DIR = REPO / "outputs" / "graphml"
CHARACTERIZATION_DIR = REPO / "outputs" / "obj1_characterization"   # values published in F1
DIRECTED_GRAPHML = GRAPHML_DIR / "ogbn_arxiv_directed.graphml"
UNDIRECTED_GRAPHML = GRAPHML_DIR / "ogbn_arxiv_undirected.graphml"
AREA_GRAPHML = GRAPHML_DIR / "ogbn_arxiv_area_graph.graphml"

# --------------------------------------------------------------------------- toy graph
# labels: papers 0-2 -> category 0, 3-4 -> category 1, 5-7 -> category 2
TOY_LABELS = np.array([0, 0, 0, 1, 1, 2, 2, 2])
TOY_NAMES = ["cs.AA", "cs.BB", "cs.CC"]
# (u, v): paper u cites paper v
TOY_EDGES = np.array([
    (1, 0), (2, 0), (2, 1), (2, 1),   # intra (cat 0); (2, 1) duplicated
    (3, 0),                           # inter
    (4, 3), (3, 4),                   # intra reciprocal pair (cat 1)
    (5, 3), (3, 5),                   # inter reciprocal pair
    (6, 5), (7, 6),                   # intra (cat 2)
    (7, 2),                           # inter
    (6, 6),                           # self-loop (intra by definition)
])
TOY_EXPECTED = {
    "m_d": 13, "self_loops": 1, "duplicates": 1, "reciprocal_pairs": 2, "m_u": 9,
    "gu_edges": [(0, 1), (0, 2), (0, 3), (1, 2), (2, 7), (3, 4), (3, 5), (5, 6), (6, 7)],
    "gu_inter_pairs": {(0, 3), (2, 7), (3, 5)},
    "gd_intra": 9, "gd_inter": 4, "gu_intra": 6, "gu_inter": 3,
    "decomposition": {
        "intra": {"difference": 3, "self_loops_removed": 1, "extra_merged_edges": 2,
                  "reciprocal_pairs_merged": 1, "duplicate_citations_merged": 1},
        "inter": {"difference": 1, "self_loops_removed": 0, "extra_merged_edges": 1,
                  "reciprocal_pairs_merged": 1, "duplicate_citations_merged": 0},
    },
    "node_composition": {"isolated": 0, "only_intra_edges": 3, "only_inter_edges": 0,
                         "both_classes": 5},
    "per_category": {
        "G_d_intra": [4, 2, 3], "G_d_inter_out": [0, 2, 2], "G_d_inter_in": [2, 1, 1],
        "G_u_intra": [3, 1, 2], "G_u_inter_incident": [2, 2, 2],
    },
}


@pytest.fixture
def toy():
    from mo438.data import CitationData

    return CitationData(name="toy", num_nodes=8, edges=TOY_EDGES, labels=TOY_LABELS,
                        category_names=TOY_NAMES, years=np.arange(2000, 2008),
                        paper_ids=np.arange(100, 108))


# --------------------------------------------------------------------------- real data


def _real_data_available() -> bool:
    for path in (DIRECTED_GRAPHML, UNDIRECTED_GRAPHML):
        if not path.is_file():
            return False
        with path.open("rb") as handle:
            if handle.read(64).startswith(b"version https://git-lfs"):
                return False
    return True


@pytest.fixture(scope="session")
def real_instances(tmp_path_factory):
    if not _real_data_available():
        pytest.skip("ogbn-arxiv GraphML instances not available (run `git lfs pull`)")
    from mo438.instances import load_instances

    # Hermetic: parse into a temporary cache rather than reusing data/processed.
    return load_instances(DIRECTED_GRAPHML, UNDIRECTED_GRAPHML,
                          cache_dir=tmp_path_factory.mktemp("cache"))


@pytest.fixture(scope="session")
def f1_summary():
    path = CHARACTERIZATION_DIR / "summary.json"
    if not path.is_file():
        pytest.skip("outputs/obj1_characterization/summary.json not available")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def real_edge_tables(real_instances):
    from mo438.edges import classify_directed, classify_undirected

    inst = real_instances
    gd = classify_directed(inst.gd_edges, inst.labels, inst.num_nodes)
    gu = classify_undirected(inst.gu_edges, inst.gd_edges, inst.labels, inst.num_nodes)
    return gd, gu


@pytest.fixture(scope="session")
def independent_undirected(real_instances):
    from mo438.verification import undirected_summary

    return undirected_summary(real_instances.gu_edges, real_instances.num_nodes)


@pytest.fixture(scope="session")
def independent_directed(real_instances):
    from mo438.verification import directed_summary

    return directed_summary(real_instances.gd_edges, real_instances.num_nodes)


@pytest.fixture(scope="session")
def real_adjacency(real_instances):
    from mo438.verification import undirected_adjacency

    return undirected_adjacency(real_instances.gu_edges, real_instances.num_nodes)
