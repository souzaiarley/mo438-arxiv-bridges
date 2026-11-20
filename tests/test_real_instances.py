"""ogbn-arxiv instances and their intra/inter classification (regression, real data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import CHARACTERIZATION_DIR

from mo438.edges import (
    node_composition,
    per_category_table,
    reciprocal_decomposition,
    validate_classification,
)
from mo438.reference import AUDIT_EDGE_CLASSES
from mo438.verification import orientation_by_year

pytestmark = pytest.mark.realdata


def test_instance_sizes(real_instances):
    inst = real_instances
    assert inst.num_nodes == 169_343
    assert inst.gd_edges.shape == (1_166_243, 2)
    assert inst.gu_edges.shape == (1_157_799, 2)
    assert inst.num_categories == 40
    c = inst.conversion
    assert (c.num_directed_self_loops, c.num_duplicate_directed_edges,
            c.num_reciprocal_pairs) == (0, 0, 8_444)


def test_category_names_match_f1(real_instances, f1_summary):
    assert real_instances.category_names == f1_summary["dataset"]["category_names"]


def test_class_counts_match_audit(real_edge_tables):
    gd, gu = real_edge_tables
    for table in (gd, gu):
        c = table.counts()
        ref = AUDIT_EDGE_CLASSES[table.graph]
        assert (c["intra"], c["inter"], c["num_edges"]) == (ref["intra"], ref["inter"],
                                                              ref["total"])


def test_reciprocal_pairs_by_class(real_edge_tables):
    dec = reciprocal_decomposition(*real_edge_tables)
    ref = AUDIT_EDGE_CLASSES["reciprocal_pairs"]
    for cls in ("intra", "inter"):
        assert dec[cls]["reciprocal_pairs_merged"] == ref[cls]
        assert dec[cls]["difference"] == ref[cls]           # no loops, no duplicates
        assert dec[cls]["duplicate_citations_merged"] == 0
        assert dec[cls]["self_loops_removed"] == 0
    assert dec["total_reciprocal_pairs"] == ref["total"]


def test_all_consistency_checks_pass_with_f1_matrix(real_instances, real_edge_tables):
    matrix = pd.read_csv(CHARACTERIZATION_DIR / "area_citation_matrix.csv", index_col=0).to_numpy()
    checks = validate_classification(*real_edge_tables, real_instances.labels,
                                     real_instances.num_nodes, area_matrix=matrix)
    assert all(c["passed"] for c in checks), [c for c in checks if not c["passed"]]


def test_per_category_table_is_consistent(real_instances, real_edge_tables):
    gd, gu = real_edge_tables
    table = per_category_table(gd, gu, real_instances.labels, real_instances.category_names)
    area = pd.read_csv(CHARACTERIZATION_DIR / "area_summary.csv")
    assert table["num_papers"].tolist() == area["num_papers"].tolist()
    assert table["G_d_intra"].tolist() == area["intra_area_citations"].tolist()
    assert (table["G_d_intra"] + table["G_d_inter_out"]).tolist() == \
        area["citations_made"].tolist()
    assert (table["G_d_intra"] + table["G_d_inter_in"]).tolist() == \
        area["citations_received"].tolist()
    assert table["G_u_intra"].sum() == AUDIT_EDGE_CLASSES["G_u"]["intra"]
    assert table["G_u_inter_incident"].sum() == 2 * AUDIT_EDGE_CLASSES["G_u"]["inter"]


def test_node_composition(real_instances, real_edge_tables):
    comp = node_composition(real_edge_tables[1], real_instances.num_nodes)
    # Audit probe (2026-11-06): isolated nodes after removing ALL inter (resp. intra) edges.
    assert comp["only_inter_edges"] == 20_799
    assert comp["only_intra_edges"] == 48_014
    assert comp["isolated"] == 0
    assert sum(comp[k] for k in ("isolated", "only_intra_edges", "only_inter_edges",
                                 "both_classes")) == real_instances.num_nodes


def test_largest_category_bounds_the_lcc_without_inter_edges(real_instances):
    sizes = np.bincount(real_instances.labels)
    assert real_instances.category_names[int(sizes.argmax())] == "cs.CV"
    assert sizes.max() == 27_321


def test_orientation_u_cites_v(real_instances):
    out = orientation_by_year(real_instances.gd_edges, real_instances.years)
    assert out["frac_citing_newer_than_cited"] == pytest.approx(0.8714513184645053, rel=1e-12)
    assert out["frac_same_year"] == pytest.approx(0.11062531565034045, rel=1e-12)
    assert out["frac_citing_older_than_cited"] == pytest.approx(0.017923365885154294, rel=1e-12)
