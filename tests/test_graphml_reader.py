"""Strict GraphML reader and instance assembly (mo438.instances)."""

from __future__ import annotations

import igraph as ig
import numpy as np
import pytest
from conftest import TOY_EXPECTED

from mo438.graph_build import undirected_edge_array
from mo438.graph_io import write_graphml
from mo438.instances import (
    GraphMLFormatError,
    build_instances,
    load_instances,
    read_graphml,
)


def _node_attrs(data):
    return {"node_id": np.arange(data.num_nodes), "category": data.labels,
            "category_name": [data.category_names[k] for k in data.labels],
            "year": data.years, "paper_id": data.paper_ids}


@pytest.fixture
def toy_files(toy, tmp_path):
    uedges, _ = undirected_edge_array(toy.edges, toy.num_nodes)
    d = write_graphml(tmp_path / "d.graphml", toy.num_nodes, toy.edges, True, _node_attrs(toy),
                      graph_attributes={"name": "toy_directed"})
    u = write_graphml(tmp_path / "u.graphml", toy.num_nodes, uedges, False, _node_attrs(toy),
                      graph_attributes={"name": "toy_undirected"})
    return d, u


def test_round_trip_matches_writer_and_igraph(toy, toy_files):
    d_path, u_path = toy_files
    d = read_graphml(d_path)
    assert d.directed and d.num_nodes == toy.num_nodes
    assert np.array_equal(d.edges, toy.edges)
    assert np.array_equal(d.node_attributes["category"], toy.labels)
    assert d.node_attributes["category_name"].tolist() == [toy.category_names[k]
                                                           for k in toy.labels]
    assert d.graph_attributes["name"] == "toy_directed"
    # independent reader: igraph
    g = ig.Graph.Read_GraphML(str(d_path))
    assert g.is_directed() and g.vcount() == toy.num_nodes
    assert [tuple(e) for e in g.get_edgelist()] == [tuple(e) for e in toy.edges.tolist()]
    assert [int(x) for x in g.vs["year"]] == d.node_attributes["year"].tolist()
    u = read_graphml(u_path)
    assert not u.directed and [tuple(e) for e in u.edges.tolist()] == TOY_EXPECTED["gu_edges"]


def test_build_instances_rebuilds_gu(toy, toy_files):
    inst = build_instances(read_graphml(toy_files[0]), read_graphml(toy_files[1]))
    assert inst.category_names == toy.category_names
    assert [tuple(e) for e in inst.gu_edges.tolist()] == TOY_EXPECTED["gu_edges"]
    assert inst.conversion.num_reciprocal_pairs == TOY_EXPECTED["reciprocal_pairs"]
    data = inst.to_citation_data()
    assert np.array_equal(data.edges, toy.edges)


def test_build_instances_rejects_a_different_undirected_graph(toy, tmp_path):
    uedges, _ = undirected_edge_array(toy.edges, toy.num_nodes)
    d = write_graphml(tmp_path / "d.graphml", toy.num_nodes, toy.edges, True, _node_attrs(toy))
    u = write_graphml(tmp_path / "u.graphml", toy.num_nodes, uedges[:-1], False, _node_attrs(toy))
    with pytest.raises(GraphMLFormatError, match="differs"):
        build_instances(read_graphml(d), read_graphml(u))


def test_reader_rejects_lfs_pointer(tmp_path):
    path = tmp_path / "pointer.graphml"
    path.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1\n")
    with pytest.raises(GraphMLFormatError, match="LFS"):
        read_graphml(path)


def test_reader_rejects_unexpected_schema(toy_files, tmp_path):
    text = toy_files[0].read_text()
    bad_order = tmp_path / "bad_order.graphml"
    bad_order.write_text(text.replace('<node id="n1">', '<node id="n9">', 1))
    with pytest.raises(GraphMLFormatError, match="node ids"):
        read_graphml(bad_order)
    edge_with_data = tmp_path / "edge_data.graphml"
    edge_with_data.write_text(text.replace('<edge source="n1" target="n0"/>',
                                           '<edge source="n1" target="n0"><data key="w">1</data>'
                                           '</edge>', 1))
    with pytest.raises(GraphMLFormatError, match="edge"):
        read_graphml(edge_with_data)


def test_cache_is_used_and_keyed_by_content(toy, toy_files, tmp_path):
    cache = tmp_path / "cache"
    first = load_instances(*toy_files, cache_dir=cache)
    assert len(list(cache.glob("instances_*.npz"))) == 1
    second = load_instances(*toy_files, cache_dir=cache)
    assert second.sources["cache_file"] == first.sources["cache_file"]
    assert np.array_equal(first.gd_edges, second.gd_edges)
    assert np.array_equal(first.gu_edges, second.gu_edges)
    assert first.conversion == second.conversion
    # different content -> different cache key
    toy_files[0].write_text(toy_files[0].read_text().replace("toy_directed", "toy_directed2"))
    load_instances(*toy_files, cache_dir=cache)
    assert len(list(cache.glob("instances_*.npz"))) == 2
