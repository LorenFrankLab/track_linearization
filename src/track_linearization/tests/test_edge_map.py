import numpy as np
import pytest

import track_linearization.core as core
from track_linearization import make_track_graph

def _mk_line_graph():
    pos = np.array([[0.0,0.0],[1.0,0.0],[2.0,0.0]], dtype=float)
    edges = [(0,1),(1,2)]
    g = make_track_graph(pos, edges)
    # Explicit, non-index edge IDs to highlight id/index mismatch
    g.edges[(0,1)]["edge_id"] = 10
    g.edges[(1,2)]["edge_id"] = 20
    # Ensure edge distances exist
    g.edges[(0,1)]["distance"] = 1.0
    g.edges[(1,2)]["distance"] = 1.0
    return g

def test_edge_map_label_passthrough_no_change():
    g = _mk_line_graph()
    pts = np.array([[0.2,0.0],[1.7,0.0]])
    df_nomap = core.get_linearized_position(pts, g, use_HMM=False)
    df_map   = core.get_linearized_position(pts, g, edge_map={10:10, 20:20}, use_HMM=False)
    # Same geometry -> identical linear positions
    assert np.allclose(df_nomap["linear_position"], df_map["linear_position"])

def test_edge_map_merge_two_edges_to_one_label():
    g = _mk_line_graph()
    pts = np.array([[0.2,0.0],[1.7,0.0]])
    df = core.get_linearized_position(pts, g, edge_map={10:99, 20:99}, use_HMM=False)
    assert set(df["track_segment_id"].unique()) == {99}

def test_edge_map_invalid_source_ignored():
    g = _mk_line_graph()
    pts = np.array([[0.2,0.0]])
    # 999 is not a real edge_id in the graph; should be ignored
    df = core.get_linearized_position(pts, g, edge_map={999:42, 10:50}, use_HMM=False)
    # Should work and use the valid mapping (10->50) while ignoring invalid key (999)
    assert df.track_segment_id.iloc[0] == 50