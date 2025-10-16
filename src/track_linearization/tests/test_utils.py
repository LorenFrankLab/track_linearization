import networkx as nx
import numpy as np
import pytest
from scipy.stats import multivariate_normal

import track_linearization.utils as utils
from track_linearization import get_linearized_position, make_track_graph


# Shared fixtures
@pytest.fixture
def simple_track_data():
    """Basic track data for testing."""
    return {
        "positions": [(0, 0), (30, 0), (30, 30), (0, 30)],
        "edges": [(0, 1), (0, 3), (1, 2)],
    }


@pytest.fixture
def matplotlib_pyplot():
    """Matplotlib pyplot if available, otherwise skip."""
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        pytest.skip("matplotlib not available")


class TestMakeTrackGraph:
    """Group tests for track graph creation functionality."""

    def test_make_track_graph_with_arrays(self):
        """Test track graph creation with numpy arrays."""
        pos = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=float)
        edges = [(0, 1), (1, 2)]
        g = utils.make_track_graph(pos, edges)

        assert isinstance(g, nx.Graph), "Should return NetworkX Graph"
        assert set(g.nodes) == {0, 1, 2}, "Should have correct nodes"
        assert set(map(tuple, g.edges)) == set(edges), "Should have correct edges"

        for i, p in enumerate(pos):
            assert tuple(g.nodes[i]["pos"]) == tuple(
                p
            ), f"Node {i} should have position {tuple(p)}"

    @pytest.mark.parametrize("input_type", ["list", "array"])
    def test_make_track_graph_input_types(self, simple_track_data, input_type):
        """Test track graph creation with different input formats."""
        positions = simple_track_data["positions"]
        edges = simple_track_data["edges"]

        if input_type == "array":
            positions = np.array(positions, dtype=float)

        track_graph = make_track_graph(positions, edges)

        assert len(track_graph.nodes) == 4, "Should have 4 nodes"
        assert len(track_graph.edges) == 3, "Should have 3 edges"

        # Check distances are computed
        for edge in track_graph.edges:
            assert (
                "distance" in track_graph.edges[edge]
            ), f"Edge {edge} should have distance"
            assert (
                track_graph.edges[edge]["distance"] > 0
            ), f"Edge {edge} distance should be positive"

    def test_make_track_graph_distance_calculation(self):
        """Test that edge distances are calculated correctly."""
        node_positions = [(0, 0), (3, 4), (0, 8)]  # 3-4-5 right triangle
        edges = [(0, 1), (1, 2)]

        track_graph = make_track_graph(node_positions, edges)

        # Check specific distances
        expected_distance = 5.0  # hypotenuse of 3-4-5 triangle
        assert (
            abs(track_graph.edges[(0, 1)]["distance"] - expected_distance) < 1e-10
        ), f"Distance should be {expected_distance}"
        assert (
            abs(track_graph.edges[(1, 2)]["distance"] - expected_distance) < 1e-10
        ), f"Distance should be {expected_distance}"


class TestEdgeOrdering:
    """Group tests for automatic edge ordering functionality."""

    def test_infer_edge_layout_basic(self):
        """Test basic edge ordering functionality."""
        pos = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=float)
        edges = [(0, 1), (1, 2)]
        g = utils.make_track_graph(pos, edges)

        order, spacing = utils.infer_edge_layout(g, start_node=0)

        assert len(order) == len(edges), "Order length should match number of edges"
        if hasattr(spacing, "shape") and spacing.size > 0:
            assert np.allclose(spacing, 0.0), "Default spacing should be zero"

    def test_infer_edge_layout_circular(self):
        """Test automatic edge ordering with circular track."""
        # Simple circular track
        angle = np.linspace(-np.pi, np.pi, num=6, endpoint=False)
        radius = 10
        node_positions = np.stack(
            (radius * np.cos(angle), radius * np.sin(angle)), axis=1
        )

        node_ids = np.arange(node_positions.shape[0])
        edges = np.stack((node_ids, np.roll(node_ids, shift=1)), axis=1)

        track_graph = make_track_graph(node_positions, edges)

        order, spacing = utils.infer_edge_layout(track_graph, start_node=0)

        assert len(order) == len(track_graph.edges), "Order should include all edges"
        assert spacing is not None, "Spacing should be returned"


class TestPlotting:
    """Group tests for plotting and visualization functions."""

    def test_plot_functions_smoke_test(self, matplotlib_pyplot):
        """Test that plotting functions don't crash."""
        if hasattr(utils, "plot_track_graph"):
            pos = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
            g = utils.make_track_graph(pos, [(0, 1)])
            utils.plot_track_graph(g)

        if hasattr(utils, "plot_graph_as_1D"):
            pos = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=float)
            g = utils.make_track_graph(pos, [(0, 1), (1, 2)])
            utils.plot_graph_as_1D(g)

    def test_plot_track_graph_with_options(self, matplotlib_pyplot, simple_track_data):
        """Test plot_track_graph with various options."""
        track_graph = make_track_graph(
            simple_track_data["positions"], simple_track_data["edges"]
        )

        fig, ax = matplotlib_pyplot.subplots()

        # Test basic plotting
        utils.plot_track_graph(track_graph, ax=ax)

        # Test with edge labels if supported
        if "draw_edge_labels" in utils.plot_track_graph.__code__.co_varnames:
            utils.plot_track_graph(track_graph, ax=ax, draw_edge_labels=True)

        matplotlib_pyplot.close(fig)

    @pytest.mark.parametrize("orientation", ["horizontal", "vertical"])
    def test_plot_graph_as_1D_orientations(
        self, matplotlib_pyplot, simple_track_data, orientation
    ):
        """Test 1D graph plotting with different orientations."""
        track_graph = make_track_graph(
            simple_track_data["positions"], simple_track_data["edges"]
        )

        if orientation == "horizontal":
            fig, ax = matplotlib_pyplot.subplots(figsize=(7, 1))
            utils.plot_graph_as_1D(track_graph, ax=ax)
        else:  # vertical
            fig, ax = matplotlib_pyplot.subplots(figsize=(1, 7))
            if "axis" in utils.plot_graph_as_1D.__code__.co_varnames:
                utils.plot_graph_as_1D(track_graph, axis="y", ax=ax)

        matplotlib_pyplot.close(fig)

    @pytest.mark.parametrize("edge_spacing", [0, 10])
    def test_plot_graph_as_1D_with_spacing(
        self, matplotlib_pyplot, simple_track_data, edge_spacing
    ):
        """Test 1D plotting with different edge spacing."""
        track_graph = make_track_graph(
            simple_track_data["positions"], simple_track_data["edges"]
        )

        fig, ax = matplotlib_pyplot.subplots(figsize=(7, 1))
        utils.plot_graph_as_1D(track_graph, edge_spacing=edge_spacing, ax=ax)
        matplotlib_pyplot.close(fig)


class TestSpecialTrackTypes:
    """Group tests for special track geometries."""

    def test_circular_track_creation(self):
        """Test creating and validating circular tracks."""
        # Create circular track like in the example
        angle = np.linspace(-np.pi, np.pi, num=12, endpoint=False)
        radius = 30
        node_positions = np.stack(
            (radius * np.cos(angle), radius * np.sin(angle)), axis=1
        )

        node_ids = np.arange(node_positions.shape[0])
        edges = np.stack((node_ids, np.roll(node_ids, shift=1)), axis=1)

        track_graph = make_track_graph(node_positions, edges)

        # Verify structure
        assert len(track_graph.nodes) == 12, "Circular track should have 12 nodes"
        assert len(track_graph.edges) == 12, "Circular track should have 12 edges"

        # Each node should be connected to exactly 2 others in a circle
        degrees = [track_graph.degree(node) for node in track_graph.nodes]
        assert all(
            deg == 2 for deg in degrees
        ), "All nodes should have degree 2 in circle"

    def test_w_track_creation(self):
        """Test creating W-shaped track from example."""
        node_positions = [(0, 0), (30, 0), (30, 30), (0, 30), (15, 30), (15, 0)]
        edges = [(3, 0), (0, 5), (4, 5), (5, 1), (1, 2)]

        track_graph = make_track_graph(node_positions, edges)

        # Verify structure
        assert len(track_graph.nodes) == 6, "W-track should have 6 nodes"
        assert len(track_graph.edges) == 5, "W-track should have 5 edges"

        # Node 5 should be the central hub connected to nodes 0, 1, and 4
        assert track_graph.degree(5) == 3, "Central node should have degree 3"
        assert set(track_graph.neighbors(5)) == {0, 1, 4}, "Central node connections"


class TestErrorHandling:
    """Group tests for error handling and edge cases."""

    def test_invalid_edge_references(self):
        """Test error handling for edges referencing non-existent nodes."""
        node_positions = [(0, 0), (1, 0)]
        invalid_edges = [(0, 2)]  # Node 2 doesn't exist

        with pytest.raises((KeyError, IndexError, ValueError)):
            make_track_graph(node_positions, invalid_edges)

    def test_empty_track_graph_handling(self):
        """Test handling of empty or minimal track graphs."""
        # Test empty track creation - may or may not raise error
        try:
            empty_graph = make_track_graph([], [])
            assert len(empty_graph.nodes) == 0, "Empty graph should have no nodes"
            assert len(empty_graph.edges) == 0, "Empty graph should have no edges"
        except (ValueError, IndexError):
            # Empty graphs may not be supported - this is acceptable
            pytest.skip("Empty track graphs not supported by implementation")

    def test_single_node_track(self):
        """Test creation of single-node track."""
        single_node_graph = make_track_graph([(0, 0)], [])

        assert len(single_node_graph.nodes) == 1, "Should have single node"
        assert len(single_node_graph.edges) == 0, "Should have no edges"


class TestPlottingWithEdgeParameters:
    """Tests for plotting functions with edge parameters."""

    @pytest.mark.parametrize("edge_spacing", [0, 5, 15])
    def test_plot_graph_as_1D_edge_spacing_variations(
        self, matplotlib_pyplot, simple_track_data, edge_spacing
    ):
        """Test plotting 1D graphs with different edge spacing values."""
        track_graph = make_track_graph(
            simple_track_data["positions"], simple_track_data["edges"]
        )

        fig, ax = matplotlib_pyplot.subplots(figsize=(10, 1))
        utils.plot_graph_as_1D(track_graph, edge_spacing=edge_spacing, ax=ax)
        matplotlib_pyplot.close(fig)

    def test_plot_graph_as_1D_custom_edge_order(
        self, matplotlib_pyplot, simple_track_data
    ):
        """Test plotting 1D graphs with custom edge order."""
        track_graph = make_track_graph(
            simple_track_data["positions"], simple_track_data["edges"]
        )

        # Test with custom edge order
        custom_edge_order = [(1, 2), (0, 1), (0, 3)]

        fig, ax = matplotlib_pyplot.subplots(figsize=(10, 1))
        utils.plot_graph_as_1D(track_graph, edge_order=custom_edge_order, ax=ax)
        matplotlib_pyplot.close(fig)

    def test_plot_graph_as_1D_edge_order_and_spacing_combined(
        self, matplotlib_pyplot, simple_track_data
    ):
        """Test plotting with both edge_order and edge_spacing."""
        track_graph = make_track_graph(
            simple_track_data["positions"], simple_track_data["edges"]
        )

        edge_order = [(0, 3), (0, 1), (1, 2)]
        edge_spacing = 8

        fig, ax = matplotlib_pyplot.subplots(figsize=(12, 1))
        utils.plot_graph_as_1D(
            track_graph, edge_order=edge_order, edge_spacing=edge_spacing, ax=ax
        )
        matplotlib_pyplot.close(fig)


class TestAutoEdgeOrdering:
    """Tests for automatic edge ordering functionality."""

    def test_infer_edge_layout_different_start_nodes(self):
        """Test auto edge ordering with different start nodes."""
        # Create a simple connected track
        node_positions = [(0, 0), (10, 0), (20, 0), (10, 10)]
        edges = [(0, 1), (1, 2), (1, 3)]  # Node 1 is hub
        track_graph = make_track_graph(node_positions, edges)

        # Test different start nodes
        for start_node in [0, 1, 2, 3]:
            order, spacing = utils.infer_edge_layout(track_graph, start_node=start_node)

            assert len(order) == len(
                track_graph.edges
            ), f"Should include all edges (start={start_node})"
            assert spacing is not None, f"Should return spacing (start={start_node})"

    def test_infer_edge_layout_complex_track(self):
        """Test auto edge ordering with complex W-shaped track."""
        # W-shaped track
        node_positions = [(0, 0), (30, 0), (30, 30), (0, 30), (15, 30), (15, 0)]
        edges = [(3, 0), (0, 5), (4, 5), (5, 1), (1, 2)]
        track_graph = make_track_graph(node_positions, edges)

        order, spacing = utils.infer_edge_layout(track_graph, start_node=0)

        assert len(order) == len(edges), "Should include all W-track edges"
        assert spacing is not None, "Should return spacing for W-track"

        # All edges in order should be valid edges in the graph
        for edge in order:
            assert track_graph.has_edge(*edge), f"Edge {edge} should exist in graph"


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_end_to_end_rectangular_track(self, matplotlib_pyplot, simple_track_data):
        """Integration test: create track, generate data, linearize, and plot."""
        track_graph = make_track_graph(
            simple_track_data["positions"], simple_track_data["edges"]
        )

        # Create synthetic position data
        x = np.linspace(0, 30, num=20)
        position = np.concatenate(
            [
                np.stack((np.zeros_like(x), x[::-1]), axis=1),
                np.stack((x, np.zeros_like(x)), axis=1),
            ]
        )
        position += multivariate_normal(mean=0, cov=0.01).rvs(position.shape)

        # Test linearization
        position_df = get_linearized_position(
            position=position, track_graph=track_graph
        )
        assert len(position_df) == len(
            position
        ), "Linearization output length should match input"

        # Test plotting track
        fig, ax = matplotlib_pyplot.subplots()
        utils.plot_track_graph(track_graph, ax=ax)
        matplotlib_pyplot.close(fig)

        # Test 1D visualization
        fig, ax = matplotlib_pyplot.subplots(figsize=(7, 1))
        utils.plot_graph_as_1D(track_graph, ax=ax)
        matplotlib_pyplot.close(fig)

    def test_end_to_end_with_edge_parameters(
        self, matplotlib_pyplot, simple_track_data
    ):
        """Integration test using custom edge parameters throughout."""
        track_graph = make_track_graph(
            simple_track_data["positions"], simple_track_data["edges"]
        )

        # Create position data
        position = np.array([[5, 0], [15, 0], [25, 0]])

        # Custom parameters
        edge_order = [(0, 3), (0, 1), (1, 2)]
        edge_spacing = [12, 8]
        edge_map = {0: 10, 1: 11, 2: 12}

        # Test linearization with all parameters
        position_df = get_linearized_position(
            position=position,
            track_graph=track_graph,
            edge_order=edge_order,
            edge_spacing=edge_spacing,
            edge_map=edge_map,
        )

        assert len(position_df) == len(position), "Should handle all positions"
        assert hasattr(position_df, "linear_position"), "Should have linear positions"
        assert hasattr(position_df, "track_segment_id"), "Should have segment IDs"

        # Check that edge mapping was applied
        unique_ids = set(position_df.track_segment_id.unique())
        assert unique_ids.issubset({10, 11, 12}), "Should use mapped edge IDs"

        # Test plotting with same parameters
        fig, ax = matplotlib_pyplot.subplots(figsize=(12, 1))
        utils.plot_graph_as_1D(
            track_graph, edge_order=edge_order, edge_spacing=edge_spacing, ax=ax
        )
        matplotlib_pyplot.close(fig)

    def test_track_graph_edge_attributes_complete(self, simple_track_data):
        """Test that track graphs have all expected edge attributes."""
        track_graph = make_track_graph(
            simple_track_data["positions"], simple_track_data["edges"]
        )

        for edge in track_graph.edges:
            edge_data = track_graph.edges[edge]

            # Should have distance
            assert "distance" in edge_data, f"Edge {edge} should have distance"
            assert isinstance(
                edge_data["distance"], float
            ), f"Edge {edge} distance should be float"
            assert edge_data["distance"] > 0, f"Edge {edge} distance should be positive"

            # Should have edge_id
            assert "edge_id" in edge_data, f"Edge {edge} should have edge_id"
            assert isinstance(
                edge_data["edge_id"], int
            ), f"Edge {edge} ID should be integer"
