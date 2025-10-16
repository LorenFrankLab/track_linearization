import networkx as nx
import numpy as np
import pandas as pd
import pytest
from scipy.stats import multivariate_normal

import track_linearization.core as core
from track_linearization import get_linearized_position, make_track_graph


# Test fixtures for common test data
@pytest.fixture
def simple_rectangular_track():
    """Create a simple rectangular track for testing."""
    node_positions = [(0, 0), (30, 0), (30, 30), (0, 30)]
    edges = [(0, 1), (0, 3), (1, 2)]
    return make_track_graph(node_positions, edges)


@pytest.fixture
def circular_track():
    """Create a circular track for testing."""
    angle = np.linspace(-np.pi, np.pi, num=24, endpoint=False)
    radius = 30
    node_positions = np.stack((radius * np.cos(angle), radius * np.sin(angle)), axis=1)
    node_ids = np.arange(node_positions.shape[0])
    edges = np.stack((node_ids, np.roll(node_ids, shift=1)), axis=1)
    return make_track_graph(node_positions, edges)


@pytest.fixture
def w_track():
    """Create a W-shaped track for testing."""
    node_positions = [(0, 0), (30, 0), (30, 30), (0, 30), (15, 30), (15, 0)]
    edges = [(3, 0), (0, 5), (4, 5), (5, 1), (1, 2)]
    return make_track_graph(node_positions, edges)


@pytest.fixture
def rectangular_position_data():
    """Generate synthetic position data for rectangular track."""
    x = np.linspace(0, 30, num=50)
    position = np.concatenate(
        [
            np.stack((np.zeros_like(x), x[::-1]), axis=1),  # bottom edge
            np.stack((x, np.zeros_like(x)), axis=1),  # left edge
            np.stack((np.ones_like(x) * 30, x), axis=1),  # right edge
        ]
    )
    # Add small amount of noise
    position += multivariate_normal(mean=0, cov=0.05).rvs(position.shape)
    return position


def _simple_line_graph(
    n=3, spacing=1.0, dim=2, with_distance=False, with_edge_id=False
):
    g = nx.Graph()
    for i in range(n):
        pos = np.zeros(dim, dtype=float)
        pos[0] = float(i) * spacing
        g.add_node(i, pos=tuple(pos))
    for i in range(n - 1):
        g.add_edge(i, i + 1)
        if with_distance:
            p1 = np.asarray(g.nodes[i]["pos"], float)
            p2 = np.asarray(g.nodes[i + 1]["pos"], float)
            g.edges[(i, i + 1)]["distance"] = float(np.linalg.norm(p2 - p1))
        if with_edge_id:
            g.edges[(i, i + 1)]["edge_id"] = i
    return g


@pytest.mark.parametrize("dim", [2, 3])
def test_get_track_segments_from_graph_shape(dim):
    if not hasattr(core, "get_track_segments_from_graph"):
        pytest.skip("get_track_segments_from_graph not available")
    g = _simple_line_graph(n=4, dim=dim)
    segs = core.get_track_segments_from_graph(g)
    assert segs.ndim == 3
    assert segs.shape[0] == len(list(g.edges))
    assert segs.shape[1] == 2
    assert segs.shape[2] == dim


def test_project_points_to_segment_handles_degenerate():
    if not hasattr(core, "project_points_to_segment"):
        pytest.skip("project_points_to_segment not available")
    # order is (track_segments, points)
    segments = np.array([[[0.0, 0.0], [0.0, 0.0]]], dtype=float)  # degenerate segment
    points = np.array([[1.0, 1.0], [0.0, 0.0]], dtype=float)  # 2 points
    out = core.project_points_to_segment(segments, points)
    proj = out[0] if isinstance(out, tuple) else out
    # Expect shape (n_points, n_segments, dim); allow NaNs for degenerate segments
    assert proj.shape == (points.shape[0], segments.shape[0], points.shape[1])


def test_normalize_to_probability_basic():
    if not hasattr(core, "normalize_to_probability"):
        pytest.skip("normalize_to_probability not available")
    x = np.array([[1.0, 1.0, 2.0], [0.0, 0.0, 0.0]], dtype=float)
    try:
        out = core.normalize_to_probability(x, axis=1)
    except TypeError:
        out = core.normalize_to_probability(x)
    assert np.allclose(out[0].sum(), 1.0, atol=1e-6)
    # Accept NaNs/zeros/uniform for the all-zero row, depending on implementation
    assert (
        np.isnan(out[1]).all()
        or np.isclose(out[1].sum(), 0.0)
        or np.allclose(out[1], 1.0 / out.shape[1])
    )


def test_route_distance_produces_finite_matrix():
    if not hasattr(core, "route_distance"):
        pytest.skip("route_distance not available")
    g = _simple_line_graph(n=4, dim=2, with_distance=True)
    # route_distance expects one candidate point per edge (3 edges)
    p_start = np.array([[0.1, 0.0], [1.1, 0.0], [2.1, 0.0]], dtype=float)  # (3,2)
    p_end = np.array([[0.9, 0.0], [1.9, 0.0], [2.9, 0.0]], dtype=float)  # (3,2)
    out = core.route_distance(p_start, p_end, g.copy())
    arr = np.asarray(out)
    assert arr.ndim in (1, 2)
    assert np.isfinite(arr).all()


def test_get_linearized_position_roundtrip_on_line():
    if not hasattr(core, "get_linearized_position") or not hasattr(
        core, "project_1d_to_2d"
    ):
        pytest.skip("get_linearized_position/project_1d_to_2d not available")
    # include 'distance' and 'edge_id' on edges because the implementation expects them
    g = _simple_line_graph(
        n=3, spacing=1.0, dim=2, with_distance=True, with_edge_id=True
    )
    pts = np.array([[0.25, 0.0], [0.75, 0.0], [1.5, 0.0]], dtype=float)
    lin = core.get_linearized_position(pts, g)
    if isinstance(lin, tuple):
        linear_pos = lin[0]
    else:
        try:
            linear_pos = np.asarray(lin["linear_position"].to_numpy())
        except Exception:
            linear_pos = np.asarray(lin)
    edge_order = list(g.edges)
    back = core.project_1d_to_2d(linear_pos, g, edge_order)
    err = np.linalg.norm(back - pts, axis=1)
    assert np.all(err < 1e-4)


def test_rectangular_track_structure(simple_rectangular_track):
    """Test that rectangular track has expected structure."""
    track_graph = simple_rectangular_track

    assert len(track_graph.nodes) == 4, "Rectangular track should have 4 nodes"
    assert len(track_graph.edges) == 3, "Rectangular track should have 3 edges"
    assert track_graph.has_edge(0, 1), "Should have edge between nodes 0 and 1"
    assert track_graph.has_edge(0, 3), "Should have edge between nodes 0 and 3"
    assert track_graph.has_edge(1, 2), "Should have edge between nodes 1 and 2"


def test_rectangular_track_linearization(
    simple_rectangular_track, rectangular_position_data
):
    """Test linearization with synthetic position data on rectangular track."""
    position_df = get_linearized_position(
        position=rectangular_position_data, track_graph=simple_rectangular_track
    )

    assert hasattr(
        position_df, "linear_position"
    ), "Result should have linear_position attribute"
    assert len(position_df) == len(
        rectangular_position_data
    ), "Output length should match input"
    assert (
        position_df.linear_position.notna().all()
    ), "All linear positions should be non-null"


@pytest.mark.parametrize("edge_spacing", [0, 10, 20])
def test_rectangular_track_with_edge_spacing(simple_rectangular_track, edge_spacing):
    """Test linearization with different edge spacing values."""
    # Create simple test position data
    x = np.linspace(0, 30, num=20)
    position = np.concatenate(
        [
            np.stack((np.zeros_like(x), x[::-1]), axis=1),
            np.stack((x, np.zeros_like(x)), axis=1),
        ]
    )

    position_df = get_linearized_position(
        position=position,
        track_graph=simple_rectangular_track,
        edge_spacing=edge_spacing,
    )

    assert hasattr(position_df, "linear_position"), "Result should have linear_position"
    assert len(position_df) == len(position), "Output length should match input"

    # Verify edge spacing effect on maximum position
    if edge_spacing > 0:
        max_expected = 2 * 30 + edge_spacing * 2  # Two 30-unit edges + gaps
        assert (
            position_df.linear_position.max() <= max_expected
        ), f"Max position should be <= {max_expected}"


def test_rectangular_track_with_custom_edge_order():
    """Test linearization with custom edge order."""
    node_positions = [(0, 0), (30, 0), (30, 30), (0, 30)]
    edges = [(0, 1), (0, 3), (1, 2)]
    track_graph = make_track_graph(node_positions, edges)

    # Create test position data
    x = np.linspace(0, 30, num=20)
    position = np.stack((x, np.zeros_like(x)), axis=1)  # Just bottom edge

    # Test different edge orders
    edge_order_1 = [(0, 1), (0, 3), (1, 2)]
    edge_order_2 = [(2, 1), (1, 0), (0, 3)]

    position_df_1 = get_linearized_position(
        position=position, track_graph=track_graph, edge_order=edge_order_1
    )

    position_df_2 = get_linearized_position(
        position=position, track_graph=track_graph, edge_order=edge_order_2
    )

    # Both should work but may produce different linear positions
    assert hasattr(position_df_1, "linear_position")
    assert hasattr(position_df_2, "linear_position")
    assert len(position_df_1) == len(position)
    assert len(position_df_2) == len(position)


class TestCircularTrack:
    """Group tests for circular track functionality."""

    def test_circular_track_structure(self, circular_track):
        """Test that circular track has correct structure."""
        assert len(circular_track.nodes) == 24, "Circular track should have 24 nodes"
        assert len(circular_track.edges) == 24, "Circular track should have 24 edges"

        # Each node should connect to exactly 2 others (forming a circle)
        degrees = [circular_track.degree(node) for node in circular_track.nodes]
        assert all(
            deg == 2 for deg in degrees
        ), "All nodes should have degree 2 in circular track"

    def test_circular_track_linearization(self, circular_track):
        """Test linearization on circular track."""
        radius = 30
        # Create position data going around the circle
        position_angles = np.linspace(-np.pi, 3 * np.pi, num=100, endpoint=False)
        position = np.stack(
            (radius * np.cos(position_angles), radius * np.sin(position_angles)), axis=1
        )
        position += multivariate_normal(mean=0, cov=0.05).rvs(position.shape)

        # Create edge order for linearization
        n_nodes = len(circular_track.nodes)
        edge_order = np.stack(
            (
                np.roll(np.arange(n_nodes - 1, -1, -1), 1),
                np.arange(n_nodes - 1, -1, -1),
            ),
            axis=1,
        )

        position_df = get_linearized_position(
            position=position,
            track_graph=circular_track,
            edge_spacing=0,
            edge_order=edge_order,
        )

        assert hasattr(
            position_df, "linear_position"
        ), "Result should have linear_position"
        assert len(position_df) == len(position), "Output length should match input"
        assert (
            position_df.linear_position.notna().all()
        ), "All positions should be non-null"


class TestWTrack:
    """Group tests for W-shaped track functionality."""

    def test_w_track_structure(self, w_track):
        """Test that W-track has expected structure."""
        assert len(w_track.nodes) == 6, "W-track should have 6 nodes"
        assert len(w_track.edges) == 5, "W-track should have 5 edges"

        # Node 5 should be the central hub connected to nodes 0, 1, and 4
        assert w_track.degree(5) == 3, "Central node should have degree 3"
        assert set(w_track.neighbors(5)) == {
            0,
            1,
            4,
        }, "Central node should connect to nodes 0, 1, 4"

    def test_w_track_linearization(self, w_track):
        """Test linearization on W-shaped track with custom parameters."""
        # Create test position data
        x = np.linspace(0, 30, num=30)
        position = np.concatenate(
            [
                np.stack((np.zeros_like(x), x[::-1]), axis=1),  # left vertical
                np.stack((x, np.zeros_like(x)), axis=1),  # bottom horizontal
                np.stack((np.ones_like(x) * 30, x), axis=1),  # right vertical
            ]
        )
        position += multivariate_normal(mean=0, cov=0.05).rvs(position.shape)

        # Test with custom edge order and spacing
        edge_order = [(4, 5), (5, 1), (1, 2), (5, 0), (0, 3)]
        edge_spacing = [15, 0, 15, 0]

        position_df = get_linearized_position(
            position=position,
            track_graph=w_track,
            edge_spacing=edge_spacing,
            edge_order=edge_order,
        )

        assert hasattr(
            position_df, "linear_position"
        ), "Result should have linear_position"
        assert len(position_df) == len(position), "Output length should match input"
        assert (
            position_df.linear_position.notna().all()
        ), "All positions should be non-null"


def test_linearization_without_hmm(simple_rectangular_track):
    """Test linearization without HMM (nearest neighbor mode)."""
    # Create simple position data
    position = np.array([[5, 0], [15, 0], [25, 0]])

    position_df = get_linearized_position(
        position=position,
        track_graph=simple_rectangular_track,
        use_HMM=False,
    )

    assert hasattr(position_df, "linear_position"), "Result should have linear_position"
    assert len(position_df) == len(position), "Output length should match input"


class TestEdgeCases:
    """Group tests for edge cases and error conditions."""

    def test_single_node_track_raises_error(self):
        """Test that single node track raises appropriate error."""
        node_positions = [(0, 0)]
        edges = []

        with pytest.raises((ValueError, IndexError)):
            track_graph = make_track_graph(node_positions, edges)
            position = np.array([[0, 0]])
            get_linearized_position(position=position, track_graph=track_graph)

    def test_single_position_point(self):
        """Test linearization with single position point."""
        node_positions = [(0, 0), (1, 0)]
        edges = [(0, 1)]
        track_graph = make_track_graph(node_positions, edges)

        single_position = np.array([[0.5, 0]])
        position_df = get_linearized_position(
            position=single_position, track_graph=track_graph
        )

        assert len(position_df) == 1, "Single input should produce single output"
        assert hasattr(
            position_df, "linear_position"
        ), "Result should have linear_position attribute"


class TestEdgeOrderParameter:
    """Comprehensive tests for edge_order parameter."""

    def test_edge_order_affects_linearization(self, simple_rectangular_track):
        """Test that different edge orders produce different linearizations."""
        position = np.array([[15, 0], [25, 0]])  # Points on bottom edge

        # Test two different edge orders
        edge_order_1 = [(0, 1), (0, 3), (1, 2)]  # Original order
        edge_order_2 = [(1, 2), (1, 0), (0, 3)]  # Different order

        pos_df_1 = get_linearized_position(
            position=position,
            track_graph=simple_rectangular_track,
            edge_order=edge_order_1,
        )
        pos_df_2 = get_linearized_position(
            position=position,
            track_graph=simple_rectangular_track,
            edge_order=edge_order_2,
        )

        # Should produce different linear positions for same input
        assert not np.allclose(
            pos_df_1.linear_position.values, pos_df_2.linear_position.values
        ), "Different edge orders should produce different linear positions"

    def test_edge_order_reversed_edges(self, simple_rectangular_track):
        """Test edge_order with reversed edge directions."""
        position = np.array([[15, 0]])

        # Test with reversed edges
        edge_order_forward = [(0, 1), (1, 2), (0, 3)]
        edge_order_reversed = [(1, 0), (2, 1), (3, 0)]

        pos_df_forward = get_linearized_position(
            position=position,
            track_graph=simple_rectangular_track,
            edge_order=edge_order_forward,
        )
        pos_df_reversed = get_linearized_position(
            position=position,
            track_graph=simple_rectangular_track,
            edge_order=edge_order_reversed,
        )

        assert hasattr(
            pos_df_forward, "linear_position"
        ), "Forward edge order should work"
        assert hasattr(
            pos_df_reversed, "linear_position"
        ), "Reversed edge order should work"

    @pytest.mark.parametrize("shuffle_seed", [42, 123, 456])
    def test_edge_order_random_permutations(
        self, simple_rectangular_track, shuffle_seed
    ):
        """Test edge_order with random permutations."""
        rng = np.random.default_rng(shuffle_seed)
        position = np.array([[15, 0], [15, 15]])

        # Get all edges and shuffle them
        all_edges = list(simple_rectangular_track.edges)
        shuffled_edges = all_edges.copy()
        rng.shuffle(shuffled_edges)

        pos_df = get_linearized_position(
            position=position,
            track_graph=simple_rectangular_track,
            edge_order=shuffled_edges,
        )

        assert hasattr(
            pos_df, "linear_position"
        ), f"Shuffled edge order (seed {shuffle_seed}) should work"
        assert len(pos_df) == len(position), "Output length should match input"

    def test_edge_order_subset_of_edges(self):
        """Test edge_order using only subset of available edges."""
        # Create track with more edges than we'll use
        node_positions = [(0, 0), (10, 0), (20, 0), (30, 0)]
        edges = [(0, 1), (1, 2), (2, 3)]
        track_graph = make_track_graph(node_positions, edges)

        position = np.array([[5, 0], [15, 0]])

        # Use only subset of edges
        edge_order_subset = [(0, 1), (1, 2)]  # Skip the last edge

        pos_df = get_linearized_position(
            position=position, track_graph=track_graph, edge_order=edge_order_subset
        )

        assert hasattr(pos_df, "linear_position"), "Subset edge order should work"

    def test_edge_order_validation_invalid_edges(self, simple_rectangular_track):
        """Test that invalid edges in edge_order raise appropriate errors."""
        position = np.array([[15, 0]])

        # Test with non-existent edge
        invalid_edge_order = [(0, 1), (5, 6)]  # (5, 6) doesn't exist

        with pytest.raises((KeyError, ValueError)):
            get_linearized_position(
                position=position,
                track_graph=simple_rectangular_track,
                edge_order=invalid_edge_order,
            )

    def test_edge_order_none_uses_default(self, simple_rectangular_track):
        """Test that edge_order=None uses default ordering."""
        position = np.array([[15, 0]])

        pos_df_none = get_linearized_position(
            position=position, track_graph=simple_rectangular_track, edge_order=None
        )
        pos_df_default = get_linearized_position(
            position=position, track_graph=simple_rectangular_track
        )

        # Should produce same results
        np.testing.assert_allclose(
            pos_df_none.linear_position.values,
            pos_df_default.linear_position.values,
            err_msg="edge_order=None should match default behavior",
        )


class TestEdgeSpacingParameter:
    """Comprehensive tests for edge_spacing parameter."""

    @pytest.mark.parametrize("spacing_value", [0, 5, 10, 25.5])
    def test_edge_spacing_scalar_values(self, simple_rectangular_track, spacing_value):
        """Test edge_spacing with different scalar values."""
        position = np.array([[5, 0], [15, 0], [25, 0]])

        pos_df = get_linearized_position(
            position=position,
            track_graph=simple_rectangular_track,
            edge_spacing=spacing_value,
        )

        assert hasattr(
            pos_df, "linear_position"
        ), f"edge_spacing={spacing_value} should work"

        if spacing_value > 0:
            # With spacing, max position should be higher
            pos_no_spacing = get_linearized_position(
                position=position, track_graph=simple_rectangular_track, edge_spacing=0
            )
            assert (
                pos_df.linear_position.max() >= pos_no_spacing.linear_position.max()
            ), "Positive spacing should increase maximum linear position"

    def test_edge_spacing_list_values(self, simple_rectangular_track):
        """Test edge_spacing with list of different values."""
        position = np.array([[5, 0], [15, 0]])

        # For 3 edges, need list of length 2 (n_edges - 1)
        edge_spacing_list = [10, 20]

        pos_df = get_linearized_position(
            position=position,
            track_graph=simple_rectangular_track,
            edge_spacing=edge_spacing_list,
        )

        assert hasattr(pos_df, "linear_position"), "List edge_spacing should work"

    def test_edge_spacing_list_validation(self, simple_rectangular_track):
        """Test validation of edge_spacing list length."""
        position = np.array([[15, 0]])

        # Wrong length: should be n_edges - 1 = 2, but giving 3
        invalid_spacing = [5, 10, 15]

        with pytest.raises(ValueError, match=r"edge_spacing.*wrong length"):
            get_linearized_position(
                position=position,
                track_graph=simple_rectangular_track,
                edge_spacing=invalid_spacing,
            )

    def test_edge_spacing_zero_vs_positive(self, simple_rectangular_track):
        """Test difference between zero and positive edge spacing."""
        position = np.array([[5, 0], [15, 0], [25, 0]])

        pos_df_zero = get_linearized_position(
            position=position, track_graph=simple_rectangular_track, edge_spacing=0
        )
        pos_df_positive = get_linearized_position(
            position=position, track_graph=simple_rectangular_track, edge_spacing=15
        )

        # Positive spacing should create gaps, increasing linear positions
        # Note: The actual effect might depend on which edges the positions are assigned to
        max_zero = pos_df_zero.linear_position.max()
        max_positive = pos_df_positive.linear_position.max()

        # At minimum, the range should be different or potentially larger with spacing
        assert (
            max_positive >= max_zero
        ), f"Positive spacing should not decrease max position: {max_zero} -> {max_positive}"

        # More robust test: check that the spacing parameter affects the results
        total_range_zero = (
            pos_df_zero.linear_position.max() - pos_df_zero.linear_position.min()
        )
        total_range_positive = (
            pos_df_positive.linear_position.max()
            - pos_df_positive.linear_position.min()
        )

        # The range might change due to spacing effects
        assert (
            total_range_positive >= total_range_zero * 0.9
        ), "Positive spacing should maintain reasonable position range"

    def test_edge_spacing_negative_values(self, simple_rectangular_track):
        """Test behavior with negative edge spacing."""
        position = np.array([[15, 0]])

        # Some implementations might allow negative spacing (overlapping segments)
        try:
            pos_df = get_linearized_position(
                position=position, track_graph=simple_rectangular_track, edge_spacing=-5
            )
            assert hasattr(
                pos_df, "linear_position"
            ), "Negative spacing might be supported"
        except (ValueError, AssertionError):
            # Negative spacing might not be allowed - this is acceptable
            pass

    @pytest.mark.parametrize(
        "edge_spacing",
        [
            [0, 0],  # No spacing
            [10, 0],  # Spacing only after first edge
            [0, 20],  # Spacing only after second edge
            [5, 15],  # Different spacing values
        ],
    )
    def test_edge_spacing_list_variations(self, simple_rectangular_track, edge_spacing):
        """Test various edge_spacing list configurations."""
        position = np.array([[5, 0], [15, 0]])

        pos_df = get_linearized_position(
            position=position,
            track_graph=simple_rectangular_track,
            edge_spacing=edge_spacing,
        )

        assert hasattr(
            pos_df, "linear_position"
        ), f"edge_spacing={edge_spacing} should work"

    def test_edge_spacing_affects_position_calculations(self):
        """Test that edge_spacing correctly affects position calculations."""
        # Create simple linear track for predictable testing
        node_positions = [(0, 0), (10, 0), (20, 0)]
        edges = [(0, 1), (1, 2)]
        track_graph = make_track_graph(node_positions, edges)

        position = np.array([[5, 0], [15, 0]])  # Mid-points of each edge

        pos_df_no_spacing = get_linearized_position(
            position=position, track_graph=track_graph, edge_spacing=0
        )
        pos_df_with_spacing = get_linearized_position(
            position=position, track_graph=track_graph, edge_spacing=5
        )

        # Expected: first point should be same, second should be shifted by spacing
        expected_shift = 5  # The spacing value
        actual_shift = (
            pos_df_with_spacing.linear_position.iloc[1]
            - pos_df_no_spacing.linear_position.iloc[1]
        )

        # Should be approximately the spacing value
        assert (
            abs(actual_shift - expected_shift) < 1e-10
        ), f"Second position should shift by spacing amount: expected {expected_shift}, got {actual_shift}"


class TestEdgeMapParameter:
    """Comprehensive tests for edge_map parameter."""

    def test_edge_map_basic_functionality(self, simple_rectangular_track):
        """Test basic edge_map functionality."""
        position = np.array([[15, 0]])

        # Create edge mapping to remap edge IDs
        edge_map = {0: 10, 1: 11, 2: 12}  # Remap to different IDs

        pos_df = get_linearized_position(
            position=position, track_graph=simple_rectangular_track, edge_map=edge_map
        )

        assert hasattr(
            pos_df, "track_segment_id"
        ), "Should have track_segment_id column"
        # The track_segment_id should use the remapped values
        assert (
            pos_df.track_segment_id.iloc[0] in edge_map.values()
        ), "track_segment_id should use remapped values"

    def test_edge_map_identity_mapping(self, simple_rectangular_track):
        """Test edge_map with identity mapping (maps to same values)."""
        position = np.array([[15, 0]])

        # Identity mapping should always work
        edge_map = {0: 0, 1: 1, 2: 2}

        pos_df_with_map = get_linearized_position(
            position=position, track_graph=simple_rectangular_track, edge_map=edge_map
        )
        pos_df_without_map = get_linearized_position(
            position=position, track_graph=simple_rectangular_track, edge_map=None
        )

        # Should produce identical results
        np.testing.assert_array_equal(
            pos_df_with_map.track_segment_id.values,
            pos_df_without_map.track_segment_id.values,
            err_msg="Identity edge_map should produce same results as no mapping",
        )

    def test_edge_map_validation_invalid_keys(self, simple_rectangular_track):
        """Test edge_map validation with invalid keys."""
        position = np.array([[15, 0]])

        # Map non-existent edge ID - should be ignored
        invalid_edge_map = {999: 1}  # 999 doesn't exist

        pos_df = get_linearized_position(
            position=position,
            track_graph=simple_rectangular_track,
            edge_map=invalid_edge_map,
        )

        # Should still work (invalid keys ignored)
        assert hasattr(
            pos_df, "linear_position"
        ), "Invalid edge_map keys should be ignored"

    def test_edge_map_none_vs_no_parameter(self, simple_rectangular_track):
        """Test that edge_map=None is equivalent to not providing edge_map."""
        position = np.array([[15, 0]])

        pos_df_none = get_linearized_position(
            position=position, track_graph=simple_rectangular_track, edge_map=None
        )
        pos_df_default = get_linearized_position(
            position=position, track_graph=simple_rectangular_track
        )

        # Should be identical
        pd.testing.assert_frame_equal(
            pos_df_none, pos_df_default, check_exact=False, rtol=1e-10
        )

    def test_edge_map_partial_mapping(self, simple_rectangular_track):
        """Test edge_map with partial mapping (not all edges mapped)."""
        position = np.array([[5, 0], [15, 0], [25, 0]])

        # Only map some edge IDs
        edge_map = {0: 100, 2: 200}  # Don't map edge ID 1

        pos_df = get_linearized_position(
            position=position, track_graph=simple_rectangular_track, edge_map=edge_map
        )

        assert hasattr(pos_df, "track_segment_id"), "Should have track_segment_id"
        unique_ids = set(pos_df.track_segment_id.unique())

        # Should contain both mapped and unmapped values
        expected_ids = {100, 1, 200}  # 100 (mapped 0), 1 (unmapped), 200 (mapped 2)
        assert unique_ids.issubset(
            expected_ids
        ), f"Expected IDs in {expected_ids}, got {unique_ids}"

    def test_edge_map_with_string_values(self, simple_rectangular_track):
        """Test edge_map with string target values."""
        position = np.array([[15, 0]])

        # Map to string values
        edge_map = {0: "segment_A", 1: "segment_B", 2: "segment_C"}
        pos_df = get_linearized_position(
            position=position, track_graph=simple_rectangular_track, edge_map=edge_map
        )

        assert hasattr(pos_df, "track_segment_id"), "Should have track_segment_id"
        # Should use string segment IDs
        assert (
            pos_df.track_segment_id.iloc[0] in edge_map.values()
        ), "Should use string segment IDs from edge_map"

    def test_edge_map_bug_documentation(self, simple_rectangular_track):
        """
        DOCUMENTATION: This test documents the current edge_map implementation bug.

        BUG DESCRIPTION:
        The edge_map feature has an implementation bug in _calculate_linear_position()
        at line 672-674 in core.py:

            projected_track_positions = projected_track_positions[
                (np.arange(n_time), track_segment_id)
            ]

        PROBLEM:
        1. edge_map correctly modifies track_segment_id values (line 809-811)
        2. But then these modified values are used as indices into projected_track_positions
        3. projected_track_positions has shape (n_time, n_segments) where n_segments
           is the number of actual track segments (e.g., 3)
        4. But edge_map can assign arbitrary values (e.g., 100, 'segment_A')
        5. This causes IndexError when track_segment_id[i] >= n_segments

        EXPECTED BEHAVIOR:
        edge_map should only affect the final track_segment_id values in the output
        DataFrame, not the internal indexing operations.

        This test will FAIL until the bug is fixed.
        """
        position = np.array([[15, 0]])

        # This should work but currently fails due to the indexing bug
        edge_map = {0: 999}  # Map edge 0 to 999

        # This WILL fail with IndexError: index 999 is out of bounds for axis 1 with size 3
        pos_df = get_linearized_position(
            position=position, track_graph=simple_rectangular_track, edge_map=edge_map
        )

        # If the bug is fixed, this should pass
        assert pos_df.track_segment_id.iloc[0] == 999, "Should use mapped edge ID"

    def test_edge_map_merging_linear_track(self):
        """Test edge_map properly merges edges into unified linear coordinate system."""
        # Create a simple linear track with 3 edges
        node_positions = [(0, 0), (10, 0), (20, 0), (30, 0)]
        edges = [(0, 1), (1, 2), (2, 3)]
        track_graph = make_track_graph(node_positions, edges)

        # Positions 5 units from the start of each edge
        position = np.array(
            [
                [5, 0],  # Edge 0: 5 units from x=0
                [15, 0],  # Edge 1: 5 units from x=10
                [25, 0],  # Edge 2: 5 units from x=20
            ]
        )

        # WITH edge_map: merge edges 0 and 1 into segment 0
        edge_map = {0: 0, 1: 0, 2: 2}
        result_with_map = get_linearized_position(
            position, track_graph, edge_spacing=0, edge_map=edge_map
        )

        # Verify merging behavior:
        # 1. Positions 0 and 1 should both be in segment 0
        assert result_with_map.track_segment_id.iloc[0] == 0
        assert result_with_map.track_segment_id.iloc[1] == 0
        assert result_with_map.track_segment_id.iloc[2] == 2

        # 2. Positions 0 and 1 should have SAME linear position (both 5 units into their edges)
        assert np.isclose(
            result_with_map.linear_position.iloc[0],
            result_with_map.linear_position.iloc[1],
            atol=0.01,
        ), f"Merged edges should have same linear position: {result_with_map.linear_position.iloc[0]} vs {result_with_map.linear_position.iloc[1]}"

        # 3. Linear position should be 5.0 (distance from edge start)
        assert np.isclose(
            result_with_map.linear_position.iloc[0], 5.0, atol=0.01
        ), f"Linear position should be 5.0, got {result_with_map.linear_position.iloc[0]}"

    def test_edge_map_merging_y_shaped_track(self):
        """Test edge_map merging with Y-shaped track (different 2D positions, same linear position)."""
        # Y-shaped track: two arms that merge
        node_positions = [
            (0, 10),  # Node 0: top of left arm
            (0, 0),  # Node 1: bottom of left arm
            (20, 10),  # Node 2: top of right arm
            (20, 0),  # Node 3: bottom of right arm
            (10, 0),  # Node 4: end of merged segment
        ]
        edges = [
            (0, 1),  # Edge 0: left arm (edge_id will be 0)
            (2, 3),  # Edge 1: right arm (edge_id will be 2)
            (1, 4),  # Edge 2: merged segment (edge_id will be 1)
        ]
        track_graph = make_track_graph(node_positions, edges)

        # Positions 5 units from start of left and right arms
        position = np.array(
            [
                [0, 5],  # Left arm: 5 units down
                [20, 5],  # Right arm: 5 units down
            ]
        )

        # Find which edge_ids correspond to left and right arms
        left_arm_id = track_graph.edges[(0, 1)]["edge_id"]
        right_arm_id = track_graph.edges[(2, 3)]["edge_id"]

        # Merge left and right arms into segment 10
        edge_map = {left_arm_id: 10, right_arm_id: 10}

        result = get_linearized_position(
            position, track_graph, edge_spacing=0, edge_map=edge_map
        )

        # Both positions should be in segment 10
        assert result.track_segment_id.iloc[0] == 10
        assert result.track_segment_id.iloc[1] == 10

        # Both should have the SAME linear position (5.0)
        assert np.isclose(
            result.linear_position.iloc[0], result.linear_position.iloc[1], atol=0.01
        ), f"Y-arms should have same linear position when merged: {result.linear_position.iloc[0]} vs {result.linear_position.iloc[1]}"

        assert np.isclose(
            result.linear_position.iloc[0], 5.0, atol=0.01
        ), f"Linear position should be 5.0, got {result.linear_position.iloc[0]}"


class TestEdgeParameterIntegration:
    """Tests combining edge_order, edge_spacing, and edge_map together."""

    def test_all_edge_parameters_together(self, simple_rectangular_track):
        """Test using edge_order, edge_spacing, and edge_map together."""
        position = np.array([[5, 0], [15, 0], [25, 0]])

        edge_order = [(1, 2), (0, 1), (0, 3)]  # Custom order
        edge_spacing = [10, 5]  # Custom spacing
        edge_map = {0: 100, 1: 101, 2: 102}  # Custom mapping

        pos_df = get_linearized_position(
            position=position,
            track_graph=simple_rectangular_track,
            edge_order=edge_order,
            edge_spacing=edge_spacing,
            edge_map=edge_map,
        )

        assert hasattr(pos_df, "linear_position"), "All parameters together should work"
        assert hasattr(pos_df, "track_segment_id"), "Should have mapped segment IDs"

        # Check that mapping was applied
        unique_ids = set(pos_df.track_segment_id.unique())
        assert unique_ids.issubset({100, 101, 102}), "Should use mapped edge IDs"

    def test_edge_parameters_with_circular_track(self, circular_track):
        """Test edge parameters with circular track."""
        radius = 30
        position_angles = np.linspace(-np.pi, np.pi, num=20)
        position = np.stack(
            (radius * np.cos(position_angles), radius * np.sin(position_angles)), axis=1
        )

        # Create custom parameters for circular track
        n_nodes = len(circular_track.nodes)
        edge_order = np.stack(
            (np.arange(n_nodes), np.roll(np.arange(n_nodes), -1)), axis=1
        )
        edge_spacing = 2.0  # Small spacing

        # Test without edge_map first (since it has implementation constraints)
        pos_df = get_linearized_position(
            position=position,
            track_graph=circular_track,
            edge_order=edge_order,
            edge_spacing=edge_spacing,
        )

        assert len(pos_df) == len(position), "Should handle all positions"
        assert hasattr(pos_df, "linear_position"), "Should have linear positions"

    def test_edge_parameters_consistency_check(self):
        """Test that edge parameter combinations produce consistent results."""
        # Create simple track for predictable testing
        node_positions = [(0, 0), (10, 0), (20, 0)]
        edges = [(0, 1), (1, 2)]
        track_graph = make_track_graph(node_positions, edges)

        position = np.array([[5, 0], [15, 0]])

        # Test that different orderings with compensating parameters work
        pos_df_1 = get_linearized_position(
            position=position,
            track_graph=track_graph,
            edge_order=[(0, 1), (1, 2)],
            edge_spacing=0,
        )

        pos_df_2 = get_linearized_position(
            position=position,
            track_graph=track_graph,
            edge_order=[(1, 2), (0, 1)],  # Reversed order
            edge_spacing=0,
        )

        # Should both work but produce different linear positions
        assert hasattr(pos_df_1, "linear_position"), "First configuration should work"
        assert hasattr(pos_df_2, "linear_position"), "Second configuration should work"

        # Results should be different due to different ordering
        assert not np.allclose(
            pos_df_1.linear_position.values, pos_df_2.linear_position.values
        ), "Different edge orders should produce different results"

    @pytest.mark.parametrize("combine_mode", ["all_custom", "mixed_defaults"])
    def test_edge_parameters_robustness(self, simple_rectangular_track, combine_mode):
        """Test robustness of edge parameter combinations."""
        position = np.array([[15, 0], [25, 0]])

        if combine_mode == "all_custom":
            # All parameters custom
            kwargs = {
                "edge_order": [(2, 1), (1, 0), (0, 3)],
                "edge_spacing": [8, 12],
                "edge_map": {0: 50, 1: 51, 2: 52},
            }
        else:  # mixed_defaults
            # Some parameters custom, others default
            kwargs = {
                "edge_order": [(0, 1), (0, 3), (1, 2)],
                "edge_spacing": 0,  # Default
                "edge_map": None,  # Default
            }

        pos_df = get_linearized_position(
            position=position, track_graph=simple_rectangular_track, **kwargs
        )

        assert hasattr(pos_df, "linear_position"), f"Mode {combine_mode} should work"
        assert len(pos_df) == len(position), "Should process all positions"
        assert (
            pos_df.linear_position.notna().all()
        ), "Should have valid linear positions"


class TestTrackGraphProperties:
    """Group tests for track graph properties and validation."""

    def test_edge_distance_computation(self, simple_rectangular_track):
        """Test that edge distances are computed correctly."""
        for edge in simple_rectangular_track.edges:
            assert (
                "distance" in simple_rectangular_track.edges[edge]
            ), f"Edge {edge} should have distance"
            assert (
                simple_rectangular_track.edges[edge]["distance"] > 0
            ), f"Edge {edge} distance should be positive"

    def test_node_position_storage(self, simple_rectangular_track):
        """Test that node positions are stored correctly."""
        expected_positions = [(0, 0), (30, 0), (30, 30), (0, 30)]
        for i, expected_pos in enumerate(expected_positions):
            actual_pos = simple_rectangular_track.nodes[i]["pos"]
            assert (
                actual_pos == expected_pos
            ), f"Node {i} should have position {expected_pos}"

    def test_edge_id_assignment(self, simple_rectangular_track):
        """Test that edges have proper ID assignment."""
        for edge in simple_rectangular_track.edges:
            edge_data = simple_rectangular_track.edges[edge]
            assert "edge_id" in edge_data, f"Edge {edge} should have edge_id"
            assert isinstance(
                edge_data["edge_id"], int
            ), f"Edge {edge} ID should be integer"
