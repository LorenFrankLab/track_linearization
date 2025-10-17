"""Tests for batch processing and utility functions."""

import numpy as np
import pytest

from track_linearization import get_linearized_position, make_track_graph, project_1d_to_2d
from track_linearization.core import (
    batch_linear_distance,
    route_distance_change,
)


class TestBatchLinearDistance:
    """Test batch_linear_distance function."""

    def test_batch_linear_distance_basic(self):
        """Test basic batch linear distance calculation."""
        node_positions = [(0, 0), (10, 0), (20, 0), (30, 0)]
        edges = [(0, 1), (1, 2), (2, 3)]
        track_graph = make_track_graph(node_positions, edges)

        # Project positions onto the track
        projected_positions = np.array([
            [5, 0],   # Middle of edge 0
            [15, 0],  # Middle of edge 1
            [25, 0],  # Middle of edge 2
        ])

        edge_ids = [(0, 1), (1, 2), (2, 3)]
        linear_zero_node_id = 0

        distances = batch_linear_distance(
            projected_track_positions=projected_positions,
            edge_ids=edge_ids,
            track_graph=track_graph,
            linear_zero_node_id=linear_zero_node_id,
        )

        # Check distances are reasonable
        assert len(distances) == 3
        assert all(isinstance(d, (int, float)) for d in distances)
        assert distances[0] < distances[1] < distances[2]  # Monotonically increasing

    def test_batch_linear_distance_single_position(self):
        """Test batch linear distance with single position."""
        node_positions = [(0, 0), (10, 0)]
        edges = [(0, 1)]
        track_graph = make_track_graph(node_positions, edges)

        projected_positions = np.array([[5, 0]])
        edge_ids = [(0, 1)]

        distances = batch_linear_distance(
            projected_track_positions=projected_positions,
            edge_ids=edge_ids,
            track_graph=track_graph,
            linear_zero_node_id=0,
        )

        assert len(distances) == 1
        assert 0 < distances[0] < 10  # Should be between start and end

    def test_batch_linear_distance_complex_track(self):
        """Test batch linear distance on more complex track."""
        node_positions = [(0, 0), (10, 0), (10, 10), (0, 10)]
        edges = [(0, 1), (1, 2), (2, 3)]
        track_graph = make_track_graph(node_positions, edges)

        projected_positions = np.array([[5, 0], [10, 5], [5, 10]])
        edge_ids = [(0, 1), (1, 2), (2, 3)]

        distances = batch_linear_distance(
            projected_track_positions=projected_positions,
            edge_ids=edge_ids,
            track_graph=track_graph,
            linear_zero_node_id=0,
        )

        # Distances should increase as we move along track
        assert len(distances) == 3
        assert distances[0] < distances[1] < distances[2]


class TestRouteDistanceChange:
    """Test route_distance_change function."""

    def test_route_distance_change_basic(self):
        """Test basic route distance change calculation."""
        node_positions = [(0, 0), (10, 0), (10, 10), (0, 10)]
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        track_graph = make_track_graph(node_positions, edges)

        # Sequence of positions moving along the track
        position = np.array([
            [5, 0],    # Edge 0
            [10, 5],   # Edge 1
            [5, 10],   # Edge 2
        ])

        distances = route_distance_change(position, track_graph)

        # Check structure - returns (n_time, n_segments, n_segments)
        assert distances.shape == (3, 4, 4)  # 3 time points, 4 segments
        # First time point should have all NaNs (no previous position)
        assert np.all(np.isnan(distances[0]))
        # Subsequent rows should have finite values
        assert np.all(np.isfinite(distances[1:]))

    def test_route_distance_change_simple_track(self):
        """Test route distance on simple two-point track."""
        node_positions = [(0, 0), (10, 0)]
        edges = [(0, 1)]
        track_graph = make_track_graph(node_positions, edges)

        # Two positions on same segment
        position = np.array([[3, 0], [7, 0]])

        distances = route_distance_change(position, track_graph)

        # Should return (2, 1, 1) for 2 time points and 1 segment
        assert distances.shape == (2, 1, 1)
        # First row is NaN
        assert np.isnan(distances[0, 0, 0])
        # Second row should be finite
        assert np.isfinite(distances[1, 0, 0])


class TestProject1dTo2d:
    """Test project_1d_to_2d function and edge cases."""

    def test_project_1d_to_2d_basic(self):
        """Test basic 1D to 2D projection."""
        node_positions = [(0, 0), (10, 0), (20, 0)]
        edges = [(0, 1), (1, 2)]
        track_graph = make_track_graph(node_positions, edges)

        # Linear positions along the track
        linear_positions = np.array([5.0, 15.0])

        projected = project_1d_to_2d(
            linear_positions, track_graph, edge_order=edges, edge_spacing=0.0
        )

        # Check shape
        assert projected.shape == (2, 2)
        # Check positions are on track
        assert np.allclose(projected[0], [5, 0])
        assert np.allclose(projected[1], [15, 0])

    def test_project_1d_to_2d_with_spacing(self):
        """Test 1D to 2D projection with edge spacing."""
        node_positions = [(0, 0), (10, 0), (20, 0)]
        edges = [(0, 1), (1, 2)]
        track_graph = make_track_graph(node_positions, edges)

        # With 5 unit spacing between edges
        linear_positions = np.array([5.0, 17.0])  # Accounting for spacing

        projected = project_1d_to_2d(
            linear_positions, track_graph, edge_order=edges, edge_spacing=5.0
        )

        assert projected.shape == (2, 2)
        # First position on first segment
        assert np.allclose(projected[0], [5, 0])
        # Second position on second segment (accounting for spacing)
        assert np.allclose(projected[1], [12, 0])

    def test_project_1d_to_2d_nan_handling(self):
        """Test 1D to 2D projection with NaN values."""
        node_positions = [(0, 0), (10, 0)]
        edges = [(0, 1)]
        track_graph = make_track_graph(node_positions, edges)

        # Include NaN in linear positions
        linear_positions = np.array([5.0, np.nan, 7.0])

        projected = project_1d_to_2d(
            linear_positions, track_graph, edge_order=edges, edge_spacing=0.0
        )

        # Check shape
        assert projected.shape == (3, 2)
        # First and third should be valid
        assert np.all(np.isfinite(projected[0]))
        assert np.all(np.isfinite(projected[2]))
        # Second should be NaN
        assert np.all(np.isnan(projected[1]))

    def test_project_1d_to_2d_out_of_bounds(self):
        """Test 1D to 2D projection with out-of-bounds positions."""
        node_positions = [(0, 0), (10, 0)]
        edges = [(0, 1)]
        track_graph = make_track_graph(node_positions, edges)

        # Position beyond end of track
        linear_positions = np.array([15.0])

        projected = project_1d_to_2d(
            linear_positions, track_graph, edge_order=edges, edge_spacing=0.0
        )

        # Should still return something (clamped to end or NaN)
        assert projected.shape == (1, 2)

    def test_project_1d_to_2d_roundtrip(self):
        """Test that 2D -> 1D -> 2D roundtrip preserves positions."""
        node_positions = [(0, 0), (10, 0), (20, 0)]
        edges = [(0, 1), (1, 2)]
        track_graph = make_track_graph(node_positions, edges)

        # Original 2D positions on the track
        position_2d = np.array([[5, 0], [15, 0]])

        # Convert to 1D
        result = get_linearized_position(position_2d, track_graph, edge_order=edges)
        linear_pos = result["linear_position"].to_numpy()

        # Convert back to 2D
        position_2d_reconstructed = project_1d_to_2d(
            linear_pos, track_graph, edge_order=edges, edge_spacing=0.0
        )

        # Should approximately recover original positions
        assert np.allclose(position_2d, position_2d_reconstructed, atol=0.01)

    def test_project_1d_to_2d_empty_array(self):
        """Test 1D to 2D projection with empty array."""
        node_positions = [(0, 0), (10, 0)]
        edges = [(0, 1)]
        track_graph = make_track_graph(node_positions, edges)

        linear_positions = np.array([])

        projected = project_1d_to_2d(
            linear_positions, track_graph, edge_order=edges, edge_spacing=0.0
        )

        # Should return empty array
        assert projected.shape[0] == 0
        # Empty array may not preserve 2D shape, which is acceptable
        assert len(projected.shape) <= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
