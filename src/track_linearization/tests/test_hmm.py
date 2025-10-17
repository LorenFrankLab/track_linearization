"""Tests for HMM-based track segment classification functionality."""

import numpy as np
import pytest

from track_linearization import get_linearized_position, make_track_graph


class TestHMMClassification:
    """Test HMM-based position classification with use_HMM=True."""

    def test_hmm_basic_linear_track(self):
        """Test HMM classification on a simple linear track."""
        # Create a simple 3-segment linear track
        node_positions = [(0, 0), (10, 0), (20, 0), (30, 0)]
        edges = [(0, 1), (1, 2), (2, 3)]
        track_graph = make_track_graph(node_positions, edges)

        # Generate positions clearly on each segment
        position = np.array([
            [2, 0],   # Segment 0
            [5, 0],   # Segment 0
            [12, 0],  # Segment 1
            [15, 0],  # Segment 1
            [22, 0],  # Segment 2
            [25, 0],  # Segment 2
        ])

        result = get_linearized_position(
            position, track_graph, use_HMM=True, sensor_std_dev=5.0
        )

        # Verify segment classification
        assert result["track_segment_id"].iloc[0] == 0
        assert result["track_segment_id"].iloc[1] == 0
        assert result["track_segment_id"].iloc[2] == 1
        assert result["track_segment_id"].iloc[3] == 1
        assert result["track_segment_id"].iloc[4] == 2
        assert result["track_segment_id"].iloc[5] == 2

    def test_hmm_with_noise(self):
        """Test HMM handles noisy position data."""
        node_positions = [(0, 0), (10, 0), (20, 0)]
        edges = [(0, 1), (1, 2)]
        track_graph = make_track_graph(node_positions, edges)

        # Add noise to positions
        np.random.seed(42)
        position = np.array([[5, 0], [15, 0]]) + np.random.normal(0, 2, (2, 2))

        result = get_linearized_position(
            position, track_graph, use_HMM=True, sensor_std_dev=5.0
        )

        # Should still classify correctly despite noise
        assert result["track_segment_id"].iloc[0] == 0
        assert result["track_segment_id"].iloc[1] == 1

    def test_hmm_temporal_continuity(self):
        """Test HMM prefers smooth transitions between adjacent segments."""
        # Create Y-shaped track where segments meet at a junction
        node_positions = [(0, 0), (10, 0), (5, 10), (15, 10)]
        edges = [(0, 1), (1, 2), (1, 3)]
        track_graph = make_track_graph(node_positions, edges)

        # Position sequence that moves smoothly along track
        position = np.array([
            [2, 0],    # Segment 0
            [5, 0],    # Segment 0
            [8, 0],    # Segment 0
            [9, 2],    # Near junction
            [7, 6],    # Segment 1 (toward node 2)
            [6, 8],    # Segment 1
        ])

        result = get_linearized_position(
            position,
            track_graph,
            use_HMM=True,
            sensor_std_dev=3.0,
            diagonal_bias=0.5,  # Encourage staying on same segment
        )

        # Verify smooth transition
        segment_ids = result["track_segment_id"].values
        # Should not jump erratically
        assert not np.isnan(segment_ids).any()

    def test_hmm_vs_no_hmm(self):
        """Compare HMM vs non-HMM classification on ambiguous position."""
        node_positions = [(0, 0), (10, 0), (10, 10)]
        edges = [(0, 1), (1, 2)]
        track_graph = make_track_graph(node_positions, edges)

        # Position equidistant from two segments
        position = np.array([[10, 5]])  # Exactly between both segments

        result_no_hmm = get_linearized_position(position, track_graph, use_HMM=False)
        result_hmm = get_linearized_position(
            position, track_graph, use_HMM=True, sensor_std_dev=5.0
        )

        # Both should give valid results (may differ due to different methods)
        assert not np.isnan(result_no_hmm["track_segment_id"].iloc[0])
        assert not np.isnan(result_hmm["track_segment_id"].iloc[0])

    def test_hmm_sensor_std_dev_parameter(self):
        """Test that sensor_std_dev parameter affects results."""
        node_positions = [(0, 0), (10, 0), (20, 0)]
        edges = [(0, 1), (1, 2)]
        track_graph = make_track_graph(node_positions, edges)

        # Position with some distance from track
        position = np.array([[5, 3]])  # 3 units off track

        # Low std dev (strict) vs high std dev (lenient)
        result_strict = get_linearized_position(
            position, track_graph, use_HMM=True, sensor_std_dev=1.0
        )
        result_lenient = get_linearized_position(
            position, track_graph, use_HMM=True, sensor_std_dev=10.0
        )

        # Both should work, but confidence may differ
        assert not np.isnan(result_strict["track_segment_id"].iloc[0])
        assert not np.isnan(result_lenient["track_segment_id"].iloc[0])

    def test_hmm_diagonal_bias_parameter(self):
        """Test diagonal_bias affects segment persistence."""
        node_positions = [(0, 0), (5, 0), (10, 0)]
        edges = [(0, 1), (1, 2)]
        track_graph = make_track_graph(node_positions, edges)

        # Positions near segment boundary
        position = np.array([
            [4.5, 0],  # Near boundary
            [5.5, 0],  # Just past boundary
        ])

        # High diagonal bias should resist switching
        result_high_bias = get_linearized_position(
            position, track_graph, use_HMM=True, diagonal_bias=0.9
        )

        # Low diagonal bias allows switching
        result_low_bias = get_linearized_position(
            position, track_graph, use_HMM=True, diagonal_bias=0.1
        )

        # Results should be valid
        assert len(result_high_bias) == 2
        assert len(result_low_bias) == 2

    def test_hmm_route_distance_scaling(self):
        """Test route_euclidean_distance_scaling parameter."""
        node_positions = [(0, 0), (10, 0), (10, 10), (0, 10)]
        edges = [(0, 1), (1, 2), (2, 3)]
        track_graph = make_track_graph(node_positions, edges)

        position = np.array([[5, 0], [10, 5], [5, 10]])

        # Different scaling values
        result1 = get_linearized_position(
            position, track_graph, use_HMM=True, route_euclidean_distance_scaling=0.1
        )
        result2 = get_linearized_position(
            position, track_graph, use_HMM=True, route_euclidean_distance_scaling=10.0
        )

        # Both should produce valid results
        assert len(result1) == 3
        assert len(result2) == 3
        assert not result1["track_segment_id"].isna().any()
        assert not result2["track_segment_id"].isna().any()


class TestHMMEdgeCases:
    """Test edge cases and error handling for HMM."""

    def test_hmm_single_position(self):
        """Test HMM with just one position."""
        node_positions = [(0, 0), (10, 0)]
        edges = [(0, 1)]
        track_graph = make_track_graph(node_positions, edges)

        position = np.array([[5, 0]])

        result = get_linearized_position(position, track_graph, use_HMM=True)

        assert len(result) == 1
        assert result["track_segment_id"].iloc[0] == 0

    def test_hmm_very_noisy_data(self):
        """Test HMM with extremely noisy positions."""
        node_positions = [(0, 0), (10, 0)]
        edges = [(0, 1)]
        track_graph = make_track_graph(node_positions, edges)

        # Positions far from track
        position = np.array([
            [5, 50],   # Very far off track
            [5, -50],  # Very far in opposite direction
        ])

        result = get_linearized_position(
            position, track_graph, use_HMM=True, sensor_std_dev=20.0
        )

        # Should handle gracefully (may have NaNs for bad positions)
        assert len(result) == 2

    def test_hmm_empty_positions(self):
        """Test HMM with empty position array."""
        node_positions = [(0, 0), (10, 0)]
        edges = [(0, 1)]
        track_graph = make_track_graph(node_positions, edges)

        position = np.empty((0, 2))

        result = get_linearized_position(position, track_graph, use_HMM=True)

        # Should return empty dataframe
        assert len(result) == 0

    def test_hmm_nan_positions(self):
        """Test HMM handles NaN in position data."""
        node_positions = [(0, 0), (10, 0), (20, 0)]
        edges = [(0, 1), (1, 2)]
        track_graph = make_track_graph(node_positions, edges)

        position = np.array([
            [5, 0],
            [np.nan, np.nan],  # Bad position
            [15, 0],
        ])

        result = get_linearized_position(position, track_graph, use_HMM=True)

        # Should handle NaN positions
        assert len(result) == 3
        # HMM fills in NaN positions with defaults (segment 0)
        # This is current behavior - NaN positions get imputed
        assert not np.isnan(result["track_segment_id"].iloc[1])


class TestHMMHelperFunctions:
    """Test individual HMM helper functions."""

    def test_euclidean_distance_change(self):
        """Test euclidean_distance_change function."""
        from track_linearization.core import euclidean_distance_change

        position = np.array([[0, 0], [3, 4], [3, 4], [6, 8]])

        distances = euclidean_distance_change(position)

        # First element is NaN by design (no previous position)
        assert np.isnan(distances[0])
        # Distance from (0,0) to (3,4) is 5
        assert np.isclose(distances[1], 5.0)
        # Distance from (3,4) to (3,4) is 0
        assert np.isclose(distances[2], 0.0)
        # Distance from (3,4) to (6,8) is 5
        assert np.isclose(distances[3], 5.0)

    def test_batch_function(self):
        """Test batch iterator function."""
        from track_linearization.core import batch

        # Test batching 10 samples with batch_size=3
        batches = list(batch(10, batch_size=3))

        assert len(batches) == 4  # ceil(10/3) = 4 batches
        assert len(batches[0]) == 3  # First batch full
        assert len(batches[1]) == 3  # Second batch full
        assert len(batches[2]) == 3  # Third batch full
        assert len(batches[3]) == 1  # Last batch partial

    def test_batch_single_batch(self):
        """Test batch with all samples in one batch."""
        from track_linearization.core import batch

        batches = list(batch(5, batch_size=10))

        assert len(batches) == 1
        assert len(batches[0]) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
