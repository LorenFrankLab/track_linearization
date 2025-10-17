"""Tests for track_builders module."""

import numpy as np
import pytest

from track_linearization.track_builders import (
    make_circular_track,
    make_figure8_track,
    make_linear_track,
    make_plus_maze_track,
    make_rectangular_track,
    make_tmaze_track,
    make_wtrack,
    make_ymaze_track,
)


class TestMakeLinearTrack:
    """Tests for make_linear_track function."""

    def test_basic_linear_track(self):
        """Test creation of basic linear track."""
        track = make_linear_track(length=100.0)

        assert track.number_of_nodes() == 2
        assert track.number_of_edges() == 1

        # Check length
        edge_length = track.edges[(0, 1)]["distance"]
        assert np.isclose(edge_length, 100.0)

    def test_custom_start_position(self):
        """Test linear track with custom start position."""
        track = make_linear_track(length=50.0, start_pos=(10, 20))

        node_pos = list(track.nodes(data="pos"))
        assert node_pos[0][1] == (10, 20)
        assert node_pos[1][1] == (60, 20)


class TestMakeCircularTrack:
    """Tests for make_circular_track function."""

    def test_basic_circular_track(self):
        """Test creation of basic circular track."""
        track = make_circular_track(radius=30.0, n_segments=24)

        assert track.number_of_nodes() == 24
        assert track.number_of_edges() == 24

        # Check approximate circumference
        total_length = sum(track.edges[e]["distance"] for e in track.edges())
        expected_circumference = 2 * np.pi * 30.0
        assert np.abs(total_length - expected_circumference) < 1.0

    def test_custom_center(self):
        """Test circular track with custom center."""
        track = make_circular_track(radius=20.0, center=(10, 15), n_segments=12)

        # Get node positions and check they're centered around (10, 15)
        positions = [data["pos"] for _, data in track.nodes(data=True)]
        mean_x = np.mean([p[0] for p in positions])
        mean_y = np.mean([p[1] for p in positions])

        assert np.isclose(mean_x, 10.0, atol=0.5)
        assert np.isclose(mean_y, 15.0, atol=0.5)

    def test_too_few_segments_raises(self):
        """Test that too few segments raises ValueError."""
        with pytest.raises(ValueError, match="n_segments must be >= 3"):
            make_circular_track(radius=10.0, n_segments=2)


class TestMakeTMazeTrack:
    """Tests for make_tmaze_track function."""

    def test_basic_tmaze(self):
        """Test creation of basic T-maze."""
        track = make_tmaze_track(stem_length=50.0, arm_length=30.0)

        assert track.number_of_nodes() == 4
        assert track.number_of_edges() == 3

        # Check that stem has correct length
        stem_edge = (0, 1)
        stem_length = track.edges[stem_edge]["distance"]
        assert np.isclose(stem_length, 50.0)

    def test_custom_arm_spacing(self):
        """Test T-maze with custom arm spacing."""
        track = make_tmaze_track(stem_length=40.0, arm_length=20.0, arm_spacing=60.0)

        # Get left and right arm end positions
        positions = [data["pos"] for _, data in track.nodes(data=True)]

        # Left arm (node 2) and right arm (node 3) should be 60 units apart
        left_x = positions[2][0]
        right_x = positions[3][0]
        spacing = abs(right_x - left_x)

        assert np.isclose(spacing, 60.0)


class TestMakePlusMazeTrack:
    """Tests for make_plus_maze_track function."""

    def test_basic_plus_maze(self):
        """Test creation of basic plus maze."""
        track = make_plus_maze_track(arm_length=40.0)

        assert track.number_of_nodes() == 5  # Center + 4 arms
        assert track.number_of_edges() == 4

        # All edges should have same length
        lengths = [track.edges[e]["distance"] for e in track.edges()]
        assert np.allclose(lengths, 40.0)

    def test_custom_center(self):
        """Test plus maze with custom center."""
        track = make_plus_maze_track(arm_length=30.0, center=(15, 20))

        # Center node should be at (15, 20)
        center_pos = track.nodes[0]["pos"]
        assert center_pos == (15, 20)


class TestMakeFigure8Track:
    """Tests for make_figure8_track function."""

    def test_basic_figure8(self):
        """Test creation of basic figure-8 track."""
        track = make_figure8_track(loop_radius=20.0, n_segments_per_loop=12)

        # Should have 2 loops + connection
        expected_edges = 2 * 12 + 1
        assert track.number_of_edges() == expected_edges

    def test_too_few_segments_raises(self):
        """Test that too few segments raises ValueError."""
        with pytest.raises(ValueError, match="n_segments_per_loop must be >= 3"):
            make_figure8_track(loop_radius=10.0, n_segments_per_loop=2)


class TestMakeWtrack:
    """Tests for make_wtrack function."""

    def test_basic_wtrack(self):
        """Test creation of basic W-track."""
        track = make_wtrack(width=60.0, height=40.0)

        assert track.number_of_nodes() == 6
        assert track.number_of_edges() == 5

    def test_wtrack_dimensions(self):
        """Test W-track has correct dimensions."""
        track = make_wtrack(width=60.0, height=40.0)

        positions = [data["pos"] for _, data in track.nodes(data=True)]

        # Check width
        x_coords = [p[0] for p in positions]
        width = max(x_coords) - min(x_coords)
        assert np.isclose(width, 60.0)

        # Check height
        y_coords = [p[1] for p in positions]
        height = max(y_coords) - min(y_coords)
        assert np.isclose(height, 40.0)


class TestMakeRectangularTrack:
    """Tests for make_rectangular_track function."""

    def test_basic_rectangular_track(self):
        """Test creation of basic rectangular track."""
        track = make_rectangular_track(width=50.0, height=30.0)

        assert track.number_of_nodes() == 4
        assert track.number_of_edges() == 4

        # Check total perimeter
        total_length = sum(track.edges[e]["distance"] for e in track.edges())
        expected_perimeter = 2 * (50.0 + 30.0)
        assert np.isclose(total_length, expected_perimeter)

    def test_custom_start_corner(self):
        """Test rectangular track with custom start corner."""
        track = make_rectangular_track(width=40.0, height=30.0, start_corner=(10, 20))

        # First node should be at start corner
        first_node_pos = track.nodes[0]["pos"]
        assert first_node_pos == (10, 20)


class TestMakeYMazeTrack:
    """Tests for make_ymaze_track function."""

    def test_basic_ymaze(self):
        """Test creation of basic Y-maze."""
        track = make_ymaze_track(arm_length=40.0)

        assert track.number_of_nodes() == 4  # Center + 3 arms
        assert track.number_of_edges() == 3

        # All arms should have same length
        lengths = [track.edges[e]["distance"] for e in track.edges()]
        assert np.allclose(lengths, 40.0)

    def test_ymaze_angles(self):
        """Test Y-maze has correct angles between arms."""
        track = make_ymaze_track(arm_length=30.0, arm_angle_deg=120.0)

        # Get arm end positions
        positions = [data["pos"] for _, data in track.nodes(data=True)][
            1:
        ]  # Skip center

        # Calculate angles from center
        angles = [np.arctan2(p[1], p[0]) for p in positions]

        # Check angle differences (should be ~120° = 2π/3 radians)
        angle_diffs = [np.abs(angles[i] - angles[i - 1]) for i in range(1, len(angles))]
        expected_diff = np.deg2rad(120.0)

        # At least one pair should be close to 120°
        assert any(np.abs(diff - expected_diff) < 0.1 for diff in angle_diffs)


class TestTrackBuilderIntegration:
    """Integration tests for track builders."""

    def test_all_builders_create_valid_graphs(self):
        """Test that all builders create valid NetworkX graphs."""
        builders = [
            lambda: make_linear_track(100.0),
            lambda: make_circular_track(30.0),
            lambda: make_tmaze_track(50.0, 30.0),
            lambda: make_plus_maze_track(40.0),
            lambda: make_figure8_track(20.0),
            lambda: make_wtrack(60.0, 40.0),
            lambda: make_rectangular_track(50.0, 30.0),
            lambda: make_ymaze_track(40.0),
        ]

        for builder in builders:
            track = builder()

            # Check basic graph properties
            assert track.number_of_nodes() > 0
            assert track.number_of_edges() > 0

            # Check all nodes have 'pos' attribute
            for node in track.nodes():
                assert "pos" in track.nodes[node]

            # Check all edges have 'distance' and 'edge_id'
            for edge in track.edges():
                assert "distance" in track.edges[edge]
                assert "edge_id" in track.edges[edge]

    def test_tracks_have_finite_coordinates(self):
        """Test that all track builders produce finite coordinates."""
        track = make_figure8_track(20.0, n_segments_per_loop=8)

        for _, data in track.nodes(data=True):
            pos = data["pos"]
            assert all(np.isfinite(pos))

        for edge in track.edges():
            distance = track.edges[edge]["distance"]
            assert np.isfinite(distance)
            assert distance > 0
