"""Tests for validation module."""

import numpy as np
import pytest

from track_linearization import get_linearized_position, make_track_graph
from track_linearization.validation import (
    check_track_graph_validity,
    detect_linearization_outliers,
    get_projection_confidence,
    validate_linearization,
)


class TestCheckTrackGraphValidity:
    """Tests for check_track_graph_validity function."""

    def test_valid_simple_track(self):
        """Test validation of a valid simple track."""
        track = make_track_graph([(0, 0), (10, 0)], [(0, 1)])
        report = check_track_graph_validity(track)

        assert report["valid"]
        assert len(report["errors"]) == 0
        assert report["n_nodes"] == 2
        assert report["n_edges"] == 1
        assert report["is_connected"]

    def test_valid_complex_track(self):
        """Test validation of a valid complex track."""
        node_positions = [(0, 0), (10, 0), (10, 10), (0, 10)]
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        track = make_track_graph(node_positions, edges)
        report = check_track_graph_validity(track)

        assert report["valid"]
        assert report["n_nodes"] == 4
        assert report["n_edges"] == 4
        assert report["is_connected"]

    def test_disconnected_graph_warning(self):
        """Test that disconnected graphs generate warnings."""
        node_positions = [(0, 0), (10, 0), (20, 0), (30, 0)]
        edges = [(0, 1), (2, 3)]  # Two separate components
        track = make_track_graph(node_positions, edges)
        report = check_track_graph_validity(track)

        assert report["valid"]  # Still valid, just has warnings
        assert not report["is_connected"]
        assert any("not connected" in w for w in report["warnings"])


class TestGetProjectionConfidence:
    """Tests for get_projection_confidence function."""

    def test_perfect_projection(self):
        """Test confidence for positions exactly on track."""
        track = make_track_graph([(0, 0), (100, 0)], [(0, 1)])
        positions = np.array([[25, 0], [50, 0], [75, 0]])
        result = get_linearized_position(positions, track)

        confidence = get_projection_confidence(positions, track, result)

        # All positions exactly on track should have high confidence
        assert np.all(confidence > 0.95)

    def test_noisy_positions(self):
        """Test confidence for positions with noise."""
        track = make_track_graph([(0, 0), (100, 0)], [(0, 1)])

        # Some positions on track, some far from it
        positions = np.array([[25, 0], [50, 20], [75, 0]])  # Middle one is far
        result = get_linearized_position(positions, track)

        confidence = get_projection_confidence(positions, track, result)

        # On-track positions should have higher confidence
        assert confidence[0] > confidence[1]
        assert confidence[2] > confidence[1]

    def test_confidence_range(self):
        """Test that confidence values are in [0, 1] range."""
        track = make_track_graph([(0, 0), (50, 0), (50, 50)], [(0, 1), (1, 2)])
        positions = np.array([[10, 0], [50, 25], [30, 30]])
        result = get_linearized_position(positions, track)

        confidence = get_projection_confidence(positions, track, result)

        assert np.all(confidence >= 0.0)
        assert np.all(confidence <= 1.0)


class TestDetectLinearizationOutliers:
    """Tests for detect_linearization_outliers function."""

    def test_no_outliers_clean_data(self):
        """Test that clean data produces no outliers."""
        track = make_track_graph([(0, 0), (100, 0)], [(0, 1)])
        positions = np.linspace([0, 0], [100, 0], num=20)
        result = get_linearized_position(positions, track)

        report = detect_linearization_outliers(positions, track, result)

        assert report["n_outliers"] == 0
        assert len(report["outlier_indices"]) == 0

    def test_detects_far_positions(self):
        """Test detection of positions far from track."""
        track = make_track_graph([(0, 0), (100, 0)], [(0, 1)])

        # Mix of on-track and far-from-track positions
        positions = np.array(
            [
                [10, 0],
                [20, 0],
                [30, 0],
                [40, 50],  # Last one is outlier
                [50, 0],
                [60, 0],
                [70, 0],
                [80, 0],
            ]
        )
        result = get_linearized_position(positions, track)

        report = detect_linearization_outliers(positions, track, result)

        assert report["n_outliers"] > 0
        assert 3 in report["outlier_indices"]  # Position at [40, 50]

    def test_detects_large_jumps(self):
        """Test detection of large jumps in linear position."""
        track = make_track_graph([(0, 0), (100, 0)], [(0, 1)])

        # Positions with a large jump
        positions = np.array(
            [[10, 0], [15, 0], [20, 0], [90, 0], [92, 0], [94, 0]]  # Large jump to 90
        )
        result = get_linearized_position(positions, track)

        report = detect_linearization_outliers(positions, track, result, threshold=2.0)

        # Should detect the jump at index 3
        assert report["n_outliers"] > 0


class TestValidateLinearization:
    """Tests for validate_linearization function."""

    def test_validates_clean_data(self):
        """Test that clean data passes validation."""
        track = make_track_graph([(0, 0), (50, 0), (50, 50)], [(0, 1), (1, 2)])
        positions = np.array([[10, 0], [30, 0], [50, 20], [50, 40]])
        result = get_linearized_position(positions, track)

        report = validate_linearization(positions, track, result)

        assert report["passed"] or report["quality_score"] > 0.5
        assert "metrics" in report
        assert "quality_score" in report

    def test_detects_quality_issues(self):
        """Test that poor quality linearization is detected."""
        track = make_track_graph([(0, 0), (10, 0)], [(0, 1)])

        # Positions far from track
        positions = np.array([[5, 50], [7, 50], [3, 60]])
        result = get_linearized_position(positions, track)

        report = validate_linearization(positions, track, result)

        # Should have warnings due to low confidence
        assert len(report["warnings"]) > 0 or report["quality_score"] < 0.9

    def test_strict_mode_raises(self):
        """Test that strict mode raises ValueError on quality issues."""
        track = make_track_graph([(0, 0), (10, 0)], [(0, 1)])

        # Positions far from track
        positions = np.array([[5, 100], [7, 100], [3, 120]])
        result = get_linearized_position(positions, track)

        # Strict mode should raise on poor quality
        with pytest.raises(ValueError, match="validation failed"):
            validate_linearization(positions, track, result, strict=True)

    def test_validation_report_structure(self):
        """Test that validation report has expected structure."""
        track = make_track_graph([(0, 0), (50, 0)], [(0, 1)])
        positions = np.array([[10, 0], [25, 0], [40, 0]])
        result = get_linearized_position(positions, track)

        report = validate_linearization(positions, track, result)

        # Check report structure
        assert "passed" in report
        assert "quality_score" in report
        assert "warnings" in report
        assert "recommendations" in report
        assert "metrics" in report

        # Check metrics structure
        assert "mean_confidence" in report["metrics"]
        assert "outlier_count" in report["metrics"]
        assert "mean_projection_distance" in report["metrics"]
