"""Validation and quality control functions for track linearization.

This module provides tools to validate linearization quality, detect outliers,
and assess confidence in position projections.
"""

from typing import Any

import networkx as nx
import numpy as np
import pandas as pd


def check_track_graph_validity(track_graph: nx.Graph) -> dict[str, Any]:
    """Validate that a track graph has the required structure and attributes.

    Checks for:
    - Node 'pos' attributes (2D or 3D coordinates)
    - Edge 'distance' attributes
    - Edge 'edge_id' attributes
    - Graph connectivity
    - Valid coordinate values (no NaN/inf)

    Parameters
    ----------
    track_graph : networkx.Graph
        The track graph to validate.

    Returns
    -------
    report : dict
        Validation report with keys:
        - 'valid' (bool): Whether graph passes all checks
        - 'errors' (list[str]): List of error messages
        - 'warnings' (list[str]): List of warning messages
        - 'n_nodes' (int): Number of nodes
        - 'n_edges' (int): Number of edges
        - 'is_connected' (bool): Whether graph is connected

    Examples
    --------
    >>> from track_linearization import make_track_graph
    >>> track = make_track_graph([(0, 0), (10, 0)], [(0, 1)])
    >>> report = check_track_graph_validity(track)
    >>> report['valid']
    True
    """
    errors = []
    warnings_list = []

    # Check basic structure
    n_nodes = track_graph.number_of_nodes()
    n_edges = track_graph.number_of_edges()

    if n_nodes == 0:
        errors.append("Graph has no nodes")
    if n_edges == 0:
        errors.append("Graph has no edges")

    # Check node positions
    node_pos = nx.get_node_attributes(track_graph, "pos")

    if not node_pos:
        errors.append("No nodes have 'pos' attribute")
    elif len(node_pos) < n_nodes:
        missing = n_nodes - len(node_pos)
        errors.append(f"{missing} node(s) missing 'pos' attribute")
    else:
        # Check for valid coordinates
        for node, pos in node_pos.items():
            if not isinstance(pos, (tuple, list, np.ndarray)):
                errors.append(f"Node {node} 'pos' is not array-like: {type(pos)}")
            elif not all(np.isfinite(pos)):
                errors.append(f"Node {node} has invalid coordinates: {pos}")

    # Check edge attributes
    edge_distances = nx.get_edge_attributes(track_graph, "distance")
    edge_ids = nx.get_edge_attributes(track_graph, "edge_id")

    if len(edge_distances) < n_edges:
        missing = n_edges - len(edge_distances)
        errors.append(f"{missing} edge(s) missing 'distance' attribute")
    else:
        # Check for valid distances
        for edge, dist in edge_distances.items():
            if not np.isfinite(dist):
                errors.append(f"Edge {edge} has invalid distance: {dist}")
            elif dist <= 0:
                warnings_list.append(
                    f"Edge {edge} has zero or negative distance: {dist}"
                )

    if len(edge_ids) < n_edges:
        missing = n_edges - len(edge_ids)
        warnings_list.append(f"{missing} edge(s) missing 'edge_id' attribute")

    # Check connectivity
    is_connected = nx.is_connected(track_graph) if n_nodes > 0 else False
    if not is_connected and n_nodes > 1:
        n_components = nx.number_connected_components(track_graph)
        warnings_list.append(
            f"Graph is not connected ({n_components} components). "
            "Some positions may not be reachable."
        )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings_list,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "is_connected": is_connected,
    }


def get_projection_confidence(
    position: np.ndarray,
    track_graph: nx.Graph,
    linearization_result: pd.DataFrame,
) -> np.ndarray:
    """Calculate confidence scores for position projections onto track.

    Confidence is based on the distance from the original position to its
    projection on the track. Closer projections have higher confidence.

    Parameters
    ----------
    position : np.ndarray, shape (n_time, n_space)
        Original 2D or 3D positions.
    track_graph : networkx.Graph
        The track graph used for linearization.
    linearization_result : pd.DataFrame
        Output from get_linearized_position() containing projected coordinates.

    Returns
    -------
    confidence : np.ndarray, shape (n_time,)
        Confidence scores between 0 and 1, where:
        - 1.0 = position exactly on track
        - Values decrease as distance from track increases
        - Confidence = exp(-distance^2 / (2 * scale^2))
        Scale is set to median distance to avoid extreme values.

    Examples
    --------
    >>> position = np.array([[10, 5], [50, 0]])  # Second point on track
    >>> confidence = get_projection_confidence(position, track, result)
    >>> confidence[1] > confidence[0]  # On-track position has higher confidence
    True
    """
    position = np.asarray(position)

    # Get projected positions
    proj_cols = [c for c in linearization_result.columns if "projected" in c]
    projected = linearization_result[proj_cols].to_numpy()

    # Calculate distances
    distances = np.linalg.norm(position - projected, axis=1)

    # Use median distance as scale (robust to outliers)
    scale = np.median(distances)
    if scale == 0:
        scale = 1.0  # All points on track

    # Calculate confidence using Gaussian-like decay
    confidence: np.ndarray = np.exp(-(distances**2) / (2 * scale**2))

    return confidence


def detect_linearization_outliers(
    position: np.ndarray,
    track_graph: nx.Graph,
    linearization_result: pd.DataFrame,
    threshold: float = 3.0,
) -> dict[str, Any]:
    """Detect suspicious positions that may be poorly linearized.

    Uses multiple criteria to identify outliers:
    1. Large projection distance (far from track)
    2. Rapid jumps in linear position
    3. Frequent segment switching

    Parameters
    ----------
    position : np.ndarray, shape (n_time, n_space)
        Original 2D or 3D positions.
    track_graph : networkx.Graph
        The track graph used for linearization.
    linearization_result : pd.DataFrame
        Output from get_linearized_position().
    threshold : float, optional
        Number of standard deviations to use as outlier threshold (default: 3.0).

    Returns
    -------
    report : dict
        Dictionary containing:
        - 'outlier_indices' (np.ndarray): Indices of outlier positions
        - 'outlier_reasons' (list[str]): Reason for each outlier
        - 'projection_distances' (np.ndarray): Distance from position to track
        - 'linear_jumps' (np.ndarray): Change in linear position between steps
        - 'n_outliers' (int): Total number of outliers

    Examples
    --------
    >>> report = detect_linearization_outliers(position, track, result)
    >>> if report['n_outliers'] > 0:
    ...     print(f"Found {report['n_outliers']} outliers")
    ...     print(f"At indices: {report['outlier_indices']}")
    """
    position = np.asarray(position)

    # Get projected positions
    proj_cols = [c for c in linearization_result.columns if "projected" in c]
    projected = linearization_result[proj_cols].to_numpy()

    # Calculate projection distances
    proj_distances = np.linalg.norm(position - projected, axis=1)

    # Calculate linear position jumps
    linear_pos = linearization_result["linear_position"].to_numpy()
    linear_jumps = np.abs(np.diff(linear_pos))

    # Note: segment switches could be used for more sophisticated outlier detection
    # segment_ids = linearization_result["track_segment_id"].to_numpy()

    # Identify outliers based on projection distance
    # Use robust statistics (median + MAD) instead of mean + std
    median_dist = np.median(proj_distances)
    mad_dist = np.median(np.abs(proj_distances - median_dist))

    # Only flag distance outliers if there's meaningful variation
    if mad_dist > 1e-10:
        dist_threshold = median_dist + threshold * mad_dist * 1.4826  # Scale MAD to std
        dist_outliers = proj_distances > dist_threshold
    else:
        # For very uniform distances, use absolute threshold
        dist_threshold = median_dist + threshold
        dist_outliers = proj_distances > dist_threshold

    # Identify outliers based on large jumps
    mad = np.median(np.abs(linear_jumps - np.median(linear_jumps)))
    # Avoid detecting outliers when data is too uniform (MAD close to 0)
    if mad > 1e-10:  # Only detect jump outliers if there's variation
        jump_threshold = np.median(linear_jumps) + threshold * mad
        jump_outliers = np.concatenate([[False], linear_jumps > jump_threshold])
    else:
        jump_outliers = np.zeros(len(proj_distances), dtype=bool)

    # Combine outlier criteria
    outlier_mask = dist_outliers | jump_outliers
    outlier_indices = np.where(outlier_mask)[0]

    # Generate reasons
    outlier_reasons = []
    for idx in outlier_indices:
        reasons = []
        if dist_outliers[idx]:
            reasons.append(
                f"far from track (distance={proj_distances[idx]:.2f} > {dist_threshold:.2f})"
            )
        if jump_outliers[idx]:
            reasons.append(
                f"large jump (delta={linear_jumps[idx-1]:.2f} > {jump_threshold:.2f})"
            )
        outlier_reasons.append("; ".join(reasons))

    return {
        "outlier_indices": outlier_indices,
        "outlier_reasons": outlier_reasons,
        "projection_distances": proj_distances,
        "linear_jumps": np.concatenate([[0], linear_jumps]),
        "n_outliers": len(outlier_indices),
    }


def validate_linearization(
    position: np.ndarray,
    track_graph: nx.Graph,
    linearization_result: pd.DataFrame,
    strict: bool = False,
) -> dict[str, Any]:
    """Comprehensive validation of linearization quality.

    Performs multiple quality checks and returns a detailed report with
    warnings and recommendations.

    Parameters
    ----------
    position : np.ndarray, shape (n_time, n_space)
        Original 2D or 3D positions.
    track_graph : networkx.Graph
        The track graph used for linearization.
    linearization_result : pd.DataFrame
        Output from get_linearized_position().
    strict : bool, optional
        If True, raise ValueError on quality issues. If False (default),
        return report with warnings.

    Returns
    -------
    report : dict
        Validation report containing:
        - 'passed' (bool): Whether validation passed
        - 'quality_score' (float): Overall quality score (0-1)
        - 'warnings' (list[str]): List of warning messages
        - 'recommendations' (list[str]): Suggestions for improvement
        - 'metrics' (dict): Detailed quality metrics

    Raises
    ------
    ValueError
        If strict=True and validation fails.

    Examples
    --------
    >>> report = validate_linearization(position, track, result)
    >>> if not report['passed']:
    ...     print(f"Quality score: {report['quality_score']:.2f}")
    ...     for warning in report['warnings']:
    ...         print(f"  ⚠️  {warning}")
    >>> else:
    ...     print("✓ Linearization validated!")
    """
    position = np.asarray(position)
    warnings_list = []
    recommendations = []

    # Check track validity
    track_report = check_track_graph_validity(track_graph)
    if not track_report["valid"]:
        for error in track_report["errors"]:
            warnings_list.append(f"Track graph issue: {error}")
    warnings_list.extend([f"Track: {w}" for w in track_report["warnings"]])

    # Get confidence scores
    confidence = get_projection_confidence(position, track_graph, linearization_result)
    mean_confidence = np.mean(confidence)
    low_confidence_pct = np.mean(confidence < 0.5) * 100

    if mean_confidence < 0.7:
        warnings_list.append(
            f"Low mean confidence: {mean_confidence:.2f} "
            f"({low_confidence_pct:.1f}% of positions have confidence < 0.5)"
        )
        recommendations.append("Consider using HMM mode (use_HMM=True) for noisy data")

    # Detect outliers
    outlier_report = detect_linearization_outliers(
        position, track_graph, linearization_result
    )
    outlier_pct = (outlier_report["n_outliers"] / len(position)) * 100

    if outlier_report["n_outliers"] > 0:
        warnings_list.append(
            f"Found {outlier_report['n_outliers']} outliers "
            f"({outlier_pct:.1f}% of positions)"
        )
        if outlier_pct > 10:
            recommendations.append(
                "High outlier rate suggests track structure may not match actual paths"
            )
            recommendations.append(
                "Verify track graph structure with plot_track_graph()"
            )

    # Check for NaN values
    if linearization_result.isna().any().any():
        n_nan = linearization_result.isna().any(axis=1).sum()
        warnings_list.append(f"Found {n_nan} positions with NaN values")

    # Calculate quality score
    quality_components = [
        mean_confidence,  # Weight: 0.4
        1.0 - min(outlier_pct / 100, 1.0),  # Weight: 0.3
        1.0 if track_report["valid"] else 0.0,  # Weight: 0.3
    ]
    weights = [0.4, 0.3, 0.3]
    quality_score = sum(
        w * c for w, c in zip(weights, quality_components, strict=False)
    )

    # Determine pass/fail
    passed = quality_score > 0.7 and len(warnings_list) == 0

    # Collect metrics
    metrics = {
        "mean_confidence": mean_confidence,
        "low_confidence_percentage": low_confidence_pct,
        "outlier_count": outlier_report["n_outliers"],
        "outlier_percentage": outlier_pct,
        "mean_projection_distance": np.mean(outlier_report["projection_distances"]),
        "median_projection_distance": np.median(outlier_report["projection_distances"]),
    }

    report = {
        "passed": passed,
        "quality_score": quality_score,
        "warnings": warnings_list,
        "recommendations": recommendations,
        "metrics": metrics,
    }

    if strict and not passed:
        error_msg = "Linearization validation failed:\n"
        error_msg += "\n".join(f"  - {w}" for w in warnings_list)
        raise ValueError(error_msg)

    return report
