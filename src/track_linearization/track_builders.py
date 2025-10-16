"""Helper functions for constructing common track geometries.

This module provides convenient builders for standard experimental track
designs, reducing setup time from hours to minutes.
"""

import numpy as np
import networkx as nx

from track_linearization.utils import make_track_graph


def make_linear_track(length: float, start_pos: tuple[float, float] = (0, 0)) -> nx.Graph:
    """Create a simple straight linear track.

    Parameters
    ----------
    length : float
        Length of the track in your units (e.g., cm).
    start_pos : tuple[float, float], optional
        (x, y) coordinates of the start position (default: (0, 0)).

    Returns
    -------
    track_graph : networkx.Graph
        Track graph with 2 nodes and 1 edge.

    Examples
    --------
    >>> track = make_linear_track(length=100.0)
    >>> track.number_of_nodes()
    2
    >>> track.number_of_edges()
    1
    """
    x0, y0 = start_pos
    node_positions = [
        (x0, y0),
        (x0 + length, y0),
    ]
    edges = [(0, 1)]
    return make_track_graph(node_positions, edges)


def make_circular_track(
    radius: float,
    center: tuple[float, float] = (0, 0),
    n_segments: int = 24,
) -> nx.Graph:
    """Create a circular track.

    The circle is approximated using multiple short line segments.
    More segments = smoother circle, but more computational cost.

    Parameters
    ----------
    radius : float
        Radius of the circle.
    center : tuple[float, float], optional
        (x, y) coordinates of circle center (default: (0, 0)).
    n_segments : int, optional
        Number of line segments to approximate the circle (default: 24).
        Must be >= 3.

    Returns
    -------
    track_graph : networkx.Graph
        Track graph with n_segments nodes and edges forming a closed loop.

    Examples
    --------
    >>> track = make_circular_track(radius=30.0, n_segments=24)
    >>> track.number_of_nodes()
    24
    >>> # Approximate circumference
    >>> total_length = sum(track.edges[e]['distance'] for e in track.edges())
    >>> abs(total_length - 2 * np.pi * 30) < 1.0  # Close to theoretical
    True
    """
    if n_segments < 3:
        raise ValueError("n_segments must be >= 3")

    cx, cy = center

    # Generate evenly spaced angles
    angles = np.linspace(-np.pi, np.pi, num=n_segments, endpoint=False)

    # Convert to (x, y) coordinates
    node_positions = [
        (cx + radius * np.cos(angle), cy + radius * np.sin(angle)) for angle in angles
    ]

    # Connect consecutive nodes in a loop
    edges = [(i, (i + 1) % n_segments) for i in range(n_segments)]

    return make_track_graph(node_positions, edges)


def make_tmaze_track(
    stem_length: float,
    arm_length: float,
    arm_spacing: float | None = None,
) -> nx.Graph:
    """Create a T-maze track.

    The T-maze has a stem and two arms (left and right). The layout is:
    ```
    Left Arm    Right Arm
        |            |
        +------------+  <- Junction
             |
           Stem
             |
           Start
    ```

    Parameters
    ----------
    stem_length : float
        Length of the stem from start to junction.
    arm_length : float
        Length of each arm from junction to end.
    arm_spacing : float, optional
        Horizontal spacing between left and right arms (default: stem_length).

    Returns
    -------
    track_graph : networkx.Graph
        Track graph with T-maze structure.

    Examples
    --------
    >>> track = make_tmaze_track(stem_length=50.0, arm_length=30.0)
    >>> track.number_of_nodes()
    4
    >>> track.number_of_edges()
    3
    """
    if arm_spacing is None:
        arm_spacing = stem_length

    # Define nodes: start, junction, left end, right end
    node_positions = [
        (0, 0),  # Start
        (0, stem_length),  # Junction
        (-arm_spacing / 2, stem_length + arm_length),  # Left arm end
        (arm_spacing / 2, stem_length + arm_length),  # Right arm end
    ]

    edges = [
        (0, 1),  # Stem
        (1, 2),  # Left arm
        (1, 3),  # Right arm
    ]

    return make_track_graph(node_positions, edges)


def make_plus_maze_track(
    arm_length: float,
    center: tuple[float, float] = (0, 0),
) -> nx.Graph:
    """Create a plus-maze (4-arm cross) track.

    The plus maze has 4 arms extending from a central junction point:
    ```
         North
           |
    West---+---East
           |
         South
    ```

    Parameters
    ----------
    arm_length : float
        Length of each arm from center to end.
    center : tuple[float, float], optional
        (x, y) coordinates of center junction (default: (0, 0)).

    Returns
    -------
    track_graph : networkx.Graph
        Track graph with 5 nodes (center + 4 arm ends) and 4 edges.

    Examples
    --------
    >>> track = make_plus_maze_track(arm_length=40.0)
    >>> track.number_of_nodes()
    5
    >>> track.number_of_edges()
    4
    """
    cx, cy = center

    node_positions = [
        (cx, cy),  # Center
        (cx, cy + arm_length),  # North
        (cx + arm_length, cy),  # East
        (cx, cy - arm_length),  # South
        (cx - arm_length, cy),  # West
    ]

    edges = [
        (0, 1),  # North arm
        (0, 2),  # East arm
        (0, 3),  # South arm
        (0, 4),  # West arm
    ]

    return make_track_graph(node_positions, edges)


def make_figure8_track(
    loop_radius: float,
    center_spacing: float | None = None,
    n_segments_per_loop: int = 12,
) -> nx.Graph:
    """Create a figure-8 track.

    Two circular loops connected at a central junction point.

    Parameters
    ----------
    loop_radius : float
        Radius of each circular loop.
    center_spacing : float, optional
        Horizontal distance between loop centers (default: 2 * loop_radius).
    n_segments_per_loop : int, optional
        Number of segments per loop (default: 12). More = smoother.

    Returns
    -------
    track_graph : networkx.Graph
        Track graph forming a figure-8 shape.

    Examples
    --------
    >>> track = make_figure8_track(loop_radius=20.0)
    >>> track.number_of_edges() == 2 * 12  # Two loops
    True
    """
    if center_spacing is None:
        center_spacing = 2 * loop_radius

    if n_segments_per_loop < 3:
        raise ValueError("n_segments_per_loop must be >= 3")

    # Left loop center
    left_cx = -center_spacing / 2
    # Right loop center
    right_cx = center_spacing / 2

    # Generate angles for each loop
    angles = np.linspace(0, 2 * np.pi, num=n_segments_per_loop, endpoint=False)

    node_positions = []
    edges = []

    # Left loop
    for angle in angles:
        x = left_cx + loop_radius * np.cos(angle)
        y = loop_radius * np.sin(angle)
        node_positions.append((x, y))

    # Connect left loop
    for i in range(n_segments_per_loop):
        edges.append((i, (i + 1) % n_segments_per_loop))

    # Right loop
    offset = n_segments_per_loop
    for angle in angles:
        x = right_cx + loop_radius * np.cos(angle)
        y = loop_radius * np.sin(angle)
        node_positions.append((x, y))

    # Connect right loop
    for i in range(n_segments_per_loop):
        edges.append((offset + i, offset + (i + 1) % n_segments_per_loop))

    # Connect loops at center (find closest nodes to origin on each loop)
    left_nodes = list(range(n_segments_per_loop))
    right_nodes = list(range(offset, offset + n_segments_per_loop))

    # Find nodes closest to y=0
    left_center_node = min(
        left_nodes, key=lambda i: abs(node_positions[i][1])
    )
    right_center_node = min(
        right_nodes, key=lambda i: abs(node_positions[i][1])
    )

    edges.append((left_center_node, right_center_node))

    return make_track_graph(node_positions, edges)


def make_wtrack(
    width: float,
    height: float,
    center_arm_offset: float | None = None,
) -> nx.Graph:
    """Create a W-shaped track.

    The W-track has two outer vertical arms and one center vertical arm,
    all connected by a horizontal base.

    ```
    Left   Center  Right
     |       |       |
     +-------+-------+
    ```

    Parameters
    ----------
    width : float
        Total width of the W (distance from left to right arm).
    height : float
        Height of the vertical arms.
    center_arm_offset : float, optional
        Vertical offset for center arm bottom (default: 0, aligned with base).

    Returns
    -------
    track_graph : networkx.Graph
        Track graph with W structure.

    Examples
    --------
    >>> track = make_wtrack(width=60.0, height=40.0)
    >>> track.number_of_nodes()
    6
    >>> track.number_of_edges()
    5
    """
    if center_arm_offset is None:
        center_arm_offset = 0.0

    # Define nodes
    node_positions = [
        (0, 0),  # Bottom-left
        (width, 0),  # Bottom-right
        (0, height),  # Top-left
        (width, height),  # Top-right
        (width / 2, height),  # Top-center
        (width / 2, center_arm_offset),  # Bottom-center (junction)
    ]

    edges = [
        (0, 5),  # Left base to center
        (5, 1),  # Center to right base
        (0, 2),  # Left vertical
        (1, 3),  # Right vertical
        (4, 5),  # Center vertical
    ]

    return make_track_graph(node_positions, edges)


def make_rectangular_track(
    width: float,
    height: float,
    start_corner: tuple[float, float] = (0, 0),
) -> nx.Graph:
    """Create a rectangular loop track.

    Parameters
    ----------
    width : float
        Width of the rectangle.
    height : float
        Height of the rectangle.
    start_corner : tuple[float, float], optional
        (x, y) coordinates of bottom-left corner (default: (0, 0)).

    Returns
    -------
    track_graph : networkx.Graph
        Track graph with 4 nodes and 4 edges forming a closed rectangle.

    Examples
    --------
    >>> track = make_rectangular_track(width=50.0, height=30.0)
    >>> track.number_of_nodes()
    4
    >>> track.number_of_edges()
    4
    """
    x0, y0 = start_corner

    node_positions = [
        (x0, y0),  # Bottom-left
        (x0 + width, y0),  # Bottom-right
        (x0 + width, y0 + height),  # Top-right
        (x0, y0 + height),  # Top-left
    ]

    edges = [
        (0, 1),  # Bottom
        (1, 2),  # Right
        (2, 3),  # Top
        (3, 0),  # Left (close loop)
    ]

    return make_track_graph(node_positions, edges)


def make_ymaze_track(
    arm_length: float,
    arm_angle_deg: float = 120.0,
) -> nx.Graph:
    """Create a Y-maze track with three arms.

    The Y-maze has three arms at equal angles (default 120° apart).

    Parameters
    ----------
    arm_length : float
        Length of each arm from center to end.
    arm_angle_deg : float, optional
        Angle between arms in degrees (default: 120° for symmetric Y).

    Returns
    -------
    track_graph : networkx.Graph
        Track graph with 4 nodes (center + 3 arm ends) and 3 edges.

    Examples
    --------
    >>> track = make_ymaze_track(arm_length=40.0)
    >>> track.number_of_nodes()
    4
    >>> track.number_of_edges()
    3
    """
    arm_angle = np.deg2rad(arm_angle_deg)

    # Center at origin
    node_positions = [(0, 0)]

    # Add three arms at equal angles
    for i in range(3):
        angle = i * arm_angle
        x = arm_length * np.cos(angle)
        y = arm_length * np.sin(angle)
        node_positions.append((x, y))

    edges = [
        (0, 1),  # Arm 1
        (0, 2),  # Arm 2
        (0, 3),  # Arm 3
    ]

    return make_track_graph(node_positions, edges)
