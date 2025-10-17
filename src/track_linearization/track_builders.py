"""Helper functions for constructing common track geometries.

This module provides convenient builders for standard experimental track
designs, reducing setup time from hours to minutes.
"""

from pathlib import Path
from typing import Any, TypedDict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from track_linearization.utils import make_track_graph


class TrackBuilderState(TypedDict):
    """State dictionary for interactive track builder."""

    nodes: list[tuple[float, float]]
    edges: list[tuple[int, int]]
    mode: str
    edge_start_node: int | None
    edge_preview_line: Any | None
    finished: bool
    cancelled: bool


def make_linear_track(
    length: float, start_pos: tuple[float, float] = (0, 0)
) -> nx.Graph:
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
    left_center_node = min(left_nodes, key=lambda i: abs(node_positions[i][1]))
    right_center_node = min(right_nodes, key=lambda i: abs(node_positions[i][1]))

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


def make_track_from_points(
    node_positions: list[tuple[float, float]],
    edges: list[tuple[int, int]],
    scale: float = 1.0,
) -> dict[str, Any]:
    """Create a track from pre-defined node positions and edges.

    This is a non-interactive alternative to make_track_from_image_interactive()
    for cases where you already know the node positions and edges.

    Parameters
    ----------
    node_positions : list[tuple[float, float]]
        List of (x, y) coordinates for each node.
    edges : list[tuple[int, int]]
        List of (node_i, node_j) pairs defining edges.
    scale : float, optional
        Scale factor for coordinates (default: 1.0).

    Returns
    -------
    result : dict
        Dictionary with 'track_graph', 'node_positions', 'edges', 'pixel_positions'.

    Examples
    --------
    >>> nodes = [(0, 0), (100, 0), (100, 100)]
    >>> edges = [(0, 1), (1, 2)]
    >>> result = make_track_from_points(nodes, edges)
    >>> track = result['track_graph']
    """
    scaled_positions = [(x * scale, y * scale) for x, y in node_positions]

    try:
        track_graph = make_track_graph(scaled_positions, edges)
        print(
            f"✓ Created track with {len(scaled_positions)} nodes and {len(edges)} edges"
        )
    except Exception as e:
        print(f"✗ Error creating track: {e}")
        track_graph = None

    return {
        "track_graph": track_graph,
        "node_positions": scaled_positions,
        "edges": edges,
        "pixel_positions": node_positions,
    }


def make_track_from_image_interactive(
    image_path: str | Path | None = None,
    image_array: np.ndarray | None = None,
    scale: float = 1.0,
    instructions: bool = True,
) -> dict[str, Any]:
    """Interactively build a track graph by clicking on an image.

    This function displays an image and allows you to:
    1. Click to add nodes
    2. Shift+Click and drag between nodes to create edges
    3. Click on a node while holding Ctrl/Cmd to delete it
    4. Press 'f' to finish and create the track

    Parameters
    ----------
    image_path : str or Path, optional
        Path to image file (PNG, JPG, etc.). Either image_path or image_array
        must be provided.
    image_array : np.ndarray, optional
        Image as numpy array (e.g., from matplotlib.image.imread()).
    scale : float, optional
        Scale factor to convert pixel coordinates to real-world units
        (e.g., cm per pixel). Default: 1.0 (use pixel coordinates).
    instructions : bool, optional
        Whether to display instructions (default: True).

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'track_graph': The created networkx.Graph (None if cancelled)
        - 'node_positions': List of (x, y) positions in scaled coordinates
        - 'edges': List of (node_i, node_j) tuples
        - 'pixel_positions': Original positions in pixel coordinates

    Examples
    --------
    >>> # Interactive mode (in Jupyter notebook)
    >>> result = make_track_from_image_interactive('maze_photo.jpg', scale=0.5)
    >>> track = result['track_graph']

    >>> # With pre-loaded image
    >>> import matplotlib.image as mpimg
    >>> img = mpimg.imread('track.png')
    >>> result = make_track_from_image_interactive(image_array=img, scale=0.1)

    Notes
    -----
    - Works best in Jupyter notebooks with %matplotlib widget or notebook
    - Press 'h' to show help/instructions
    - Press 'f' when finished to create the track
    - Press 'q' to quit without creating a track
    - Node numbers are displayed on the image
    - Edges are shown as blue lines
    - A mode indicator shows current action (ADD NODE / CREATE EDGE / DELETE)
    """
    # Load image
    if image_path is not None:
        try:
            img = plt.imread(str(image_path))
        except Exception as e:
            raise ValueError(f"Failed to load image from {image_path}: {e}") from e
    elif image_array is not None:
        img = image_array
    else:
        raise ValueError("Either image_path or image_array must be provided")

    # Detect if we're in Jupyter early
    try:
        from IPython import get_ipython

        ipython = get_ipython()
        is_jupyter = ipython is not None and "IPKernelApp" in ipython.config
    except Exception:
        is_jupyter = False

    # State variables
    state: TrackBuilderState = {
        "nodes": [],  # List of (x, y) pixel coordinates
        "edges": [],  # List of (i, j) node index pairs
        "edge_start_node": None,  # For edge creation
        "edge_preview_line": None,  # Preview line while dragging
        "finished": False,
        "cancelled": False,
        "mode": "ADD",  # 'ADD', 'EDGE', 'DELETE'
    }

    # Create figure
    fig = plt.figure(figsize=(12, 11))

    # Main image axes
    ax = fig.add_axes((0.05, 0.15, 0.9, 0.8))
    ax.imshow(img, origin="upper")

    # Title with mode indicator
    ax.text(
        0.5,
        1.02,
        "Interactive Track Builder",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        ha="center",
    )
    mode_text = ax.text(
        0.5,
        0.98,
        "Mode: ADD NODE (Click to add)",
        transform=ax.transAxes,
        fontsize=11,
        ha="center",
        style="italic",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "lightgreen", "alpha": 0.7},
    )

    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")

    # Add buttons at bottom (only for non-Jupyter environments)
    if not is_jupyter:
        from matplotlib.widgets import Button

        ax_finish = fig.add_axes((0.7, 0.02, 0.1, 0.05))
        btn_finish = Button(ax_finish, "Finish", color="lightgreen", hovercolor="green")

        ax_cancel = fig.add_axes((0.82, 0.02, 0.1, 0.05))
        btn_cancel = Button(ax_cancel, "Cancel", color="lightcoral", hovercolor="red")

        def on_finish_click(event: Any) -> None:
            state["finished"] = True
            print("\n✓ Finished! (via button) Creating track graph...")
            plt.close(fig)

        def on_cancel_click(event: Any) -> None:
            state["cancelled"] = True
            print("\n✗ Cancelled (via button).")
            plt.close(fig)

        btn_finish.on_clicked(on_finish_click)
        btn_cancel.on_clicked(on_cancel_click)

    # Storage for plot elements
    node_scatter = ax.scatter(
        [], [], c="red", s=100, zorder=10, marker="o", edgecolors="white", linewidths=2
    )
    node_labels: list[Any] = []
    edge_lines: list[Any] = []

    def update_mode_display() -> None:
        """Update mode indicator text and color."""
        if state["mode"] == "ADD":
            mode_text.set_text("Mode: ADD NODE (Click anywhere)")
            mode_text.set_bbox(
                {"boxstyle": "round,pad=0.5", "facecolor": "lightgreen", "alpha": 0.7}
            )
        elif state["mode"] == "EDGE":
            mode_text.set_text("Mode: CREATE EDGE (Click & drag between nodes)")
            mode_text.set_bbox(
                {"boxstyle": "round,pad=0.5", "facecolor": "lightblue", "alpha": 0.7}
            )
        elif state["mode"] == "DELETE":
            mode_text.set_text("Mode: DELETE NODE (Click on node)")
            mode_text.set_bbox(
                {"boxstyle": "round,pad=0.5", "facecolor": "salmon", "alpha": 0.7}
            )

    def update_display() -> None:
        """Redraw nodes and edges."""
        # Update nodes
        if state["nodes"]:
            xs, ys = zip(*state["nodes"], strict=False)
            node_scatter.set_offsets(np.c_[xs, ys])
        else:
            node_scatter.set_offsets(np.empty((0, 2)))

        # Clear old labels
        for label in node_labels:
            label.remove()
        node_labels.clear()

        # Add node labels
        for i, (x, y) in enumerate(state["nodes"]):
            label = ax.text(
                x,
                y + 15,
                str(i),
                color="white",
                fontsize=10,
                ha="center",
                va="bottom",
                fontweight="bold",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "red", "alpha": 0.7},
            )
            node_labels.append(label)

        # Clear old edges
        for line in edge_lines:
            line.remove()
        edge_lines.clear()

        # Draw edges
        for i, j in state["edges"]:
            if i < len(state["nodes"]) and j < len(state["nodes"]):
                x1, y1 = state["nodes"][i]
                x2, y2 = state["nodes"][j]
                (line,) = ax.plot(
                    [x1, x2], [y1, y2], "b-", linewidth=3, alpha=0.6, zorder=5
                )
                edge_lines.append(line)

        # Highlight node being used for edge
        if state["edge_start_node"] is not None and state["edge_start_node"] < len(
            state["nodes"]
        ):
            x, y = state["nodes"][state["edge_start_node"]]
            circle = plt.Circle(
                (x, y), 20, color="yellow", fill=False, linewidth=3, zorder=11
            )
            ax.add_patch(circle)
            edge_lines.append(circle)

        update_mode_display()
        fig.canvas.draw_idle()

    def find_closest_node(x: float, y: float, max_distance: float = 25) -> int | None:
        """Find closest node to coordinates within max_distance pixels."""
        if not state["nodes"]:
            return None
        distances = [
            np.sqrt((x - nx) ** 2 + (y - ny) ** 2) for nx, ny in state["nodes"]
        ]
        closest_idx = int(np.argmin(distances))
        if distances[closest_idx] < max_distance:
            return closest_idx
        return None

    def on_press(event: Any) -> None:
        """Handle mouse press."""
        if event.inaxes != ax:
            return

        if state["finished"] or state["cancelled"]:
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        # Check which mode we're in
        if state["mode"] == "ADD":
            # Add a new node
            state["nodes"].append((x, y))
            print(f"✓ Added node {len(state['nodes'])-1} at ({x:.1f}, {y:.1f})")
            update_display()

        elif state["mode"] == "EDGE":
            # Start edge creation by finding nearest node
            closest = find_closest_node(x, y)
            if closest is not None:
                state["edge_start_node"] = closest
                print(f"• Starting edge from node {closest}...")
                update_display()

        elif state["mode"] == "DELETE":
            # Delete closest node
            closest = find_closest_node(x, y)
            if closest is not None:
                state["nodes"].pop(closest)
                print(f"✗ Deleted node {closest}")

                # Remove and renumber edges
                state["edges"] = [
                    (i if i < closest else i - 1, j if j < closest else j - 1)
                    for i, j in state["edges"]
                    if i != closest and j != closest
                ]
                state["edges"] = [(min(i, j), max(i, j)) for i, j in state["edges"]]
                update_display()

    def on_motion(event: Any) -> None:
        """Handle mouse motion for edge preview."""
        # Note: Don't print debug here - too spammy during mouse movement
        if event.inaxes != ax or state["finished"] or state["cancelled"]:
            return
        if state["mode"] != "EDGE" or state["edge_start_node"] is None:
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        # Remove old preview line
        if state["edge_preview_line"] is not None:
            state["edge_preview_line"].remove()
            state["edge_preview_line"] = None

        # Draw preview line from start node to cursor
        x1, y1 = state["nodes"][state["edge_start_node"]]
        (line,) = ax.plot([x1, x], [y1, y], "y--", linewidth=2, alpha=0.7, zorder=6)
        state["edge_preview_line"] = line
        fig.canvas.draw_idle()

    def on_release(event: Any) -> None:
        """Handle mouse release."""
        if event.inaxes != ax:
            return

        if state["finished"] or state["cancelled"]:
            return

        if state["mode"] != "EDGE" or state["edge_start_node"] is None:
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            state["edge_start_node"] = None
            if state["edge_preview_line"] is not None:
                state["edge_preview_line"].remove()
                state["edge_preview_line"] = None
            update_display()
            return

        # Find end node
        closest = find_closest_node(x, y)
        if closest is not None and closest != state["edge_start_node"]:
            # Create edge
            node1, node2 = state["edge_start_node"], closest
            edge = (min(node1, node2), max(node1, node2))
            if edge not in state["edges"]:
                state["edges"].append(edge)
                print(f"✓ Created edge: {node1} ↔ {node2}")
            else:
                print(f"⚠ Edge {node1} ↔ {node2} already exists")

        # Clean up
        state["edge_start_node"] = None
        if state["edge_preview_line"] is not None:
            state["edge_preview_line"].remove()
            state["edge_preview_line"] = None
        update_display()

    def on_key(event: Any) -> None:
        """Handle key presses."""

        if event.key == "f":  # Finish
            state["finished"] = True
            print("\n✓ Finished! Creating track graph...")
            print("   Closing window...")
            plt.close(fig)
            print("   Window closed.")
        elif event.key == "q":  # Quit
            state["cancelled"] = True
            print("\n✗ Cancelled.")
            plt.close(fig)
        elif event.key == "h":  # Help
            print_instructions()
        elif event.key == "a":  # Switch to ADD mode
            state["mode"] = "ADD"
            state["edge_start_node"] = None
            if state["edge_preview_line"] is not None:
                state["edge_preview_line"].remove()
                state["edge_preview_line"] = None
            print("→ Switched to ADD NODE mode")
            update_display()
        elif event.key == "e":  # Switch to EDGE mode
            state["mode"] = "EDGE"
            print("→ Switched to CREATE EDGE mode (click and drag between nodes)")
            update_display()
        elif event.key == "x":  # Switch to DELETE mode
            state["mode"] = "DELETE"
            state["edge_start_node"] = None
            if state["edge_preview_line"] is not None:
                state["edge_preview_line"].remove()
                state["edge_preview_line"] = None
            print("→ Switched to DELETE NODE mode")
            update_display()
        elif event.key == "u":  # Undo last node
            if state["nodes"]:
                state["nodes"].pop()
                print(f"↶ Undid last node. {len(state['nodes'])} nodes remaining.")
                update_display()
        elif event.key == "backspace" or event.key == "d":  # Delete last edge
            if state["edges"]:
                edge = state["edges"].pop()
                print(f"↶ Deleted last edge {edge}")
                update_display()

    def print_instructions() -> None:
        """Print usage instructions."""
        print("\n" + "=" * 70)
        print("INTERACTIVE TRACK BUILDER - CONTROLS")
        print("=" * 70)

        if not is_jupyter:
            print("BUTTONS (at bottom of window):")
            print("  [Finish] button:  Finish and create track graph")
            print("  [Cancel] button:  Quit without creating track")
            print()

        print("MODES (press key to switch):")
        print("  'a' key:        ADD NODE mode - Click anywhere to add node")
        print("  'e' key:        CREATE EDGE mode - Click & drag between nodes")
        print("  'x' key:        DELETE NODE mode - Click on node to delete")
        print()
        print("ACTIONS:")
        print("  Click:          Action depends on current mode (see above)")
        print("  Click & Drag:   In EDGE mode, creates edge between nodes")
        print()
        print("KEYBOARD SHORTCUTS:")
        if not is_jupyter:
            print("  'f' key:        Finish and create track graph")
            print("  'q' key:        Quit without creating track")
        print("  'u' key:        Undo last node")
        print("  'Backspace':    Delete last edge")
        print("  'h' key:        Show this help")
        print()
        print("VISUAL CUES:")
        print("  Green banner:   ADD NODE mode")
        print("  Blue banner:    CREATE EDGE mode")
        print("  Red banner:     DELETE NODE mode")
        print("  Yellow circle:  Selected node during edge creation")
        print("  Yellow dashed:  Edge preview while dragging")
        print()

        if is_jupyter:
            print("📌 JUPYTER: Build your track, then retrieve results in a new cell")
            print("   (See instructions below after running this cell)")
        else:
            print("⚠️  If keyboard doesn't work, use [Finish] button at bottom!")

        print("=" * 70 + "\n")

    # Connect event handlers
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("key_press_event", on_key)

    # Show instructions
    if instructions:
        print_instructions()

    # Initial mode display
    update_mode_display()

    # Show plot
    plt.tight_layout()

    # Show the plot
    if is_jupyter:
        # JUPYTER MODE: Non-blocking approach
        # The widget is already interactive, just display it and return immediately
        # Store state in figure so user can retrieve it later
        fig._track_builder_state = state  # type: ignore[attr-defined]

        plt.show()
        print()
        print("=" * 70)
        print("🎯 JUPYTER MODE - INTERACTIVE BUILDER")
        print("=" * 70)
        print("The plot is displayed above. It is ALREADY INTERACTIVE!")
        print()
        print("To build your track:")
        print("  1. Click on the image to add nodes (red dots appear)")
        print("  2. Press 'e' then click & drag to create edges (blue lines)")
        print("  3. Press 'a' to return to ADD mode")
        print()
        print("To get your results, run this in a NEW cell:")
        print()
        print("  import matplotlib.pyplot as plt")
        print("  from track_linearization import _build_track_from_state")
        print("  state = plt.gcf()._track_builder_state")
        print("  result = _build_track_from_state(state, scale=" + str(scale) + ")")
        print("  print(result)")
        print()
        print("=" * 70)
        print()

        # Return a partial result with instructions
        return {
            "track_graph": None,
            "node_positions": [],
            "edges": [],
            "pixel_positions": [],
            "_jupyter_note": "In Jupyter: Interact with plot above, then retrieve results using _build_track_from_state()",
            "_state_location": "Access state via: plt.gcf()._track_builder_state",
        }

    else:
        # NON-JUPYTER: Regular blocking mode
        try:
            plt.show(block=True)  # Block until window is closed
        except Exception as e:
            print(f"Warning: matplotlib show() error: {e}")
            print("Attempting to continue...")

    # Print status after window closes
    print("\n" + "=" * 70)
    if state["finished"]:
        print("✓ Window closed via Finish command")
    elif state["cancelled"]:
        print("✗ Window closed via Cancel command")
    else:
        print("✓ Window closed manually (that's fine!)")
    print("=" * 70 + "\n")

    # Use the helper function to build results
    return _build_track_from_state(state, scale)


def _build_track_from_state(
    state: TrackBuilderState | dict[str, Any], scale: float = 1.0
) -> dict[str, Any]:
    """
    Helper function to build track graph from interactive builder state.

    This is used internally and can also be called by users in Jupyter
    to retrieve results after interacting with the plot.

    Parameters
    ----------
    state : dict
        State dictionary from make_track_from_image_interactive containing:
        - 'nodes': List of (x, y) pixel coordinates
        - 'edges': List of (i, j) node index pairs
        - 'cancelled': Whether user cancelled
    scale : float, optional
        Scale factor to convert pixel coordinates to real units (default 1.0)

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'track_graph': NetworkX graph or None
        - 'node_positions': List of scaled (x, y) coordinates
        - 'edges': List of (i, j) edge pairs
        - 'pixel_positions': Original pixel coordinates
    """
    # Handle cancelled or empty state
    if state.get("cancelled", False) or not state.get("nodes", []):
        return {
            "track_graph": None,
            "node_positions": [],
            "edges": [],
            "pixel_positions": [],
        }

    # Scale coordinates
    pixel_positions = state["nodes"].copy()
    scaled_positions = [(x * scale, y * scale) for x, y in state["nodes"]]

    # Create track graph
    try:
        track_graph = make_track_graph(scaled_positions, state["edges"])
        print(
            f"\n✓ Created track with {len(scaled_positions)} nodes and {len(state['edges'])} edges"
        )
        print(f"   Scale factor: {scale} (units per pixel)")
    except Exception as e:
        print(f"\n✗ Error creating track graph: {e}")
        track_graph = None

    return {
        "track_graph": track_graph,
        "node_positions": scaled_positions,
        "edges": state["edges"],
        "pixel_positions": pixel_positions,
    }
