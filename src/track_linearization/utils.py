from collections.abc import Sequence
from typing import Any

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from track_linearization.core import (
    get_track_segments_from_graph,
    project_points_to_segment,
)


Edge = tuple[Any, Any]


def make_track_graph(
    node_positions: np.ndarray | Sequence[tuple[float, ...]],
    edges: np.ndarray | Sequence[tuple[int, int]],
) -> nx.Graph:
    """Constructs a graph representation of a 2D track.

    This is the recommended way to create track graphs for linearization.
    It automatically computes edge distances and assigns edge IDs, ensuring
    the graph has all required attributes for get_linearized_position().

    Node positions determine the name of the node by order in array.
    Edges determine the connections between the nodes by name.

    Parameters
    ----------
    node_positions : numpy.ndarray or list, shape (n_nodes, 2)
        Coordinates of each node in 2D space. Can be a list of tuples
        [(x1, y1), (x2, y2), ...] or a numpy array.
        Node IDs are assigned as indices 0, 1, 2, ...
    edges : numpy.ndarray or list, shape (n_edges, 2)
        Pairs of node IDs defining connections. Each edge is [node1_id, node2_id].

    Returns
    -------
    track_graph : networkx.Graph
        A NetworkX graph with:
        - Node attribute 'pos': (x, y) coordinates
        - Edge attribute 'distance': Euclidean length
        - Edge attribute 'edge_id': unique integer identifier

    Examples
    --------
    Create a simple L-shaped track:

    >>> from track_linearization import make_track_graph
    >>> node_positions = [(0, 0), (10, 0), (10, 10)]
    >>> edges = [(0, 1), (1, 2)]
    >>> track_graph = make_track_graph(node_positions, edges)
    >>> print(track_graph.nodes[0]['pos'])
    (0, 0)
    >>> print(track_graph.edges[(0, 1)]['distance'])
    10.0

    Create a Y-shaped branching track:

    >>> import numpy as np
    >>> node_positions = np.array([
    ...     [0, 0],    # Node 0: base
    ...     [10, 0],   # Node 1: junction
    ...     [15, 5],   # Node 2: upper branch
    ...     [15, -5]   # Node 3: lower branch
    ... ])
    >>> edges = [(0, 1), (1, 2), (1, 3)]
    >>> track_graph = make_track_graph(node_positions, edges)
    >>> len(track_graph.edges)
    3

    See Also
    --------
    get_linearized_position : Use the track graph for linearization
    plot_track_graph : Visualize the track structure
    infer_edge_layout : Automatically determine edge ordering

    """
    track_graph = nx.Graph()

    for node_id, node_position in enumerate(node_positions):
        track_graph.add_node(node_id, pos=tuple(node_position))

    for node1, node2 in edges:
        pos1 = np.asarray(track_graph.nodes[node1]["pos"])
        pos2 = np.asarray(track_graph.nodes[node2]["pos"])
        distance = np.linalg.norm(pos1 - pos2)
        track_graph.add_edge(node1, node2, distance=distance)

    for edge_id, edge in enumerate(track_graph.edges):
        track_graph.edges[edge]["edge_id"] = edge_id

    return track_graph


def plot_track_graph(
    track_graph: nx.Graph,
    ax: plt.Axes | None = None,
    draw_edge_labels: bool = False,
    show: bool = False,
    figsize: tuple[float, float] = (8, 8),
    **kwds: Any,
) -> plt.Axes:
    """Plot a 2D visualization of the track graph.

    Creates a spatial plot showing nodes as circles with their IDs labeled,
    and edges as lines connecting them. This is useful for visualizing the
    track structure before linearization.

    Parameters
    ----------
    track_graph : networkx.Graph
        The track graph to plot. Must have 'pos' attributes on nodes.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes to plot on. If None, creates a new figure and axes.
    draw_edge_labels : bool, optional
        If True, displays edge IDs at the midpoint of each edge. Useful for
        understanding which edges correspond to which segments. Default is False.
    show : bool, optional
        If True, calls plt.show() to display the plot immediately. Default is False.
    figsize : tuple of float, optional
        Figure size (width, height) in inches if creating a new figure.
        Only used when ax=None. Default is (8, 8).
    **kwds
        Additional keyword arguments passed to nx.draw_networkx_nodes() for
        customizing node appearance (e.g., node_size, node_color).

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object containing the plot, allowing for further customization.

    Examples
    --------
    Basic usage:

    >>> from track_linearization import make_track_graph, plot_track_graph
    >>> node_positions = [(0, 0), (10, 0), (10, 10), (0, 10)]
    >>> edges = [(0, 1), (1, 2), (2, 3)]
    >>> track_graph = make_track_graph(node_positions, edges)
    >>> plot_track_graph(track_graph, show=True)

    Customize appearance and add edge labels:

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(figsize=(10, 10))
    >>> plot_track_graph(
    ...     track_graph,
    ...     ax=ax,
    ...     draw_edge_labels=True,
    ...     node_size=500,
    ...     node_color='lightblue'
    ... )
    >>> ax.set_title('My Track Layout')
    >>> plt.show()

    See Also
    --------
    plot_graph_as_1D : Visualize the 1D linearized representation
    make_track_graph : Create a track graph from positions and edges
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    node_position = nx.get_node_attributes(track_graph, "pos")
    nx.draw_networkx_nodes(track_graph, node_position, ax=ax, **kwds)
    for node_id, pos in node_position.items():
        ax.text(
            pos[0],
            pos[1],
            str(node_id),
            fontsize=8,
            ha="center",
            va="center",
        )
    for node_id1, node_id2 in track_graph.edges:
        pos = np.stack((node_position[node_id1], node_position[node_id2]))
        ax.plot(pos[:, 0], pos[:, 1], color="black", zorder=-1)

    if draw_edge_labels:
        edge_ids = {edge: ind for ind, edge in enumerate(track_graph.edges)}
        nx.draw_networkx_edge_labels(
            track_graph, node_position, edge_labels=edge_ids, ax=ax
        )

    if show:
        plt.show()

    return ax


def _plot_linear_segment(
    ax: plt.Axes,
    start_pos: float,
    end_pos: float,
    other_pos: float,
    edge: Edge,
    edge_id: int,
    axis: str,
    node_size: int,
    node_color: str,
    draw_edge_labels: bool,
) -> None:
    """Helper function to plot a single segment in 1D layout.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    start_pos : float
        Starting position along the main axis.
    end_pos : float
        Ending position along the main axis.
    other_pos : float
        Position on the perpendicular axis.
    edge : tuple
        Edge tuple (node1, node2).
    edge_id : int
        Edge identifier for labeling.
    axis : str
        'x' for horizontal or 'y' for vertical.
    node_size : int
        Size of node markers.
    node_color : str
        Color for node markers.
    draw_edge_labels : bool
        Whether to draw edge ID labels.
    """
    # Determine coordinate order based on axis
    if axis == "x":
        start_coords = (start_pos, other_pos)
        end_coords = (end_pos, other_pos)
        line_x = (start_pos, end_pos)
        line_y = (other_pos, other_pos)
    else:  # axis == "y"
        start_coords = (other_pos, start_pos)
        end_coords = (other_pos, end_pos)
        line_x = (other_pos, other_pos)
        line_y = (start_pos, end_pos)

    # Plot start and end nodes
    ax.scatter(*start_coords, zorder=8, s=node_size, clip_on=False, color=node_color)
    ax.scatter(*end_coords, zorder=8, s=node_size, clip_on=False, color=node_color)

    # Plot line connecting nodes
    ax.plot(line_x, line_y, color="black", clip_on=False, zorder=7)

    # Draw edge label if requested
    if draw_edge_labels:
        edge_midpoint = start_pos + (end_pos - start_pos) / 2
        if axis == "x":
            label_coords = (edge_midpoint, other_pos)
        else:
            label_coords = (other_pos, edge_midpoint)

        ax.scatter(*label_coords, color="white", zorder=9, s=node_size, clip_on=False)
        ax.text(*label_coords, str(edge_id), ha="center", va="center", zorder=10)

    # Add node labels
    ax.text(*start_coords, edge[0], ha="center", va="center", zorder=10)
    ax.text(*end_coords, edge[1], ha="center", va="center", zorder=10)


def plot_graph_as_1D(
    track_graph: nx.Graph,
    edge_order: list[Edge] | None = None,
    edge_spacing: float | Sequence[float] = 0,
    ax: plt.Axes | None = None,
    axis: str = "x",
    other_axis_start: float = 0.0,
    draw_edge_labels: bool = False,
    node_size: int = 300,
    node_color: str = "#1f77b4",
    show: bool = False,
    figsize: tuple[float, float] | None = None,
) -> plt.Axes:
    """Plot the track graph as a 1D linear representation.

    Visualizes how the 2D track will appear after linearization, showing
    segments laid out in a line with spacing between them. This helps verify
    that edge_order and edge_spacing produce the desired layout.

    Parameters
    ----------
    track_graph : nx.Graph
        The track graph to visualize.
    edge_order : list of 2-tuples, optional
        Order of edges in the linearization. If None, uses the order edges
        were added to the graph. Same parameter as get_linearized_position().
    edge_spacing : float or list of float, optional
        Spacing between consecutive edges. Same parameter as get_linearized_position().
        Default is 0 (no spacing).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure and axes.
    axis : str, optional
        Orientation: 'x' for horizontal (default) or 'y' for vertical layout.
    other_axis_start : float, optional
        Position on the perpendicular axis. Default is 0.0.
    draw_edge_labels : bool, optional
        If True, shows edge IDs at segment midpoints. Default is False.
    node_size : int, optional
        Size of node markers in points. Default is 300.
    node_color : str, optional
        Color for node markers. Default is "#1f77b4" (matplotlib blue).
    show : bool, optional
        If True, calls plt.show() to display immediately. Default is False.
    figsize : tuple of float, optional
        Figure size (width, height) in inches if creating new figure.
        If None, uses (7, 1) for horizontal or (1, 7) for vertical.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object containing the plot.

    Examples
    --------
    Visualize a track's linearization layout:

    >>> from track_linearization import make_track_graph, plot_graph_as_1D
    >>> node_positions = [(0, 0), (10, 0), (10, 10), (0, 10)]
    >>> edges = [(0, 1), (1, 2), (2, 3)]
    >>> track_graph = make_track_graph(node_positions, edges)
    >>>
    >>> # Show the default linearization
    >>> plot_graph_as_1D(track_graph, show=True)

    Compare different layouts:

    >>> import matplotlib.pyplot as plt
    >>> fig, axes = plt.subplots(2, 1, figsize=(10, 3))
    >>>
    >>> # Layout 1: default order, no spacing
    >>> plot_graph_as_1D(track_graph, ax=axes[0])
    >>> axes[0].set_title('No spacing')
    >>>
    >>> # Layout 2: same order, with spacing
    >>> plot_graph_as_1D(track_graph, edge_spacing=5, ax=axes[1])
    >>> axes[1].set_title('5-unit spacing')
    >>>
    >>> plt.tight_layout()
    >>> plt.show()

    Vertical orientation with edge labels:

    >>> plot_graph_as_1D(
    ...     track_graph,
    ...     axis='y',
    ...     draw_edge_labels=True,
    ...     show=True
    ... )

    See Also
    --------
    plot_track_graph : Visualize the 2D track structure
    get_linearized_position : Main linearization function
    infer_edge_layout : Automatically determine edge_order and spacing
    """
    if ax is None:
        if figsize is None:
            figsize = (7, 1) if axis == "x" else (1, 7)
        _, ax = plt.subplots(figsize=figsize)

    # If no edge_order is given, then arange edges in the order passed to
    # construct the track graph
    if edge_order is None:
        edge_order = list(track_graph.edges)

    n_edges = len(edge_order)
    if isinstance(edge_spacing, (int, float)):
        edge_spacing = [
            edge_spacing,
        ] * (n_edges - 1)

    # Plot all segments using helper function
    start_node_linear_position = 0.0
    end_node_linear_position = 0.0

    for ind, edge in enumerate(edge_order):
        end_node_linear_position = (
            start_node_linear_position + track_graph.edges[edge]["distance"]
        )

        _plot_linear_segment(
            ax=ax,
            start_pos=start_node_linear_position,
            end_pos=end_node_linear_position,
            other_pos=other_axis_start,
            edge=edge,
            edge_id=track_graph.edges[edge]["edge_id"],
            axis=axis,
            node_size=node_size,
            node_color=node_color,
            draw_edge_labels=draw_edge_labels,
        )

        try:
            start_node_linear_position += (
                track_graph.edges[edge]["distance"] + edge_spacing[ind]
            )
        except IndexError:
            pass

    # Configure axis-specific styling
    if axis == "x":
        ax.set_xlim((other_axis_start, end_node_linear_position))
        ax.set_xlabel("Linear Position [cm]")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_yticks([])
    elif axis == "y":
        ax.set_ylabel("Linear Position [cm]")

    if show:
        plt.show()

    return ax


def _get_projected_track_position(
    track_graph: nx.Graph, track_segment_id: np.ndarray, position: np.ndarray
) -> np.ndarray:
    track_segment_id[np.isnan(track_segment_id)] = 0
    track_segment_id = track_segment_id.astype(int)

    track_segments = get_track_segments_from_graph(track_graph)
    projected_track_position = project_points_to_segment(track_segments, position)
    n_time = projected_track_position.shape[0]
    result: np.ndarray = projected_track_position[(np.arange(n_time), track_segment_id)]
    return result


def make_actual_vs_linearized_position_movie(
    track_graph: nx.Graph,
    position_df: pd.DataFrame,
    time_slice: slice | None = None,
    movie_name: str = "actual_vs_linearized",
    frame_rate: float = 33,
) -> None:
    """

    Parameters
    ----------
    track_graph : networkx.Graph
    position_df : pandas.DataFrame
    time_slice : slice or None, optional
    movie_name : str, optional
    frame_rate : float, optional
        Frames per second.
    """

    all_position = position_df.loc[:, ["x_position", "y_position"]].to_numpy()
    all_linear_position = position_df.linear_position.to_numpy()
    all_time = position_df.index.to_numpy() / np.timedelta64(1, "s")

    if time_slice is None:
        position = all_position
        track_segment_id = position_df.track_segment_id.to_numpy()
        linear_position = all_linear_position
        time = all_time
    else:
        position = all_position[time_slice]
        track_segment_id = position_df.iloc[time_slice].track_segment_id.to_numpy()
        linear_position = all_linear_position[time_slice]
        time = all_time[time_slice]

    projected_track_position = _get_projected_track_position(
        track_graph, track_segment_id, position
    )

    # Set up formatting for the movie files
    Writer = animation.writers["ffmpeg"]
    writer = Writer(fps=15, metadata={"artist": "Me"}, bitrate=1800)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(21, 7),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [2, 1]},
    )

    # Subplot 1
    axes[0].scatter(all_time, all_linear_position, color="lightgrey", zorder=0, s=10)
    axes[0].set_xlim((all_time.min(), all_time.max()))
    axes[0].set_ylim((all_linear_position.min(), all_linear_position.max()))
    linear_head = axes[0].scatter([], [], s=100, zorder=101, color="b")

    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Position [cm]")
    axes[0].set_title("Linearized Position")

    axes[1].plot(all_position[:, 0], all_position[:, 1], color="lightgrey", zorder=-10)
    plot_track_graph(track_graph, ax=axes[1])
    plt.axis("off")

    # Subplot 2
    axes[1].set_xlim(all_position[:, 0].min() - 10, all_position[:, 0].max() + 10)
    axes[1].set_ylim(all_position[:, 1].min() - 10, all_position[:, 1].max() + 10)

    (actual_line,) = axes[1].plot(
        [], [], "g-", label="actual position", linewidth=3, zorder=101
    )
    actual_head = axes[1].scatter([], [], s=80, zorder=101, color="g")

    (predicted_line,) = axes[1].plot(
        [], [], "b-", label="linearized position", linewidth=3, zorder=102
    )
    predicted_head = axes[1].scatter([], [], s=80, zorder=102, color="b")

    axes[1].legend()
    axes[1].set_xlabel("x-position")
    axes[1].set_ylabel("y-position")
    axes[1].set_title("Linearized vs. Actual Position")

    def _update_plot(time_ind: int) -> tuple[Any, ...]:
        start_ind = max(0, time_ind - 33)
        time_slice = slice(start_ind, time_ind)

        linear_head.set_offsets(np.array((time[time_ind], linear_position[time_ind])))

        actual_line.set_data(position[time_slice, 0], position[time_slice, 1])
        actual_head.set_offsets(position[time_ind])

        predicted_line.set_data(
            projected_track_position[time_slice, 0],
            projected_track_position[time_slice, 1],
        )
        predicted_head.set_offsets(projected_track_position[time_ind])

        return actual_line, predicted_line

    n_time = position.shape[0]
    line_ani = animation.FuncAnimation(
        fig, _update_plot, frames=n_time, interval=1000 / frame_rate, blit=True
    )
    line_ani.save(movie_name + ".mp4", writer=writer)


def infer_edge_layout(
    track_graph: nx.Graph,
    start_node: object = None,
    spacing_between_unconnected_components: float = 15.0,
) -> tuple[list, np.ndarray]:
    """Automatically determine edge order and spacing for track linearization.

    Uses a depth-first search starting from `start_node` to traverse the track graph
    and create a sensible 1D layout. When consecutive edges in the layout are not
    directly connected in the graph, a gap is inserted between them.

    This is useful when you want to linearize a track but don't want to manually
    specify the edge order. The algorithm attempts to create a spatially coherent
    layout by following connected paths.

    Note: All spatial quantities use the same units as the track graph coordinates
    (commonly centimeters in neuroscience applications).

    Parameters
    ----------
    track_graph : nx.Graph
        The track graph to linearize. Must have been created with make_track_graph()
        or have properly formatted edge attributes.
    start_node : object, optional
        Node to start the traversal from. If None, starts from an arbitrary node.
        Choosing different start nodes can produce different layouts for graphs
        with multiple components or complex connectivity.
    spacing_between_unconnected_components : float, optional
        Gap size (in same units as graph coordinates) to insert between segments
        that are not directly connected. Default is 15.0.

    Returns
    -------
    edge_order : list of tuples
        Ordered list of edges (node1, node2) defining the linearization path.
        Can be passed directly to get_linearized_position().
    edge_spacing : np.ndarray
        Array of spacing values between consecutive edges. Length is len(edge_order) - 1.
        Values are 0.0 for connected edges, spacing_between_unconnected_components otherwise.

    Examples
    --------
    Automatically layout a track and use it for linearization:

    >>> from track_linearization import make_track_graph, infer_edge_layout, get_linearized_position
    >>> import numpy as np
    >>>
    >>> # Create a branching track (Y-shape)
    >>> node_positions = [(0, 0), (10, 0), (15, 5), (15, -5)]
    >>> edges = [(0, 1), (1, 2), (1, 3)]
    >>> track_graph = make_track_graph(node_positions, edges)
    >>>
    >>> # Automatically determine layout starting from node 0
    >>> edge_order, edge_spacing = infer_edge_layout(track_graph, start_node=0)
    >>> print(edge_order)
    [(0, 1), (1, 2), (1, 3)]
    >>>
    >>> # Use the inferred layout for linearization
    >>> position = np.array([[5, 0], [12, 3], [12, -3]])
    >>> result = get_linearized_position(
    ...     position, track_graph,
    ...     edge_order=edge_order,
    ...     edge_spacing=edge_spacing
    ... )

    Compare different start nodes:

    >>> order1, spacing1 = infer_edge_layout(track_graph, start_node=0)
    >>> order2, spacing2 = infer_edge_layout(track_graph, start_node=2)
    >>> # Different start nodes may produce different but equally valid layouts

    See Also
    --------
    get_linearized_position : Main function for linearizing positions
    make_track_graph : Create a properly formatted track graph
    """
    linear_edge_order = list(nx.traversal.edge_bfs(track_graph, source=start_node))
    is_connected_component = ~(
        np.abs(np.array(linear_edge_order)[:-1, 1] - np.array(linear_edge_order)[1:, 0])
        > 0
    )
    linear_edge_spacing = (
        ~is_connected_component * spacing_between_unconnected_components
    )

    return linear_edge_order, linear_edge_spacing
