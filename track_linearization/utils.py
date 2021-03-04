import matplotlib.animation as animation
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from track_linearization.core import (get_track_segments_from_graph,
                                      project_points_to_segment)


def make_track_graph(node_positions, edges):
    '''Constructs a graph representation of a 2D track.

    Node positions determine the name of the node by order in array.
    Edges determine the connections between the nodes by name.

    Parameters
    ----------
    node_positions : numpy.ndarray, shape (n_nodes, 2)
    edges : numpy.ndarray, shape (n_edges, 2)

    Returns
    -------
    track_graph : networkx.Graph

    '''
    track_graph = nx.Graph()

    for node_id, node_position in enumerate(node_positions):
        track_graph.add_node(node_id, pos=tuple(node_position))

    for (node1, node2) in edges:
        pos1 = np.asarray(track_graph.nodes[node1]["pos"])
        pos2 = np.asarray(track_graph.nodes[node2]["pos"])
        distance = np.linalg.norm(pos1 - pos2)
        track_graph.add_edge(node1, node2, distance=distance)

    for edge_id, edge in enumerate(track_graph.edges):
        track_graph.edges[edge]["edge_id"] = edge_id

    return track_graph


def plot_track_graph(track_graph, ax=None, draw_edge_labels=False, **kwds):
    '''

    Parameters
    ----------
    track_graph : networkx.Graph
    ax : matplotlib axis, optional
    draw_edge_labels : bool, optional
        Plots the names of the edges
    kwds : additional plotting keyworks for `draw_networkx`

    '''
    if ax is None:
        ax = plt.gca()
    node_position = nx.get_node_attributes(track_graph, 'pos')
    nx.draw_networkx(track_graph, node_position, ax=ax, **kwds)

    if draw_edge_labels:
        edge_ids = {edge: ind for ind, edge in enumerate(track_graph.edges)}
        nx.draw_networkx_edge_labels(track_graph, node_position,
                                     edge_labels=edge_ids, ax=ax)


def plot_graph_as_1D(track_graph, edge_order=None, edge_spacing=0,
                     ax=None, axis="x", other_axis_start=0.0,
                     draw_edge_labels=False, node_size=300,
                     node_color="#1f77b4"):

    if ax is None:
        ax = plt.gca()
    # If no edge_order is given, then arange edges in the order passed to
    # construct the track graph
    if edge_order is None:
        edge_order = np.asarray(track_graph.edges)

    n_edges = len(edge_order)
    if isinstance(edge_spacing, int) | isinstance(edge_spacing, float):
        edge_spacing = [edge_spacing, ] * (n_edges - 1)

    if axis == "x":
        start_node_linear_position = 0.0

        for ind, edge in enumerate(edge_order):
            end_node_linear_position = (start_node_linear_position +
                                        track_graph.edges[edge]["distance"])
            ax.scatter(start_node_linear_position, other_axis_start,
                       zorder=8, s=node_size, clip_on=False, color=node_color)
            ax.scatter(end_node_linear_position, other_axis_start,
                       zorder=8, s=node_size, clip_on=False, color=node_color)
            ax.plot((start_node_linear_position, end_node_linear_position),
                    (other_axis_start, other_axis_start),
                    color="black", clip_on=False, zorder=7)

            if draw_edge_labels:
                edge_midpoint = start_node_linear_position + \
                    (end_node_linear_position - start_node_linear_position) / 2
                ax.scatter(edge_midpoint, other_axis_start, color="white",
                           zorder=9, s=node_size, clip_on=False)
                ax.text(edge_midpoint, other_axis_start,
                        track_graph.edges[edge]["edge_id"],
                        ha="center", va="center", zorder=10)

            ax.text(start_node_linear_position, other_axis_start,
                    edge[0], ha="center", va="center", zorder=10)
            ax.text(end_node_linear_position, other_axis_start,
                    edge[1], ha="center", va="center", zorder=10)

            try:
                start_node_linear_position += (
                    track_graph.edges[edge]["distance"] + edge_spacing[ind])
            except IndexError:
                pass

        ax.set_xlim((other_axis_start, end_node_linear_position))
        ax.set_xlabel("Linear Position [cm]")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_yticks([])
    elif axis == "y":
        start_node_linear_position = 0.0

        for ind, edge in enumerate(edge_order):
            end_node_linear_position = (start_node_linear_position +
                                        track_graph.edges[edge]["distance"])
            ax.scatter(other_axis_start, start_node_linear_position,
                       zorder=8, s=node_size, clip_on=False, color=node_color)
            ax.scatter(other_axis_start, end_node_linear_position,
                       zorder=8, s=node_size, clip_on=False, color=node_color)
            ax.plot((other_axis_start, other_axis_start),
                    (start_node_linear_position, end_node_linear_position),
                    color="black", clip_on=False, zorder=7)

            if draw_edge_labels:
                edge_midpoint = start_node_linear_position + \
                    (end_node_linear_position - start_node_linear_position) / 2
                ax.scatter(other_axis_start, edge_midpoint, color="white",
                           zorder=9, s=node_size, clip_on=False)
                ax.text(other_axis_start, edge_midpoint,
                        track_graph.edges[edge]["edge_id"],
                        ha="center", va="center", zorder=10)

            ax.text(other_axis_start, start_node_linear_position,
                    edge[0], ha="center", va="center", zorder=10)
            ax.text(other_axis_start, end_node_linear_position,
                    edge[1], ha="center", va="center", zorder=10)

            try:
                start_node_linear_position += (
                    track_graph.edges[edge]["distance"] + edge_spacing[ind])
            except IndexError:
                pass
        ax.set_ylabel("Linear Position [cm]")


def _get_projected_track_position(track_graph, track_segment_id, position):
    track_segment_id[np.isnan(track_segment_id)] = 0
    track_segment_id = track_segment_id.astype(int)

    track_segments = get_track_segments_from_graph(track_graph)
    projected_track_position = project_points_to_segment(
        track_segments, position)
    n_time = projected_track_position.shape[0]
    return projected_track_position[(
        np.arange(n_time), track_segment_id)]


def make_actual_vs_linearized_position_movie(
        track_graph, position_df, time_slice=None,
        movie_name='actual_vs_linearized', frame_rate=33):
    '''

    Parameters
    ----------
    track_graph : networkx.Graph
    position_df : pandas.DataFrame
    time_slice : slice or None, optional
    movie_name : str, optional
    frame_rate : float, optional
        Frames per second.
    '''

    all_position = position_df.loc[:, ['x_position', 'y_position']].values
    all_linear_position = position_df.linear_position.values
    all_time = position_df.index.values / np.timedelta64(1, 's')

    if time_slice is None:
        position = all_position
        track_segment_id = position_df.track_segment_id.values
        linear_position = all_linear_position
        time = all_time
    else:
        position = all_position[time_slice]
        track_segment_id = position_df.iloc[time_slice].track_segment_id.values
        linear_position = all_linear_position[time_slice]
        time = all_time[time_slice]

    projected_track_position = _get_projected_track_position(
        track_graph, track_segment_id, position)

    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    fig, axes = plt.subplots(1, 2, figsize=(21, 7), constrained_layout=True,
                             gridspec_kw={'width_ratios': [2, 1]})

    # Subplot 1
    axes[0].scatter(all_time, all_linear_position, color='lightgrey',
                    zorder=0, s=10)
    axes[0].set_xlim((all_time.min(), all_time.max()))
    axes[0].set_ylim((all_linear_position.min(), all_linear_position.max()))
    linear_head = axes[0].scatter([], [], s=100, zorder=101, color='b')

    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('Position [cm]')
    axes[0].set_title('Linearized Position')

    axes[1].plot(all_position[:, 0], all_position[:, 1], color='lightgrey',
                 zorder=-10)
    plot_track_graph(track_graph, ax=axes[1])
    plt.axis('off')

    # Subplot 2
    axes[1].set_xlim(all_position[:, 0].min() - 10,
                     all_position[:, 0].max() + 10)
    axes[1].set_ylim(all_position[:, 1].min() - 10,
                     all_position[:, 1].max() + 10)

    actual_line, = axes[1].plot(
        [], [], 'g-', label='actual position', linewidth=3, zorder=101)
    actual_head = axes[1].scatter([], [], s=80, zorder=101, color='g')

    predicted_line, = axes[1].plot(
        [], [], 'b-', label='linearized position', linewidth=3, zorder=102)
    predicted_head = axes[1].scatter([], [], s=80, zorder=102, color='b')

    axes[1].legend()
    axes[1].set_xlabel('x-position')
    axes[1].set_ylabel('y-position')
    axes[1].set_title('Linearized vs. Actual Position')

    def _update_plot(time_ind):
        start_ind = max(0, time_ind - 33)
        time_slice = slice(start_ind, time_ind)

        linear_head.set_offsets(
            np.array((time[time_ind], linear_position[time_ind])))

        actual_line.set_data(position[time_slice, 0], position[time_slice, 1])
        actual_head.set_offsets(position[time_ind])

        predicted_line.set_data(projected_track_position[time_slice, 0],
                                projected_track_position[time_slice, 1])
        predicted_head.set_offsets(projected_track_position[time_ind])

        return actual_line, predicted_line

    n_time = position.shape[0]
    line_ani = animation.FuncAnimation(fig, _update_plot, frames=n_time,
                                       interval=1000 / frame_rate, blit=True)
    line_ani.save(movie_name + '.mp4', writer=writer)
