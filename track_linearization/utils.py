import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from track_linearization.core import get_graph_1D_2D_relationships


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

    for edge in edges:
        node1, node2 = edge
        pos1 = np.asarray(track_graph.nodes[node1]["pos"])
        pos2 = np.asarray(track_graph.nodes[node2]["pos"])
        distance = np.linalg.norm(pos1 - pos2)
        nx.add_path(track_graph, edge, distance=distance)

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
    node_position = nx.get_node_attributes(track_graph, 'pos')
    nx.draw_networkx(track_graph, node_position, ax, **kwds)

    if draw_edge_labels:
        edge_ids = {edge: ind for ind, edge in enumerate(track_graph.edges)}
        nx.draw_networkx_edge_labels(track_graph, node_position,
                                     edge_labels=edge_ids)


def plot_graph_as_1D(track_graph, edge_order=None, edge_spacing=0,
                     linear_zero_node_id=None, ax=None):
    if ax is None:
        ax = plt.gca()
    # If no edge_order is given, then arange edges in the order passed to
    # construct the track graph
    if edge_order is None:
        edge_order = np.arange(len(track_graph.edges), dtype=int)

    # If no linear zero node id is given,
    if linear_zero_node_id is None:
        linear_zero_node_id = list(track_graph.edges)[0][0]

    node_linear_position, _ = get_graph_1D_2D_relationships(
        track_graph, edge_order, edge_spacing, linear_zero_node_id)
    edge_node_id = np.asarray(list(track_graph.edges))[edge_order]
    ax.scatter(node_linear_position.ravel(), np.zeros_like(
        node_linear_position.ravel()), zorder=8, s=300, clip_on=False)
    for nodes_pos, edge_id, (node1, node2) in zip(node_linear_position,
                                                  edge_order, edge_node_id):
        plt.plot(nodes_pos, np.zeros_like(nodes_pos), color="black")
        edge_midpoint = nodes_pos[0] + (nodes_pos[1] - nodes_pos[0]) / 2
        plt.scatter(edge_midpoint, 0, color="white",
                    zorder=9, s=300, clip_on=False)
        plt.text(edge_midpoint, 0, edge_id,
                 ha="center", va="center", zorder=10)
        plt.text(nodes_pos[0], 0.0, node1, ha="center", va="center", zorder=10)
        plt.text(nodes_pos[1], 0.0, node2, ha="center", va="center", zorder=10)
    ax.set_xlim((0, node_linear_position.max()))
    ax.set_xlabel("Linear Position [cm]")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])
