import networkx as nx
import numpy as np


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
