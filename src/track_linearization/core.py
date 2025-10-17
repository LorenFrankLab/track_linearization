from collections.abc import Iterator, Mapping, Sequence
from math import sqrt
from typing import Any

import dask
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse
import scipy.stats
from networkx import Graph


Edge = tuple[Any, Any]

try:
    import numba

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


def validate_track_graph(
    track_graph: Graph, edge_order: list[Edge] | None = None
) -> None:
    """Validate that track_graph has the required structure for linearization.

    Parameters
    ----------
    track_graph : networkx.Graph
        The track graph to validate.
    edge_order : list of 2-tuples, optional
        If provided, validates that these edges exist in the graph.

    Raises
    ------
    ValueError
        If the graph structure is invalid, with specific guidance on how to fix it.

    Examples
    --------
    >>> import networkx as nx
    >>> g = nx.Graph()
    >>> g.add_node(0, pos=(0, 0))
    >>> g.add_node(1, pos=(1, 0))
    >>> g.add_edge(0, 1, distance=1.0, edge_id=0)
    >>> validate_track_graph(g)  # No error - valid graph
    """
    # Check for edges
    if len(track_graph.edges) == 0:
        raise ValueError(
            "Track graph has no edges.\n\n"
            "WHAT: The track graph must contain at least one edge connecting two nodes.\n"
            "WHY: Linearization requires track segments to project positions onto.\n"
            "HOW: Create a valid track using make_track_graph(node_positions, edges)\n\n"
            "Example:\n"
            "    from track_linearization import make_track_graph\n"
            "    node_positions = [(0, 0), (10, 0), (10, 10)]\n"
            "    edges = [(0, 1), (1, 2)]\n"
            "    track_graph = make_track_graph(node_positions, edges)"
        )

    # Check node attributes
    nodes_without_pos = [
        n for n in track_graph.nodes if "pos" not in track_graph.nodes[n]
    ]
    if nodes_without_pos:
        raise ValueError(
            f"Nodes {nodes_without_pos[:5]} are missing 'pos' attribute "
            f"({len(nodes_without_pos)} total).\n\n"
            "WHAT: All nodes must have a 'pos' attribute containing spatial coordinates.\n"
            "WHY: The 'pos' attribute defines where each node is located in space.\n"
            "HOW: Use make_track_graph() which automatically adds this attribute:\n\n"
            "Example:\n"
            "    track_graph = make_track_graph(node_positions, edges)\n\n"
            "Or add manually:\n"
            "    track_graph.nodes[node_id]['pos'] = (x, y)"
        )

    # Check edge attributes - distance
    edges_without_distance = [
        e for e in track_graph.edges if "distance" not in track_graph.edges[e]
    ]
    if edges_without_distance:
        raise ValueError(
            f"{len(edges_without_distance)} edges are missing 'distance' attribute.\n\n"
            "WHAT: All edges must have a 'distance' attribute (Euclidean length).\n"
            "WHY: Distances are required for calculating linear positions.\n"
            "HOW: Use make_track_graph() which automatically computes distances:\n\n"
            "Example:\n"
            "    track_graph = make_track_graph(node_positions, edges)\n\n"
            "Or compute manually:\n"
            "    import numpy as np\n"
            "    for node1, node2 in track_graph.edges:\n"
            "        pos1 = np.array(track_graph.nodes[node1]['pos'])\n"
            "        pos2 = np.array(track_graph.nodes[node2]['pos'])\n"
            "        track_graph.edges[(node1, node2)]['distance'] = np.linalg.norm(pos2 - pos1)"
        )

    # Check edge attributes - edge_id
    edges_without_id = [
        e for e in track_graph.edges if "edge_id" not in track_graph.edges[e]
    ]
    if edges_without_id:
        raise ValueError(
            f"{len(edges_without_id)} edges are missing 'edge_id' attribute.\n\n"
            "WHAT: All edges must have an 'edge_id' attribute (unique integer).\n"
            "WHY: Edge IDs are used to identify which track segment positions belong to.\n"
            "HOW: Use make_track_graph() which automatically assigns edge IDs:\n\n"
            "Example:\n"
            "    track_graph = make_track_graph(node_positions, edges)\n\n"
            "Or assign manually:\n"
            "    for edge_id, edge in enumerate(track_graph.edges):\n"
            "        track_graph.edges[edge]['edge_id'] = edge_id"
        )

    # Validate edge_order if provided
    if edge_order is not None:
        invalid_edges = [e for e in edge_order if not track_graph.has_edge(*e)]
        if invalid_edges:
            available_edges = list(track_graph.edges)
            raise ValueError(
                f"{len(invalid_edges)} edges in edge_order are not in the graph.\n\n"
                f"WHAT: edge_order contains edges that don't exist in track_graph.\n"
                f"WHY: Can only linearize using edges that are actually in the graph.\n"
                f"HOW: Check that your edge_order only uses valid edges.\n\n"
                f"Invalid edges (showing first 5): {invalid_edges[:5]}\n"
                f"Available edges in graph: {available_edges[:10]}\n"
                f"{'...' if len(available_edges) > 10 else ''}"
            )


def get_track_segments_from_graph(
    track_graph: "Graph",
) -> np.ndarray:
    """Extracts track segments as pairs of node positions from a graph.

    Parameters
    ----------
    track_graph : networkx.Graph
        A graph where nodes have a "pos" attribute representing their spatial coordinates.
        The "pos" attribute should be an iterable (e.g., tuple or list) of numbers.

    Returns
    -------
    track_segments : np.ndarray, shape (n_segments, 2, n_space)
        An array where each row represents a track segment, defined by the
        start and end node positions. `n_space` is the dimensionality of the positions.
    """
    node_positions = nx.get_node_attributes(track_graph, "pos")
    return np.asarray(
        [
            (node_positions[node1], node_positions[node2])
            for node1, node2 in track_graph.edges()
        ]
    )


def project_points_to_segment(
    track_segments: np.ndarray, position: np.ndarray
) -> np.ndarray:
    """Finds the closest point on each track segment to given positions.

    Parameters
    ----------
    track_segments : np.ndarray, shape (n_segments, 2, n_space)
        Array of line segments, where each segment is defined by two points (start, end).
        `n_space` is the dimensionality of the coordinates.
        Original code implies n_nodes is 2 for a segment, and n_space is 2 (from `pos` attributes).
        This docstring uses `n_space` generally.
    position : np.ndarray, shape (n_time, n_space)
        Array of points to project. `n_space` should match `track_segments`.

    Returns
    -------
    projected_positions : np.ndarray, shape (n_time, n_segments, n_space)
        The projection of each point onto each track segment.
    """
    segment_diff = np.diff(track_segments, axis=1).squeeze(axis=1)
    sum_squares = np.sum(segment_diff**2, axis=1)
    node1 = track_segments[:, 0, :]
    nx_param = (
        np.sum(segment_diff * (position[:, np.newaxis, :] - node1), axis=2)
        / sum_squares
    )

    np.clip(nx_param, 0.0, 1.0, out=nx_param)

    result: np.ndarray = node1[np.newaxis, ...] + (
        nx_param[:, :, np.newaxis] * segment_diff[np.newaxis, ...]
    )
    return result


def find_projected_point_distance(
    track_segments: np.ndarray, position: np.ndarray
) -> np.ndarray:
    """Calculates Euclidean distance from points to their projections on track segments.

    Parameters
    ----------
    track_segments : np.ndarray, shape (n_segments, 2, n_space)
        Array of line segments.
    position : np.ndarray, shape (n_time, n_space)
        Array of points.

    Returns
    -------
    distances : np.ndarray, shape (n_time, n_segments)
        Euclidean distance from each point to its projection on each segment.
    """
    result: np.ndarray = np.linalg.norm(
        position[:, np.newaxis, :]
        - project_points_to_segment(track_segments, position),
        axis=2,
    )
    return result


def find_nearest_segment(
    track_segments: np.ndarray, position: np.ndarray
) -> np.ndarray:
    """Returns the track segment that is closest to the position
    at each time point.

    Parameters
    ----------
    track_segments : np.ndarray, shape (n_segments, 2, n_space) # n_space added
        Array of line segments.
    position : np.ndarray, shape (n_time, n_space) # n_space added
        Array of points.

    Returns
    -------
    segment_id : np.ndarray, shape (n_time,)
        Index of the nearest track segment for each time point.
    """
    distance = find_projected_point_distance(track_segments, position)
    result: np.ndarray = np.argmin(distance, axis=1)
    return result


def euclidean_distance_change(position: np.ndarray) -> np.ndarray:
    """Distance between position at successive time points.

    Parameters
    ----------
    position : np.ndarray, shape (n_time, n_space) # n_space added
        Array of spatial positions over time.

    Returns
    -------
    distance : np.ndarray, shape (n_time,)
        Euclidean distance between position[t] and position[t-1].
        The first element is `np.nan`.
    """
    distance = np.linalg.norm(position[1:] - position[:-1], axis=1)
    return np.concatenate(([np.nan], distance))


def route_distance(
    candidates_t_1: np.ndarray,
    candidates_t: np.ndarray,
    track_graph: Graph,
) -> np.ndarray:
    """Calculates route distances between sets of candidate points on a track graph.

    Warning: This function modifies the input `track_graph` by adding and
    removing temporary nodes. Pass a copy if the original graph should be preserved.
    (e.g., `track_graph.copy()`).

    Parameters
    ----------
    candidates_t_1 : np.ndarray, shape (n_segments_or_points, n_space)
        Candidate points at time t-1. `n_space` usually 2 for (x,y).
    candidates_t : np.ndarray, shape (n_segments_or_points, n_space)
        Candidate points at time t. `n_space` usually 2 for (x,y).
    track_graph : networkx.Graph
        The base track graph. Nodes must have a "pos" attribute (tuple of coordinates).
        This graph will be modified in place.

    Returns
    -------
    route_distances : np.ndarray, shape (n_segments_or_points, n_segments_or_points)
        A matrix where element (i, j) is the shortest route distance along the
        modified graph from the i-th candidate at t-1 to the j-th candidate at t.
        Returns NaNs if any input candidate is NaN.
    """
    # TODO: speedup function. This takes the most time
    len(track_graph.edges)  # Based on original graph's edges
    if np.any(np.isnan(candidates_t) | np.isnan(candidates_t_1)):
        return np.full((candidates_t_1.shape[0], candidates_t.shape[0]), np.nan)

    node_names: list[str] = []
    edges = list(track_graph.edges.keys())
    n_original_nodes = len(track_graph.nodes)

    # insert virtual node
    # Assuming len(candidates_t) == len(candidates_t_1) == len(edges)
    for edge_number, (position_t, position_t_1, (node1, node2)) in enumerate(
        zip(candidates_t, candidates_t_1, edges, strict=True)
    ):
        node_name_t, node_name_t_1 = f"t_0_{edge_number}", f"t_1_{edge_number}"
        node_names.append(node_name_t)
        node_names.append(node_name_t_1)
        nx.add_path(track_graph, [node1, node_name_t, node2])
        nx.add_path(track_graph, [node1, node_name_t_1, node2])
        nx.add_path(
            track_graph, [node_name_t, node_name_t_1]
        )  # Connects candidates directly
        track_graph.nodes[node_name_t]["pos"] = tuple(position_t)
        track_graph.nodes[node_name_t_1]["pos"] = tuple(position_t_1)

    # calculate distance for all edges in the modified graph
    for node1, node2 in track_graph.edges:
        # Assumes 'pos' is a 2-tuple (x,y)
        x1, y1 = track_graph.nodes[node1]["pos"]
        x2, y2 = track_graph.nodes[node2]["pos"]
        track_graph.edges[(node1, node2)]["distance"] = sqrt(
            (x2 - x1) ** 2 + (y2 - y1) ** 2
        )

    # calculate path distance
    path_distance = scipy.sparse.csgraph.dijkstra(
        nx.to_scipy_sparse_array(track_graph, weight="distance")
    )
    n_total_nodes = len(track_graph.nodes)  # After adding temporary nodes
    node_ind = np.arange(n_total_nodes)
    # These indices assume temporary nodes are added contiguously at the end
    # and in pairs (t, t-1) for each original edge.
    start_node_ind = node_ind[n_original_nodes::2]  # Corresponds to t_0_ nodes
    end_node_ind = node_ind[n_original_nodes + 1 :: 2]  # Corresponds to t_1_ nodes

    dist_matrix_slice: np.ndarray = path_distance[start_node_ind][:, end_node_ind]

    track_graph.remove_nodes_from(node_names)  # Clean up graph

    return dist_matrix_slice


def batch(n_samples: int, batch_size: int = 1) -> Iterator[range]:
    """Generates batches of indices.

    Parameters
    ----------
    n_samples : int
        Total number of samples.
    batch_size : int, optional
        Number of samples per batch, by default 1.

    Yields
    ------
    Iterator[range]
        An iterator yielding ranges of indices for each batch.
    """
    for ind in range(0, n_samples, batch_size):
        yield range(ind, min(ind + batch_size, n_samples))


@dask.delayed  # type: ignore[misc]
def batch_route_distance(
    track_graph: "Graph",
    projected_track_position_t: np.ndarray,
    projected_track_position_t_1: np.ndarray,
) -> np.ndarray:
    """Computes route distances for a batch of projected positions using Dask.

    Note: `route_distance` (called internally) modifies the `track_graph`.
    The `copy_graph` here is made once. If `route_distance` is called multiple
    times in the loop, it will operate on an increasingly modified graph from
    the previous iteration's changes. This is likely not the intended behavior
    and can lead to incorrect results or errors.

    Parameters
    ----------
    track_graph : networkx.Graph
        The network graph representing the track.
    projected_track_position_t : np.ndarray, shape (n_batch_time, n_segments, n_space)
        Projected positions on track segments at time t for a batch.
    projected_track_position_t_1 : np.ndarray, shape (n_batch_time, n_segments, n_space)
        Projected positions on track segments at time t-1 for a batch.

    Returns
    -------
    np.ndarray, shape (n_batch_time, n_segments_t-1, n_segments_t)
        Stacked route distances for each time pair in the batch.
    """
    copy_graph = track_graph.copy()
    distances = [
        route_distance(p_t, p_t_1, copy_graph)
        for p_t, p_t_1 in zip(
            projected_track_position_t, projected_track_position_t_1, strict=True
        )
    ]
    return np.stack(distances)


def route_distance_change(position: np.ndarray, track_graph: Graph) -> np.ndarray:
    """Calculates route distances between projected positions at successive time points.

    Parameters
    ----------
    position : np.ndarray, shape (n_time, n_space)
        Spatial positions over time.
    track_graph : networkx.Graph
        The track graph.

    Returns
    -------
    np.ndarray, shape (n_time, n_segments, n_segments)
        Route distances. The first time point (index 0) will have NaNs.
        `route_distance_change[t, i, j]` is the route distance from segment `i`
        (where position was at t-1) to segment `j` (where position was at t).
    """
    track_segments = get_track_segments_from_graph(track_graph)
    projected_track_position = project_points_to_segment(track_segments, position)
    n_segments = len(track_segments)

    all_distances_results: list[np.ndarray | Any] = [
        np.full((1, n_segments, n_segments), np.nan)
    ]

    projected_track_position_t = projected_track_position[1:]
    projected_track_position_t_1 = projected_track_position[:-1]
    n_time = projected_track_position_t.shape[0]

    for time_ind in batch(n_time, batch_size=10_000):
        all_distances_results.append(
            batch_route_distance(
                track_graph,
                projected_track_position_t[time_ind],
                projected_track_position_t_1[time_ind],
            )
        )

    computed_results: tuple = dask.compute(*all_distances_results)
    return np.concatenate(computed_results, axis=0)


def calculate_position_likelihood(
    position: np.ndarray, track_graph: Graph, sigma: float = 10.0
) -> np.ndarray:
    r"""Calculates the likelihood of a position being associated with track segments.

    Assumes a Gaussian error model for the position sensor relative to the track.
    Likelihood $L(pos | segment) = PDF(N(0, \sigma^2))$ evaluated at $d(pos, segment)$.

    Parameters
    ----------
    position : np.ndarray, shape (n_time, n_space)
        Array of spatial positions over time.
    track_graph : networkx.Graph
        The track graph.
    sigma : float, optional
        Standard deviation of the sensor noise (in the same units as position
        and graph coordinates), by default 10.0.

    Returns
    -------
    likelihood : np.ndarray, shape (n_time, n_segments)
        Likelihood of each position belonging to each track segment.
    """
    track_segments = get_track_segments_from_graph(track_graph)
    projected_position_distance = find_projected_point_distance(
        track_segments, position
    )
    result: np.ndarray = np.exp(-0.5 * (projected_position_distance / sigma) ** 2) / (
        np.sqrt(2 * np.pi) * sigma
    )
    return result


def normalize_to_probability(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Ensure the array axis sum to 1.

    Parameters
    ----------
    x : np.ndarray
        Input array.
    axis : int, optional
        Axis along which to normalize, by default -1.

    Returns
    -------
    normalized_x : np.ndarray
        Array with values normalized to sum to 1 along the specified axis.
        If a sum along the axis is 0, the original values in that slice will
        result in NaNs or Infs after division.
    """
    result: np.ndarray = x / x.sum(axis=axis, keepdims=True)
    return result


def calculate_empirical_state_transition(
    position: np.ndarray,
    track_graph: Graph,
    scaling: float = 1e-1,
    diagonal_bias: float = 1e-1,
) -> np.ndarray:
    """Calculates the state transition probability between track segments by
    favoring route distances that are similar to euclidean distances between
    successive time points.

    Parameters
    ----------
    position : np.ndarray, shape (n_time, n_space) # n_space added
        Spatial positions over time.
    track_graph : networkx.Graph
        The track graph.
    scaling : float, optional
        Scale parameter for the exponential PDF. Default is 1e-1.
    diagonal_bias : float, optional
        Value added to the diagonal of the transition matrix (before final
        normalization) to bias towards staying on the same segment. Default is 1e-1.

    Returns
    -------
    state_transition_matrix : np.ndarray, shape (n_time, n_segments, n_segments)
        Time-dependent state transition probabilities.
        `state_transition_matrix[t, i, j]` is P(next_segment=j | current_segment=i) at time t.
        The first time slice (index 0) will contain NaNs.

    References
    ----------
    .. [1] Newson, P., and Krumm, J. (2009). Hidden Markov map matching through
       noise and sparseness. In Proceedings of the 17th ACM SIGSPATIAL
       International Conference on Advances in Geographic Information Systems,
       (ACM), pp. 336-343.
    """
    route_and_euclidean_distance_similarity = np.abs(
        route_distance_change(position, track_graph)
        - euclidean_distance_change(position)[:, np.newaxis, np.newaxis]
    )
    exponential_pdf = scipy.stats.expon.pdf(
        route_and_euclidean_distance_similarity, scale=scaling
    )
    exponential_pdf = normalize_to_probability(exponential_pdf, axis=2)

    n_states = route_and_euclidean_distance_similarity.shape[1]
    # Add bias after first normalization
    if n_states > 0:  # Avoid error with identity matrix if n_segments is 0
        exponential_pdf += np.identity(n_states)[np.newaxis] * diagonal_bias

    return normalize_to_probability(exponential_pdf, axis=2)


def viterbi_no_numba(
    initial_conditions: np.ndarray,
    state_transition: np.ndarray,
    likelihood: np.ndarray,
) -> np.ndarray:
    """Find the most likely sequence of paths using the Viterbi algorithm.

    Note that the state_transition matrix is time-dependent. NaNs are removed
    and placed back in at the end.

    Parameters
    ----------
    initial_conditions : np.ndarray, shape (n_states,)
        Initial probabilities (or log-probabilities, depending on usage) of states.
        Original code takes `np.log(initial_conditions + LOG_EPS)`.
    state_transition : np.ndarray, shape (n_time, n_states, n_states)
        State transition probability matrix. `state_transition[t, i, j]` is
        P(to_state_j | from_state_i) for observations at time `t`.
        Original code takes `np.log(state_transition + LOG_EPS)`.
    likelihood : np.ndarray, shape (n_time, n_states)
        Likelihood of observations given states. `likelihood[t, s]` is P(obs_t | state_s).

    Returns
    -------
    state_id : np.ndarray, shape (n_time,)
        The most likely sequence of state IDs (indices).
    """

    LOG_EPS = 1e-16
    log_likelihood = np.log(likelihood.copy())
    log_state_transition = np.log(state_transition.copy() + LOG_EPS)

    n_time, n_states = log_likelihood.shape
    posterior_prob = np.zeros((n_time, n_states))
    path_pointers = np.zeros((n_time, n_states), dtype=int)

    # initialization
    posterior_prob[0] = np.log(initial_conditions + LOG_EPS) + log_likelihood[0]

    # recursion
    for time_ind in range(1, n_time):
        prior = (
            posterior_prob[time_ind - 1][:, np.newaxis] + log_state_transition[time_ind]
        )

        for state_ind in range(n_states):  # current state `j`
            path_pointers[time_ind, state_ind] = np.argmax(prior[:, state_ind])
            posterior_prob[time_ind, state_ind] = (
                np.max(prior[:, state_ind]) + log_likelihood[time_ind, state_ind]
            )

    # termination
    most_probable_state_ind = np.zeros((n_time,), dtype=float)
    if n_time > 0:
        most_probable_state_ind[n_time - 1] = np.argmax(posterior_prob[n_time - 1])

        # path back-tracking
        for time_ind in range(n_time - 2, -1, -1):
            most_probable_state_ind[time_ind] = path_pointers[
                time_ind + 1, int(most_probable_state_ind[time_ind + 1])
            ]

    return most_probable_state_ind.astype(int)


if NUMBA_AVAILABLE:

    @numba.njit(cache=True)  # type: ignore[misc]
    def viterbi(
        initial_conditions: np.ndarray,
        state_transition: np.ndarray,
        likelihood: np.ndarray,
    ) -> np.ndarray:
        """Find the most likely sequence of paths using the Viterbi algorithm (numba version).

        This is the numba-compiled version that duplicates the logic from viterbi_no_numba.
        """
        LOG_EPS = 1e-16
        log_likelihood = np.log(likelihood.copy())
        log_state_transition = np.log(state_transition.copy() + LOG_EPS)

        n_time, n_states = log_likelihood.shape
        posterior_prob = np.zeros((n_time, n_states))
        path_pointers = np.zeros((n_time, n_states), dtype=np.int64)

        # initialization
        posterior_prob[0] = np.log(initial_conditions + LOG_EPS) + log_likelihood[0]

        # recursion
        for time_ind in range(1, n_time):
            prior = (
                posterior_prob[time_ind - 1][:, np.newaxis]
                + log_state_transition[time_ind]
            )

            for state_ind in range(n_states):  # current state `j`
                path_pointers[time_ind, state_ind] = np.argmax(prior[:, state_ind])
                posterior_prob[time_ind, state_ind] = (
                    np.max(prior[:, state_ind]) + log_likelihood[time_ind, state_ind]
                )

        # termination
        most_probable_state_ind = np.zeros((n_time,), dtype=np.float64)
        if n_time > 0:
            most_probable_state_ind[n_time - 1] = np.argmax(posterior_prob[n_time - 1])

            # path back-tracking
            for time_ind in range(n_time - 2, -1, -1):
                most_probable_state_ind[time_ind] = path_pointers[
                    time_ind + 1, int(most_probable_state_ind[time_ind + 1])
                ]

        return most_probable_state_ind.astype(np.int64)

else:
    viterbi = viterbi_no_numba


def classify_track_segments(
    track_graph: Graph,
    position: np.ndarray,
    sensor_std_dev: float = 10.0,
    route_euclidean_distance_scaling: float = 1e-1,
    diagonal_bias: float = 1e-1,
) -> np.ndarray:
    """Find the most likely track segment for a given position.

    Tries to make sure the euclidean distance between successive time points
    is similar to the route distance along the graph.

    Parameters
    ----------
    track_graph : networkx.Graph
        The track graph.
    position : np.ndarray, shape (n_time, n_space) # n_space added
        Spatial positions over time.
    sensor_std_dev : float, optional
        Uncertainty of position sensor. Default is 10.0.
    route_euclidean_distance_scaling : float, optional
        How much to prefer route distances between successive time points
        that are closer to the euclidean distance. Smaller numbers mean the
        route distance is more likely to be close to the euclidean distance.
        Default is 1e-1.
    diagonal_bias : float, optional
        Biases the transition matrix to prefer the current track segment.
        Default is 1e-1.

    Returns
    -------
    segment_id : np.ndarray, shape (n_time,)
        The most likely track segment ID (index corresponding to order of edges
        in `track_graph.edges()`). Contains NaNs for time points where
        classification was problematic.

    References
    ----------
    .. [1] Newson, P., and Krumm, J. (2009). Hidden Markov map matching through
       noise and sparseness. In Proceedings of the 17th ACM SIGSPATIAL
       International Conference on Advances in Geographic Information Systems,
       (ACM), pp. 336-343.
    """
    n_segments = len(track_graph.edges)
    if n_segments == 0:  # Handle case with no segments
        return np.full(position.shape[0], np.nan)

    initial_conditions = np.ones((n_segments,))
    state_transition = calculate_empirical_state_transition(
        position,
        track_graph,
        scaling=route_euclidean_distance_scaling,
        diagonal_bias=diagonal_bias,
    )
    likelihood = calculate_position_likelihood(
        position, track_graph, sigma=sensor_std_dev
    )
    is_bad = np.any(np.isnan(likelihood), axis=1) | np.any(
        np.isinf(np.log(likelihood)), axis=1
    )
    likelihood = likelihood[~is_bad]
    state_transition = state_transition[~is_bad]
    track_segment_id = viterbi(initial_conditions, state_transition, likelihood)
    track_segment_id_with_nan = np.full((is_bad.size,), np.nan)
    track_segment_id_with_nan[~is_bad] = track_segment_id
    return track_segment_id_with_nan


def batch_linear_distance(
    track_graph: Graph,
    projected_track_positions: np.ndarray,
    edge_ids: list[tuple[Any, Any]],
    linear_zero_node_id: Any,
) -> list[float]:
    """Calculates linear distances for a batch of projected positions.

    Warning: This function modifies the `track_graph` copy internally by adding
    and removing a temporary node "projected".

    Parameters
    ----------
    track_graph : networkx.Graph
        The base track graph. Must have "pos" attributes for nodes.
        Edge "distance" attributes will be calculated if not present.
    projected_track_positions : np.ndarray, shape (n_batch, n_space)
        Batch of positions, each projected onto a track segment.
    edge_ids : List[Tuple[Any, Any]]
        List of edge tuples (node1, node2) corresponding to each projected position,
        indicating which segment the point lies on. Length must match `n_batch`.
    linear_zero_node_id : Any
        The node in `track_graph` considered the origin (zero point) for
        linear distance measurements.

    Returns
    -------
    List[float]
        A list of linear distances for each projected position in the batch.
        Contains `np.nan` if a shortest path cannot be found.
    """
    copy_graph = track_graph.copy()
    linear_distance: list[float] = []

    for (x3, y3), (node1, node2) in zip(
        projected_track_positions, edge_ids, strict=True
    ):
        x1, y1 = copy_graph.nodes[node1]["pos"]
        left_distance = sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
        nx.add_path(copy_graph, [node1, "projected"], distance=left_distance)

        x2, y2 = copy_graph.nodes[node2]["pos"]
        right_distance = sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
        nx.add_path(copy_graph, ["projected", node2], distance=right_distance)

        linear_distance.append(
            nx.shortest_path_length(
                copy_graph,
                source=linear_zero_node_id,
                target="projected",
                weight="distance",
            )
        )
        copy_graph.remove_node("projected")

    return linear_distance


def _normalize_edge_spacing(
    edge_spacing: float | Sequence[float], n_edges: int
) -> list[float]:
    """Convert edge_spacing to list format with validation.

    Parameters
    ----------
    edge_spacing : float or list of float
        Spacing between consecutive edges.
    n_edges : int
        Number of edges in the edge_order.

    Returns
    -------
    spacing_list : list of float
        List of spacing values with length n_edges - 1.

    Raises
    ------
    ValueError
        If edge_spacing list has incorrect length.
    TypeError
        If edge_spacing has invalid type.
    """
    if isinstance(edge_spacing, (int, float)):
        return [float(edge_spacing)] * (n_edges - 1) if n_edges > 1 else []
    elif isinstance(edge_spacing, list):
        expected_length = max(0, n_edges - 1)
        if len(edge_spacing) != expected_length:
            raise ValueError(
                f"edge_spacing list has wrong length: got {len(edge_spacing)}, expected {expected_length}\n\n"
                f"WHAT: The edge_spacing parameter has {len(edge_spacing)} values but needs {expected_length}.\n"
                f"WHY: edge_spacing controls gaps between consecutive edges in the linearization.\n"
                f"      For {n_edges} edges, you need {expected_length} spacing values (one between each pair).\n"
                f"HOW: Either use a single value for uniform spacing, or provide {expected_length} values.\n\n"
                f"Examples:\n"
                f"    edge_spacing=10.0              # Uniform 10-unit gaps\n"
                f"    edge_spacing=[10, 5, 15, ...]  # Custom gaps ({expected_length} values)"
            )
        return [float(es) for es in edge_spacing]
    else:
        raise TypeError(
            f"edge_spacing has invalid type: {type(edge_spacing).__name__}\n\n"
            "WHAT: edge_spacing must be either a number or a list of numbers.\n"
            "WHY: This parameter controls the spacing between track segments.\n"
            "HOW: Pass either a single value or a list.\n\n"
            "Examples:\n"
            "    edge_spacing=10.0        # All gaps are 10 units\n"
            "    edge_spacing=[5, 10, 5]  # Custom gaps between segments"
        )


def _apply_edge_map_to_positions(
    linear_position: np.ndarray,
    track_segment_id: np.ndarray,
    edge_map: Mapping[int, int | str],
    track_graph: Graph,
    edge_order: list[Edge],
    edge_spacing: float | Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply edge_map to adjust linear positions and segment IDs.

    This function handles the merging of track segments by adjusting linear positions
    so that merged edges share the same linear coordinate system.

    Parameters
    ----------
    linear_position : np.ndarray, shape (n_time,)
        Original linear positions calculated from physical edges.
    track_segment_id : np.ndarray, shape (n_time,)
        Original track segment IDs.
    edge_map : dict
        Mapping from original edge IDs to target edge IDs.
    track_graph : networkx.Graph
        The track graph.
    edge_order : list of 2-tuples
        Ordered list of edges.
    edge_spacing : float or list of float
        Spacing between consecutive edges.

    Returns
    -------
    adjusted_linear_position : np.ndarray, shape (n_time,)
        Linear positions adjusted for merged coordinate systems.
    output_track_segment_id : np.ndarray, shape (n_time,)
        Track segment IDs after applying edge_map.
    """
    spacing_list = _normalize_edge_spacing(edge_spacing, len(edge_order))

    # Calculate starting linear positions for each target in merged space
    target_start_positions = {}
    cumulative_position = 0.0

    for i, edge in enumerate(edge_order):
        edge_id = track_graph.edges[edge]["edge_id"]
        target = edge_map.get(edge_id, edge_id)

        # Only record start position for first occurrence of each target
        if target not in target_start_positions:
            target_start_positions[target] = cumulative_position
            # Add this edge's length and spacing to cumulative position
            cumulative_position += track_graph.edges[edge]["distance"]
            if i < len(spacing_list):
                cumulative_position += spacing_list[i]

    # Adjust linear positions based on merged coordinate systems
    # For each position, find its offset within its original edge,
    # then add that to the merged edge's starting position
    adjusted_linear_position = np.zeros_like(linear_position)

    # Build lookup from edge_id to starting position
    edge_id_to_start_pos = {}
    cumulative = 0.0
    for i, edge in enumerate(edge_order):
        edge_id = track_graph.edges[edge]["edge_id"]
        edge_id_to_start_pos[edge_id] = cumulative
        cumulative += track_graph.edges[edge]["distance"]
        if i < len(spacing_list):
            cumulative += spacing_list[i]

    for i in range(len(track_segment_id)):
        orig_edge_id = (
            int(track_segment_id[i]) if not np.isnan(track_segment_id[i]) else 0
        )
        orig_start = edge_id_to_start_pos.get(orig_edge_id, 0.0)
        offset_within_edge = linear_position[i] - orig_start

        target = edge_map.get(orig_edge_id, orig_edge_id)
        new_start = target_start_positions.get(target, orig_start)
        adjusted_linear_position[i] = new_start + offset_within_edge

    # Output uses the target segment IDs from edge_map
    # Check if edge_map contains non-integer values for dtype handling
    has_non_int_values = any(
        not isinstance(v, (int, np.integer)) for v in edge_map.values()
    )
    if has_non_int_values:
        # Convert to object dtype to support mixed types
        output_track_segment_id = np.empty(len(track_segment_id), dtype=object)
        output_track_segment_id[:] = track_segment_id
    else:
        output_track_segment_id = track_segment_id.copy()

    # Apply mapping
    for cur_edge, new_edge in edge_map.items():
        output_track_segment_id[track_segment_id == cur_edge] = new_edge

    return adjusted_linear_position, output_track_segment_id


def _calculate_linear_position(
    track_graph: Graph,
    position: np.ndarray,
    track_segment_id: np.ndarray,
    edge_order: list[Edge],
    edge_spacing: float | Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Determines the linear position given a 2D position and a track graph.

    Parameters
    ----------
    track_graph : nx.Graph
        The track graph. Nodes need "pos". Edges need "distance" and "edge_id".
    position : np.ndarray, shape (n_time, n_space)
        Spatial positions.
    track_segment_id : np.ndarray, shape (n_time,)
        Integer 'edge_id' for each time point. NaNs should be pre-handled or
        will lead to errors/defaulting to edge_id 0.
    edge_order : list of 2-tuples
        Ordered list of edge tuples (node1, node2) defining the linearization path.
        These tuples are keys in `track_graph.edges`.
    edge_spacing : float or list of float
        Spacing to insert between consecutive segments in `edge_order`.
        If list, length `len(edge_order) - 1`.

    Returns
    -------
    linear_position : np.ndarray, shape (n_time,)
    projected_track_positions_x : np.ndarray, shape (n_time,)
    projected_track_positions_y : np.ndarray, shape (n_time,)
        (Assumes n_space=2 for x,y return).
    """
    is_nan = np.isnan(track_segment_id)
    track_segment_id[is_nan] = 0  # need to check
    track_segment_id = track_segment_id.astype(int)

    track_segments = get_track_segments_from_graph(track_graph)
    projected_track_positions = project_points_to_segment(track_segments, position)
    n_time = projected_track_positions.shape[0]
    projected_track_positions = projected_track_positions[
        (np.arange(n_time), track_segment_id)
    ]

    edge_spacing_list = _normalize_edge_spacing(edge_spacing, len(edge_order))

    counter = 0.0
    start_node_linear_position_list: list[float] = []

    for ind, edge in enumerate(edge_order):
        start_node_linear_position_list.append(counter)

        try:
            counter += track_graph.edges[edge]["distance"] + edge_spacing_list[ind]
        except IndexError:
            pass

    start_node_linear_position = np.asarray(start_node_linear_position_list)

    track_segment_id_to_start_node_linear_position = {
        track_graph.edges[e]["edge_id"]: snlp
        for e, snlp in zip(edge_order, start_node_linear_position_list, strict=True)
    }

    start_node_linear_position = np.asarray(
        [
            track_segment_id_to_start_node_linear_position[edge_id]
            for edge_id in track_segment_id
        ]
    )

    track_segment_id_to_edge = {track_graph.edges[e]["edge_id"]: e for e in edge_order}
    start_node_id = np.asarray(
        [track_segment_id_to_edge[edge_id][0] for edge_id in track_segment_id]
    )
    start_node_2D_position = np.asarray(
        [track_graph.nodes[node]["pos"] for node in start_node_id]
    )

    linear_position = start_node_linear_position + (
        np.linalg.norm(start_node_2D_position - projected_track_positions, axis=1)
    )
    linear_position[is_nan] = np.nan

    return (
        linear_position,
        projected_track_positions[:, 0],
        projected_track_positions[:, 1],
    )


def get_linearized_position(
    position: np.ndarray,
    track_graph: Graph,
    edge_order: list[Edge] | None = None,
    edge_spacing: float | Sequence[float] = 0.0,
    use_HMM: bool = False,
    route_euclidean_distance_scaling: float = 1.0,
    sensor_std_dev: float = 5.0,
    diagonal_bias: float = 0.1,
    edge_map: Mapping[int, int | str] | None = None,
) -> pd.DataFrame:
    """Linearize 2D position based on graph representation of track.

    This function converts 2D spatial trajectories into 1D linear positions along a track.
    It's commonly used in neuroscience to analyze animal movement on mazes or tracks.

    Note: All spatial quantities (position, distance, sensor_std_dev) must use consistent
    units. The package is unit-agnostic, but centimeters are commonly used in neuroscience.

    Parameters
    ----------
    position : numpy.ndarray, shape (n_time, n_space)
        2D or 3D position of the animal over time.
        Shape: (n_time, 2) for 2D positions, (n_time, 3) for 3D.
    track_graph : networkx.Graph
        Graph representation of the 2D track. Must be created with make_track_graph()
        or have the required attributes:
        - Nodes: 'pos' attribute with spatial coordinates (x, y)
        - Edges: 'distance' attribute (Euclidean length) and 'edge_id' attribute (unique integer)
    edge_order : list of 2-tuples, optional
        Controls the order of track segments in the 1D linearization. Each element should be
        an edge from the graph as a tuple (node1, node2).

        If None (default), uses the order edges were added to the graph. This is deterministic
        but may not be spatially intuitive. For reproducible results with complex tracks,
        explicitly specify edge_order or use infer_edge_layout() to generate it automatically.

        Example: [(0, 1), (1, 2), (2, 3)] specifies the path through nodes 0→1→2→3.
    edge_spacing : float or list of float, optional
        Controls gaps between track segments in the 1D representation.
        - If float: uniform spacing applied between all consecutive segments
        - If list: custom spacing for each gap (length must be len(edge_order) - 1)
        Default is 0.0 (no gaps).
    use_HMM : bool, optional
        If True, uses Hidden Markov Model to classify which edge the animal is on,
        considering temporal continuity. If False, simply finds the closest edge
        using Euclidean distance. Default is False (faster, good for clean data).
    route_euclidean_distance_scaling : float, optional
        (Only used when use_HMM=True) Controls how much to penalize jumps between
        non-adjacent segments. Smaller values (e.g., 0.1) strongly prefer smooth paths
        with fewer jumps. Larger values (e.g., 10.0) allow more jumps. Default is 1.0.
    sensor_std_dev : float, optional
        (Only used when use_HMM=True) Standard deviation of position sensor noise,
        in the same units as position data. Typical values: 5-10 for camera tracking
        in centimeters. Default is 5.0.
    diagonal_bias : float, optional
        (Only used when use_HMM=True) Biases the HMM to stay on the current segment.
        Range 0.0-1.0, where larger values increase the probability of staying on
        the same segment. Default is 0.1.
    edge_map : dict, optional
        Maps edge IDs to new segment IDs, enabling merging or relabeling of track segments.
        This is useful for treating different spatial paths as equivalent behavioral segments.

        **Merging behavior**: When multiple edges map to the same target ID, they are merged
        into a unified linear coordinate system. Positions on merged edges are projected onto
        their original physical edges, but their linear positions are calculated from a shared
        reference point. This means:
        - Different 2D paths can have the same 1D linear position
        - Positions N units from the start of any merged edge will have the same linear position
        - Useful for T-mazes, figure-8 tracks, or symmetric choice points

        Examples:
        - Relabeling: {0: 100, 1: 101, 2: 102} changes segment IDs without merging
        - Merging: {0: 10, 1: 10, 2: 20} merges edges 0 & 1 into segment 10
        - String labels: {0: 'left_arm', 1: 'right_arm', 2: 'center'} for semantic names

        Default is None (no remapping).

    Returns
    -------
    position_df : pandas.DataFrame
        DataFrame with one row per time point, containing:
        - 'linear_position': 1D position along the linearized track
        - 'track_segment_id': which edge/segment the position is on (after edge_map applied)
        - 'projected_x_position', 'projected_y_position': coordinates projected onto track

    Raises
    ------
    ValueError
        If track_graph is missing required attributes or edge_order contains invalid edges.

    Examples
    --------
    Basic usage with a simple linear track:

    >>> import numpy as np
    >>> from track_linearization import make_track_graph, get_linearized_position
    >>>
    >>> # Create a simple L-shaped track
    >>> node_positions = [(0, 0), (10, 0), (10, 10)]
    >>> edges = [(0, 1), (1, 2)]
    >>> track_graph = make_track_graph(node_positions, edges)
    >>>
    >>> # Animal positions moving along the track
    >>> position = np.array([[2, 0], [5, 0], [10, 3], [10, 7]])
    >>>
    >>> # Linearize the positions
    >>> result = get_linearized_position(position, track_graph)
    >>> result[['linear_position', 'track_segment_id']]
       linear_position  track_segment_id
    0              2.0                 0
    1              5.0                 0
    2             13.0                 1
    3             17.0                 1

    Using custom edge spacing to add gaps:

    >>> result = get_linearized_position(
    ...     position, track_graph, edge_spacing=5.0
    ... )
    >>> # Now positions on segment 1 will be offset by 5 units

    Using HMM for noisy data with temporal continuity:

    >>> result = get_linearized_position(
    ...     position, track_graph,
    ...     use_HMM=True,
    ...     sensor_std_dev=10.0,  # Higher noise tolerance
    ...     diagonal_bias=0.5     # Strong preference to stay on same segment
    ... )

    See Also
    --------
    make_track_graph : Create a properly formatted track graph
    infer_edge_layout : Automatically determine edge_order and edge_spacing
    project_1d_to_2d : Convert linear positions back to 2D coordinates
    """
    import warnings

    # If no edge_order is given, then arange edges in the order passed to
    # construct the track graph
    if edge_order is None:
        edge_order = list(track_graph.edges)

    # Validate track graph structure
    validate_track_graph(track_graph, edge_order)

    # Warning for extremely large edge_spacing values
    if len(edge_order) > 0:
        max_edge_length = max(track_graph.edges[e]["distance"] for e in edge_order)
        if (
            isinstance(edge_spacing, (int, float))
            and edge_spacing > 10 * max_edge_length
        ):
            warnings.warn(
                f"edge_spacing ({edge_spacing:.1f}) is very large compared to typical edge length ({max_edge_length:.1f}).\n"
                f"This may create unexpectedly large gaps in the linearized representation.\n"
                f"Did you mean edge_spacing={edge_spacing/10:.1f}?",
                UserWarning,
                stacklevel=2,
            )

    # Figure out the most probable track segement that correponds to
    # 2D position
    if use_HMM:
        track_segment_id = classify_track_segments(
            track_graph,
            position,
            route_euclidean_distance_scaling=route_euclidean_distance_scaling,
            sensor_std_dev=sensor_std_dev,
            diagonal_bias=diagonal_bias,
        )
    else:
        track_segments = get_track_segments_from_graph(track_graph)
        track_segment_id = find_nearest_segment(track_segments, position)

    # Calculate linear positions using original edge_order
    # This ensures positions are projected onto their actual physical edges
    (
        linear_position,
        projected_x_position,
        projected_y_position,
    ) = _calculate_linear_position(
        track_graph, position, track_segment_id, edge_order, edge_spacing
    )

    # Apply edge_map to create merged edge_order and spacing
    # This ensures merged edges share the same linear position space
    if edge_map is not None:
        linear_position, output_track_segment_id = _apply_edge_map_to_positions(
            linear_position,
            track_segment_id,
            edge_map,
            track_graph,
            edge_order,
            edge_spacing,
        )
    else:
        output_track_segment_id = track_segment_id

    return pd.DataFrame(
        {
            "linear_position": linear_position,
            "track_segment_id": output_track_segment_id,
            "projected_x_position": projected_x_position,
            "projected_y_position": projected_y_position,
        }
    )


def project_1d_to_2d(
    linear_position: np.ndarray,
    track_graph: nx.Graph,
    edge_order: list[Edge],
    edge_spacing: float | Sequence[float] = 0.0,
) -> np.ndarray:
    """
    Map 1-D linear positions back to 2-D coordinates on the track graph.

    Parameters
    ----------
    linear_position : np.ndarray, shape (n_time,)
    track_graph : networkx.Graph
        Same graph you passed to `get_linearized_position`.
        Nodes must have `"pos"`; edges must have `"distance"`.
    edge_order : list[tuple(node, node)]
        Same order you used for linearisation.
    edge_spacing : float or list of float, optional
        Controls the spacing between track segments in 1D position.
        If float, applied uniformly. If list, length must be `len(edge_order) - 1`.

    Returns
    -------
    coords : np.ndarray, shape (n_time, n_space)
        2-D (or 3-D) coordinates corresponding to each 1-D input.
        Positions that fall beyond the last edge are clipped to the last node.
        NaNs in `linear_position` propagate to rows of NaNs.
    """
    linear_position = np.asarray(linear_position, dtype=float)
    n_edges = len(edge_order)

    # --- edge lengths & spacing ------------------------------------------------
    edge_lengths = np.array(
        [track_graph.edges[e]["distance"] for e in edge_order], dtype=float
    )

    if isinstance(edge_spacing, (int, float)):
        gaps = np.full(max(0, n_edges - 1), float(edge_spacing))
    else:
        gaps = np.asarray(edge_spacing, dtype=float)
        if gaps.size != max(0, n_edges - 1):
            raise ValueError("edge_spacing length must be len(edge_order)-1")

    # cumulative start position of each edge
    cumulative = np.concatenate(
        [[0.0], np.cumsum(edge_lengths[:-1] + gaps)]
    )  # shape (n_edges,)

    idx = np.searchsorted(cumulative, linear_position, side="right") - 1
    idx = np.clip(idx, 0, n_edges - 1)  # clamp to valid edge index

    nan_mask = ~np.isfinite(linear_position)
    idx[nan_mask] = 0  # dummy index, will overwrite later

    # param along each chosen edge
    t = (linear_position - cumulative[idx]) / edge_lengths[idx]
    t = np.clip(t, 0.0, 1.0)  # project extremes onto endpoints

    # gather endpoint coordinates
    node_pos = nx.get_node_attributes(track_graph, "pos")
    u = np.array([node_pos[edge_order[i][0]] for i in idx])
    v = np.array([node_pos[edge_order[i][1]] for i in idx])

    coords: np.ndarray = (1.0 - t[:, None]) * u + t[:, None] * v

    # propagate NaNs from the input
    coords[nan_mask] = np.nan
    return coords
