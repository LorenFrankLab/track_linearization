from math import sqrt

import dask
import networkx as nx
import numba
import numpy as np
import pandas as pd
import scipy.stats

np.warnings.filterwarnings('ignore')


def get_track_segments_from_graph(track_graph):
    '''

    Parameters
    ----------
    track_graph : networkx Graph

    Returns
    -------
    track_segments : ndarray, shape (n_segments, n_nodes, n_space)

    '''
    node_positions = nx.get_node_attributes(track_graph, 'pos')
    return np.asarray([(node_positions[node1], node_positions[node2])
                       for node1, node2 in track_graph.edges()])


def project_points_to_segment(track_segments, position):
    '''Finds the closet point on a track segment in terms of Euclidean distance

    Parameters
    ----------
    track_segments : ndarray, shape (n_segments, n_nodes, 2)
    position : ndarray, shape (n_time, 2)

    Returns
    -------
    projected_positions : ndarray, shape (n_time, n_segments, n_space)

    '''
    segment_diff = np.diff(track_segments, axis=1).squeeze(axis=1)
    sum_squares = np.sum(segment_diff ** 2, axis=1)
    node1 = track_segments[:, 0, :]
    nx = (np.sum(segment_diff *
                 (position[:, np.newaxis, :] - node1), axis=2) /
          sum_squares)
    nx[np.where(nx < 0)] = 0.0
    nx[np.where(nx > 1)] = 1.0
    return node1[np.newaxis, ...] + (
        nx[:, :, np.newaxis] * segment_diff[np.newaxis, ...])


def find_projected_point_distance(track_segments, position):
    '''
    '''
    return np.linalg.norm(
        position[:, np.newaxis, :] -
        project_points_to_segment(track_segments, position), axis=2)


def find_nearest_segment(track_segments, position):
    '''Returns the track segment that is closest to the position
    at each time point.

    Parameters
    ----------
    track_segments : ndarray, shape (n_segments, n_nodes, n_space)
    position : ndarray, shape (n_time, n_space)

    Returns
    -------
    segment_id : ndarray, shape (n_time,)

    '''
    distance = find_projected_point_distance(track_segments, position)
    return np.argmin(distance, axis=1)


def euclidean_distance_change(position):
    '''Distance between position at successive time points

    Parameters
    ----------
    position : ndarray, shape (n_time, n_space)

    Returns
    -------
    distance : ndarray, shape (n_time,)

    '''
    distance = np.linalg.norm(position[1:] - position[:-1], axis=1)
    return np.concatenate(([np.nan], distance))


def route_distance(candidates_t_1, candidates_t, track_graph):
    '''

    Parameters
    ----------
    candidates_t_1 : ndarray, shape (n_segments, n_space)
    candidates_t : ndarray, shape (n_segments, n_space)
    track_graph : networkx Graph

    Returns
    -------
    route_distance : ndarray, shape (n_segments, n_segments)

    '''
    # TODO: speedup function. This takes the most time
    n_segments = len(track_graph.edges)
    if np.any(np.isnan(candidates_t) | np.isnan(candidates_t)):
        return np.full((n_segments, n_segments), np.nan)
    node_names = []
    edges = list(track_graph.edges.keys())
    n_original_nodes = len(track_graph.nodes)
    # insert virtual node
    for edge_number, (position_t, position_t_1, (node1, node2)) in enumerate(
            zip(candidates_t, candidates_t_1, edges)):
        node_name_t, node_name_t_1 = f't_0_{edge_number}', f't_1_{edge_number}'
        node_names.append(node_name_t)
        node_names.append(node_name_t_1)
        nx.add_path(track_graph, [node1, node_name_t, node2])
        nx.add_path(track_graph, [node1, node_name_t_1, node2])
        nx.add_path(track_graph, [node_name_t, node_name_t_1])
        track_graph.nodes[node_name_t]['pos'] = tuple(position_t)
        track_graph.nodes[node_name_t_1]['pos'] = tuple(position_t_1)

    # calculate distance
    for node1, node2 in track_graph.edges:
        x1, y1 = track_graph.nodes[node1]['pos']
        x2, y2 = track_graph.nodes[node2]['pos']
        track_graph.edges[(node1, node2)]['distance'] = sqrt(
            (x2 - x1)**2 + (y2 - y1)**2)

    # calculate path distance
    path_distance = scipy.sparse.csgraph.dijkstra(
        nx.to_scipy_sparse_matrix(track_graph, weight='distance'))
    n_total_nodes = len(track_graph.nodes)
    node_ind = np.arange(n_total_nodes)
    start_node_ind = node_ind[n_original_nodes::2]
    end_node_ind = node_ind[n_original_nodes + 1::2]

    track_graph.remove_nodes_from(node_names)

    return path_distance[start_node_ind][:, end_node_ind]


def batch(n_samples, batch_size=1):
    for ind in range(0, n_samples, batch_size):
        yield range(ind, min(ind + batch_size, n_samples))


@dask.delayed
def batch_route_distance(track_graph, projected_track_position_t,
                         projected_track_position_t_1):
    copy_graph = track_graph.copy()
    distances = [route_distance(p_t, p_t_1, copy_graph)
                 for p_t, p_t_1 in zip(projected_track_position_t,
                                       projected_track_position_t_1)]
    return np.stack(distances)


def route_distance_change(position, track_graph):
    track_segments = get_track_segments_from_graph(track_graph)
    projected_track_position = project_points_to_segment(
        track_segments, position)
    n_segments = len(track_segments)

    distances = [np.full((1, n_segments, n_segments), np.nan)]
    projected_track_position_t = projected_track_position[1:]
    projected_track_position_t_1 = projected_track_position[:-1]
    n_time = projected_track_position_t.shape[0]

    for time_ind in batch(n_time, batch_size=10_000):
        distances.append(
            batch_route_distance(
                track_graph, projected_track_position_t[time_ind],
                projected_track_position_t_1[time_ind]))

    return np.concatenate(dask.compute(*distances), axis=0)


def calculate_position_likelihood(position, track_graph, sigma=10):
    track_segments = get_track_segments_from_graph(track_graph)
    projected_position_distance = find_projected_point_distance(
        track_segments, position)
    return (np.exp(-0.5 * (projected_position_distance / sigma) ** 2) /
            (np.sqrt(2 * np.pi) * sigma))


def normalize_to_probability(x, axis=-1):
    '''Ensure the array axis sum to 1

    Parameters
    ----------
    x : ndarray

    Returns
    -------
    normalized_x : ndarray

    '''
    return x / x.sum(axis=axis, keepdims=True)


def calculate_empirical_state_transition(position, track_graph,
                                         scaling=1E-1, diagonal_bias=1E-1):
    '''Calculates the state transition probabilty between track segments by
    favoring route distances that are similar to euclidean distances between
    successive time points.

    Parameters
    ----------
    position : ndarray, shape (n_time, n_space)
    track_graph : networkx Graph
    scaling : float, optional

    Returns
    -------
    state_transition_matrix : shape (n_time, n_segments, n_segments)

    References
    ----------
    .. [1] Newson, P., and Krumm, J. (2009). Hidden Markov map matching through
    noise and sparseness. In Proceedings of the 17th ACM SIGSPATIAL
    International Conference on Advances in Geographic Information Systems,
    (ACM), pp. 336-343.

    '''
    route_and_euclidean_distance_similarity = np.abs(
        route_distance_change(position, track_graph) -
        euclidean_distance_change(position)[:, np.newaxis, np.newaxis])
    exponential_pdf = scipy.stats.expon.pdf(
        route_and_euclidean_distance_similarity, scale=scaling)
    exponential_pdf = normalize_to_probability(exponential_pdf, axis=2)

    n_states = route_and_euclidean_distance_similarity.shape[1]
    exponential_pdf += np.identity(n_states)[np.newaxis] * diagonal_bias

    return normalize_to_probability(exponential_pdf, axis=2)


@numba.njit(cache=True)
def viterbi(initial_conditions, state_transition, likelihood):
    '''Find the most likely sequence of paths using the Viterbi algorithm.

    Note that the state_transition matrix is time-dependent. NaNs are removed
    and placed back in at the end.

    Parameters
    ----------
    initial_conditions : ndarray, shape (n_states,)
    state_transition : ndarray, shape (n_time, n_states, n_states)
    likelihood : ndarray, shape (n_time, n_states)

    Returns
    -------
    state_id : ndarray, shape (n_time,)

    '''

    LOG_EPS = 1e-16
    log_likelihood = np.log(likelihood.copy())
    state_transition = np.log(state_transition.copy() + LOG_EPS)

    n_time, n_states = log_likelihood.shape
    posterior = np.zeros((n_time, n_states))
    max_state_ind = np.zeros((n_time, n_states))

    # initialization
    posterior[0] = np.log(initial_conditions + LOG_EPS) + log_likelihood[0]

    # recursion
    for time_ind in range(1, n_time):
        prior = posterior[time_ind - 1] + state_transition[time_ind]
        for state_ind in range(n_states):
            max_state_ind[time_ind, state_ind] = np.argmax(prior[state_ind])
            posterior[time_ind, state_ind] = (
                prior[state_ind, int(max_state_ind[time_ind, state_ind])]
                + log_likelihood[time_ind, state_ind])

    # termination
    most_probable_state_ind = np.zeros((n_time,))
    most_probable_state_ind[n_time - 1] = np.argmax(posterior[n_time - 1])

    # path back-tracking
    for time_ind in range(n_time - 2, -1, -1):
        most_probable_state_ind[time_ind] = max_state_ind[
            time_ind + 1, int(most_probable_state_ind[time_ind + 1])]

    return most_probable_state_ind


def classify_track_segments(track_graph, position, sensor_std_dev=10,
                            route_euclidean_distance_scaling=1E-1,
                            diagonal_bias=1E-1):
    '''Find the most likely track segment for a given position.

    Tries to make sure the euclidean distance between successive time points
    is similar to the route distance along the graph.

    Parameters
    ----------
    track_graph : networkx Graph
    position : ndarray, shape (n_time, n_space)
    sensor_std_dev : float, optional
        Uncertainty of position sensor.
    route_euclidean_distance_scaling : float, optional
        How much to prefer route distances between successive time points
        that are closer to the euclidean distance. Smaller numbers mean the
        route distance is more likely to be close to the euclidean distance.
    diagonal_bias : float, optional
        Biases the transition matrix to prefer the current track segment.

    Returns
    -------
    segment_id : ndarray, shape (n_time,)

    References
    ----------
    .. [1] Newson, P., and Krumm, J. (2009). Hidden Markov map matching through
    noise and sparseness. In Proceedings of the 17th ACM SIGSPATIAL
    International Conference on Advances in Geographic Information Systems,
    (ACM), pp. 336-343.

    '''
    n_segments = len(track_graph.edges)
    initial_conditions = np.ones((n_segments,))
    state_transition = calculate_empirical_state_transition(
        position, track_graph, scaling=route_euclidean_distance_scaling,
        diagonal_bias=diagonal_bias)
    likelihood = calculate_position_likelihood(
        position, track_graph, sigma=sensor_std_dev)
    is_bad = (np.any(np.isnan(likelihood), axis=1) |
              np.any(np.isinf(np.log(likelihood)), axis=1))
    likelihood = likelihood[~is_bad]
    state_transition = state_transition[~is_bad]
    track_segment_id = viterbi(
        initial_conditions, state_transition, likelihood)
    track_segment_id_with_nan = np.full((is_bad.size,), np.nan)
    track_segment_id_with_nan[~is_bad] = track_segment_id
    return track_segment_id_with_nan


def batch_linear_distance(track_graph, projected_track_positions, edge_ids,
                          linear_zero_node_id):

    copy_graph = track_graph.copy()
    linear_distance = []

    for (x3, y3), (node1, node2) in zip(
            projected_track_positions, edge_ids):

        x1, y1 = copy_graph.nodes[node1]['pos']
        left_distance = sqrt((x3 - x1)**2 + (y3 - y1)**2)
        nx.add_path(copy_graph, [node1, 'projected'], distance=left_distance)

        x2, y2 = copy_graph.nodes[node2]['pos']
        right_distance = sqrt((x3 - x2)**2 + (y3 - y2)**2)
        nx.add_path(copy_graph, ['projected', node2], distance=right_distance)

        linear_distance.append(
            nx.shortest_path_length(copy_graph, source=linear_zero_node_id,
                                    target='projected', weight='distance'))
        copy_graph.remove_node('projected')

    return linear_distance


def _calulcate_linear_position(track_graph, position, track_segment_id,
                               edge_order, edge_spacing):
    is_nan = np.isnan(track_segment_id)
    track_segment_id[is_nan] = 0  # need to check
    track_segment_id = track_segment_id.astype(int)

    track_segments = get_track_segments_from_graph(track_graph)
    projected_track_positions = project_points_to_segment(
        track_segments, position)
    n_time = projected_track_positions.shape[0]
    projected_track_positions = projected_track_positions[(
        np.arange(n_time), track_segment_id)]

    n_edges = len(edge_order)
    if isinstance(edge_spacing, int) | isinstance(edge_spacing, float):
        edge_spacing = [edge_spacing, ] * (n_edges - 1)

    counter = 0.0
    start_node_linear_position = []

    for ind, edge in enumerate(edge_order):
        start_node_linear_position.append(counter)

        try:
            counter += track_graph.edges[edge]["distance"] + edge_spacing[ind]
        except IndexError:
            pass

    start_node_linear_position = np.asarray(start_node_linear_position)

    track_segment_id_to_start_node_linear_position = {
        track_graph.edges[e]["edge_id"]: snlp
        for e, snlp in zip(edge_order, start_node_linear_position)}

    start_node_linear_position = np.asarray([
        track_segment_id_to_start_node_linear_position[edge_id]
        for edge_id in track_segment_id
    ])

    track_segment_id_to_edge = {
        track_graph.edges[e]["edge_id"]: e for e in edge_order}
    start_node_id = np.asarray([track_segment_id_to_edge[edge_id][0]
                                for edge_id in track_segment_id])
    start_node_2D_position = np.asarray(
        [track_graph.nodes[node]["pos"] for node in start_node_id])

    linear_position = start_node_linear_position + (
        np.linalg.norm(start_node_2D_position -
                       projected_track_positions, axis=1))
    linear_position[is_nan] = np.nan

    return (linear_position,
            projected_track_positions[:, 0],
            projected_track_positions[:, 1])


def get_linearized_position(position,
                            track_graph,
                            edge_order=None,
                            edge_spacing=0,
                            use_HMM=False,
                            route_euclidean_distance_scaling=1.0,
                            sensor_std_dev=5.0,
                            diagonal_bias=0.1,
                            ):
    """Linearize 2D position based on graph representation of track.

    Parameters
    ----------
    position : numpy.ndarray, shape (n_time, 2)
        2D position of the animal.
    track_graph : networkx.Graph
        Graph representation of the 2D track.
    edge_order : numpy.ndarray, shape (n_edges, 2), optional
        Controls order of track segments in 1D position. Specify as edges as
        node pairs such as [(node1, node2), (node2, node3)]
    edge_spacing : float or numpy.ndarray, shape (n_edges - 1,), optional
        Controls the spacing between track segments in 1D position
    use_HMM : bool
        If True, then uses HMM to classify the edge the animal is on.
        If False, then finds the closest edge (using euclidean distance).
    route_euclidean_distance_scaling : float, optional
        Used with HMM. How much to prefer route distances between successive
        time points that are closer to the euclidean distance. Smaller
        numbers mean the route distance is more likely to be close to the
        euclidean distance. This favors less jumps. Larger numbers favor
        more jumps.
    sensor_std_dev : float, optional
        Used with HMM. The variability of the sensor used to track position
    diagonal_bias : float between 0.0 and 1.0, optional
        Used with HMM. Bigger values mean the linear position is more likely
        to stick to the current track segment.

    Returns
    -------
    position_df : pandas.DataFrame, shape (n_time, 5)
        'linear_position' - linear position of animal
        'track_segment_id' - the edge the animal is on
        'projected_x_position' - the 2D position projected to the track_graph
        'projected_y_position' - the 2D position projected to the track_graph

    """
    # If no edge_order is given, then arange edges in the order passed to
    # construct the track graph
    if edge_order is None:
        edge_order = list(track_graph.edges)

    # Figure out the most probable track segement that correponds to
    # 2D position
    if use_HMM:
        track_segment_id = classify_track_segments(
            track_graph, position,
            route_euclidean_distance_scaling=route_euclidean_distance_scaling,
            sensor_std_dev=sensor_std_dev,
            diagonal_bias=diagonal_bias)
    else:
        track_segments = get_track_segments_from_graph(track_graph)
        track_segment_id = find_nearest_segment(track_segments, position)

    (linear_position, projected_x_position,
     projected_y_position) = _calulcate_linear_position(
        track_graph, position, track_segment_id, edge_order, edge_spacing)

    return pd.DataFrame({
        'linear_position': linear_position,
        'track_segment_id': track_segment_id,
        'projected_x_position': projected_x_position,
        'projected_y_position': projected_y_position,
    })
