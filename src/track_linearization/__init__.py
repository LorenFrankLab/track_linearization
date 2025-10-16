# flake8: noqa
from track_linearization.core import (
    get_linearized_position,
    project_1d_to_2d,
    validate_track_graph,
)
from track_linearization.track_builders import (
    make_circular_track,
    make_figure8_track,
    make_linear_track,
    make_plus_maze_track,
    make_rectangular_track,
    make_tmaze_track,
    make_track_from_image_interactive,
    make_wtrack,
    make_ymaze_track,
)
from track_linearization.utils import (
    infer_edge_layout,
    make_actual_vs_linearized_position_movie,
    make_track_graph,
    plot_graph_as_1D,
    plot_track_graph,
)
from track_linearization.validation import (
    check_track_graph_validity,
    detect_linearization_outliers,
    get_projection_confidence,
    validate_linearization,
)

try:
    from ._version import __version__
except ImportError:
    # Fallback for development installs
    from importlib.metadata import version

    __version__ = version("track_linearization")
