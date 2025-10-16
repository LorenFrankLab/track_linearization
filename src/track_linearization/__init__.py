# flake8: noqa
from track_linearization.core import (
    get_linearized_position,
    project_1d_to_2d,
    validate_track_graph,
)
from track_linearization.utils import (
    infer_edge_layout,
    make_actual_vs_linearized_position_movie,
    make_track_graph,
    plot_graph_as_1D,
    plot_track_graph,
)

try:
    from ._version import __version__
except ImportError:
    # Fallback for development installs
    from importlib.metadata import version

    __version__ = version("track_linearization")
