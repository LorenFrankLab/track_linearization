"""Track Linearization Package.

A Python package for converting 2D/3D trajectories to 1D using Hidden Markov Models.

Main Functions
--------------
get_linearized_position : Convert 2D positions to 1D linear coordinates
project_1d_to_2d : Reverse mapping from 1D back to 2D positions
validate_track_graph : Validate track graph structure

Track Builders
--------------
make_linear_track : Simple straight track
make_circular_track : Circular track
make_tmaze_track : T-maze with stem and arms
make_plus_maze_track : Plus/cross maze
make_figure8_track : Figure-8 track
make_wtrack : W-shaped track
make_rectangular_track : Rectangular track
make_ymaze_track : Y-maze with three arms
make_track_from_points : Create track from manual points
make_track_from_image_interactive : Interactive track builder from images (Jupyter compatible)

Validation & Quality Control
-----------------------------
check_track_graph_validity : Validate track graph structure
get_projection_confidence : Calculate confidence scores for linearization
detect_linearization_outliers : Detect suspicious/outlier positions
validate_linearization : Comprehensive quality assessment

Utilities
---------
make_track_graph : Create NetworkX graph from positions and edges
infer_edge_layout : Automatically determine edge order and spacing
plot_track_graph : Visualize track graph in 2D
plot_graph_as_1D : Visualize track as 1D representation
make_actual_vs_linearized_position_movie : Create animation of linearization

For detailed usage, see the tutorial notebook in `notebooks/track_linearization_tutorial.ipynb`
or visit: https://github.com/edeno/track_linearization
"""

# flake8: noqa
from track_linearization.core import (
    get_linearized_position,
    project_1d_to_2d,
    validate_track_graph,
)
from track_linearization.track_builders import (
    _build_track_from_state,
    make_circular_track,
    make_figure8_track,
    make_linear_track,
    make_plus_maze_track,
    make_rectangular_track,
    make_tmaze_track,
    make_track_from_image_interactive,
    make_track_from_points,
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
