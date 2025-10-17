[![Test, Build, and Publish](https://github.com/LorenFrankLab/track_linearization/actions/workflows/release.yml/badge.svg)](https://github.com/LorenFrankLab/track_linearization/actions/workflows/release.yml)
[![codecov](https://codecov.io/gh/LorenFrankLab/track_linearization/branch/master/graph/badge.svg)](https://codecov.io/gh/LorenFrankLab/track_linearization)

# track_linearization

**Convert 2D spatial trajectories into 1D linear positions along track structures.**

`track_linearization` is a Python package for mapping animal movement on complex track environments (mazes, figure-8s, T-mazes) into simplified 1D representations. It uses Hidden Markov Models to handle noisy position data and provides powerful tools for analyzing spatial behavior in neuroscience experiments.

## Features

- **Flexible Track Representation**: Define any 2D track structure using NetworkX graphs
- **Pre-Built Track Geometries**: 8+ track builders for common experimental setups (T-maze, circular, figure-8, etc.)
- **HMM-Based Classification**: Optional temporal continuity for robust segment classification
- **Edge Merging**: Treat different spatial paths as equivalent behavioral segments
- **Automatic Layout Inference**: Smart edge ordering and spacing for intuitive linearization
- **Interactive Track Builder**: Create tracks from images with mouse clicks (Jupyter-compatible)
- **Validation & Quality Control**: Confidence scoring, outlier detection, and comprehensive quality assessment
- **Visualization Tools**: Plot tracks in 2D and linearized 1D representations

## Installation

### From PyPI (recommended)
```bash
pip install track_linearization
```

### From Conda
```bash
conda install -c franklab track_linearization
```

### From Source
```bash
git clone https://github.com/LorenFrankLab/track_linearization.git
cd track_linearization
pip install -e .
```

### Optional Dependencies
For additional features (performance & visualization):
```bash
pip install track_linearization[opt]  # Includes numba, ipympl
```

## Quick Start

**📚 For a comprehensive tutorial with detailed examples, see [notebooks/track_linearization_tutorial.ipynb](notebooks/track_linearization_tutorial.ipynb)**

### Basic Example: Linear Track

```python
import numpy as np
from track_linearization import make_track_graph, get_linearized_position

# Define a simple L-shaped track
node_positions = [(0, 0), (10, 0), (10, 10)]
edges = [(0, 1), (1, 2)]
track_graph = make_track_graph(node_positions, edges)

# Animal positions moving along the track
position = np.array([
    [2, 0],   # On first segment
    [5, 0],   # Middle of first segment
    [10, 3],  # On second segment
    [10, 7]   # Near end of second segment
])

# Linearize the positions
result = get_linearized_position(position, track_graph)

print(result)
#    linear_position  track_segment_id  projected_x_position  projected_y_position
# 0              2.0                 0                   2.0                   0.0
# 1              5.0                 0                   5.0                   0.0
# 2             13.0                 1                  10.0                   3.0
# 3             17.0                 1                  10.0                   7.0
```

### Adding Edge Spacing

Control gaps between segments in the linearized representation:

```python
# Add 5-unit spacing between segments
result = get_linearized_position(
    position,
    track_graph,
    edge_spacing=5.0
)

# Now segment 1 starts at position 15 (10 + 5 spacing)
# instead of position 10
```

### Using HMM for Noisy Data

For real-world tracking data with noise, use the HMM mode:

```python
result = get_linearized_position(
    position,
    track_graph,
    use_HMM=True,
    sensor_std_dev=10.0,  # Expected position noise in your units
    diagonal_bias=0.5      # Preference to stay on same segment
)
```

## Advanced Usage

### Edge Mapping: Merging Track Segments

The `edge_map` parameter enables powerful behavioral analysis by treating different spatial paths as equivalent:

#### Example: T-Maze Left/Right Arms

```python
# Create a T-maze
#     L        R
#     2        4
#     |        |
# 0---1--------3---5
node_positions = [(0, 0), (10, 0), (10, 10), (20, 0), (20, 10)]
edges = [
    (0, 1),  # Stem (edge_id 0)
    (1, 2),  # Left arm (edge_id 1)
    (1, 3),  # Center (edge_id 2)
    (3, 4),  # Right arm (edge_id 3)
]
track_graph = make_track_graph(node_positions, edges)

# Positions 5 units up each arm
position = np.array([
    [10, 5],  # 5 units up left arm
    [20, 5],  # 5 units up right arm
])

# Merge left and right arms into single "choice_arm" segment
edge_map = {1: 'choice_arm', 3: 'choice_arm'}

result = get_linearized_position(
    position,
    track_graph,
    edge_map=edge_map
)

print(result)
#    linear_position  track_segment_id  projected_x  projected_y
# 0              5.0        choice_arm          10.0          5.0
# 1              5.0        choice_arm          20.0          5.0

# Both positions have linear_position = 5.0!
# This treats left/right choice arms as behaviorally equivalent
```

**Key insight**: Positions equidistant from the start of merged edges have identical linear positions, enabling behavioral analyses that ignore spatial distinctions.

#### Use Cases for Edge Mapping

1. **T-mazes**: Merge left/right choice arms
2. **Figure-8 tracks**: Merge symmetric loops
3. **Multiple paths**: Treat different routes to the same goal as equivalent
4. **Semantic labeling**: Use strings instead of integers for clearer analysis

```python
# Example: Semantic labels for analysis
edge_map = {
    0: 'start',
    1: 'left_arm',
    2: 'right_arm',
    3: 'goal'
}
```

### Custom Edge Order

Control how segments are arranged in the linearization:

```python
# Specify exact edge order
edge_order = [(1, 2), (0, 1), (2, 3)]  # Custom path through nodes

result = get_linearized_position(
    position,
    track_graph,
    edge_order=edge_order
)

# Or use automatic inference
from track_linearization import infer_edge_layout

edge_order, edge_spacing = infer_edge_layout(
    track_graph,
    start_node=0  # Begin linearization at node 0
)
```

### Variable Edge Spacing

Use different spacing between each pair of segments:

```python
# Custom spacing for each gap
edge_spacing = [5, 10, 3]  # Different spacing between each pair

result = get_linearized_position(
    position,
    track_graph,
    edge_spacing=edge_spacing
)
```

## Visualization

```python
from track_linearization import plot_track_graph, plot_graph_as_1D

# Plot the 2D track structure
plot_track_graph(track_graph, show=True)

# Plot the linearized 1D representation
plot_graph_as_1D(
    track_graph,
    edge_order=edge_order,
    edge_spacing=5.0,
    show=True
)
```

## API Reference

### Core Functions

- **`get_linearized_position()`**: Main function to convert 2D positions to 1D
- **`make_track_graph()`**: Create a properly formatted track graph from node positions and edges
- **`infer_edge_layout()`**: Automatically determine optimal edge ordering and spacing

### Utility Functions

- **`plot_track_graph()`**: Visualize 2D track structure
- **`plot_graph_as_1D()`**: Visualize linearized 1D representation
- **`project_1d_to_2d()`**: Convert linear positions back to 2D coordinates

For detailed API documentation, see the docstrings in each function or visit [the documentation](https://github.com/LorenFrankLab/track_linearization).

## How It Works

1. **Graph Representation**: Tracks are represented as NetworkX graphs where nodes have 2D positions and edges represent valid paths
2. **Segment Classification**: For each position, determine which track segment (edge) the animal is on using either:
   - Nearest-neighbor (fast, for clean data)
   - HMM inference (robust, for noisy data with temporal continuity)
3. **Projection**: Project 2D position onto the identified edge
4. **Linearization**: Calculate 1D position based on distance along the edge from a reference point, with optional spacing between segments
5. **Edge Mapping**: Optionally merge or relabel segments to create unified coordinate systems for behavioral analysis

## Requirements

- Python 3.10+
- numpy
- scipy
- pandas
- matplotlib
- networkx >= 3.2.1
- dask

**Optional**:
- numba (for acceleration)
- ipympl (for interactive plots)

## Developer Installation

For development with testing and linting tools:

```bash
# Clone the repository
git clone https://github.com/LorenFrankLab/track_linearization.git
cd track_linearization

# Create conda environment
conda env create -f environment.yml
conda activate track_linearization

# Install in editable mode with dev dependencies
pip install -e .[dev]

# Run tests
pytest src/track_linearization/tests/ -v

# Run linting
ruff check src/
mypy src/track_linearization/
```

## Testing

```bash
# Run all tests
pytest src/track_linearization/tests/

# Run with coverage
pytest src/track_linearization/tests/ --cov=track_linearization --cov-report=html

# Run specific test file
pytest src/track_linearization/tests/test_core.py -v
```

## Citation

If you use this package in your research, please cite:

```bibtex
@software{track_linearization,
  title = {track_linearization: 2D to 1D position linearization using HMMs},
  author = {{Loren Frank Lab}},
  url = {https://github.com/LorenFrankLab/track_linearization},
  year = {2024}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Learning Resources

### Tutorial Notebooks
📚 **[Interactive Tutorial](notebooks/track_linearization_tutorial.ipynb)** - A comprehensive, pedagogically-designed notebook covering:
- Simple to complex track examples (linear, L-shaped, circular, W-track)
- Edge mapping for behavioral analysis (T-maze example)
- HMM-based classification for noisy data
- Best practices and troubleshooting tips

Perfect for scientists new to the package or computational spatial analysis!

🔧 **[Advanced Features Tutorial](notebooks/advanced_features_tutorial.ipynb)** - Learn about new features in v2.4+:
- Pre-built track geometries (T-maze, plus maze, figure-8, etc.)
- Validation & quality control for linearization
- Interactive track builder from images (Jupyter-compatible)
- Outlier detection and confidence scoring
- Real-world analysis workflows

## Support

- **Issues**: Report bugs or request features via [GitHub Issues](https://github.com/LorenFrankLab/track_linearization/issues)
- **Documentation**: See docstrings in the code for detailed API documentation
- **Tutorial**: Start with [track_linearization_tutorial.ipynb](notebooks/track_linearization_tutorial.ipynb)
- **Examples**: Check the `notebooks/` directory for additional Jupyter notebook examples

## Related Projects

- [replay_trajectory_classification](https://github.com/LorenFrankLab/replay_trajectory_classification) - Decode spatial trajectories from neural activity
- [spyglass](https://github.com/LorenFrankLab/spyglass) - Analysis framework for systems neuroscience

## Acknowledgments

Developed by the [Loren Frank Lab](https://www.cin.ucsf.edu/HTML/Loren_Frank.html) at UCSF for analyzing spatial behavior in neuroscience experiments.
