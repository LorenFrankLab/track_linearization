# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

track_linearization is a Python package for mapping 2D trajectories to 1D using Hidden Markov Models (HMMs). The core functionality involves converting animal movement tracks on 2D environments (like mazes) into linear representations.

## Project Structure

- `src/track_linearization/`: Main package source
  - `core.py`: Core algorithms for position linearization, track segment projection, and HMM-based analysis
  - `utils.py`: Utility functions for graph construction, plotting, and visualization
  - `__init__.py`: Package initialization with main API exports
  - `tests/`: Test suite with comprehensive coverage
- `notebooks/`: Jupyter notebooks for examples and analysis
  - `track_linearization_tutorial.ipynb`: **Comprehensive pedagogical tutorial** (recommended starting point)
  - `test_linearization.ipynb`: Original test examples
- `build/`: Build artifacts (generated)
- `htmlcov/`: Test coverage reports (generated)

## Key Architecture

The package follows a two-module design:

1. **Core Module** (`core.py`): Contains the main algorithms
   - `get_linearized_position()`: Main API function for 2D→1D conversion
   - Track segment projection and distance calculations
   - HMM-based position estimation using scipy sparse matrices
   - **Edge mapping**: Merge or relabel track segments for behavioral analysis
   - Optional numba acceleration when available

2. **Utils Module** (`utils.py`): Graph construction and visualization
   - `make_track_graph()`: Creates NetworkX graphs from node positions and edges
   - Plotting functions for track visualization and animations
   - `infer_edge_layout()`: Automatically determine edge order and spacing (renamed from `get_auto_linear_edge_order_spacing`)
   - Edge ordering and spacing utilities

The package uses NetworkX graphs to represent tracks, where nodes have `pos` attributes (spatial coordinates) and edges have `distance` and `edge_id` attributes.

### Edge Map Implementation (core.py:1022-1097)

The `edge_map` parameter enables merging track segments into unified linear coordinate systems:

**How it works:**
1. Calculate linear positions using original edge_order (preserves accurate 2D→edge projection)
2. Build merged coordinate system by identifying which edges map to same target
3. Adjust linear positions: calculate offset within original edge, add to merged edge's start position
4. Apply edge_map to output segment IDs with proper dtype handling (supports strings)

**Key behavior:** Positions N units from the start of any merged edge have the **same** linear position, even if they're at different 2D locations. This enables behavioral analyses that treat different spatial paths as equivalent (e.g., T-maze left/right arms).

## Development Commands

### Testing

```bash
# Run all tests with coverage
python -m pytest src/track_linearization/tests/ -v

# Run specific test file
python -m pytest src/track_linearization/tests/test_core.py -v

# Run with coverage report
python -m pytest src/track_linearization/tests/ -v --cov=track_linearization --cov-report=html --cov-report=term-missing

# Run single test
python -m pytest src/track_linearization/tests/test_core.py::test_specific_function -v
```

### Code Quality (requires dev dependencies)

```bash
# Install dev dependencies first
pip install -e .[dev]

# Linting with ruff
python -m ruff check src/track_linearization/

# Auto-fix ruff issues
python -m ruff check src/track_linearization/ --fix

# Type checking with mypy
python -m mypy src/track_linearization/

# Code formatting with black
python -m black src/track_linearization/

# Format check only (no changes)
python -m black --check src/track_linearization/
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually on all files
pre-commit run --all-files

# Run specific hook
pre-commit run black --all-files
```

### Building

```bash
# Build package
python -m build

# Install in development mode
pip install -e .

# Install with optional dependencies
pip install -e .[opt]  # numba, ipympl for performance/visualization
```

### Release Process

1. Update CHANGELOG.md with changes
2. Create and push a version tag:
   ```bash
   git tag v2.4.0
   git push origin v2.4.0
   ```
3. GitHub Actions automatically:
   - Runs tests across Python 3.10-3.13
   - Builds wheel and sdist
   - Publishes to PyPI via Trusted Publishing
   - Creates GitHub Release

See `.github/PYPI_TRUSTED_PUBLISHING.md` for detailed setup instructions.

## Dependencies

- **Core** (with minimum versions):
  - numpy >= 1.24
  - scipy >= 1.10
  - matplotlib >= 3.7
  - pandas >= 2.0
  - dask[array] >= 2023.5.0
  - networkx >= 3.2.1
- **Optional performance**: numba (for acceleration)
- **Optional visualization**: ipympl (for interactive plots)
- **Development**: black, pytest, pytest-cov, ruff, mypy, pandas-stubs, pre-commit

## Testing Configuration

Tests are configured in `pyproject.toml` with:

- Test path: `src/track_linearization/tests`
- Coverage reporting enabled by default
- Doctest modules included
- Python 3.10+ support

## Important Notes

- The package detects numba availability at runtime and uses acceleration when possible
- Track graphs must have nodes with `pos` attributes (spatial coordinates)
- Edge attributes include `distance` (Euclidean) and `edge_id` for identification
- Version is managed via hatch-vcs from git tags
