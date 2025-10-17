# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.4.0] - 2025-10-16

### Added
- **Track Builders Module** (`track_builders.py`):
  - `make_linear_track()` - Create simple linear tracks
  - `make_circular_track()` - Create circular/annular tracks
  - `make_tmaze_track()` - Create T-maze tracks for alternation tasks
  - `make_plus_maze_track()` - Create plus/cross maze tracks
  - `make_figure8_track()` - Create figure-8 tracks
  - `make_wtrack()` - Create W-shaped tracks
  - `make_rectangular_track()` - Create rectangular perimeter tracks
  - `make_ymaze_track()` - Create Y-maze tracks with configurable angles
  - `make_track_from_points()` - Create tracks from manual point specification
  - `make_track_from_image_interactive()` - Interactive track builder from images (Jupyter-compatible)
  - `_build_track_from_state()` - Helper for retrieving interactive builder results in Jupyter

- **Validation & QC Module** (`validation.py`):
  - `check_track_graph_validity()` - Validate track graph structure and attributes
  - `get_projection_confidence()` - Calculate confidence scores for position projections
  - `detect_linearization_outliers()` - Detect outliers using projection distance and jump detection
  - `validate_linearization()` - Comprehensive quality assessment with scoring and recommendations

- **Tutorial Notebooks**:
  - `track_linearization_tutorial.ipynb` - Comprehensive pedagogical tutorial for basic usage
  - `advanced_features_tutorial.ipynb` - Tutorial covering track builders, validation, and interactive features

- **Core Functionality**:
  - `project_1d_to_2d()` - Reverse mapping from 1D linear positions back to 2D coordinates
  - Numba-optimized Viterbi algorithm for HMM inference (when numba available)
  - Exposed `project_1d_to_2d` in package `__init__.py`

- **Infrastructure**:
  - Pre-commit hooks configuration for automated code quality checks
  - Comprehensive CI/CD pipeline with quality, test, build, and publish jobs
  - Support for Python 3.13
  - PyPI Trusted Publishing (OIDC) support for secure releases
  - Automated GitHub release creation with changelog extraction
  - Notebook execution testing in CI
  - Modern dependency pinning with lower bounds following Scientific Python SPEC 0
  - Enhanced ruff configuration with numpy-specific and pandas-vet rules
  - Improved mypy configuration with test-specific overrides
  - pandas-stubs for better type checking
  - Comprehensive test suite for track builders and validation (117 tests total)

### Changed
- **BREAKING**: Dropped support for Python 3.9 (minimum version now 3.10)
- Upgraded minimum dependency versions:
  - numpy >= 1.24 (was unpinned)
  - scipy >= 1.10 (was unpinned)
  - matplotlib >= 3.7 (was unpinned)
  - pandas >= 2.0 (was unpinned)
  - dask[array] >= 2023.5.0 (was unpinned, added array extra)
  - networkx >= 3.2.1 (explicit minimum for compatibility)
- Replaced deprecated pandas `.values` with `.to_numpy()` in utils module
- Updated CI workflow to test Python 3.10, 3.11, 3.12, and 3.13
- Replaced old test_package_build.yml with modern release.yml workflow
- Fixed non-breaking hyphen character in error message
- Enhanced `__init__.py` with comprehensive module docstring
- Improved README with new features section and tutorial links
- Interactive track builder uses two-step workflow in Jupyter (non-blocking)
- Outlier detection uses robust statistics (median + MAD) instead of mean + std

### Fixed
- **Type Annotations & IDE Support**: Improved type hints for better IDE autocomplete and type checking
  - `edge_map` now accepts both integer and string segment IDs (e.g., `{0: "left_arm", 1: "right_arm"}`)
  - `edge_spacing` now accepts tuples, lists, and numpy arrays (previously only lists)
  - Achieved zero mypy errors in source code for improved code quality
- Outlier detection false positives on uniform data (now uses robust statistics)
- Interactive builder event loop blocking in Jupyter notebooks (non-blocking two-step workflow)

### Infrastructure
- New GitHub Actions workflow structure:
  - Separate quality, test, build, and publish jobs
  - Parallel testing across Python versions
  - Artifact management for distributions
  - Integration with Codecov for coverage reporting

## [2.3.2] - 2024-10-22

### Changed
- Revamped README with detailed usage and documentation
- Fixed minor notebook formatting and output updates
- Removed unused import from __init__.py
- Refactored formatting for readability in core and tests

## [2.3.1] - 2024-10-16

### Added
- Support for merging edges in edge_map parameter

### Changed
- Improved track graph validation and documentation
- Refactored type hints and added strict zip usage
- Enhanced type hints and formatting for Python 3.10+

## [2.3.0] - 2024-09-24

### Added
- Comprehensive tests for core and utils modules
- Fallback for __version__ using importlib.metadata

### Changed
- Updated Python support matrix
- Added development tooling configurations
- Fixed edge_spacing error handling

## [2.2.0] - 2024-06-07

### Changed
- Added node ID to plot functionality
- Various CI/CD improvements

## [2.1.0] - 2024-05-28

### Changed
- Updated version management
- CI/CD workflow improvements

## [2.0.0] - 2023-08-21

### Changed
- Major refactoring and restructuring

## [1.0.0] - 2020-08-21

### Added
- Initial release with core linearization functionality
- Track graph construction
- HMM-based position classification
- Visualization tools

[Unreleased]: https://github.com/LorenFrankLab/track_linearization/compare/v2.4.0...HEAD
[2.4.0]: https://github.com/LorenFrankLab/track_linearization/compare/v2.3.2...v2.4.0
[2.3.2]: https://github.com/LorenFrankLab/track_linearization/compare/v2.3.1...v2.3.2
[2.3.1]: https://github.com/LorenFrankLab/track_linearization/compare/v2.3.0...v2.3.1
[2.3.0]: https://github.com/LorenFrankLab/track_linearization/compare/v2.2.0...v2.3.0
[2.2.0]: https://github.com/LorenFrankLab/track_linearization/compare/v2.1.0...v2.2.0
[2.1.0]: https://github.com/LorenFrankLab/track_linearization/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/LorenFrankLab/track_linearization/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/LorenFrankLab/track_linearization/releases/tag/v1.0.0
