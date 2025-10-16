# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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

### Changed
- **BREAKING**: Dropped support for Python 3.9 (minimum version now 3.10)
- Upgraded minimum dependency versions:
  - numpy >= 1.24 (was unpinned)
  - scipy >= 1.10 (was unpinned)
  - matplotlib >= 3.7 (was unpinned)
  - pandas >= 2.0 (was unpinned)
  - dask[array] >= 2023.5.0 (was unpinned, added array extra)
- Replaced deprecated pandas `.values` with `.to_numpy()` in utils module
- Updated CI workflow to test Python 3.10, 3.11, 3.12, and 3.13
- Replaced old test_package_build.yml with modern release.yml workflow
- Fixed non-breaking hyphen character in error message

### Fixed
- Unused `fig` variables in plotting functions (now use `_` prefix)
- Import sorting and organization per ruff standards
- Regex pattern in pytest match statement (now uses raw string)

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

[Unreleased]: https://github.com/LorenFrankLab/track_linearization/compare/v2.3.2...HEAD
[2.3.2]: https://github.com/LorenFrankLab/track_linearization/compare/v2.3.1...v2.3.2
[2.3.1]: https://github.com/LorenFrankLab/track_linearization/compare/v2.3.0...v2.3.1
[2.3.0]: https://github.com/LorenFrankLab/track_linearization/compare/v2.2.0...v2.3.0
[2.2.0]: https://github.com/LorenFrankLab/track_linearization/compare/v2.1.0...v2.2.0
[2.1.0]: https://github.com/LorenFrankLab/track_linearization/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/LorenFrankLab/track_linearization/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/LorenFrankLab/track_linearization/releases/tag/v1.0.0
