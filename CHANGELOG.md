# Changelog

All notable changes to the Neural Dimensionality Tracker will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased -- Supply chain

### Fixed
- **Scorecard workflow** failed on every run: `ossf/scorecard-action` v2.4.0 pulls
  its image from `gcr.io`, which now refuses the request. Moved to v2.4.4.
- **Empty matrices** (a zero dimension) crashed every estimator with `IndexError`;
  they now raise `ValueError` like the other invalid inputs. Found while writing the
  fuzz harness.

### Changed
- **Minimum versions with no known vulnerabilities**: `torch>=2.13.0` (23 advisories
  below it) and, in `requirements.txt`, `scikit-learn>=1.5.0`.
- **CI installs pinned by hash**: tests, docs, fuzzing and the build install from
  `.github/requirements/*.txt` with `--require-hashes`, and the package itself with
  `--no-deps`. Dependabot keeps those files current.
- **Releases**: a tag `v*` builds the package, publishes it with
  `pypa/gh-action-pypi-publish`, and creates the GitHub release with every file
  signed by sigstore (`.sig` + `.pem`). The `release: published` trigger is gone, so a
  release is no longer published twice.

### Added
- **Fuzzing**: `fuzz/fuzz_estimators.py` (Atheris) feeds arbitrary matrices to the
  five estimators; CI runs it for two minutes on pull requests that touch
  `src/ndt/core` and weekly.

## 2.0.0 -- The validity harness

### Changed
- **Toolchain unified to `ruff`** (formatting, linting and import sorting), replacing
  black + isort + flake8, and pinned to a single version. This ends the format-drift
  that came from three unpinned formatters on an EOL Python. CI runs `ruff check` and
  `ruff format --check`. Applied `raise ... from e` in the SVD error paths and cleaned
  unused loop variables surfaced by the linter.

### Added
- **`ndt.validity`**: a validity harness for transition detectors. Run any
  detector against ground-truth fixtures (`planted_transition`, `planted_multi`,
  `pure_noise`, `drift_no_jump`) and get a `ValidityReport` with recall on known
  transitions and false-positive rate on known nulls, plus a plain verdict.
  Synthetic fixtures cost nothing to generate; real-model fixtures (e.g. a
  grokking run) plug into the same `Fixture` interface. New public API:
  `validate_detector`, `jump_detector_as_callable`, `ValidityReport`,
  `standard_battery`.

### Changed
- **Positioning corrected.** The README no longer claims ndt reliably "detects
  discrete phase transitions." The companion study shows detectors built on
  these metrics disagree with each other (threshold vs PELT correlate at -0.029)
  and none is consistent across metrics. The included `JumpDetector` is now
  documented as one detector to be validated, not a truth oracle. ndt's
  distinctive value is the check, not the claim.
- License field in `pyproject.toml` set to Apache-2.0 to match the LICENSE file
  and the badge (was MIT).
- Repository moved from the groundlens-dev organization to the author's account
  (`Javihaus/ndt`); all URLs updated.

## [1.0.2] - 2025-11-13

### Changed
- Package name changed to `ndtracker` on PyPI (name `ndt` was already taken)
- Install command: `pip install ndtracker`
- Import remains: `import ndt`

### Fixed
- Fixed version synchronization between `pyproject.toml` and `__version__.py`
- Package now correctly reports version 1.0.2 when installed

## [1.0.1] - 2025-11-13

### Fixed
- Merge and synchronization updates for stability

## [1.0.0] - 2025-11-13

### Changed
- Package name changed to `ndtracker` on PyPI (name `ndt` was already taken)
  - Install with `pip install ndtracker`
  - Import as `import ndt` (module name unchanged)
  - Updated package name for PyPI compatibility

### Fixed
- PyPI packaging compatibility by constraining setuptools to <70.0
- Removed duplicate license classifier to avoid metadata conflicts
- Fixed GitHub Actions publish workflow for tag push events
- Package now passes twine validation checks

### Added
- Codecov integration with v5 action for test coverage reporting

## [0.1.0] - 2024-11-12

### Added
- Initial release of Neural Dimensionality Tracker
- Core dimensionality estimators:
  - Stable rank
  - Participation ratio
  - Cumulative energy 90%
  - Nuclear norm ratio
- HighFrequencyTracker class for automatic dimensionality monitoring
- Forward hook system for activation capture
- Jump detection using Z-score analysis
- Architecture handlers for:
  - Multi-Layer Perceptrons (MLP)
  - Convolutional Neural Networks (CNN)
  - Transformers
  - Vision Transformers (ViT)
- Visualization utilities:
  - Static plots with Matplotlib
  - Interactive plots with Plotly
  - Dashboard creation
- Export functionality:
  - CSV export
  - JSON export
  - HDF5 export (for large-scale data)
- Comprehensive test suite (>90% coverage)
- Examples:
  - Quickstart with MNIST
  - CNN with CIFAR-10
- CI/CD pipelines:
  - Automated testing on push/PR
  - PyPI publishing on release
- Documentation:
  - README with quickstart guide
  - Contributing guidelines
  - Issue templates
  - Example code

### Features
- Minimal intrusion (3-line integration)
- Architecture-agnostic design
- Automatic layer detection
- Context manager support
- Configuration file support
- Memory-efficient streaming computation
- Full type hints and mypy support

## [Unreleased]

### Planned
- Transformer example with real language model
- Vision Transformer example with ImageNet
- Real-time monitoring dashboard
- JAX support
- Additional dimensionality metrics
- Phase detection algorithms
- Correlation analysis tools
- Model comparison utilities
- Documentation website with MkDocs
