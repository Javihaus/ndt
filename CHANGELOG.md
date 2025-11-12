# Changelog

All notable changes to the Neural Dimensionality Tracker will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
