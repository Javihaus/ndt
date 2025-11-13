# Neural Dimensionality Tracker (NDT) - Comprehensive Repository Overview

## Project Summary

Neural Dimensionality Tracker (NDT) is a Python library for high-frequency monitoring of neural network representational dimensionality during training. It enables researchers and practitioners to track how neural network internal representations evolve, detect phase transitions, and gain insights into the learning dynamics of deep neural networks.

**Key Characteristics:**
- **Version:** 0.1.0 (Beta, just released)
- **License:** MIT
- **Author:** Javier Marín
- **Repository:** https://github.com/Javihaus/ndt
- **Python Support:** 3.8+ (tested on 3.8, 3.9, 3.10, 3.11)
- **Status:** Production-ready with >90% test coverage

---

## 1. Project Layout & Directory Structure

```
/home/user/ndt/
├── .github/                          # GitHub configuration
│   ├── workflows/
│   │   ├── tests.yml                # Multi-platform testing pipeline
│   │   └── publish.yml              # PyPI publishing automation
│   └── ISSUE_TEMPLATE/
│       ├── bug_report.md
│       └── feature_request.md
├── src/ndt/                          # Main source code (2,690 lines)
│   ├── core/                         # Core functionality
│   │   ├── tracker.py               # HighFrequencyTracker class (365 lines)
│   │   ├── estimators.py            # Dimensionality metrics (250 lines)
│   │   ├── hooks.py                 # Activation capture (131 lines)
│   │   ├── jump_detector.py         # Phase transition detection (208 lines)
│   │   └── __init__.py
│   ├── architectures/               # Architecture-specific handlers
│   │   ├── base.py                  # Base handler class (59 lines)
│   │   ├── mlp.py                   # MLP handler (72 lines)
│   │   ├── cnn.py                   # CNN handler (129 lines)
│   │   ├── transformer.py           # Transformer handler (108 lines)
│   │   ├── vit.py                   # Vision Transformer handler (126 lines)
│   │   ├── registry.py              # Architecture detection (123 lines)
│   │   └── __init__.py
│   ├── export/                      # Export functionality
│   │   ├── csv.py                   # CSV export (113 lines)
│   │   ├── json.py                  # JSON export (103 lines)
│   │   ├── hdf5.py                  # HDF5 export (167 lines)
│   │   └── __init__.py
│   ├── visualization/               # Visualization utilities
│   │   ├── plots.py                 # Matplotlib plots (273 lines)
│   │   ├── interactive.py           # Plotly dashboards (251 lines)
│   │   └── __init__.py
│   ├── utils/                       # Utilities
│   │   ├── config.py                # Configuration handling (77 lines)
│   │   ├── logging.py               # Logging setup (54 lines)
│   │   └── __init__.py
│   ├── __init__.py                  # Package exports
│   └── __version__.py               # Version: "0.1.0"
├── tests/                            # Test suite (5 test files)
│   ├── conftest.py                  # Pytest fixtures
│   ├── test_tracker.py              # HighFrequencyTracker tests
│   ├── test_estimators.py           # Dimensionality metric tests
│   ├── test_jump_detection.py       # Jump detection tests
│   └── __init__.py
├── examples/                         # Working examples (2)
│   ├── 01_quickstart_mnist.py       # MNIST + MLP example
│   ├── 02_cnn_cifar10.py            # CIFAR-10 + CNN example
│   └── README.md                    # Examples documentation
├── docs/                             # Documentation
│   └── quickstart.md                # 184-line quickstart guide
├── Configuration Files
│   ├── pyproject.toml               # Modern Python project config
│   ├── setup.py                     # Fallback setup script
│   ├── MANIFEST.in                  # Package manifest
│   └── .gitattributes               # Git attributes
├── Documentation
│   ├── README.md                    # Main project overview
│   ├── INSTALL.md                   # Installation guide
│   ├── CONTRIBUTING.md              # Contribution guidelines
│   ├── CODE_OF_CONDUCT.md           # Code of conduct
│   └── CHANGELOG.md                 # Version history
├── TDS Reference
│   └── I Measured Neural Network Training Every 5 Steps...
│       _TDS Contributor Portal.pdf  # TDS article (8 pages)
├── .gitignore                        # Git ignore rules
└── LICENSE                           # MIT License

```

---

## 2. Source Code Organization (src/ndt/)

### Core Module (src/ndt/core/)
- **tracker.py** - Main `HighFrequencyTracker` class
  - High-level API for dimensionality tracking
  - Manages activation capture and metric computation
  - Supports context manager protocol
  - 365 lines

- **estimators.py** - Dimensionality metric computations
  - `stable_rank()` - Effective dimensionality (robust to noise)
  - `participation_ratio()` - Variance distribution evenness
  - `cumulative_energy_90()` - Components for 90% variance
  - `nuclear_norm_ratio()` - Normalized rank measure
  - `compute_all_metrics()` - Batch computation
  - 250 lines

- **hooks.py** - Forward hook mechanism
  - `ActivationCapture` class for layer activation capture
  - Efficient batched computation
  - 131 lines

- **jump_detector.py** - Phase transition detection
  - `JumpDetector` class for anomaly detection
  - `Jump` dataclass for results
  - Z-score based detection algorithm
  - 208 lines

### Architecture Module (src/ndt/architectures/)
- **base.py** - Abstract base for architecture handlers
- **mlp.py** - Multi-Layer Perceptron support
- **cnn.py** - Convolutional Neural Network support
- **transformer.py** - Standard Transformer support
- **vit.py** - Vision Transformer support
- **registry.py** - Auto-detection and handler factory

### Export Module (src/ndt/export/)
- **csv.py** - Export to CSV format (113 lines)
- **json.py** - Export to JSON format (103 lines)
- **hdf5.py** - Export to HDF5 format, memory-efficient (167 lines)

### Visualization Module (src/ndt/visualization/)
- **plots.py** - Matplotlib static visualizations (273 lines)
  - `plot_phases()` - Multi-layer dimensionality plot
  - `plot_jumps()` - Phase transition visualization
  - `plot_metrics_comparison()` - Compare metrics
  - `plot_single_metric()` - Single layer analysis

- **interactive.py** - Plotly interactive dashboards (251 lines)
  - `create_interactive_plot()` - Interactive Plotly plots
  - `create_multi_layer_plot()` - Multi-layer dashboard

### Utils Module (src/ndt/utils/)
- **config.py** - Configuration file handling (77 lines)
- **logging.py** - Logging setup utilities (54 lines)

---

## 3. Configuration Files

### pyproject.toml
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]

[project]
name = "neural-dimensionality-tracker"
version = "0.1.0"
requires-python = ">=3.8"

[project.dependencies]
- torch >= 1.12.0
- numpy >= 1.21.0
- pandas >= 1.3.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- plotly >= 5.0.0
- scipy >= 1.7.0
- tqdm >= 4.62.0
- pyyaml >= 6.0
- h5py >= 3.6.0

[project.optional-dependencies]
dev = [pytest, pytest-cov, black, isort, flake8, mypy]
docs = [mkdocs, mkdocs-material, mkdocstrings]
jax = [jax, jaxlib]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --cov=ndt --cov-report=html --cov-report=term-missing"

[tool.black]
line-length = 100
target-version = ['py38', 'py39', 'py310', 'py311']

[tool.isort]
profile = "black"
line_length = 100
force_single_line = true

[tool.mypy]
python_version = "3.8"
disallow_untyped_defs = true
```

### setup.py
- Minimal fallback for older pip versions
- Delegates to pyproject.toml

### MANIFEST.in
- Specifies package data for distribution

---

## 4. CI/CD Setup (.github/workflows/)

### tests.yml
**Purpose:** Continuous integration testing on every push and PR

**Configuration:**
- **Matrix Testing:**
  - OS: ubuntu-latest, macos-latest, windows-latest
  - Python: 3.8, 3.9, 3.10, 3.11
  - Total: 12 test combinations

- **Testing Pipeline:**
  1. Code formatting check (Black)
  2. Import sorting check (isort)
  3. Linting (flake8)
  4. Type checking (mypy, non-blocking)
  5. Unit tests (pytest with coverage)
  6. Coverage upload (Codecov)

- **Jobs:**
  - `test` - Full test matrix (12 combinations)
  - `lint-checks` - Style checks on Python 3.10

### publish.yml
**Purpose:** Automated PyPI publishing on releases

**Triggers:**
- GitHub release created
- Git tags matching 'v*' pattern

**Pipeline:**
1. Checkout code
2. Setup Python 3.10
3. Install build tools
4. Build distribution (wheel + source)
5. Validate package with twine
6. Publish to PyPI using token
7. Update GitHub release notes

**Secrets Required:**
- `PYPI_API_TOKEN` for authentication

---

## 5. Documentation Structure

### Root Documentation Files

**README.md** (65 lines)
- Project tagline and feature highlights
- Quick installation instructions
- Basic usage code snippet
- Links to examples and docs

**INSTALL.md** (134 lines)
- PyPI installation
- Development installation from source
- Optional dependencies (JAX, docs)
- System requirements and supported platforms
- Troubleshooting guide
- Docker reference (coming soon)

**CONTRIBUTING.md** (166 lines)
- Development setup instructions
- Code style standards (Black, isort, flake8)
- Testing requirements (>90% coverage)
- Pull request process
- Types of contributions (bugs, features, docs)
- Release process
- Communication guidelines

**CHANGELOG.md** (67 lines)
- Version 0.1.0 (2024-11-12) - Initial release
- Complete feature list from alpha
- Planned features for unreleased version
- Semantic versioning adherence

**CODE_OF_CONDUCT.md** (160+ lines)
- Community standards
- Enforcement mechanisms

### Technical Documentation

**docs/quickstart.md** (184 lines)
- Step-by-step usage guide
- Complete working code examples
- Explanations of key components:
  - Tracker initialization
  - Training loop integration
  - Result analysis
  - Jump detection
  - Visualization
  - Export functionality
- Customization options
- Common FAQs
- Troubleshooting

**examples/README.md** (57 lines)
- Overview of 2 working examples
- Execution instructions
- Expected output descriptions
- Tips for customization
- Next steps guidance

---

## 6. Examples & Specifications

### Example 1: 01_quickstart_mnist.py
**Dataset:** MNIST
**Architecture:** Simple MLP (Flatten -> Linear -> ReLU -> Linear -> ReLU -> Linear)
**Specifications:**
- Batch size: 64
- Optimizer: Adam (lr=0.001)
- Epochs: 2 (stops at 1000 steps for demo)
- Sampling frequency: 10 steps
- Jump detection: Enabled
- Tracking: Automatic layer detection
- Outputs:
  - Console: Training progress, jumps detected, measurements per layer
  - PNG: `mnist_stable_rank.png` (stable rank across all layers)
  - CSV: `mnist_results.csv` (complete tracking data)
- Runtime: ~2 minutes on CPU

**Key Features Demonstrated:**
- Minimal intrusion (3-line integration)
- Automatic layer detection
- Jump detection
- Data export (CSV)
- Static visualization

### Example 2: 02_cnn_cifar10.py (partial view shown)
**Dataset:** CIFAR-10
**Architecture:** SimpleCNN
- Conv2d(3, 32, kernel=3)
- MaxPool2d(2, 2)
- Conv2d(32, 64, kernel=3)
- MaxPool2d(2, 2)
- Conv2d(64, 128, kernel=3)
- MaxPool2d(2, 2)
- FC: 128*4*4 -> 256 -> 10

**Specifications:**
- Batch size: 32
- Optimizer: SGD with momentum
- Data normalization applied
- Gradient norm tracking included
- Tracking: Both conv and FC layers
- Export format: JSON
- Runtime: ~10 min on CPU, ~3 min on GPU

**Key Features Demonstrated:**
- CNN layer handling
- Conv and FC layer mixing
- Gradient norm tracking
- JSON export format
- Architecture-specific handling

---

## 7. Test Suite (tests/)

### conftest.py
**Pytest Fixtures:**
- `small_mlp` - 3-layer MLP (10 -> 20 -> 15 -> 5)
- `simple_cnn` - CNN with conv and FC layers
- `sample_batch` - Dense batch (32, 10)
- `sample_image_batch` - Image batch (8, 3, 32, 32)
- `random_matrix` - Random (100, 50)
- `low_rank_matrix` - Rank-5 matrix (100, 50)
- `identity_matrix` - Identity (50, 50)
- `near_singular_matrix` - Rank-deficient (50, 50)

### test_tracker.py
Tests for `HighFrequencyTracker` and `DimensionalityMetrics`
- Initialization with different architectures
- Logging and step tracking
- Results retrieval
- Jump detection functionality
- Context manager usage
- Resource cleanup

### test_estimators.py
Tests for dimensionality metric computations
- `stable_rank()` with various matrices
- `participation_ratio()` behavior
- `cumulative_energy_90()` calculations
- `nuclear_norm_ratio()` computations
- Edge cases (singular matrices, low-rank matrices)

### test_jump_detection.py
Tests for jump detection system
- Jump detection algorithm
- Z-score threshold behavior
- Window size effects
- Edge case handling

### Coverage
- Current: >90% (as stated in README)
- Commands: `pytest tests/ -v --cov=ndt --cov-report=html`
- HTML reports generated to `htmlcov/`

---

## 8. TDS Article & Related Experiments

### TDS Publication
**File:** `I Measured Neural Network Training Every 5 Steps for 10,000 Iterations _ TDS Contributor Portal.pdf`
- **Type:** PDF document (8 pages, version 1.4)
- **Size:** ~979 KB
- **Source:** Towards Data Science (TDS) Contributor Portal
- **Relevance:** Likely documents the original experiments and motivations for NDT

### Implied Experiment Specifications from Project
Based on the PDF filename and project structure, the TDS article likely describes:
- **Measurement Frequency:** Every 5 steps for 10,000 iterations
- **Scope:** Neural network training dynamics over extended runs
- **Focus:** Dimensionality tracking insights
- **Experiments:** Multiple architectures and datasets

**References in Project:**
- No direct code references to "TDS" or "Towards Data Science"
- PDF included as research reference/motivation document
- Examples (MNIST, CIFAR-10) align with typical TDS demonstration scope
- Metrics (stable rank, participation ratio, etc.) are research-driven

---

## 9. Current Repository State

### Git Status
```
Current Branch: claude/verify-ndt-post-alignment-011CV5dLES3SVkTDuVguLPsx
Status: Clean (no uncommitted changes)

Recent Commits:
- d6860a2 Add files via upload
- 27a255f Fix Python 3.8 compatibility in hdf5.py: Use Tuple from typing
- 1a73dce Fix Python 3.8 compatibility: Use Tuple from typing
- 402c362 Enforce single-line imports for isort consistency
- 3c562f4 Add explicit isort configuration for consistency
- 748c963 Fix import ordering for isort compliance
- 9abdada Fix test expectations for random matrix stable rank and PR
- 95af956 Fix stable rank calculation and final linting issue
```

### Recent Activity
- Latest fix: Python 3.8 compatibility corrections
- Focus: Code quality enforcement (import ordering, type hints)
- Preparation phase: Setting up for verification against TDS post

### Version Info
- **Package Version:** 0.1.0 (beta)
- **Python Versions:** 3.8+
- **Release Date:** 2024-11-12

---

## 10. Key Features Summary

### Core Functionality
1. **HighFrequencyTracker** - Main tracking interface
   - Minimal 3-line integration
   - Automatic layer detection
   - Context manager support
   - Configurable sampling frequency

2. **Four Dimensionality Metrics**
   - Stable rank (robust to noise)
   - Participation ratio (variance evenness)
   - Cumulative 90% energy (component count)
   - Nuclear norm ratio (normalized rank)

3. **Jump Detection**
   - Z-score based anomaly detection
   - Configurable thresholds
   - Window-based analysis
   - Phase transition identification

4. **Architecture Support**
   - MLPs (fully connected networks)
   - CNNs (convolutional networks)
   - Transformers (attention-based)
   - Vision Transformers (ViT)
   - Custom architecture support

5. **Visualization**
   - Static matplotlib plots
   - Interactive plotly dashboards
   - Multi-layer comparison views
   - Jump visualization

6. **Export Options**
   - CSV format (standard pandas)
   - JSON format (structured data)
   - HDF5 format (large-scale storage)

7. **Quality Assurance**
   - >90% test coverage
   - Type hints throughout (mypy)
   - Code style enforcement (Black, isort)
   - Linting (flake8)
   - Multi-platform CI/CD

---

## 11. Dependencies

### Core Dependencies
- **torch** >= 1.12.0 (PyTorch neural network framework)
- **numpy** >= 1.21.0 (numerical computing)
- **pandas** >= 1.3.0 (data structures and analysis)
- **matplotlib** >= 3.5.0 (static plotting)
- **seaborn** >= 0.11.0 (statistical visualization)
- **plotly** >= 5.0.0 (interactive visualization)
- **scipy** >= 1.7.0 (scientific computing)
- **tqdm** >= 4.62.0 (progress bars)
- **pyyaml** >= 6.0 (YAML configuration)
- **h5py** >= 3.6.0 (HDF5 file handling)

### Optional Dependencies
- **JAX** (jax>=0.4.0, jaxlib>=0.4.0) - For JAX integration
- **MkDocs** (mkdocs>=1.4.0) - For documentation generation

### Development Dependencies
- pytest >= 7.0.0
- pytest-cov >= 3.0.0
- black >= 22.0.0
- isort >= 5.10.0
- flake8 >= 4.0.0
- mypy >= 0.950

---

## 12. Supported Platforms

- **Linux**: Ubuntu 18.04+, CentOS 7+
- **macOS**: 10.15+ (Catalina and later)
- **Windows**: 10 and 11

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total Lines of Code | 2,690 |
| Number of Modules | 7 (core, architectures, export, visualization, utils, etc.) |
| Test Files | 5 |
| Examples | 2 |
| Documentation Files | 6 |
| CI/CD Workflows | 2 |
| Python Versions Supported | 4 (3.8, 3.9, 3.10, 3.11) |
| Test Coverage | >90% |
| Version | 0.1.0 |
| Status | Beta, Production-Ready |

