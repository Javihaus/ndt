# Installation Guide

## From PyPI (Once Published)

The easiest way to install NDT:

```bash
pip install neural-dimensionality-tracker
```

## From Source (Development)

### Prerequisites

- Python 3.8 or higher
- pip (latest version recommended)

### Clone and Install

```bash
# Clone the repository
git clone https://github.com/Javihaus/ndt.git
cd ndt

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"
```

### Verify Installation

```python
python -c "from ndt import HighFrequencyTracker; print('NDT installed successfully!')"
```

## Optional Dependencies

### For JAX Support

```bash
pip install neural-dimensionality-tracker[jax]
```

### For Documentation Building

```bash
pip install neural-dimensionality-tracker[docs]
```

### All Optional Dependencies

```bash
pip install neural-dimensionality-tracker[dev,docs,jax]
```

## System Requirements

- **CPU**: Any modern CPU (x86_64, ARM64)
- **Memory**: Depends on model size; typically < 1GB overhead
- **GPU**: Optional, automatically used if available via PyTorch
- **Disk**: ~50MB for package, variable for results storage

## Supported Platforms

- **Linux**: Ubuntu 18.04+, CentOS 7+, other major distributions
- **macOS**: 10.15+ (Catalina and later)
- **Windows**: 10 and 11

## Dependencies

Core dependencies (automatically installed):
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

## Troubleshooting

### Import Errors

If you get import errors, ensure your Python path is correct:

```bash
export PYTHONPATH=/path/to/ndt/src:$PYTHONPATH
```

### PyTorch Installation

If PyTorch is not installed or you need GPU support:

```bash
# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu

# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# See https://pytorch.org for more options
```

### Testing Installation

Run the test suite:

```bash
pytest tests/ -v
```

## Uninstallation

```bash
pip uninstall neural-dimensionality-tracker
```

## Docker (Coming Soon)

Pre-configured Docker images will be available for easy deployment.

## Support

For installation issues:
1. Check [GitHub Issues](https://github.com/Javihaus/ndt/issues)
2. Search existing issues
3. Create a new issue with your environment details
