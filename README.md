# Neural Dimensionality Tracker (NDT)

[![Tests](https://github.com/Javihaus/ndt/workflows/Tests/badge.svg)](https://github.com/Javihaus/ndt/actions)
[![PyPI version](https://badge.fury.io/py/neural-dimensionality-tracker.svg)](https://pypi.org/project/neural-dimensionality-tracker/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

High-frequency monitoring of neural network representational dimensionality during training. Track how your network's internal representations evolve, detect phase transitions, and gain insights into the learning dynamics of deep neural networks.

## Features

- **Minimal Intrusion**: Add dimensionality tracking to any PyTorch model with just 3 lines of code
- **Architecture-Agnostic**: Automatic support for MLPs, CNNs, Transformers, and Vision Transformers
- **Multiple Metrics**: Track 4 complementary dimensionality measures
- **Jump Detection**: Automatically identify phase transitions during training
- **Rich Visualization**: Built-in plotting with Matplotlib and interactive Plotly dashboards
- **Flexible Export**: Save results as CSV, JSON, or HDF5
- **Production-Ready**: Fully typed, tested (>90% coverage), and documented

## Installation

```bash
pip install neural-dimensionality-tracker
```

## Quick Start

```python
import torch.nn as nn
from ndt import HighFrequencyTracker

# Your model
model = nn.Sequential(
    nn.Linear(784, 512), nn.ReLU(),
    nn.Linear(512, 256), nn.ReLU(),
    nn.Linear(256, 10)
)

# Create tracker
tracker = HighFrequencyTracker(model, sampling_frequency=10)

# Training loop
for step, (x, y) in enumerate(dataloader):
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    tracker.log(step, loss.item())  # One line!

# Analyze
results = tracker.get_results()
from ndt import plot_phases
plot_phases(results, metric="stable_rank")
```

## Documentation

See [examples/](examples/) for complete working examples and detailed usage guides.

## License

MIT License - see [LICENSE](LICENSE) file for details
