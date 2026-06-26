<div align="center">

<!-- logo: drop org logo here -->

# ndt
## Neural Dimensionality Tracker

High-frequency monitoring of neural-network representational dimensionality during training.

[![Python](https://img.shields.io/badge/python-3.8%20|%203.9%20|%203.10%20|%203.11-blue?style=flat-square)](https://github.com/groundlens-dev/ndt)
[![CI](https://img.shields.io/github/actions/workflow/status/groundlens-dev/ndt/tests.yml?branch=main&label=CI&style=flat-square)](https://github.com/groundlens-dev/ndt/actions)
[![Docs](https://img.shields.io/badge/docs-docs.groundlens.dev%2Fndt-blue?style=flat-square)](https://docs.groundlens.dev/ndt)
[![PyPI](https://img.shields.io/pypi/v/ndtracker?style=flat-square)](https://pypi.org/project/ndtracker/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](https://opensource.org/licenses/MIT)

[Documentation](https://docs.groundlens.dev/ndt) · [Examples](examples/) · [Contributing](CONTRIBUTING.md)

</div>

> **A Groundlens tool.** Part of the [Groundlens](https://github.com/groundlens-dev) open-source family — alongside [groundlens](https://github.com/groundlens-dev/groundlens) (geometric LLM verification) and [otwin](https://github.com/groundlens-dev/otwin) (physics-informed digital twins).

---

**ndt** (Neural Dimensionality Tracker) is a production-ready Python tool for high-frequency monitoring of neural-network representational dimensionality during training, enabling researchers and practitioners to track how internal representations evolve, detect discrete phase transitions (jumps), and gain mechanistic insight into the learning dynamics of deep neural networks across architectures (MLPs, CNNs, Transformers, Vision Transformers).

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
pip install ndtracker
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

- **Quick Start**: See above for 3-line integration
- **Examples**: [examples/](examples/) contains complete working examples:
  - [`01_quickstart_mnist.py`](examples/01_quickstart_mnist.py) - Basic MLP on MNIST
  - [`02_cnn_cifar10.py`](examples/02_cnn_cifar10.py) - CNN on CIFAR-10
  - [`03_reproduce_tds_experiment.py`](examples/03_reproduce_tds_experiment.py) - Reproduce TDS article experiment
- **Full API Documentation**: [docs/](docs/)
- **Installation Guide**: [INSTALL.md](INSTALL.md)
- **Contributing**: [CONTRIBUTING.md](CONTRIBUTING.md)

## Reproducing the TDS Article Experiment

The repository includes a complete reproduction of the experiment described in the Towards Data Science article "I Measured Neural Network Training Every 5 Steps for 10,000 Iterations":

```bash
python examples/03_reproduce_tds_experiment.py
```

This script uses the exact specifications from the article:
- Architecture: 784-256-128-10 (3-layer MLP)
- Dataset: MNIST (60k train/10k test)
- Optimizer: Adam (β1=0.9, β2=0.999)
- Learning rate: 0.001, batch size: 64
- Training: 8000 steps with measurements every 5 steps
- Expected results: 3 distinct phases (collapse, expansion, stabilization)

## Citation

If you use Neural Dimensionality Tracker in your research, please cite:

```bibtex
@software{marin2024ndt,
  author = {Marín, Javier},
  title = {Neural Dimensionality Tracker: High-Frequency Monitoring of Neural Network Training Dynamics},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/groundlens-dev/ndt},
  version = {0.1.0}
}
```

**Associated article:**
```bibtex
@article{marin2025measuring,
  author = {Marín, Javier},
  title = {I Measured Neural Network Training Every 5 Steps for 10,000 Iterations: What High-Resolution Training Dynamics Taught Me About Feature Formation},
  journal = {Towards Data Science},
  year = {2025},
  month = {November},
  url = {https://towardsdatascience.com/}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details

## Author

**Javier Marín**
- LinkedIn: [linkedin.com/in/jmarin](https://linkedin.com/in/javiermarinvalenzuela)
- Twitter: [@javihaus](https://twitter.com/javihaus)
- Email: javier@jmarin.info

## Acknowledgments

This work builds on research by:
- Achille, A., Rovere, M., & Soatto, S. (2019). Critical learning periods in deep networks. In International Conference on Learning Representations (ICLR). https://openreview.net/forum?id=BkeStsCcKQ
- Ansuini, A., Laio, A., Macke, J. H., & Zoccolan, D. (2019). Intrinsic dimension of data representations in deep neural networks. In Advances in Neural Information Processing Systems (Vol. 32, pp. 6109-6119). https://proceedings.neurips.cc/paper/2019/hash/cfcce0621b49c983991ead4c3d4d3b6b-Abstract.html
- Yang, J., Zhao, Y., & Zhu, Q. (2024). ε-rank and the staircase phenomenon: New insights into neural network training dynamics. arXiv preprint arXiv:2412.05144. https://arxiv.org/abs/2412.05144

---

An open-source tool by **Groundlens** · groundlens.dev
