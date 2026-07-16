<div align="center">

# Neural Dimensionality Tracker - ndt

High-frequency monitoring of neural-network representational dimensionality during training.

[![Python](https://img.shields.io/badge/python-3.8%20|%203.9%20|%203.10%20|%203.11-blue?style=flat-square)](https://github.com/Javihaus/ndt)
[![CI](https://img.shields.io/github/actions/workflow/status/Javihaus/ndt/tests.yml?branch=main&label=CI&style=flat-square)](https://github.com/Javihaus/ndt/actions)
[![Docs](https://img.shields.io/badge/docs-javihaus.github.io%2Fndt-blue?style=flat-square)](https://javihaus.github.io/ndt)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat-square)](LICENSE)
[![OpenSSF Scorecard](https://img.shields.io/ossf-scorecard/github.com/Javihaus/ndt?style=flat-square&label=OpenSSF%20Scorecard)](https://scorecard.dev/viewer/?uri=github.com/Javihaus/ndt)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/13530/badge)](https://www.bestpractices.dev/projects/13530)


[Documentation](https://javihaus.github.io/ndt) · [Examples](examples/) · [Contributing](CONTRIBUTING.md)

</div>

By [Javier Marin](https://jmarin.info). A companion to the paper *Phase Transitions or Continuous Evolution? Methodological Sensitivity in Neural Network Training Dynamics*.

---

**ndt** (Neural Dimensionality Tracker) tracks how a network's representational dimensionality evolves during training, across MLPs, CNNs, Transformers and Vision Transformers, and it does one thing no other tracker does: **it tells you whether a detected phase transition is real or an artifact of your detection method.**

That check matters because it is easy to get wrong. In a companion study across 55 experiments and 30,147 measurement points, transition detectors built on these same metrics disagreed with each other almost completely (a threshold detector and the threshold-free PELT algorithm correlated at -0.029), and no transition appeared consistently across metrics. A detector can report a clean "phase transition" that is nothing but its own response to noise. So before you build a theory on a detected transition, run the validity harness on your detector and see what it does on ground truth you control.

## Features

- **Validity harness**: test any transition detector against ground-truth fixtures (a known transition and a known null) and get a plain verdict on whether its detections can be trusted. Runs in milliseconds, no training required.
- **Dimensionality tracking**: add tracking to any PyTorch model in three lines, across MLPs, CNNs, Transformers and Vision Transformers.
- **Four complementary metrics**: participation ratio, stable rank, cumulative energy, nuclear-norm ratio.
- **Transition detection, honestly framed**: a z-score jump detector is included as *one* detector to be validated, not a truth oracle. Check it before you trust it.
- **Rich visualization** and **flexible export** (CSV, JSON, HDF5).
- Fully typed, tested, documented.

## Installation

```bash
pip install ndtracker
```

## Check your detector before you trust it

```python
from ndt import JumpDetector
from ndt.validity import validate_detector, jump_detector_as_callable

detector = jump_detector_as_callable(JumpDetector(z_threshold=3.0))
report = validate_detector(detector, name="JumpDetector(z=3.0)")
print(report.render())
```

```
Validity report for: JumpDetector(z=3.0)

  planted_transition   recall: 1.00   false positives:   5
  planted_multi        recall: 1.00   false positives:  14
  pure_noise           null control   false positives:   2 (5.0/1000 steps)
  drift_no_jump        null control   false positives:   4 (10.0/1000 steps)

  mean recall on planted transitions : 1.00
  false positives on null controls   : 6

  VERDICT: NOT VALID on these fixtures
  Fires on pure noise and continuous drift. It manufactures transitions where
  there are none, so a detection on real data cannot be trusted without this check.
```

The point is not that this detector is bad. It is that it recovers real transitions *and* fires on noise, so a bare detection count means little until you know its false-positive rate on a null. The harness gives you that number. Plug in your own detector (any callable from a value sequence to detected step indices), or your own fixtures, including a real grokking run where the transition is visible in test accuracy.

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
  url = {https://github.com/Javihaus/ndt},
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

Apache 2.0 - see [LICENSE](LICENSE) file for details

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
