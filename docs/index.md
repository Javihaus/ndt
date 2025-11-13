# Neural Dimensionality Tracker (NDT)

Welcome to the documentation for **Neural Dimensionality Tracker**, a production-ready Python library for high-frequency monitoring of neural network representational dimensionality during training.

## What is NDT?

NDT enables you to:

- **Track** how your neural network's internal representations evolve during training
- **Detect** discrete phase transitions (jumps) in learning dynamics
- **Understand** the mechanistic behavior of deep learning models
- **Visualize** representational capacity expansion in real-time
- **Export** comprehensive dimensionality measurements for analysis

## Key Features

✨ **Minimal Integration** - Add tracking with just 3 lines of code

🏗️ **Architecture-Agnostic** - Works with MLPs, CNNs, Transformers, ViTs, and more

📊 **Multiple Metrics** - Track 4 complementary dimensionality measures:
   - Stable Rank (effective dimensionality)
   - Participation Ratio (variance distribution)
   - Cumulative Energy 90% (components for 90% variance)
   - Nuclear Norm Ratio (normalized rank measure)

🔍 **Automatic Jump Detection** - Identify phase transitions during training

📈 **Rich Visualization** - Built-in plotting with Matplotlib and interactive Plotly dashboards

💾 **Flexible Export** - Save results as CSV, JSON, or HDF5

⚡ **Production-Ready** - Fully typed, tested (>90% coverage), <10% overhead

## Quick Start

### Installation

```bash
pip install neural-dimensionality-tracker
```

### Basic Usage

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

That's it! See [Quickstart Guide](quickstart.md) for more details.

## Why Track Dimensionality?

Recent research (Ansuini et al. 2019, Yang et al. 2024) has shown that neural networks **expand their effective representational capacity during training**, despite having fixed parameters. This challenges our understanding of how neural networks learn.

**Key insights:**

1. **Capacity expands during training** - Networks don't just explore a fixed space, they build new representational structures
2. **Transitions are discrete** - Learning happens in distinct phases with rapid jumps
3. **Critical periods exist** - Early training (first 25%) is when most structure forms
4. **High-frequency sampling reveals hidden dynamics** - Coarse checkpointing misses most transitions

NDT makes these insights accessible for any PyTorch model.

## Research Reproduction

NDT includes a complete reproduction of the experiment from the Towards Data Science article **"I Measured Neural Network Training Every 5 Steps for 10,000 Iterations"**:

```bash
python examples/03_reproduce_tds_experiment.py
```

**Expected results:**
- Phase 1 (Collapse): Steps 0-300
- Phase 2 (Expansion): Steps 300-5000
- Phase 3 (Stabilization): Steps 5000-8000

See [TDS Article Reproduction](tds_reproduction.md) for details.

## Documentation Structure

### Getting Started
- [Quickstart Guide](quickstart.md) - Get started in 5 minutes
- [Installation](../INSTALL.md) - Detailed installation instructions
- [Examples Gallery](examples_gallery.md) - Complete collection of examples

### Reference
- [API Reference](api_reference.md) - Complete API documentation
- [Architecture Support](architecture_support.md) - Supported architectures and compatibility
- [Performance Benchmarks](performance_benchmarks.md) - Overhead, memory, and scalability
- [Troubleshooting](troubleshooting.md) - Common issues and solutions

### Research
- [TDS Article Reproduction](tds_reproduction.md) - Reproduce research results

### Contributing
- [Contributing Guide](../CONTRIBUTING.md) - How to contribute

## Use Cases

### Research
- **Interpretability:** Understand when and how features form during training
- **Architecture Design:** Compare representational dynamics across architectures
- **Training Dynamics:** Study critical periods and phase transitions
- **Transfer Learning:** Analyze representation transfer between tasks

### Production
- **Monitoring:** Track representation health during large-scale training
- **Debugging:** Diagnose training instability and architectural bottlenecks
- **Optimization:** Identify when to stop training or adjust hyperparameters
- **Validation:** Ensure consistent learning dynamics across experiments

## Supported Architectures

| Architecture | Auto-Detection | Status |
|--------------|----------------|--------|
| MLP | ✅ Yes | Fully supported |
| CNN | ✅ Yes | Fully supported |
| ResNet | ✅ Yes | Fully supported |
| Transformer | ⚠️ Partial | Specify FFN layers |
| BERT | ⚠️ Partial | Specify layers |
| GPT | ⚠️ Partial | Specify layers |
| ViT | ⚠️ Partial | Specify layers |

See [Architecture Support](architecture_support.md) for complete matrix.

## Performance

| Metric | Target | Measured |
|--------|--------|----------|
| Training overhead | <10% | 2-8% (typical) |
| Memory usage | Minimal | <1MB per 1000 measurements |
| Max model size | 1B params | Tested up to 1.5B |
| Initialization | <1 second | 50-200ms |

See [Performance Benchmarks](performance_benchmarks.md) for details.

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
  title = {I Measured Neural Network Training Every 5 Steps for 10,000 Iterations},
  journal = {Towards Data Science},
  year = {2025},
  month = {November}
}
```

## Community

- **GitHub:** [github.com/Javihaus/ndt](https://github.com/Javihaus/ndt)
- **Issues:** [Report bugs or request features](https://github.com/Javihaus/ndt/issues)
- **Discussions:** [Ask questions](https://github.com/Javihaus/ndt/discussions)
- **PyPI:** [pypi.org/project/neural-dimensionality-tracker](https://pypi.org/project/neural-dimensionality-tracker/)

## License

MIT License - see [LICENSE](../LICENSE) for details.

## Acknowledgments

This work builds on research by:

- **Ansuini et al. (2019)** - Intrinsic dimension of data representations in deep neural networks
- **Yang et al. (2024)** - ε-rank and the staircase phenomenon
- **Achille et al. (2019)** - Critical learning periods in deep networks

## Next Steps

- [Install NDT](../INSTALL.md) and try the [Quickstart](quickstart.md)
- Explore [Examples](examples_gallery.md)
- Read the [API Reference](api_reference.md)
- Reproduce the [TDS experiment](tds_reproduction.md)
- [Contribute](../CONTRIBUTING.md) to the project

---

**Author:** Javier Marín | [LinkedIn](https://linkedin.com/in/jmarin) | [Twitter](https://twitter.com/javihaus)
