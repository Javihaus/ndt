## Examples

This directory contains examples demonstrating how to use the Neural Dimensionality Tracker with different architectures and datasets.

### Quickstart

**01_quickstart_mnist.py** - Minimal example showing core functionality
- Simple MLP on MNIST
- Demonstrates: initialization, tracking, analysis, visualization, export
- Run time: ~2 minutes on CPU

```bash
python examples/01_quickstart_mnist.py
```

### Architecture-Specific Examples

**02_cnn_cifar10.py** - Convolutional neural network
- CNN on CIFAR-10
- Tracks both conv and FC layers
- Includes gradient norm tracking
- Run time: ~10 minutes on CPU, ~3 minutes on GPU

```bash
python examples/02_cnn_cifar10.py
```

### Research Reproduction

**03_reproduce_tds_experiment.py** - Reproduce TDS article experiment
- Exact reproduction of "I Measured Neural Network Training Every 5 Steps for 10,000 Iterations"
- Architecture: 784-256-128-10 (3-layer MLP)
- High-frequency sampling: Every 5 steps over 8000 steps
- Demonstrates 3 distinct phases: collapse, expansion, stabilization
- Generates figures matching TDS article (Figures 2 & 3)
- Run time: ~15 minutes on CPU, ~5 minutes on GPU

```bash
python examples/03_reproduce_tds_experiment.py
```

### Requirements

Install dependencies:
```bash
pip install torch torchvision
pip install ndtracker
```

### Output Files

Each example generates:
- **PNG images**: Visualizations of dimensionality metrics
- **CSV/JSON files**: Exported tracking data
- **Console output**: Summary statistics and detected jumps

### Tips

1. **Faster experimentation**: Reduce training steps or increase `sampling_frequency`
2. **Memory**: For large models, increase `sampling_frequency` or track fewer layers
3. **GPU**: Examples automatically use GPU if available
4. **Customization**: All examples are self-contained and easy to modify

### Next Steps

After running examples:
1. Examine the generated visualizations
2. Load exported data for custom analysis
3. Try with your own models and datasets
4. See full documentation for advanced features
