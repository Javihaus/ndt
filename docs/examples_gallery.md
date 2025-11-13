# Examples Gallery

Complete collection of examples demonstrating Neural Dimensionality Tracker across different architectures, datasets, and use cases.

## Table of Contents

1. [Quickstart & Basic Examples](#quickstart--basic-examples)
2. [Architecture-Specific Examples](#architecture-specific-examples)
3. [Research Reproduction](#research-reproduction)
4. [Advanced Use Cases](#advanced-use-cases)
5. [Production Deployment](#production-deployment)

---

## Quickstart & Basic Examples

### 01: Quickstart MNIST

**File:** [`01_quickstart_mnist.py`](../examples/01_quickstart_mnist.py)

**Description:** Minimal example showing core NDT functionality with a simple MLP on MNIST.

**What you'll learn:**
- How to create a tracker
- How to integrate into training loop
- How to analyze results
- How to detect jumps
- How to visualize metrics
- How to export data

**Architecture:** 4-layer MLP (784 → 512 → 256 → 128 → 10)

**Configuration:**
- Dataset: MNIST (60k train)
- Steps: 1000
- Sampling: Every 10 steps (100 measurements)
- Runtime: ~2 minutes (CPU)

**Key code:**
```python
from ndt import HighFrequencyTracker

tracker = HighFrequencyTracker(model, sampling_frequency=10)

# Training loop
for step, (x, y) in enumerate(dataloader):
    # ... standard training ...
    tracker.log(step, loss.item())  # One line!

# Analysis
results = tracker.get_results()
jumps = tracker.detect_jumps()
```

**Expected output:**
- `mnist_stable_rank.png` - Visualization
- `mnist_results.csv` - Exported data
- Console: Detected jumps

**Run:**
```bash
python examples/01_quickstart_mnist.py
```

---

## Architecture-Specific Examples

### 02: CNN on CIFAR-10

**File:** [`02_cnn_cifar10.py`](../examples/02_cnn_cifar10.py)

**Description:** Track dimensionality in a convolutional neural network with both conv and FC layers.

**What you'll learn:**
- How to track Conv layers
- How to track multiple layer types
- How to specify layers explicitly
- How to track gradient norms
- How to compare metrics

**Architecture:** 3-layer CNN (3→32→64→128) + 2 FC layers

**Configuration:**
- Dataset: CIFAR-10 (50k train)
- Steps: 2000
- Sampling: Every 20 steps (100 measurements)
- Runtime: ~10 min (CPU), ~3 min (GPU)

**Key code:**
```python
tracker = HighFrequencyTracker(
    model,
    layers=[model.conv1, model.conv2, model.conv3, model.fc1],
    layer_names=["Conv1", "Conv2", "Conv3", "FC1"],
    sampling_frequency=20
)

# Track with gradient norm
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))
tracker.log(step, loss.item(), grad_norm=grad_norm.item())
```

**Expected output:**
- `cifar10_conv3_metrics.png` - All metrics for Conv3
- `cifar10_results.json` - JSON export
- Console: Jump statistics per layer

**Run:**
```bash
python examples/02_cnn_cifar10.py
```

---

## Research Reproduction

### 03: TDS Article Experiment

**File:** [`03_reproduce_tds_experiment.py`](../examples/03_reproduce_tds_experiment.py)

**Description:** Exact reproduction of the experiment from the TDS article "I Measured Neural Network Training Every 5 Steps for 10,000 Iterations".

**What you'll learn:**
- High-frequency sampling (every 5 steps)
- Phase detection (collapse, expansion, stabilization)
- How to reproduce research results
- How sampling frequency affects jump detection
- How to generate publication-quality figures

**Architecture:** 3-layer MLP (784 → 256 → 128 → 10) ⚠️ Note: Different from quickstart!

**Configuration (exact TDS specs):**
- Dataset: MNIST (60k train / 10k test)
- Optimizer: Adam (β1=0.9, β2=0.999)
- Learning rate: 0.001
- Batch size: 64
- Steps: 8000
- Sampling: Every 5 steps (1600 measurements)
- Runtime: ~15 min (CPU), ~5 min (GPU)

**Key findings:**
- Phase 1 (Collapse): Steps 0-300, dimensionality drops ~2500 → ~500
- Phase 2 (Expansion): Steps 300-5000, dimensionality climbs to ~1000
- Phase 3 (Stabilization): Steps 5000-8000, dimensionality plateaus
- ~85 jumps in activation space vs 1 in weight space
- Dimensionality correlates with loss (ρ = -0.951)

**Expected output:**
- `tds_figure2_activation_space.png` - Matches Figure 2 from article
- `tds_figure3_dimensionality_loss.png` - Matches Figure 3 from article
- `tds_experiment_results.csv` - Full data export
- `tds_experiment_results.h5` - HDF5 export (efficient)
- Console: Detailed phase analysis

**Key code:**
```python
# Exact architecture from TDS article
class TDSExperimentMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 256)  # Note: 256, not 512!
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 10)

# High-frequency sampling
tracker = HighFrequencyTracker(model, sampling_frequency=5)

# Train for exactly 8000 steps
for step in range(8000):
    # ... training ...
    tracker.log(step, loss.item(), grad_norm=grad_norm.item())
```

**Run:**
```bash
python examples/03_reproduce_tds_experiment.py
```

**Compare with low-frequency sampling:**
```bash
# Modify sampling_frequency=5 to sampling_frequency=50
# Run again and compare jump detection results
```

---

## Advanced Use Cases

### 04: Transformer Language Model (Coming Soon)

**Description:** Track BERT or GPT-style transformer during fine-tuning.

**What you'll learn:**
- How to track transformer layers
- How to select attention vs FFN layers
- How dimensionality evolves in language models
- How to handle large models efficiently

**Architecture:** BERT-base (12 layers, 768 hidden)

**Configuration:**
- Dataset: GLUE benchmark
- Sampling: Every 50 steps
- Track: FFN layers in layers 0, 6, 11

---

### 05: Vision Transformer (Coming Soon)

**Description:** Track Vision Transformer on image classification.

**What you'll learn:**
- How to track ViT patch embeddings
- How to track attention mechanisms
- How visual representations evolve
- How to optimize for large ViT models

**Architecture:** ViT-Base/16 (86M parameters)

**Configuration:**
- Dataset: ImageNet-1k
- Sampling: Every 100 steps
- Track: Patch embed + 4 transformer blocks

---

### 06: ResNet on ImageNet (Coming Soon)

**Description:** Production-scale tracking on ImageNet classification.

**What you'll learn:**
- How to track deep residual networks
- How to handle large-scale datasets
- How to minimize overhead in production
- How to select representative layers

**Architecture:** ResNet-50 (25M parameters)

**Configuration:**
- Dataset: ImageNet-1k (1.2M images)
- Sampling: Every 100 steps
- Track: Residual blocks in stages 1, 2, 3, 4

---

### 07: GAN Training Dynamics (Coming Soon)

**Description:** Track generator and discriminator dimensionality during GAN training.

**What you'll learn:**
- How to track multiple networks
- How dimensionality evolves in adversarial training
- How to detect mode collapse
- How to diagnose training instability

**Architecture:** DCGAN or StyleGAN

**Configuration:**
- Dataset: CelebA
- Sampling: Every 10 steps
- Track: Generator layers + discriminator layers

---

### 08: Reinforcement Learning (Coming Soon)

**Description:** Track policy network dimensionality during RL training.

**What you'll learn:**
- How to track in RL context
- How representation evolves with reward
- How to correlate dimensionality with performance
- How to detect learning plateaus

**Architecture:** PPO with CNN policy

**Configuration:**
- Environment: Atari (e.g., Breakout)
- Sampling: Every episode
- Track: Convolutional feature layers

---

## Production Deployment

### 09: Distributed Training (Coming Soon)

**Description:** Track dimensionality in multi-GPU distributed training.

**What you'll learn:**
- How to integrate with DDP
- How to aggregate results across GPUs
- How to minimize communication overhead
- How to save results from rank 0 only

**Key code:**
```python
import torch.distributed as dist

if dist.get_rank() == 0:
    tracker = HighFrequencyTracker(model, sampling_frequency=50)

# Training loop
loss = train_step(batch)
if dist.get_rank() == 0:
    tracker.log(step, loss.item())
```

---

### 10: PyTorch Lightning Integration (Coming Soon)

**Description:** Use NDT with PyTorch Lightning's LightningModule.

**What you'll learn:**
- How to integrate with Lightning's training loop
- How to log to TensorBoard/Weights & Biases
- How to use callbacks
- How to handle checkpointing

**Key code:**
```python
class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.tracker = HighFrequencyTracker(self.model)

    def training_step(self, batch, batch_idx):
        # ... training ...
        self.tracker.log(self.global_step, loss.item())
        return loss
```

---

### 11: Continuous Monitoring (Coming Soon)

**Description:** Set up continuous dimensionality monitoring for production training.

**What you'll learn:**
- How to export metrics to Prometheus
- How to create Grafana dashboards
- How to set up alerts for anomalies
- How to track long-running jobs

---

### 12: Custom Architectures (Coming Soon)

**Description:** Extend NDT to work with custom neural architecture.

**What you'll learn:**
- How to add support for custom layers
- How to write custom dimensionality metrics
- How to handle non-standard tensor shapes
- How to contribute back to NDT

---

## Example Templates

### Basic Template

```python
import torch
import torch.nn as nn
from ndt import HighFrequencyTracker, plot_phases, export_to_csv

# 1. Define model
model = nn.Sequential(...)

# 2. Create tracker
tracker = HighFrequencyTracker(model, sampling_frequency=10)

# 3. Training loop
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

for step, (x, y) in enumerate(dataloader):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    tracker.log(step, loss.item())

# 4. Analysis
results = tracker.get_results()
jumps = tracker.detect_jumps()

# 5. Visualization
fig = plot_phases(results, metric="stable_rank")
fig.savefig("results.png")

# 6. Export
export_to_csv(results, "results.csv")

# 7. Cleanup
tracker.close()
```

---

### Advanced Template (GPU, context manager, explicit layers)

```python
import torch
import torch.nn as nn
from ndt import HighFrequencyTracker, create_interactive_dashboard, export_to_hdf5

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model = MyComplexModel().to(device)

# Context manager for automatic cleanup
with HighFrequencyTracker(
    model,
    layers=[model.encoder.layer1, model.decoder.layer3],
    layer_names=["Encoder_L1", "Decoder_L3"],
    sampling_frequency=20,
    enable_jump_detection=True,
    device=device
) as tracker:

    # Training
    for step, batch in enumerate(dataloader):
        x, y = batch
        x, y = x.to(device), y.to(device)

        output = model(x)
        loss = criterion(output, y)
        loss.backward()

        # Track with gradient norm
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        tracker.log(step, loss.item(), grad_norm=grad_norm.item())

        if step >= 10000:
            break

    # Analysis & Export
    results = tracker.get_results()

    # Interactive dashboard
    fig = create_interactive_dashboard(results)
    fig.write_html("dashboard.html")

    # Efficient storage
    export_to_hdf5(results, "results.h5")

    # Jump analysis
    for layer_name, jumps in tracker.detect_jumps().items():
        print(f"{layer_name}: {len(jumps)} jumps")

# Tracker automatically closed
```

---

## Running All Examples

```bash
# Clone repository
git clone https://github.com/Javihaus/ndt.git
cd ndt

# Install dependencies
pip install -e .

# Run all examples
./scripts/run_all_examples.sh

# Or run individually
python examples/01_quickstart_mnist.py
python examples/02_cnn_cifar10.py
python examples/03_reproduce_tds_experiment.py
```

---

## Example Comparison

| Example | Architecture | Dataset | Steps | Sampling | Runtime | Difficulty |
|---------|--------------|---------|-------|----------|---------|------------|
| 01_quickstart | MLP (4-layer) | MNIST | 1k | 10 | 2 min | ⭐ Beginner |
| 02_cnn_cifar10 | CNN | CIFAR-10 | 2k | 20 | 10 min | ⭐⭐ Intermediate |
| 03_tds_experiment | MLP (3-layer) | MNIST | 8k | 5 | 15 min | ⭐⭐ Intermediate |
| 04_transformer | BERT | GLUE | 10k | 50 | 2 hours | ⭐⭐⭐ Advanced |
| 05_vit | ViT-Base | ImageNet | 100k | 100 | 10 hours | ⭐⭐⭐ Advanced |
| 06_resnet_imagenet | ResNet-50 | ImageNet | 500k | 100 | 3 days | ⭐⭐⭐⭐ Expert |

---

## Tips for Using Examples

### 1. Start Simple
Begin with `01_quickstart_mnist.py` to understand the basics.

### 2. Modify Hyperparameters
All examples are designed to be easily modified:
```python
# Change sampling frequency
tracker = HighFrequencyTracker(model, sampling_frequency=5)  # Was 10

# Change training steps
for step in range(2000):  # Was 1000
```

### 3. Add Your Own Models
Replace the model definition with your own:
```python
# Instead of example model
model = MyCustomModel()  # Your model here

# Rest stays the same
tracker = HighFrequencyTracker(model)
```

### 4. Experiment with Metrics
Try different metrics for analysis:
```python
# Instead of stable_rank
fig = plot_phases(results, metric="participation_ratio")
jumps = tracker.detect_jumps(metric="cum_energy_90")
```

### 5. Combine Examples
Mix and match techniques from different examples:
```python
# Combine TDS sampling + CNN architecture + gradient tracking
tracker = HighFrequencyTracker(cnn_model, sampling_frequency=5)
tracker.log(step, loss.item(), grad_norm=grad_norm.item())
```

---

## Testing Examples

All examples have corresponding tests in `tests/test_examples.py`:

```bash
# Test all examples (fast mode)
pytest tests/test_examples.py -v

# Test specific example
pytest tests/test_examples.py::test_quickstart_mnist -v

# Test with full runs (slow)
pytest tests/test_examples.py --run-full-examples
```

---

## Contributing Examples

Have an interesting use case? Contribute an example!

**Guidelines:**
1. **Runtime:** Should complete in <30 minutes on CPU
2. **Comments:** Explain every step clearly
3. **Output:** Include visualization and export
4. **Test:** Add test in `tests/test_examples.py`
5. **Documentation:** Add entry to this gallery

**Template:** Use `examples/00_template.py` as starting point.

**Submit:** Open PR with:
- Example file (`examples/XX_your_example.py`)
- Test (`tests/test_examples.py`)
- Documentation update (this file)
- README update (if applicable)

---

## See Also

- [API Reference](api_reference.md)
- [Quickstart Guide](quickstart.md)
- [Architecture Support](architecture_support.md)
- [Troubleshooting](troubleshooting.md)

---

## Questions?

- **Issues:** [GitHub Issues](https://github.com/Javihaus/ndt/issues)
- **Discussions:** [GitHub Discussions](https://github.com/Javihaus/ndt/discussions)
- **Email:** javier@jmarin.info
