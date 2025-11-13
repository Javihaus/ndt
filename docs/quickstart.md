# Quickstart Guide

Get started with Neural Dimensionality Tracker in 5 minutes.

## Installation

```bash
pip install ndt
```

## Basic Usage

### 1. Import and Create Tracker

```python
import torch
import torch.nn as nn
from ndt import HighFrequencyTracker

# Your PyTorch model
model = nn.Sequential(
    nn.Linear(784, 512), nn.ReLU(),
    nn.Linear(512, 256), nn.ReLU(),
    nn.Linear(256, 10)
)

# Create tracker (automatically detects layers)
tracker = HighFrequencyTracker(
    model,
    sampling_frequency=10  # Record every 10 steps
)
```

### 2. Training Loop

Add a single line to your training loop:

```python
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

for step, (x, y) in enumerate(dataloader):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    # Track dimensionality - just one line!
    tracker.log(step, loss.item())
```

### 3. Analyze Results

```python
# Get results as pandas DataFrames
results = tracker.get_results()

# results is a dictionary: layer_name -> DataFrame
for layer_name, df in results.items():
    print(f"{layer_name}: {len(df)} measurements")
    print(df.head())
```

### 4. Detect Jumps

```python
# Detect dimensionality jumps (phase transitions)
jumps = tracker.detect_jumps(metric="stable_rank")

for layer_name, layer_jumps in jumps.items():
    print(f"{layer_name}: {len(layer_jumps)} jumps detected")
    for jump in layer_jumps:
        print(f"  Step {jump.step}: z-score={jump.z_score:.2f}")
```

### 5. Visualize

```python
from ndt import plot_phases, plot_metrics_comparison

# Plot stable rank across all layers
fig = plot_phases(results, metric="stable_rank")
fig.savefig("stable_rank.png")

# Plot all metrics for one layer
layer_results = tracker.get_results(layer_name="Linear_0")
fig = plot_metrics_comparison(layer_results, layer_name="Linear_0")
fig.savefig("all_metrics.png")
```

### 6. Export

```python
from ndt import export_to_csv, export_to_json

# Export to CSV
export_to_csv(results, "results.csv")

# Export to JSON
export_to_json(results, "results.json")
```

### 7. Cleanup

```python
# Remove hooks and free resources
tracker.close()

# Or use as context manager (automatic cleanup)
with HighFrequencyTracker(model) as tracker:
    # Training loop
    pass
# Hooks automatically removed
```

## What Gets Tracked?

For each layer, at each sampled step, NDT computes:

1. **Stable Rank**: Effective dimensionality (robust to noise)
2. **Participation Ratio**: How evenly variance is distributed
3. **Cumulative Energy 90%**: Components needed for 90% variance
4. **Nuclear Norm Ratio**: Normalized measure of rank

Plus training loss and optional gradient norm.

## Customization

### Specify Layers Explicitly

```python
tracker = HighFrequencyTracker(
    model,
    layers=[model[0], model[2], model[4]],
    layer_names=["Input", "Hidden", "Output"]
)
```

### Adjust Sampling

```python
# Record every step (more data, slower)
tracker = HighFrequencyTracker(model, sampling_frequency=1)

# Record every 100 steps (less data, faster)
tracker = HighFrequencyTracker(model, sampling_frequency=100)
```

### Configure Jump Detection

```python
tracker = HighFrequencyTracker(
    model,
    enable_jump_detection=True,
    jump_z_threshold=3.0,  # Higher = fewer false positives
    jump_window_size=50     # Window for rolling statistics
)
```

## Next Steps

- See [examples/](https://github.com/Javihaus/ndt/tree/main/examples) for complete working code
- Check API documentation for advanced features
- Try with your own models and datasets

## Common Issues

**Q: How do I reduce memory usage?**

A: Increase `sampling_frequency` or track fewer layers.

**Q: Can I use this with distributed training?**

A: Yes, but create separate trackers per process or track on rank 0 only.

**Q: Does this work with custom architectures?**

A: Yes! Specify layers explicitly if auto-detection doesn't work.

**Q: How much overhead does tracking add?**

A: < 10% with `sampling_frequency=1`, negligible with higher values.
