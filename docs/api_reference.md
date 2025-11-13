# API Reference

Complete API documentation for Neural Dimensionality Tracker.

## Core Classes

### `HighFrequencyTracker`

Main class for tracking representational dimensionality during training.

```python
from ndt import HighFrequencyTracker

tracker = HighFrequencyTracker(
    model,
    layers=None,
    layer_names=None,
    sampling_frequency=10,
    enable_jump_detection=True,
    jump_z_threshold=2.0,
    jump_window_size=20,
    device=None
)
```

#### Parameters

- **model** (`torch.nn.Module`): The PyTorch model to track
- **layers** (`List[torch.nn.Module]`, optional): Specific layers to track. If `None`, auto-detects Linear and Conv layers
- **layer_names** (`List[str]`, optional): Names for tracked layers. Must match length of `layers`
- **sampling_frequency** (`int`, default=10): Record measurements every N steps
- **enable_jump_detection** (`bool`, default=True): Whether to enable automatic jump detection
- **jump_z_threshold** (`float`, default=2.0): Z-score threshold for detecting jumps
- **jump_window_size** (`int`, default=20): Window size for rolling statistics in jump detection
- **device** (`torch.device`, optional): Device to use for computations

#### Methods

##### `log(step: int, loss: float, grad_norm: Optional[float] = None) -> None`

Record dimensionality measurements at the current training step.

**Parameters:**
- `step`: Current training step
- `loss`: Training loss value
- `grad_norm`: Optional gradient norm value

**Example:**
```python
tracker.log(step=100, loss=0.5, grad_norm=2.3)
```

##### `get_results(layer_name: Optional[str] = None) -> Union[Dict[str, pd.DataFrame], pd.DataFrame]`

Get tracking results as pandas DataFrames.

**Parameters:**
- `layer_name`: If specified, return results for single layer. Otherwise return all layers.

**Returns:**
- Dictionary mapping layer names to DataFrames (all layers)
- Single DataFrame (specific layer)

**DataFrame columns:**
- `step`: Training step
- `loss`: Training loss
- `grad_norm`: Gradient norm (if tracked)
- `stable_rank`: Stable rank metric
- `participation_ratio`: Participation ratio metric
- `cum_energy_90`: Cumulative energy 90% metric
- `nuclear_norm_ratio`: Nuclear norm ratio metric

**Example:**
```python
# Get all results
all_results = tracker.get_results()

# Get specific layer
layer_results = tracker.get_results(layer_name="Linear_0")
```

##### `detect_jumps(metric: str = "stable_rank", threshold_z: Optional[float] = None) -> Dict[str, List[JumpDetection]]`

Detect phase transitions (jumps) in dimensionality metrics.

**Parameters:**
- `metric`: Which metric to use for detection (`"stable_rank"`, `"participation_ratio"`, etc.)
- `threshold_z`: Z-score threshold override

**Returns:**
- Dictionary mapping layer names to lists of `JumpDetection` objects

**Example:**
```python
jumps = tracker.detect_jumps(metric="stable_rank", threshold_z=3.0)
for layer_name, layer_jumps in jumps.items():
    print(f"{layer_name}: {len(layer_jumps)} jumps")
    for jump in layer_jumps:
        print(f"  Step {jump.step}: z={jump.z_score:.2f}")
```

##### `close() -> None`

Remove forward hooks and cleanup resources.

**Example:**
```python
tracker.close()

# Or use as context manager
with HighFrequencyTracker(model) as tracker:
    # training loop
    pass  # Automatically closed
```

---

## Visualization Functions

### `plot_phases`

Plot dimensionality metric over training for all layers.

```python
from ndt import plot_phases

fig = plot_phases(
    results,
    metric="stable_rank",
    title=None,
    figsize=(12, 6)
)
```

**Parameters:**
- `results`: Dictionary of layer results from `tracker.get_results()`
- `metric`: Which metric to plot
- `title`: Custom plot title
- `figsize`: Figure size tuple

**Returns:** `matplotlib.figure.Figure`

---

### `plot_metrics_comparison`

Plot all metrics for a single layer.

```python
from ndt import plot_metrics_comparison

fig = plot_metrics_comparison(
    layer_results,
    layer_name,
    figsize=(14, 10)
)
```

**Parameters:**
- `layer_results`: DataFrame for single layer
- `layer_name`: Name of the layer
- `figsize`: Figure size tuple

**Returns:** `matplotlib.figure.Figure`

---

### `create_interactive_dashboard`

Create interactive Plotly dashboard.

```python
from ndt import create_interactive_dashboard

fig = create_interactive_dashboard(
    results,
    metric="stable_rank"
)
fig.show()  # Opens in browser
```

**Parameters:**
- `results`: Dictionary of layer results
- `metric`: Primary metric to display

**Returns:** `plotly.graph_objects.Figure`

---

## Export Functions

### `export_to_csv`

Export results to CSV file.

```python
from ndt import export_to_csv

export_to_csv(results, "output.csv")
```

**Parameters:**
- `results`: Dictionary of layer results
- `filepath`: Output CSV file path

**Format:** Multi-indexed CSV with layer names as first column

---

### `export_to_json`

Export results to JSON file.

```python
from ndt import export_to_json

export_to_json(results, "output.json")
```

**Parameters:**
- `results`: Dictionary of layer results
- `filepath`: Output JSON file path

**Format:** Nested JSON with layers as top-level keys

---

### `export_to_hdf5`

Export results to HDF5 file (efficient for large datasets).

```python
from ndt import export_to_hdf5

export_to_hdf5(results, "output.h5")
```

**Parameters:**
- `results`: Dictionary of layer results
- `filepath`: Output HDF5 file path

**Format:** HDF5 with one dataset per layer

---

## Estimator Classes

### `DimensionalityEstimator`

Base class for dimensionality estimation.

```python
from ndt.core.estimators import DimensionalityEstimator

estimator = DimensionalityEstimator()
dims = estimator.estimate(activation_matrix)
```

#### Methods

##### `estimate(X: np.ndarray) -> Dict[str, float]`

Compute all dimensionality metrics from activation matrix.

**Parameters:**
- `X`: Activation matrix of shape `(batch_size, features)`

**Returns:** Dictionary with keys:
- `stable_rank`: Stable rank value
- `participation_ratio`: Participation ratio value
- `cum_energy_90`: Components for 90% energy
- `nuclear_norm_ratio`: Nuclear norm ratio value

---

### `StableRankEstimator`

Compute stable rank (effective dimensionality).

```python
from ndt.core.estimators import StableRankEstimator

estimator = StableRankEstimator()
sr = estimator.compute(X)
```

**Formula:** `stable_rank = ||X||_F^2 / ||X||_2^2`

Where:
- `||X||_F` is the Frobenius norm (sum of squared singular values)
- `||X||_2` is the spectral norm (largest singular value)

---

### `ParticipationRatioEstimator`

Compute participation ratio (variance distribution).

```python
from ndt.core.estimators import ParticipationRatioEstimator

estimator = ParticipationRatioEstimator()
pr = estimator.compute(X)
```

**Formula:** `PR = (sum(λ_i))^2 / sum(λ_i^2)`

Where `λ_i` are the eigenvalues of the covariance matrix.

---

## Architecture Detection

### `detect_architecture`

Automatically detect model architecture type.

```python
from ndt.architectures import detect_architecture

arch_type = detect_architecture(model)
# Returns: "mlp", "cnn", "transformer", "vit", or "unknown"
```

---

## Configuration

### `TrackerConfig`

Configuration dataclass for tracker settings.

```python
from ndt.utils.config import TrackerConfig

config = TrackerConfig(
    sampling_frequency=10,
    enable_jump_detection=True,
    jump_z_threshold=2.0,
    jump_window_size=20,
    track_gradients=True
)

tracker = HighFrequencyTracker.from_config(model, config)
```

---

## Data Classes

### `JumpDetection`

Dataclass representing a detected jump.

**Attributes:**
- `step` (`int`): Training step where jump occurred
- `metric` (`str`): Metric used for detection
- `z_score` (`float`): Z-score of the jump
- `magnitude` (`float`): Magnitude of the change
- `direction` (`str`): "increase" or "decrease"

---

## Constants

### Metrics

```python
from ndt import METRICS

METRICS = [
    "stable_rank",
    "participation_ratio",
    "cum_energy_90",
    "nuclear_norm_ratio"
]
```

### Architecture Types

```python
from ndt import ARCHITECTURE_TYPES

ARCHITECTURE_TYPES = [
    "mlp",
    "cnn",
    "transformer",
    "vit"
]
```

---

## Type Hints

All functions and classes are fully typed. Import types:

```python
from typing import Dict, List, Optional, Union
import pandas as pd
import torch
import numpy as np

from ndt.types import (
    LayerResults,      # Dict[str, pd.DataFrame]
    JumpDict,          # Dict[str, List[JumpDetection]]
    MetricName,        # Literal["stable_rank", ...]
    ArchitectureType   # Literal["mlp", "cnn", ...]
)
```

---

## Examples

### Complete Example

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ndt import HighFrequencyTracker, plot_phases, export_to_csv

# Model
model = nn.Sequential(
    nn.Linear(784, 256), nn.ReLU(),
    nn.Linear(256, 128), nn.ReLU(),
    nn.Linear(128, 10)
)

# Tracker
tracker = HighFrequencyTracker(model, sampling_frequency=5)

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for step, (x, y) in enumerate(dataloader):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    tracker.log(step, loss.item())

    if step >= 1000:
        break

# Analysis
results = tracker.get_results()
jumps = tracker.detect_jumps()

# Visualization
fig = plot_phases(results, metric="stable_rank")
fig.savefig("results.png")

# Export
export_to_csv(results, "results.csv")

# Cleanup
tracker.close()
```

---

## Advanced Usage

### Custom Layer Selection

```python
# Track specific layers
tracker = HighFrequencyTracker(
    model,
    layers=[model.encoder.layer1, model.decoder.layer3],
    layer_names=["Encoder_L1", "Decoder_L3"]
)
```

### GPU Support

```python
device = torch.device("cuda")
model = model.to(device)

tracker = HighFrequencyTracker(model, device=device)
```

### Distributed Training

```python
import torch.distributed as dist

# Only track on rank 0
if dist.get_rank() == 0:
    tracker = HighFrequencyTracker(model)

# In training loop
if dist.get_rank() == 0:
    tracker.log(step, loss.item())
```

### Context Manager

```python
with HighFrequencyTracker(model, sampling_frequency=10) as tracker:
    for step, (x, y) in enumerate(dataloader):
        # training code
        tracker.log(step, loss.item())
# Automatically cleaned up
```

---

## Performance Considerations

1. **Sampling Frequency**: Higher frequency = more overhead
   - `1-5`: High resolution, ~5-10% overhead
   - `10-20`: Balanced, ~2-5% overhead
   - `50-100`: Low overhead, < 1%

2. **Layer Selection**: Track fewer layers to reduce overhead

3. **Device**: Keep tracker on same device as model

4. **Memory**: Each measurement ~200 bytes per layer

5. **Export**: Use HDF5 for large datasets (>10k measurements)

---

## See Also

- [Quickstart Guide](quickstart.md)
- [Examples](../examples/)
- [Troubleshooting](troubleshooting.md)
- [Architecture Support](architecture_support.md)
