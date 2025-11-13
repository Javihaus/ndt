# Troubleshooting Guide

Common issues and solutions when using Neural Dimensionality Tracker.

## Installation Issues

### Issue: `pip install ndtracker` fails

**Error message:**
```
ERROR: Could not find a version that satisfies the requirement ndt
```

**Causes & Solutions:**

1. **Python version too old**
   ```bash
   python --version  # Check version
   ```
   - **Requirement:** Python 3.8+
   - **Solution:** Upgrade Python: `pyenv install 3.10` or use conda

2. **pip too old**
   ```bash
   pip install --upgrade pip
   pip install ndtracker
   ```

3. **Network/proxy issues**
   ```bash
   pip install --index-url https://pypi.org/simple/ ndt
   ```

---

### Issue: Import error after installation

**Error message:**
```python
ImportError: No module named 'ndt'
```

**Solutions:**

1. **Check installation:**
   ```bash
   pip list | grep ndt
   ```

2. **Wrong Python environment:**
   ```bash
   which python  # Verify correct Python
   pip install ndtracker  # Reinstall in correct env
   ```

3. **Development installation:**
   ```bash
   git clone https://github.com/Javihaus/ndt.git
   cd ndt
   pip install -e .
   ```

---

## Tracker Initialization Issues

### Issue: "No layers detected" warning

**Error message:**
```
Warning: No layers detected for tracking. Model might not contain Linear or Conv layers.
```

**Cause:** Auto-detection found no compatible layers.

**Solutions:**

1. **Check model architecture:**
   ```python
   print(model)  # Verify model has Linear or Conv layers
   ```

2. **Specify layers explicitly:**
   ```python
   tracker = HighFrequencyTracker(
       model,
       layers=[model.custom_layer1, model.custom_layer2],
       layer_names=["Layer1", "Layer2"]
   )
   ```

3. **Check for custom layers:**
   ```python
   # Custom layers need explicit specification
   for name, module in model.named_modules():
       print(f"{name}: {type(module)}")
   ```

---

### Issue: "Layer names must match length of layers"

**Error message:**
```
ValueError: layer_names must have same length as layers
```

**Cause:** Mismatch between `layers` and `layer_names` parameters.

**Solution:**
```python
# Wrong
tracker = HighFrequencyTracker(
    model,
    layers=[model.layer1, model.layer2, model.layer3],
    layer_names=["Layer1", "Layer2"]  # Only 2 names for 3 layers!
)

# Correct
tracker = HighFrequencyTracker(
    model,
    layers=[model.layer1, model.layer2, model.layer3],
    layer_names=["Layer1", "Layer2", "Layer3"]  # 3 names for 3 layers
)
```

---

## Runtime Errors

### Issue: "CUDA out of memory" during tracking

**Error message:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Reduce sampling frequency:**
   ```python
   tracker = HighFrequencyTracker(model, sampling_frequency=50)  # Was 10
   ```

2. **Track fewer layers:**
   ```python
   # Instead of all layers
   tracker = HighFrequencyTracker(model)

   # Track only key layers
   tracker = HighFrequencyTracker(
       model,
       layers=[model.layer1, model.layer4],  # Just 2 layers
       layer_names=["First", "Last"]
   )
   ```

3. **Reduce batch size:**
   ```python
   dataloader = DataLoader(dataset, batch_size=32)  # Was 128
   ```

4. **Move tracker to CPU:**
   ```python
   tracker = HighFrequencyTracker(model, device=torch.device("cpu"))
   ```

---

### Issue: "RuntimeError: shape mismatch in dimensionality estimation"

**Error message:**
```
RuntimeError: Expected 2D tensor, got 4D tensor of shape (32, 128, 7, 7)
```

**Cause:** Conv layer outputs 4D tensors, but estimator expects 2D.

**Solution:** NDT automatically flattens activations. If this error occurs:

```python
# Check layer output shape
def check_shape_hook(module, input, output):
    print(f"Output shape: {output.shape}")

model.conv1.register_forward_hook(check_shape_hook)

# If needed, manually flatten in hook
class CustomTracker(HighFrequencyTracker):
    def _process_activation(self, activation, layer_name):
        if activation.dim() > 2:
            activation = activation.flatten(1)
        super()._process_activation(activation, layer_name)
```

---

### Issue: "Hook already registered" error

**Error message:**
```
RuntimeError: Cannot register multiple hooks on the same module
```

**Cause:** Created multiple trackers on same model.

**Solution:**
```python
# Wrong
tracker1 = HighFrequencyTracker(model)
tracker2 = HighFrequencyTracker(model)  # Error!

# Correct: Close first tracker before creating new one
tracker1 = HighFrequencyTracker(model)
# ... use tracker1 ...
tracker1.close()
tracker2 = HighFrequencyTracker(model)  # OK now

# Or use context manager
with HighFrequencyTracker(model) as tracker:
    # training
    pass  # Automatically closed
```

---

## Data Collection Issues

### Issue: "No measurements recorded"

**Symptoms:** `tracker.get_results()` returns empty DataFrames

**Causes & Solutions:**

1. **Forgot to call `tracker.log()`:**
   ```python
   for step, (x, y) in enumerate(dataloader):
       output = model(x)
       loss = criterion(output, y)
       # Missing: tracker.log(step, loss.item())  # Add this!
   ```

2. **Sampling frequency too high:**
   ```python
   tracker = HighFrequencyTracker(model, sampling_frequency=10000)
   # If training only 1000 steps, no samples!

   # Solution: Lower sampling frequency
   tracker = HighFrequencyTracker(model, sampling_frequency=10)
   ```

3. **Model not in forward pass:**
   ```python
   # Trackers only work during forward pass
   model.eval()  # Evaluation mode still works
   with torch.no_grad():
       output = model(x)  # Hooks still fire
       tracker.log(step, loss.item())  # OK
   ```

---

### Issue: Measurements inconsistent or noisy

**Symptoms:** Erratic dimensionality values, unexpected jumps

**Causes & Solutions:**

1. **Batch size too small:**
   ```python
   # Small batch = noisy estimates
   dataloader = DataLoader(dataset, batch_size=4)  # Too small!

   # Solution: Use larger batches (32-128)
   dataloader = DataLoader(dataset, batch_size=64)
   ```

2. **Numerical instability:**
   ```python
   # Check for NaN/Inf
   results = tracker.get_results()
   for layer_name, df in results.items():
       print(f"{layer_name} NaN count: {df.isna().sum()}")

   # If NaN present, check model for:
   # - Exploding gradients
   # - Division by zero
   # - Log of negative numbers
   ```

3. **Mixed precision training:**
   ```python
   # AMP can cause instability in dimensionality estimates
   # Solution: Ensure tracker on same dtype as model
   from torch.cuda.amp import autocast

   with autocast():
       output = model(x)
   # Tracker handles mixed precision automatically
   ```

---

## Visualization Issues

### Issue: "Empty or incorrect plots"

**Symptoms:** Blank plots or plots missing data

**Solutions:**

1. **Check results exist:**
   ```python
   results = tracker.get_results()
   for layer_name, df in results.items():
       print(f"{layer_name}: {len(df)} measurements")
       if len(df) == 0:
           print(f"  WARNING: No data for {layer_name}")
   ```

2. **Verify metric name:**
   ```python
   # Wrong
   fig = plot_phases(results, metric="stable-rank")  # Wrong hyphen!

   # Correct
   fig = plot_phases(results, metric="stable_rank")  # Underscore
   ```

3. **Check matplotlib backend:**
   ```python
   import matplotlib
   print(matplotlib.get_backend())

   # If 'agg' (non-interactive), use:
   fig = plot_phases(results)
   fig.savefig("output.png")  # Don't use plt.show()
   ```

---

### Issue: Plotly dashboard doesn't display

**Symptoms:** `create_interactive_dashboard()` returns but nothing shows

**Solution:**
```python
from ndt import create_interactive_dashboard

fig = create_interactive_dashboard(results)

# In Jupyter notebook:
fig.show()

# In script:
fig.write_html("dashboard.html")
import webbrowser
webbrowser.open("dashboard.html")
```

---

## Export Issues

### Issue: "Permission denied" when exporting

**Error message:**
```
PermissionError: [Errno 13] Permission denied: 'results.csv'
```

**Solutions:**

1. **File is open in another program:**
   - Close Excel, text editor, etc.
   - Use different filename

2. **Directory doesn't exist:**
   ```python
   import os
   os.makedirs("outputs", exist_ok=True)
   export_to_csv(results, "outputs/results.csv")
   ```

3. **Insufficient permissions:**
   ```bash
   # Check write permissions
   ls -la .

   # Use writable directory
   export_to_csv(results, "/tmp/results.csv")
   ```

---

### Issue: "HDF5 file is corrupted"

**Error message:**
```
OSError: Unable to open file (file signature not found)
```

**Solutions:**

1. **File write was interrupted:**
   ```python
   # Use context manager for safety
   with h5py.File("results.h5", "w") as f:
       export_to_hdf5(results, f)
   ```

2. **Use built-in export function:**
   ```python
   # This handles errors automatically
   export_to_hdf5(results, "results.h5")
   ```

3. **Check disk space:**
   ```bash
   df -h .
   ```

---

## Performance Issues

### Issue: Training is very slow with tracker

**Symptoms:** >20% overhead, training much slower

**Solutions:**

1. **Check sampling frequency:**
   ```python
   # If sampling_frequency=1, try increasing
   tracker = HighFrequencyTracker(model, sampling_frequency=10)  # 10x faster
   ```

2. **Reduce tracked layers:**
   ```python
   # Count layers being tracked
   results = tracker.get_results()
   print(f"Tracking {len(results)} layers")

   # If >20 layers, reduce:
   tracker = HighFrequencyTracker(
       model,
       layers=[model.layer1, model.layer4],  # Track only 2
       layer_names=["First", "Last"]
   )
   ```

3. **Disable jump detection:**
   ```python
   tracker = HighFrequencyTracker(
       model,
       enable_jump_detection=False  # 10-15% faster
   )
   ```

4. **Profile overhead:**
   ```python
   import cProfile

   profiler = cProfile.Profile()
   profiler.enable()

   # Training loop
   tracker.log(step, loss.item())

   profiler.disable()
   profiler.print_stats(sort='cumtime')
   ```

---

### Issue: High memory usage

**Symptoms:** System runs out of RAM

**Solutions:**

1. **Increase sampling frequency:**
   ```python
   # Fewer measurements = less memory
   tracker = HighFrequencyTracker(model, sampling_frequency=100)
   ```

2. **Periodically export and clear:**
   ```python
   for epoch in range(100):
       # Training...
       tracker.log(step, loss.item())

       # Export every 10 epochs and clear
       if epoch % 10 == 0:
           results = tracker.get_results()
           export_to_hdf5(results, f"results_epoch{epoch}.h5")
           tracker.clear()  # Clear internal buffers
   ```

3. **Track fewer layers:**
   ```python
   # Reduce number of tracked layers
   tracker = HighFrequencyTracker(
       model,
       layers=[model.layer1],  # Only one layer
       layer_names=["Layer1"]
   )
   ```

---

## Multi-GPU / Distributed Training Issues

### Issue: "Multiple trackers created in distributed setup"

**Symptoms:** Duplicate measurements, errors about hooks

**Solution:**
```python
import torch.distributed as dist

# Only create tracker on rank 0
if dist.get_rank() == 0:
    tracker = HighFrequencyTracker(model)

# In training loop
loss = criterion(output, target)
if dist.get_rank() == 0:
    tracker.log(step, loss.item())
```

---

### Issue: "Device mismatch" in multi-GPU training

**Error message:**
```
RuntimeError: Expected all tensors to be on the same device
```

**Solution:**
```python
import torch.nn as nn

# For DataParallel
model = nn.DataParallel(model)
tracker = HighFrequencyTracker(model.module, device=torch.device("cuda:0"))

# For DistributedDataParallel
model = nn.parallel.DistributedDataParallel(model)
local_rank = int(os.environ["LOCAL_RANK"])
tracker = HighFrequencyTracker(model.module, device=torch.device(f"cuda:{local_rank}"))
```

---

## Framework Integration Issues

### Issue: PyTorch Lightning integration

**Problem:** Tracker not working with Lightning

**Solution:**
```python
import pytorch_lightning as pl
from ndt import HighFrequencyTracker

class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(...)
        self.tracker = None

    def on_train_start(self):
        # Initialize tracker when training starts
        self.tracker = HighFrequencyTracker(self.model)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x)
        loss = F.cross_entropy(output, y)

        # Log tracking
        self.tracker.log(self.global_step, loss.item())

        return loss

    def on_train_end(self):
        # Export results
        results = self.tracker.get_results()
        export_to_hdf5(results, "results.h5")
        self.tracker.close()
```

---

### Issue: Hugging Face Transformers integration

**Problem:** Can't find layers to track

**Solution:**
```python
from transformers import BertModel
from ndt import HighFrequencyTracker

model = BertModel.from_pretrained("bert-base-uncased")

# Print model structure to find layers
for name, module in model.named_modules():
    print(name, type(module))

# Track specific layers
tracker = HighFrequencyTracker(
    model,
    layers=[
        model.encoder.layer[0].intermediate.dense,
        model.encoder.layer[11].intermediate.dense
    ],
    layer_names=["L0_FFN", "L11_FFN"]
)
```

---

## Jump Detection Issues

### Issue: "No jumps detected" (but visual inspection shows jumps)

**Cause:** Z-score threshold too high

**Solution:**
```python
# Default threshold is 2.0, try lowering
jumps = tracker.detect_jumps(metric="stable_rank", threshold_z=1.5)

# Or adjust during initialization
tracker = HighFrequencyTracker(
    model,
    jump_z_threshold=1.5,  # More sensitive
    jump_window_size=30     # Larger window
)
```

---

### Issue: "Too many jumps detected" (false positives)

**Cause:** Z-score threshold too low or noisy data

**Solutions:**

1. **Increase threshold:**
   ```python
   jumps = tracker.detect_jumps(metric="stable_rank", threshold_z=3.0)
   ```

2. **Increase batch size:**
   ```python
   # Larger batches = less noisy estimates
   dataloader = DataLoader(dataset, batch_size=128)
   ```

3. **Increase window size:**
   ```python
   tracker = HighFrequencyTracker(
       model,
       jump_window_size=50  # Smoother detection
   )
   ```

---

## Common Mistakes

### 1. Forgetting to close tracker

```python
# Wrong: Hooks remain registered
tracker = HighFrequencyTracker(model)
# ... training ...
# (no tracker.close())

# Correct: Always close
tracker = HighFrequencyTracker(model)
# ... training ...
tracker.close()

# Better: Use context manager
with HighFrequencyTracker(model) as tracker:
    # ... training ...
    pass  # Automatically closed
```

---

### 2. Calling log() before forward pass

```python
# Wrong order
tracker.log(step, loss.item())  # No activations captured yet!
output = model(x)

# Correct order
output = model(x)
loss = criterion(output, y)
tracker.log(step, loss.item())  # Activations from forward pass
```

---

### 3. Modifying model after tracker creation

```python
# Wrong: Model changed after tracker created
tracker = HighFrequencyTracker(model)
model.add_module("new_layer", nn.Linear(128, 64))  # Tracker doesn't know!

# Correct: Create tracker after model is finalized
model.add_module("new_layer", nn.Linear(128, 64))
tracker = HighFrequencyTracker(model)  # Now tracks all layers
```

---

### 4. Using wrong metric names

```python
# Wrong
fig = plot_phases(results, metric="stable-rank")  # Hyphen!
jumps = tracker.detect_jumps(metric="pr")  # Abbreviation!

# Correct
fig = plot_phases(results, metric="stable_rank")  # Underscore
jumps = tracker.detect_jumps(metric="participation_ratio")  # Full name
```

---

## Getting Help

### 1. Enable debug logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)

# NDT will print detailed information
tracker = HighFrequencyTracker(model)
```

---

### 2. Check version

```python
import ndt
print(ndt.__version__)

# Ensure you have latest version
# pip install --upgrade ndt
```

---

### 3. Create minimal reproducer

```python
import torch
import torch.nn as nn
from ndt import HighFrequencyTracker

# Minimal model
model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))

# Minimal tracker
tracker = HighFrequencyTracker(model, sampling_frequency=1)

# Minimal training
x = torch.randn(4, 10)
y = torch.randint(0, 2, (4,))

output = model(x)
loss = nn.CrossEntropyLoss()(output, y)

tracker.log(0, loss.item())

# Check results
results = tracker.get_results()
for name, df in results.items():
    print(f"{name}: {len(df)} measurements")

tracker.close()
```

---

### 4. Report issues

If you encounter a bug:

1. **Check existing issues:** [GitHub Issues](https://github.com/Javihaus/ndt/issues)
2. **Create new issue with:**
   - NDT version
   - PyTorch version
   - Python version
   - Minimal reproducer
   - Full error traceback

---

## FAQ

**Q: Does NDT work with TorchScript?**

A: No, TorchScript doesn't support forward hooks. Use eager mode.

**Q: Can I use NDT with torch.compile?**

A: Partial support. Hooks work but may disable some optimizations.

**Q: Does NDT work on CPU?**

A: Yes! Works on CPU, GPU, and Apple Silicon (MPS).

**Q: Can I track custom modules?**

A: Yes, specify layers explicitly in the `layers` parameter.

**Q: How do I track only specific layers?**

A: Use the `layers` and `layer_names` parameters explicitly.

**Q: Can I pause/resume tracking?**

A: Not directly, but you can close and recreate the tracker.

---

## See Also

- [API Reference](api_reference.md)
- [Architecture Support](architecture_support.md)
- [Performance Benchmarks](performance_benchmarks.md)
- [Examples](https://github.com/Javihaus/ndt/tree/main/examples)
