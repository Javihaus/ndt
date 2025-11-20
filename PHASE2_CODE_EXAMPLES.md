# Phase 2 Mechanistic Interpretability - Code Examples & Quick Reference

## Table of Contents
1. Using Existing Infrastructure
2. Capturing Activations at Critical Moments
3. Analyzing Activation Geometry
4. Building Phase 2 Modules
5. Integration Patterns

---

## 1. USING EXISTING INFRASTRUCTURE

### 1.1 High-Frequency Tracking with Jump Detection

```python
import torch
import torch.nn as nn
from ndt import HighFrequencyTracker

# Create model
model = nn.Sequential(
    nn.Linear(784, 512), nn.ReLU(),
    nn.Linear(512, 256), nn.ReLU(),
    nn.Linear(256, 10)
)

# Create tracker with fine-grained sampling
tracker = HighFrequencyTracker(
    model, 
    sampling_frequency=1,  # Log every step
    enable_jump_detection=True,
    jump_z_threshold=2.5  # Lower threshold for earlier detection
)

# Standard training loop
for step, (x, y) in enumerate(dataloader):
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
    
    # Track dimensionality
    tracker.log(step, loss.item(), grad_norm=None)

# Analyze results
results = tracker.get_results()  # Dict[layer_name -> DataFrame]

# Detect jumps in all layers
all_jumps = tracker.detect_jumps(metric="stable_rank")

for layer_name, jumps in all_jumps.items():
    if jumps:
        print(f"\n{layer_name}:")
        for jump in jumps:
            print(f"  Step {jump.step}: {jump.value_before:.2f} -> {jump.value_after:.2f} "
                  f"(z={jump.z_score:.2f})")

# Export for later analysis
from ndt import export_to_hdf5
export_to_hdf5(results, "phase1_results.h5", 
               metadata={"model": "MLP", "dataset": "MNIST"})
```

### 1.2 Identifying Critical Moments from Existing Phase 1 Data

```python
import json
import numpy as np
from pathlib import Path

# Load Phase 1 results
phase1_results = Path("/home/user/ndt/experiments/new/results/phase1_full/")
results_file = phase1_results / "mlp_deep_10_mnist.json"

with open(results_file) as f:
    data = json.load(f)

# Extract dimensionality curves per layer
layers = data.keys()
for layer in layers:
    if layer.startswith("Linear"):
        steps = np.array(data[layer]["step"])
        stable_rank = np.array(data[layer]["stable_rank"])
        
        # Find jumps manually using derivative
        gradients = np.gradient(stable_rank)
        threshold = np.std(gradients) * 2.5
        jump_indices = np.where(np.abs(gradients) > threshold)[0]
        
        if len(jump_indices) > 0:
            print(f"\n{layer} - Potential jump points:")
            for idx in jump_indices:
                print(f"  Step {steps[idx]}: gradient = {gradients[idx]:.4f}")
```

### 1.3 Auto-Detect Architecture and Layers

```python
from ndt.architectures import get_handler, detect_architecture

# Supports: MLP, CNN, Transformer, ViT
model = your_model

# Auto-detect architecture
arch_name = detect_architecture(model)
print(f"Detected architecture: {arch_name}")

# Get handler with layer info
handler = get_handler(model)
layers_to_monitor = handler.get_activation_layers(model)
layer_names = handler.get_layer_names(model)

print(f"Monitoring {len(layers_to_monitor)} layers:")
for name in layer_names:
    print(f"  - {name}")

# Use with tracker
tracker = HighFrequencyTracker(
    model,
    layers=layers_to_monitor,
    layer_names=layer_names,
    sampling_frequency=5
)
```

---

## 2. CAPTURING ACTIVATIONS AT CRITICAL MOMENTS

### 2.1 Capture Before/After Dimensionality Jump

```python
import torch
import torch.nn as nn
from ndt.core.hooks import ActivationCapture

def capture_around_jump(model, dataloader, jump_step, capture_window=5):
    """
    Capture activations before and after a dimensionality jump.
    
    Args:
        jump_step: Step where jump was detected
        capture_window: Number of steps before/after to capture
    """
    
    activations_before = {}
    activations_after = {}
    
    # Register hooks on all Linear layers
    capture = ActivationCapture()
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    capture.register_hooks(model, linear_layers)
    
    model.eval()
    with torch.no_grad():
        for step, (x, y) in enumerate(dataloader):
            if step == jump_step - capture_window:
                # Capture "before" activation
                _ = model(x.to(device))
                activations_before = {
                    name: tensor.cpu().numpy() 
                    for name, tensor in capture.get_all_activations().items()
                }
                print(f"Captured before-jump at step {step}")
                
            elif step == jump_step + capture_window:
                # Capture "after" activation
                _ = model(x.to(device))
                activations_after = {
                    name: tensor.cpu().numpy() 
                    for name, tensor in capture.get_all_activations().items()
                }
                print(f"Captured after-jump at step {step}")
                break
    
    capture.remove_hooks()
    return activations_before, activations_after
```

### 2.2 Full Activation Matrix Capture at Specific Steps

```python
import h5py
from ndt.core.hooks import ActivationCapture

def capture_activations_at_steps(model, dataloader, target_steps, save_path="activations.h5"):
    """
    Capture full activation matrices at specific training steps.
    Saves to HDF5 for efficient storage.
    """
    
    capture = ActivationCapture()
    layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    capture.register_hooks(model, layers)
    
    activations_by_step = {}
    model.eval()
    
    with torch.no_grad():
        for step, (x, y) in enumerate(dataloader):
            if step in target_steps:
                # Clear previous activations
                capture.clear_activations()
                
                # Forward pass
                _ = model(x.to(device))
                
                # Store activations
                step_activations = {}
                for layer_idx, (name, module) in enumerate(
                    [(f"layer_{i}", m) for i, m in enumerate(layers)]
                ):
                    act = capture.get_activation(f"Linear_{layer_idx}_out")
                    if act is not None:
                        step_activations[name] = act.cpu().numpy()
                
                activations_by_step[step] = step_activations
                print(f"Captured activations at step {step}")
    
    # Save to HDF5
    with h5py.File(save_path, 'w') as f:
        for step, acts in activations_by_step.items():
            step_group = f.create_group(f"step_{step}")
            for layer_name, activation in acts.items():
                step_group.create_dataset(
                    layer_name, 
                    data=activation,
                    compression='gzip'
                )
    
    capture.remove_hooks()
    print(f"Saved {len(activations_by_step)} snapshots to {save_path}")
    return activations_by_step
```

---

## 3. ANALYZING ACTIVATION GEOMETRY

### 3.1 PCA Per Training Phase

```python
import numpy as np
from sklearn.decomposition import PCA

def analyze_activation_phases(activations_by_step, n_components=10):
    """
    Perform PCA on activations across training phases.
    """
    
    # Group steps into phases (e.g., before/during/after jump)
    steps = sorted(activations_by_step.keys())
    
    results = {}
    for layer_name in activations_by_step[steps[0]].keys():
        print(f"\nAnalyzing {layer_name}...")
        
        # Collect all activations for this layer
        all_activations = []
        step_indices = []
        
        for step in steps:
            act = activations_by_step[step][layer_name]
            # Flatten to 2D: (batch * spatial, features)
            if act.ndim > 2:
                act = act.reshape(act.shape[0], -1)
            all_activations.append(act)
            step_indices.extend([step] * len(act))
        
        # Combine into single matrix
        activation_matrix = np.vstack(all_activations)
        
        # Apply PCA
        pca = PCA(n_components=min(n_components, activation_matrix.shape[1]))
        transformed = pca.fit_transform(activation_matrix)
        
        print(f"  Explained variance: {pca.explained_variance_ratio_[:3]}")
        print(f"  Total variance explained: {pca.explained_variance_ratio_.sum():.2%}")
        
        results[layer_name] = {
            'pca': pca,
            'transformed': transformed,
            'step_indices': np.array(step_indices),
            'explained_variance': pca.explained_variance_ratio_
        }
    
    return results

# Usage:
pca_results = analyze_activation_phases(
    activations_by_step={step: acts for step, acts in ...},
    n_components=15
)

# Visualize explained variance
import matplotlib.pyplot as plt
for layer_name, result in pca_results.items():
    plt.figure()
    plt.bar(range(len(result['explained_variance'])), 
            result['explained_variance'])
    plt.title(f"{layer_name} - Explained Variance per Component")
    plt.xlabel("Principal Component")
    plt.ylabel("Variance Explained")
    plt.savefig(f"pca_{layer_name}.png", dpi=150)
```

### 3.2 Clustering Activations by Phase

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def cluster_activations_by_phase(activations_by_step, n_clusters=5):
    """
    Cluster activation patterns across training phases.
    """
    
    results = {}
    
    for layer_name in activations_by_step[list(activations_by_step.keys())[0]].keys():
        # Collect activations
        steps = sorted(activations_by_step.keys())
        all_activations = []
        step_labels = []
        
        for step in steps:
            act = activations_by_step[step][layer_name]
            if act.ndim > 2:
                act = act.reshape(act.shape[0], -1)
            all_activations.append(act)
            step_labels.extend([step] * len(act))
        
        activation_matrix = np.vstack(all_activations)
        
        # Standardize
        scaler = StandardScaler()
        activation_matrix_scaled = scaler.fit_transform(activation_matrix)
        
        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(activation_matrix_scaled)
        
        results[layer_name] = {
            'kmeans': kmeans,
            'cluster_labels': cluster_labels,
            'step_labels': np.array(step_labels),
            'scaler': scaler
        }
        
        # Analyze: which steps produce which clusters?
        print(f"\n{layer_name} Cluster Distribution:")
        for step in steps:
            mask = np.array(step_labels) == step
            clusters_at_step = cluster_labels[mask]
            unique, counts = np.unique(clusters_at_step, return_counts=True)
            print(f"  Step {step}: {dict(zip(unique, counts))}")
    
    return results
```

### 3.3 Neuron-Level Statistics

```python
def compute_neuron_statistics(activations_before, activations_after):
    """
    Compare neuron statistics before and after phase transition.
    """
    
    stats = {}
    
    for layer_name in activations_before.keys():
        act_before = activations_before[layer_name]
        act_after = activations_after[layer_name]
        
        if act_before.ndim > 2:
            # Flatten spatial dims
            act_before = act_before.reshape(act_before.shape[0], -1)
            act_after = act_after.reshape(act_after.shape[0], -1)
        
        # Per-neuron statistics
        mean_before = act_before.mean(axis=0)
        mean_after = act_after.mean(axis=0)
        
        std_before = act_before.std(axis=0)
        std_after = act_after.std(axis=0)
        
        sparsity_before = (act_before == 0).mean(axis=0)
        sparsity_after = (act_after == 0).mean(axis=0)
        
        # Find neurons with large changes
        mean_change = np.abs(mean_after - mean_before)
        mean_change_normalized = mean_change / (np.abs(mean_before) + 1e-8)
        
        # Dead neurons
        dead_before = sparsity_before > 0.9
        dead_after = sparsity_after > 0.9
        became_dead = ~dead_before & dead_after
        became_alive = dead_before & ~dead_after
        
        stats[layer_name] = {
            'mean_before': mean_before,
            'mean_after': mean_after,
            'mean_change': mean_change,
            'std_before': std_before,
            'std_after': std_after,
            'sparsity_before': sparsity_before,
            'sparsity_after': sparsity_after,
            'became_dead': became_dead,
            'became_alive': became_alive,
            'n_dead_before': dead_before.sum(),
            'n_dead_after': dead_after.sum(),
            'n_became_dead': became_dead.sum(),
            'n_became_alive': became_alive.sum(),
        }
    
    # Report
    for layer_name, stat in stats.items():
        print(f"\n{layer_name}:")
        print(f"  Neurons that became dead: {stat['n_became_dead']}")
        print(f"  Neurons that became alive: {stat['n_became_alive']}")
        print(f"  Mean activation change (avg): {stat['mean_change'].mean():.4f}")
        print(f"  Max activation change: {stat['mean_change'].max():.4f}")
    
    return stats
```

---

## 4. BUILDING PHASE 2 MODULES

### 4.1 Phase 2 Analyzer Class (Starting Template)

```python
import torch
import torch.nn as nn
from pathlib import Path
from ndt.core.hooks import ActivationCapture
from ndt.core.estimators import stable_rank, participation_ratio

class Phase2AnalyzerBase:
    """
    Base class for Phase 2 mechanistic analysis.
    Extends the tracking capability with detailed activation analysis.
    """
    
    def __init__(self, model, device=None):
        self.model = model
        self.device = device or (torch.device("cuda") 
                                if torch.cuda.is_available() 
                                else torch.device("cpu"))
        self.activations_cache = {}
        self.hooks = []
    
    def register_activation_hooks(self, layers=None):
        """Register hooks on specified layers."""
        if layers is None:
            layers = [m for m in self.model.modules() 
                     if isinstance(m, (nn.Linear, nn.Conv2d))]
        
        self.capture = ActivationCapture()
        self.capture.register_hooks(self.model, layers)
    
    def capture_activations(self, x):
        """Capture activations for input x."""
        self.model.eval()
        with torch.no_grad():
            _ = self.model(x.to(self.device))
            return self.capture.get_all_activations()
    
    def analyze_dimensionality(self, activations):
        """Compute dimensionality for all captured activations."""
        results = {}
        
        for layer_name, activation in activations.items():
            # Convert to 2D matrix
            if activation.ndim == 4:  # Conv2d
                activation = activation.view(activation.size(0), -1)
            elif activation.ndim == 3:  # Transformer
                activation = activation.view(-1, activation.size(-1))
            
            # Compute metrics
            results[layer_name] = {
                'stable_rank': stable_rank(activation),
                'participation_ratio': participation_ratio(activation),
                'shape': tuple(activation.shape),
                'mean': activation.mean(dim=0).cpu().numpy(),
                'std': activation.std(dim=0).cpu().numpy(),
            }
        
        return results
    
    def cleanup(self):
        """Remove hooks."""
        self.capture.remove_hooks()

# Usage:
analyzer = Phase2AnalyzerBase(model)
analyzer.register_activation_hooks()

# At critical moment
activations = analyzer.capture_activations(x_batch)
dims = analyzer.analyze_dimensionality(activations)

analyzer.cleanup()
```

### 4.2 Checkpoint Management for Phase 2

```python
import torch
import json
from pathlib import Path

class Phase2Checkpointer:
    """Manage checkpoints and activations during Phase 2 training."""
    
    def __init__(self, checkpoint_dir="phase2_checkpoints", 
                 capture_interval=50):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.capture_interval = capture_interval
        self.captured_steps = set()
    
    def should_capture(self, step):
        """Check if we should capture at this step."""
        return step % self.capture_interval == 0
    
    def save_checkpoint(self, model, step, metadata=None):
        """Save model checkpoint."""
        ckpt_path = self.checkpoint_dir / f"model_step_{step}.pt"
        torch.save(model.state_dict(), ckpt_path)
        
        if metadata:
            meta_path = self.checkpoint_dir / f"meta_step_{step}.json"
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        return ckpt_path
    
    def load_checkpoint(self, model, step):
        """Load model checkpoint."""
        ckpt_path = self.checkpoint_dir / f"model_step_{step}.pt"
        model.load_state_dict(torch.load(ckpt_path))
        return model
    
    def save_activations(self, activations, step):
        """Save activation snapshots."""
        import h5py
        
        act_path = self.checkpoint_dir / f"activations_step_{step}.h5"
        with h5py.File(act_path, 'w') as f:
            for layer_name, activation in activations.items():
                f.create_dataset(layer_name, data=activation,
                               compression='gzip', compression_opts=4)
        
        self.captured_steps.add(step)
        return act_path

# Usage:
checkpointer = Phase2Checkpointer(capture_interval=100)

for step, (x, y) in enumerate(dataloader):
    # ... training step ...
    
    if checkpointer.should_capture(step):
        checkpoint_path = checkpointer.save_checkpoint(model, step)
        activations = analyzer.capture_activations(x)
        checkpointer.save_activations(activations, step)
        print(f"Captured at step {step}")
```

---

## 5. INTEGRATION PATTERNS

### 5.1 Full Phase 2 Training Pipeline

```python
def phase2_mechanistic_experiment(model, dataloader, num_steps=5000):
    """
    Complete Phase 2 training with activation analysis.
    """
    
    # Setup
    tracker = HighFrequencyTracker(model, sampling_frequency=5)
    analyzer = Phase2AnalyzerBase(model)
    checkpointer = Phase2Checkpointer()
    
    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    critical_steps = set()
    
    for step in range(num_steps):
        # Get batch (cycle through dataloader)
        try:
            x, y = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            x, y = next(dataloader_iter)
        
        # Training step
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        
        # Track dimensionality
        tracker.log(step, loss.item())
        
        # Capture at critical moments
        if checkpointer.should_capture(step):
            activations = analyzer.capture_activations(x)
            checkpointer.save_activations(activations, step)
            critical_steps.add(step)
    
    # Post-analysis
    results = tracker.get_results()
    jumps = tracker.detect_jumps(metric="stable_rank")
    
    # Identify jump steps
    jump_steps = set()
    for layer_jumps in jumps.values():
        for jump in layer_jumps:
            jump_steps.add(jump.step)
    
    print(f"\nPhase 2 Complete:")
    print(f"  Training steps: {num_steps}")
    print(f"  Jump points: {jump_steps}")
    print(f"  Captured checkpoints: {critical_steps}")
    
    return {
        'tracker': tracker,
        'analyzer': analyzer,
        'checkpointer': checkpointer,
        'results': results,
        'jumps': jumps,
        'critical_steps': critical_steps
    }
```

### 5.2 Batch Analysis on Phase 1 Results

```python
def batch_phase2_analysis(phase1_results_dir, architectures=None):
    """
    Run Phase 2 analysis on a batch of Phase 1 results.
    """
    
    phase1_dir = Path(phase1_results_dir)
    json_files = list(phase1_dir.glob("*.json"))
    
    if architectures:
        json_files = [f for f in json_files 
                     if any(a in f.name for a in architectures)]
    
    all_results = {}
    
    for json_file in json_files:
        print(f"\nAnalyzing {json_file.name}...")
        
        with open(json_file) as f:
            data = json.load(f)
        
        # Identify jumps across layers
        layer_jumps = {}
        
        for layer_name in [k for k in data.keys() if 'Linear' in k]:
            steps = np.array(data[layer_name]['step'])
            sr = np.array(data[layer_name]['stable_rank'])
            
            # Detect jumps via derivative
            grad = np.gradient(sr)
            threshold = np.std(grad) * 2.5
            jump_indices = np.where(np.abs(grad) > threshold)[0]
            
            layer_jumps[layer_name] = {
                'jump_indices': jump_indices.tolist(),
                'jump_steps': steps[jump_indices].tolist()
            }
        
        all_results[json_file.stem] = {
            'file': json_file.name,
            'layer_jumps': layer_jumps,
            'n_jumps': sum(len(j['jump_indices']) for j in layer_jumps.values())
        }
    
    # Summary
    print("\n" + "="*60)
    print("Phase 2 Batch Analysis Summary")
    print("="*60)
    for exp_name, result in sorted(all_results.items()):
        print(f"{exp_name}: {result['n_jumps']} total jumps")
        for layer, jumps in result['layer_jumps'].items():
            if jumps['jump_indices']:
                print(f"  {layer}: {len(jumps['jump_indices'])} jumps at steps "
                      f"{jumps['jump_steps']}")
    
    return all_results
```

---

## Key Imports for Phase 2

```python
# Core tracking
from ndt import HighFrequencyTracker
from ndt.core.hooks import ActivationCapture
from ndt.core.jump_detector import JumpDetector
from ndt.core.estimators import (
    stable_rank, participation_ratio, 
    cumulative_energy_90, nuclear_norm_ratio
)
from ndt.architectures import get_handler

# Visualization
from ndt.visualization import (
    plot_single_metric, plot_phases, 
    plot_jumps, plot_correlation_heatmap
)
from ndt.visualization.interactive import create_interactive_plot

# Export
from ndt import export_to_csv, export_to_hdf5

# Scientific
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
```

