# Phase 2 Mechanistic Interpretability Infrastructure Analysis

## Executive Summary

The codebase has **strong foundational infrastructure** for Phase 2 mechanistic interpretability experiments, particularly for:
- High-frequency dimensionality tracking
- Activation capture at critical moments
- Jump detection at phase transitions
- Multi-scale analysis across architectures

**However, it lacks specialized tools** for feature visualization, intervention experiments, and detailed activation analysis needed for mechanistic interpretation work.

---

## 1. EXISTING MODEL ARCHITECTURES

### Location: `/home/user/ndt/src/ndt/architectures/`

**Available Handlers:**
- `MLPHandler` - Multi-layer perceptrons (Linear layers only)
- `CNNHandler` - CNNs with ResNet support (Conv2d + Linear)
- `TransformerHandler` - Transformer models (MultiheadAttention + Linear)
- `ViTHandler` - Vision Transformers
- `ArchitectureHandler` (abstract base)

**Registry System:**
- `registry.py` - Auto-detection of architecture type
- `get_handler()` function for automatic architecture detection
- Each handler has:
  - `validate_model()` - Check if model matches type
  - `get_activation_layers()` - Returns monitored layers
  - `get_layer_names()` - Generate descriptive names

**What You Can Do:**
```python
from ndt.architectures import get_handler
handler = get_handler(model)  # Auto-detects type
layers = handler.get_activation_layers(model)
layer_names = handler.get_layer_names(model)
```

**Status:** ✅ READY - Supports MLPs, CNNs, Transformers, ViTs

---

## 2. TRAINING INFRASTRUCTURE

### Location: `/home/user/ndt/examples/` and `/home/user/ndt/experiments/new/`

**Core Components:**

#### 2.1 High-Frequency Tracking
- **File:** `/home/user/ndt/src/ndt/core/tracker.py`
- **Class:** `HighFrequencyTracker`
- **Features:**
  - Logs at specified `sampling_frequency` (e.g., every 5 steps)
  - Captures 4 dimensionality metrics per layer per step
  - Auto-detects layers via architecture handlers
  - Stores: `DimensionalityMetrics` dataclass with step, metrics, loss, grad_norm
  - Can force logging at specific steps
  
**Usage:**
```python
tracker = HighFrequencyTracker(model, sampling_frequency=5)
for step, (x, y) in enumerate(dataloader):
    loss = train_step(model, x, y)
    tracker.log(step, loss.item())  # One line!
results = tracker.get_results()  # Dict[layer_name -> DataFrame]
```

**Status:** ✅ READY - Fully functional with auto layer detection

#### 2.2 Activation Capture Hooks
- **File:** `/home/user/ndt/src/ndt/core/hooks.py`
- **Class:** `ActivationCapture`
- **Features:**
  - Register forward hooks on arbitrary layers
  - Auto-reshape (handles 2D, 3D, 4D tensors)
  - Detach to avoid graph overhead
  - Context manager support
  
**Can capture:**
- Linear layer outputs: (batch, features)
- Conv outputs: (batch, channels, height, width)
- Transformer outputs: (batch, seq_len, hidden_dim)

**Status:** ✅ READY - Used by tracker internally

#### 2.3 Jump Detection at Phase Transitions
- **File:** `/home/user/ndt/src/ndt/core/jump_detector.py`
- **Class:** `JumpDetector`
- **Features:**
  - Z-score based detection
  - Rolling window statistics
  - Configurable threshold (default: 3.0)
  - Returns `Jump` dataclass with step, z_score, values

**Usage:**
```python
jumps = tracker.detect_jumps(layer_name="Linear_0", metric="stable_rank")
# Returns: Dict[layer_name -> List[Jump]]
for jump in jumps:
    print(f"Jump at step {jump.step}: {jump.value_before} -> {jump.value_after}")
```

**Limitations:** Only detects jumps in metrics, not semantic changes

**Status:** ✅ PARTIAL - Good for dimensionality jumps, needs enhancement for other signals

---

## 3. DIMENSIONALITY METRICS

### Location: `/home/user/ndt/src/ndt/core/estimators.py`

**Four Metrics Implemented:**

1. **Stable Rank** (`stable_rank()`)
   - Formula: ||A||²_F / ||A||²_2
   - Interpretation: Effective number of linearly independent dimensions
   
2. **Participation Ratio** (`participation_ratio()`)
   - Formula: (Σσᵢ)² / Σσ²ᵢ
   - Interpretation: How uniformly distributed singular values are

3. **Cumulative Energy 90** (`cumulative_energy_90()`)
   - Formula: # components for 90% variance
   - Interpretation: Intrinsic dimensionality (discrete)

4. **Nuclear Norm Ratio** (`nuclear_norm_ratio()`)
   - Formula: ||A||_* / ||A||_2
   - Interpretation: Normalized sum of singular values

**All functions:**
- Take 2D matrix (batch_size, features)
- Use SVD internally
- Include numerical stability (eps=1e-10)
- Raise ValueError on invalid input

**Status:** ✅ READY - Stable, well-tested

---

## 4. VISUALIZATION TOOLS

### Location: `/home/user/ndt/src/ndt/visualization/`

#### 4.1 Matplotlib Plots (plots.py)
Functions available:
- `plot_single_metric()` - Single metric + loss dual-axis
- `plot_metrics_comparison()` - 4 metrics stacked
- `plot_phases()` - Metric across multiple layers
- `plot_jumps()` - Metric with jump annotations
- `plot_correlation_heatmap()` - Inter-metric correlations

**Status:** ✅ READY - Good for static analysis

#### 4.2 Interactive Plotly (interactive.py)
- `create_interactive_plot()` - Interactive subplots
- Hovertips for precise values
- Save as HTML

**Status:** ✅ READY - Good for exploration

**What's Missing:**
- Feature maps/channel visualizations
- Attention map visualizations
- Activation distributions
- Correlation with performance metrics

---

## 5. DATA EXPORT & CHECKPOINT MANAGEMENT

### Location: `/home/user/ndt/src/ndt/export/`

#### 5.1 CSV Export (csv.py)
```python
export_to_csv(results, "results.csv")  # Combined
export_to_csv(results, "results/", separate_files=True)  # Per-layer
results = load_from_csv("results.csv")
```

#### 5.2 JSON Export (json.py)
- Metadata support
- Human-readable format

#### 5.3 HDF5 Export (hdf5.py)
```python
export_to_hdf5(results, "results.h5", metadata={...}, compression="gzip")
results, metadata = load_from_hdf5("results.h5")
```

**Status:** ✅ READY - Multiple formats, can store metadata

**Checkpoint Saving:**
- Not explicitly implemented in core
- Can checkpoint models separately via PyTorch
- Results can be saved at multiple intervals

---

## 6. EXISTING MECHANISTIC INTERPRETABILITY FEATURES

### Limited Tools Available:

1. **Activation Dimensionality Analysis**
   - Can measure dimensionality at any layer
   - Can compare before/after jumps
   - ✅ Ready to use

2. **Multi-Layer Comparison**
   - Can track all layers simultaneously
   - Can correlate across depth
   - ✅ Ready to use

3. **Time-Series Analysis**
   - High-frequency sampling (every N steps)
   - Can align with performance jumps
   - ✅ Ready to use

### Missing Tools:

1. **Feature Visualization** ❌
   - No class activation maps (CAM)
   - No gradient-based visualization
   - No attention visualization
   - No feature importance scoring

2. **Activation Analysis** ❌
   - No clustering of activations
   - No manifold visualization (t-SNE, UMAP)
   - No PCA per phase
   - No neuron statistics

3. **Intervention/Ablation** ❌
   - No pruning utilities
   - No weight perturbation
   - No channel ablation
   - No causal analysis

4. **Semantic Analysis** ❌
   - No decision boundary tracking
   - No representation collapse detection
   - No feature interaction analysis

---

## 7. TRAINING EXAMPLES & REFERENCE IMPLEMENTATIONS

### Location: `/home/user/ndt/examples/` and `/home/user/ndt/experiments/new/`

#### Available Examples:
1. **01_quickstart_mnist.py** - Minimal 3-layer MLP on MNIST
2. **02_cnn_cifar10.py** - CNN on CIFAR-10
3. **03_reproduce_tds_experiment.py** - Full reproduction (784-256-128-10 MLP)

#### Experiment Scripts (production quality):
1. **phase1_calibration.py**
   - 17 architectures × 5 datasets
   - Training loop with measurement every 5 steps
   - Captures: stable_rank, participation_ratio, effective_rank_90, nuclear_norm_ratio
   - Saves results as JSON (1MB+ per experiment)

2. **phase1_analysis.py**
   - TAP model fitting
   - α parameter extraction
   - Dimensionality jump detection

3. **phase2_prediction.py**
   - Predicts D(t) before training
   - Uses fitted α models
   - Compares predicted vs actual

4. **phase3_realtime_monitor.py** & **phase4_capability_emergence.py**
   - Real-time monitoring
   - Performance correlation analysis

**Status:** ✅ Reference implementations available, can fork for Phase 2

---

## 8. DATA LOADING UTILITIES

### Location: `/home/user/ndt/experiments/new/`

**Files:**
- `dataset_downloader.py` - Downloads MNIST, CIFAR-10, Fashion-MNIST, QMNIST, AG News
- `modern_dataset_loaders.py` - Configurable DataLoader creation
- `architecture_dataset_mapping.py` - Recommended architecture-dataset pairs

**Available Datasets:**
- MNIST (10 classes, 28×28 grayscale)
- Fashion-MNIST (10 classes, 28×28 grayscale)
- QMNIST (variants of MNIST)
- CIFAR-10 (10 classes, 32×32 RGB)
- SVHN (Street View House Numbers)
- AG News (text classification, 4 classes)

**Status:** ✅ READY - Can load standard datasets

---

## 9. RESULTS & METRICS COMPUTED

### Current Results Structure:

**Files:** `/home/user/ndt/experiments/new/results/phase1_full/`

Example: `mlp_deep_10_mnist.json`
```
{
  "step": [0, 5, 10, ...],
  "stable_rank": [...],
  "participation_ratio": [...],
  "cumulative_90": [...],
  "nuclear_norm_ratio": [...],
  "loss": [...]
}
```

**Per Layer:** Separate measurements for each monitored layer
**Per Step:** High-frequency measurements (every 5 steps)

**Saved Models:** `/home/user/ndt/experiments/new/results/mechanistic/`

**Status:** ✅ READY to extend with new metrics

---

## 10. COMPREHENSIVE INFRASTRUCTURE CHECKLIST

### What EXISTS and is READY:

- ✅ Model architecture detection (MLP, CNN, Transformer, ViT)
- ✅ Automatic layer selection for monitoring
- ✅ Forward hooks for activation capture
- ✅ High-frequency logging (every N steps)
- ✅ 4 complementary dimensionality metrics
- ✅ Jump detection via z-score
- ✅ 2D activation matrix handling (auto-reshape)
- ✅ CSV/JSON/HDF5 export
- ✅ Matplotlib + Plotly visualization
- ✅ Production training examples
- ✅ 17 architectures × 5 datasets tested
- ✅ Jump detection at critical moments

### What NEEDS TO BE BUILT for Phase 2:

#### Priority 1 (Core Mechanistic Tools):
- ❌ Class Activation Maps (CAM) for CNNs
- ❌ Activation clustering/PCA per phase
- ❌ Layer-wise PCA throughout training
- ❌ Feature importance scoring
- ❌ Attention visualization for Transformers

#### Priority 2 (Intervention Tools):
- ❌ Channel-level ablation framework
- ❌ Neuron-level pruning utilities
- ❌ Weight perturbation analysis
- ❌ Causal intervention framework

#### Priority 3 (Analysis Tools):
- ❌ Dead neuron detection
- ❌ Representation collapse detection
- ❌ Decision boundary tracking
- ❌ Batch comparison utilities (before/after jump)

---

## 11. RECOMMENDED PHASE 2 ARCHITECTURE

### Module Structure:

```
/home/user/ndt/src/ndt/
├── architectures/          # ✅ EXISTING
│   ├── base.py
│   ├── mlp.py
│   ├── cnn.py
│   ├── transformer.py
│   ├── vit.py
│   └── registry.py
│
├── core/                   # ✅ MOSTLY COMPLETE
│   ├── hooks.py            # ✅ Activation capture
│   ├── tracker.py          # ✅ High-frequency tracking
│   ├── estimators.py       # ✅ 4 dimensionality metrics
│   ├── jump_detector.py    # ✅ Phase transition detection
│   └── interventions.py    # ❌ NEW: Ablation/perturbation
│
├── analysis/               # ❌ NEW MODULE: Phase 2 analysis
│   ├── __init__.py
│   ├── feature_viz.py      # CAM, attention maps
│   ├── activation_analysis.py  # PCA, clustering
│   ├── neuron_importance.py    # Feature scoring
│   └── interventions.py    # Pruning, perturbations
│
├── visualization/          # ✅ EXTENDED
│   ├── plots.py            # ✅ Core plots
│   ├── interactive.py      # ✅ Interactive plots
│   ├── feature_maps.py     # ❌ NEW: Feature viz
│   └── attention_maps.py   # ❌ NEW: Attention viz
│
└── export/                 # ✅ READY
    ├── csv.py
    ├── json.py
    └── hdf5.py
```

---

## 12. KEY FILES FOR PHASE 2 REFERENCE

### Foundation Files to Understand:

1. **`/home/user/ndt/src/ndt/core/tracker.py`** (366 lines)
   - Core tracking infrastructure
   - Shows how to integrate hooks
   - Example of adding custom metrics

2. **`/home/user/ndt/src/ndt/core/hooks.py`** (132 lines)
   - Activation capture mechanism
   - Forward hook pattern
   - Context manager pattern

3. **`/home/user/ndt/experiments/new/phase1_calibration.py`** (600+ lines)
   - Complete training loop with measurement
   - ActivationCapture usage
   - Dimensionality measurement at each step
   - Architecture/dataset iteration

4. **`/home/user/ndt/src/ndt/visualization/plots.py`** (274 lines)
   - Matplotlib plotting patterns
   - Multi-layer comparison examples
   - Styling conventions

### Reference Experiments:

1. **`/home/user/ndt/experiments/new/phase1_analysis.py`** (600+ lines)
   - TAP model fitting
   - Jump detection implementation
   - Batch processing patterns

2. **`/home/user/ndt/experiments/new/phase2_prediction.py`** (400+ lines)
   - How to compare predicted vs actual
   - Results aggregation patterns

---

## 13. DATA ALREADY COLLECTED

### Phase 1 Results Available:

**Location:** `/home/user/ndt/experiments/new/results/phase1_full/`

**Content:**
- 18 JSON files with training trajectories
- Each ~500KB, contains ~5000 measurements
- Multi-layer per experiment
- All 4 dimensionality metrics

**Example Experiments:**
- `mlp_deep_10_mnist.json` - 10-layer MLP
- `cnn_deep_fashion_mnist.json` - Deep CNN
- `transformer_medium_ag_news.json` - Transformer on text

**What This Enables:**
- Can re-analyze existing data with new Phase 2 methods
- Can identify critical moments (jumps) retrospectively
- Can correlate metrics without re-training

**Validation Data:**
- `/home/user/ndt/experiments/validation_data.pkl` (150KB)
- Pre-loaded for quick testing

---

## 14. IMPLEMENTATION STRATEGY FOR PHASE 2

### Step 1: Identify Critical Moments (0-1 week)
- Use existing jump detector on all phase1 results
- Identify step ranges around dimensionality jumps
- Save checkpoint locations

### Step 2: Activate Capture Around Jumps (1-2 weeks)
- Rerun training with checkpointing at jump moments
- Capture full activation matrices at before/after jump
- Store as HDF5 (space-efficient for large matrices)

### Step 3: Activation Analysis (2-3 weeks)
- PCA per phase
- Clustering of activation patterns
- Layer-wise correlation analysis
- Neuron importance scoring

### Step 4: Visualization & Interpretation (3-4 weeks)
- Feature maps for CNNs
- Attention visualizations for Transformers
- Before/after jump comparisons
- Manifold visualizations

### Step 5: Intervention Experiments (4-6 weeks)
- Design targeted ablations
- Test importance of discovered features
- Causal analysis

---

## 15. QUICK START FOR PHASE 2 DEVELOPMENT

### To Extend Core Infrastructure:

```python
# 1. Use existing tracker to capture at specific moments
from ndt import HighFrequencyTracker
tracker = HighFrequencyTracker(model, sampling_frequency=1)

# 2. Detect jumps first
jumps = tracker.detect_jumps(metric="stable_rank")

# 3. Identify when to capture detailed activations
critical_steps = [jump.step for jump in jumps["Linear_0"]]

# 4. Rerun training and capture activations at critical steps
# (Build this in Phase 2 module)
for step, (x, y) in enumerate(dataloader):
    # ... training ...
    tracker.log(step, loss.item())
    
    if step in critical_steps:
        # Capture full activation matrix
        activation_capture.register_hooks(model, layers)
        with torch.no_grad():
            _ = model(x)
            activations = activation_capture.get_all_activations()
            # Save to HDF5 or pickle
```

### Example Phase 2 Class Structure:

```python
from ndt.core.hooks import ActivationCapture
from ndt.core.estimators import compute_all_metrics

class Phase2Analyzer:
    """Mechanistic analysis at critical moments."""
    
    def __init__(self, model, critical_steps):
        self.model = model
        self.critical_steps = critical_steps
        self.activations_by_step = {}
    
    def capture_at_steps(self, dataloader):
        """Capture activations at critical moments."""
        for step, (x, y) in enumerate(dataloader):
            if step in self.critical_steps:
                # Capture and analyze
                pass
    
    def analyze_activation_geometry(self, layer_name):
        """PCA, clustering, etc."""
        pass
    
    def compute_feature_importance(self):
        """Neuron importance scores."""
        pass
```

---

## SUMMARY: WHAT YOU HAVE vs WHAT YOU NEED

### READY TO USE:
1. Model detection & layer selection
2. High-frequency activation capture
3. Dimensionality measurement (4 metrics)
4. Jump detection
5. Export/visualization
6. Training infrastructure
7. Phase 1 reference implementations

### YOU NEED TO BUILD:
1. Feature visualization (CAM, attention maps)
2. Activation PCA per phase
3. Neuron importance scoring
4. Ablation frameworks
5. Causal analysis tools
6. Batch comparison utilities
7. Representation geometry analysis

### EFFORT ESTIMATE:
- **Foundation:** 1 week (understand existing code)
- **Core Phase 2 Tools:** 2-3 weeks
- **Experimentation:** 2-4 weeks
- **Writing/Visualization:** 1-2 weeks
- **Total:** 6-10 weeks for comprehensive Phase 2

---

## KEY DOCUMENTATION FILES

- `/home/user/ndt/README.md` - Library overview
- `/home/user/ndt/examples/` - Working examples
- `/home/user/ndt/experiments/new/README.md` - Experiment guide
- `/home/user/ndt/experiments/new/EXPERIMENT_PLAN.md` - Full research plan

