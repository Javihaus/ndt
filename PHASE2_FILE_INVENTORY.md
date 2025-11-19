# Phase 2 Mechanistic Interpretability - Complete File Inventory

## ABSOLUTE FILE PATHS & STRUCTURE

### Core Infrastructure Files (READY FOR PHASE 2)

#### 1. Architecture Detection & Layer Selection
```
/home/user/ndt/src/ndt/architectures/__init__.py
/home/user/ndt/src/ndt/architectures/base.py                 (Abstract handler - 60 lines)
/home/user/ndt/src/ndt/architectures/mlp.py                   (MLP handler - 73 lines)
/home/user/ndt/src/ndt/architectures/cnn.py                   (CNN handler with ResNet - 130 lines)
/home/user/ndt/src/ndt/architectures/transformer.py           (Transformer handler - 109 lines)
/home/user/ndt/src/ndt/architectures/vit.py                   (ViT handler - ~80 lines)
/home/user/ndt/src/ndt/architectures/registry.py              (Auto-detection system)
```

**Key Classes:**
- `ArchitectureHandler` - Abstract base with 3 required methods
- `MLPHandler`, `CNNHandler`, `TransformerHandler`, `ViTHandler`
- `detect_architecture(model)` - Returns architecture name
- `get_handler(model)` - Returns appropriate handler instance

#### 2. Activation Capture (Forward Hooks)
```
/home/user/ndt/src/ndt/core/hooks.py                          (132 lines)
```

**Key Class: `ActivationCapture`**
- Methods:
  - `register_hooks(model, layers, layer_names)` - Register on layers
  - `remove_hooks()` - Cleanup
  - `clear_activations()` - Reset without removing hooks
  - `get_activation(layer_name)` - Get single activation
  - `get_all_activations()` - Get dict of all
- Features:
  - Auto-detach to avoid graph overhead
  - Works as context manager
  - Auto-reshape support (2D/3D/4D tensors)

#### 3. High-Frequency Tracking
```
/home/user/ndt/src/ndt/core/tracker.py                        (366 lines)
```

**Key Class: `HighFrequencyTracker`**
- Methods:
  - `__init__(model, layers, sampling_frequency, enable_jump_detection, ...)`
  - `log(step, loss, grad_norm, force)` - Called during training
  - `get_results(layer_name)` - Returns DataFrame(s)
  - `detect_jumps(layer_name, metric)` - Find phase transitions
  - `close()` - Cleanup
- Features:
  - Auto-layer detection
  - 4 dimensionality metrics per layer
  - Jump detection integrated
  - Works as context manager

**Data Structure: `DimensionalityMetrics` dataclass**
- Fields: step, stable_rank, participation_ratio, cumulative_90, nuclear_norm_ratio, loss, grad_norm

#### 4. Dimensionality Metrics (SVD-based)
```
/home/user/ndt/src/ndt/core/estimators.py                     (150+ lines)
```

**Functions:**
- `stable_rank(matrix)` - ||A||²_F / ||A||²_2
- `participation_ratio(matrix)` - (Σσᵢ)² / Σσ²ᵢ
- `cumulative_energy_90(matrix, threshold)` - # components for 90% variance
- `nuclear_norm_ratio(matrix)` - ||A||_* / ||A||_2

**All:** Take torch.Tensor, return float, handle 2D matrices

#### 5. Phase Transition Detection
```
/home/user/ndt/src/ndt/core/jump_detector.py                  (200+ lines)
```

**Key Classes:**
- `Jump` dataclass: step, z_score, value_before, value_after, metric_name
- `JumpDetector` class:
  - Methods: `detect_jumps(values, metric_name, step_offset)` → List[Jump]
  - Configurable: window_size, z_threshold, min_samples

#### 6. Visualization Tools
```
/home/user/ndt/src/ndt/visualization/plots.py                 (274 lines)
/home/user/ndt/src/ndt/visualization/interactive.py           (80+ lines)
```

**Matplotlib Functions:**
- `plot_single_metric(df, metric, layer_name, show_loss)` - Line plot + loss
- `plot_metrics_comparison(df, metrics, layer_name)` - 4 metrics stacked
- `plot_phases(results_dict, metric)` - Multi-layer comparison
- `plot_jumps(df, jumps, metric, layer_name)` - With annotations
- `plot_correlation_heatmap(df, metrics)` - Seaborn heatmap

**Plotly Functions:**
- `create_interactive_plot(df, metrics, layer_name, show_loss)` - HTML interactive

#### 7. Data Export
```
/home/user/ndt/src/ndt/export/csv.py                          (114 lines)
/home/user/ndt/src/ndt/export/json.py                         (120+ lines)
/home/user/ndt/src/ndt/export/hdf5.py                         (200+ lines)
```

**Functions:**
- CSV: `export_to_csv(results, path, separate_files)`, `load_from_csv(path, separate_files)`
- JSON: `export_to_json(results, path)`, `load_from_json(path)`
- HDF5: `export_to_hdf5(results, path, compression, metadata)`, `load_from_hdf5(path)`

---

### Example & Reference Implementations

#### Working Examples
```
/home/user/ndt/examples/01_quickstart_mnist.py                (101 lines)
/home/user/ndt/examples/02_cnn_cifar10.py                     (~150 lines)
/home/user/ndt/examples/03_reproduce_tds_experiment.py        (~200 lines)
```

**01_quickstart_mnist.py** - Perfect starting point
- Simple 4-layer MLP on MNIST
- Shows complete pipeline: track → detect jumps → visualize → export
- Uses HighFrequencyTracker directly

#### Phase 1 Experimental Framework
```
/home/user/ndt/experiments/new/phase1_calibration.py          (600+ lines)
/home/user/ndt/experiments/new/phase1_analysis.py             (600+ lines)
/home/user/ndt/experiments/new/phase2_prediction.py           (400+ lines)
/home/user/ndt/experiments/new/phase3_realtime_monitor.py     (500+ lines)
/home/user/ndt/experiments/new/phase4_capability_emergence.py (600+ lines)
```

**phase1_calibration.py Key Components:**
- `DimensionalityEstimator` class (4 metrics)
- `ActivationCapture` class (duplicate in experiments)
- `SimpleMLP`, `SimpleCNN` model definitions
- `measure_network_dimensionality()` function
- Complete training loops with measurement every 5 steps
- Support for MNIST, CIFAR-10, Fashion-MNIST, QMNIST, AG News

**phase1_analysis.py:**
- TAP model fitting
- `logistic_growth()` function
- `detect_jumps()` for dimensionality analysis
- Batch analysis on multiple experiments
- Jump visualization patterns

**phase2_prediction.py:**
- `DimensionalityPredictor` class
- Loads α models from phase1
- `estimate_alpha()` from architecture params
- `predict_curve()` using TAP model
- Result comparison (predicted vs actual)

#### Utility Scripts
```
/home/user/ndt/experiments/new/visualization_utils.py         (500+ lines)
/home/user/ndt/experiments/new/dataset_downloader.py
/home/user/ndt/experiments/new/modern_dataset_loaders.py
/home/user/ndt/experiments/new/architecture_dataset_mapping.py
```

**visualization_utils.py:**
- `setup_plot_style()` - Consistent styling
- `plot_dimensionality_curve()` - With jumps and predictions
- `plot_multi_layer_dimensionality()` - Multi-layer comparison
- Comparison and result plotting functions

---

### Results & Data Files

#### Phase 1 Experimental Results
```
/home/user/ndt/experiments/new/results/phase1_full/           (18 JSON files)
  - mlp_deep_10_mnist.json                                     (~1 MB)
  - mlp_deep_10_fashion_mnist.json
  - mlp_deep_10_qmnist.json
  - mlp_deep_10_ag_news.json
  - mlp_medium_5_mnist.json
  - mlp_medium_5_fashion_mnist.json
  - mlp_medium_5_qmnist.json
  - mlp_medium_5_ag_news.json
  - cnn_deep_mnist.json
  - cnn_deep_fashion_mnist.json
  - cnn_deep_qmnist.json
  - cnn_medium_mnist.json
  - cnn_medium_fashion_mnist.json
  - cnn_medium_qmnist.json
  - cnn_shallow_mnist.json
  - cnn_shallow_fashion_mnist.json
  - cnn_shallow_qmnist.json
  - transformer_medium_ag_news.json
```

**JSON Structure:**
```json
{
  "Linear_0": {
    "step": [0, 5, 10, ..., 5000],
    "stable_rank": [0.1, 0.12, ..., 50.5],
    "participation_ratio": [...],
    "cumulative_90": [...],
    "nuclear_norm_ratio": [...],
    "loss": [...]
  },
  "Linear_1": { ... },
  "Linear_2": { ... }
}
```

#### Checkpoint & Validation Data
```
/home/user/ndt/experiments/validation_data.pkl                (150 KB)
/home/user/ndt/experiments/new/results/phase1_analysis/
  - alpha_models.pkl                                           (Fitted α models)
/home/user/ndt/experiments/new/results/mechanistic/
  - CLAUDE.md                                                  (Critical assessment)
```

---

### Documentation Files

#### Main Library Documentation
```
/home/user/ndt/README.md                                       (130 lines)
/home/user/ndt/INSTALL.md
/home/user/ndt/CONTRIBUTING.md
/home/user/ndt/REPOSITORY_OVERVIEW.md
```

#### Experiment Documentation
```
/home/user/ndt/experiments/new/README.md                       (200+ lines)
/home/user/ndt/experiments/new/EXPERIMENT_PLAN.md              (200+ lines)
/home/user/ndt/experiments/new/QUICK_START.txt
/home/user/ndt/experiments/new/LOCAL_EXECUTION_GUIDE.md
/home/user/ndt/experiments/new/MODERN_DATASETS_GUIDE.md
/home/user/ndt/experiments/new/READY_TO_RUN.md
/home/user/ndt/experiments/new/EXPERIMENTS_COMPLETED.md
/home/user/ndt/experiments/CLAUDE.md                           (Critical assessment)
```

---

## WHAT EXISTS - DETAILED CHECKLIST

### Phase 2 Infrastructure Available

#### Architecture Support (✅ READY)
- [✅] MLP detection and layer extraction
- [✅] CNN detection with ResNet support
- [✅] Transformer detection and attention tracking
- [✅] ViT detection
- [✅] Auto-architecture detection
- [✅] Layer name generation

#### Activation Capture (✅ READY)
- [✅] Forward hook registration
- [✅] Multi-layer capture simultaneously
- [✅] Auto-reshape (2D/3D/4D support)
- [✅] Activation storage and retrieval
- [✅] Cleanup/context manager

#### Dimensionality Measurement (✅ READY)
- [✅] Stable rank (SVD-based)
- [✅] Participation ratio (SVD-based)
- [✅] Cumulative energy 90 (PCA-like)
- [✅] Nuclear norm ratio (SVD-based)
- [✅] Numerical stability built-in
- [✅] Per-layer, per-step tracking

#### High-Frequency Tracking (✅ READY)
- [✅] Configurable sampling frequency
- [✅] Loss logging
- [✅] Gradient norm logging
- [✅] Force logging at specific steps
- [✅] Results as pandas DataFrame
- [✅] Export to CSV/JSON/HDF5

#### Jump/Phase Detection (✅ PARTIAL)
- [✅] Z-score based jump detection
- [✅] Configurable threshold
- [✅] Multi-metric detection
- [✅] Return Jump objects with metadata
- [❌] Limited to metric jumps (not semantic)

#### Visualization (✅ READY)
- [✅] Single metric plots with loss dual-axis
- [✅] Multi-metric comparison (stacked)
- [✅] Multi-layer phase comparison
- [✅] Jump annotations on plots
- [✅] Correlation heatmaps
- [✅] Interactive Plotly plots
- [✅] HTML export

#### Data Management (✅ READY)
- [✅] CSV export/load (combined or separate)
- [✅] JSON export/load
- [✅] HDF5 export/load with compression
- [✅] Metadata support
- [✅] Multiple result formats

---

## WHAT NEEDS TO BE BUILT FOR PHASE 2

### Priority 1: Critical Mechanistic Tools

#### Feature Visualization (NEW MODULE NEEDED)
Location to create: `/home/user/ndt/src/ndt/analysis/feature_visualization.py`

Features needed:
- Class Activation Maps (CAM) for CNNs
- Gradient-based saliency maps
- Feature importance scoring
- Attention maps for transformers
- Activation distribution plots

#### Activation Geometry Analysis (NEW MODULE NEEDED)
Location to create: `/home/user/ndt/src/ndt/analysis/activation_analysis.py`

Features needed:
- PCA per training phase
- Activation clustering (k-means, DBSCAN)
- Manifold visualization (t-SNE, UMAP wrapper)
- Neuron statistics (dead neurons, sparsity)
- Correlation analysis between layers

### Priority 2: Intervention Experiments (NEW MODULE NEEDED)
Location to create: `/home/user/ndt/src/ndt/core/interventions.py`

Features needed:
- Channel ablation framework
- Neuron-level pruning
- Weight perturbation analysis
- Causal intervention tools

### Priority 3: Batch Analysis Tools (NEW MODULE NEEDED)
Location to create: `/home/user/ndt/analysis/batch_analysis.py`

Features needed:
- Before/after jump comparison utilities
- Batch processing runner
- Result aggregation
- Statistical significance testing

---

## RECOMMENDED PHASE 2 MODULE STRUCTURE

```
/home/user/ndt/src/ndt/analysis/                    (NEW PACKAGE)
├── __init__.py
├── feature_visualization.py                        (CAM, saliency, attention)
├── activation_analysis.py                          (PCA, clustering, statistics)
├── neuron_importance.py                            (Feature scoring)
└── interventions.py                                (Ablation, pruning)

/home/user/ndt/src/ndt/visualization/               (EXTENDED)
├── feature_maps.py                                 (CNN feature visualization)
├── attention_maps.py                               (Transformer attention)
└── manifold.py                                     (t-SNE, UMAP wrappers)

/home/user/ndt/experiments/phase2/                  (NEW EXPERIMENT DIR)
├── __init__.py
├── phase2_main.py                                  (Main orchestrator)
├── phase2_activation_capture.py
├── phase2_analysis.py
├── phase2_visualization.py
└── phase2_interventions.py
```

---

## QUICK REFERENCE: KEY CLASSES & FUNCTIONS

### Essential Classes to Use

1. **`HighFrequencyTracker`** - Start here
   - File: `/home/user/ndt/src/ndt/core/tracker.py`
   - Use for: Tracking dimensionality during training

2. **`ActivationCapture`** - For detailed analysis
   - File: `/home/user/ndt/src/ndt/core/hooks.py`
   - Use for: Capturing activations at specific moments

3. **`JumpDetector`** - Find critical moments
   - File: `/home/user/ndt/src/ndt/core/jump_detector.py`
   - Use for: Identifying phase transitions

4. **Architecture Handlers** - Auto-detection
   - File: `/home/user/ndt/src/ndt/architectures/`
   - Use for: Get layers to monitor from any model

### Essential Functions to Use

1. **Dimensionality metrics**: 
   - `stable_rank()`, `participation_ratio()`, `cumulative_energy_90()`, `nuclear_norm_ratio()`
   - File: `/home/user/ndt/src/ndt/core/estimators.py`

2. **Visualization**:
   - `plot_phases()`, `plot_jumps()`, `plot_metrics_comparison()`
   - File: `/home/user/ndt/src/ndt/visualization/plots.py`

3. **Export**:
   - `export_to_csv()`, `export_to_hdf5()`
   - File: `/home/user/ndt/src/ndt/export/`

---

## GETTING STARTED CHECKLIST FOR PHASE 2

1. **Understand Existing Code**
   - [ ] Read `/home/user/ndt/src/ndt/core/tracker.py` (main orchestrator)
   - [ ] Read `/home/user/ndt/src/ndt/core/hooks.py` (activation capture)
   - [ ] Read `/home/user/ndt/examples/01_quickstart_mnist.py` (example usage)
   - [ ] Read `/home/user/ndt/experiments/new/phase1_calibration.py` (reference)

2. **Set Up Phase 2 Environment**
   - [ ] Create `/home/user/ndt/src/ndt/analysis/` package
   - [ ] Create Phase 2 experiment directory
   - [ ] Install dependencies: scikit-learn, plotly, umap-learn (optional)

3. **Load & Analyze Phase 1 Data**
   - [ ] Load JSON files from `/home/user/ndt/experiments/new/results/phase1_full/`
   - [ ] Use existing jump detector on phase 1 curves
   - [ ] Identify critical moments for detailed analysis

4. **Start Building Phase 2 Modules**
   - [ ] Create `feature_visualization.py` with CAM implementation
   - [ ] Create `activation_analysis.py` with PCA, clustering
   - [ ] Create `neuron_importance.py` with importance scoring

5. **Run First Phase 2 Experiment**
   - [ ] Pick one Phase 1 result (e.g., mlp_deep_10_mnist)
   - [ ] Use Phase2AnalyzerBase template to capture activations
   - [ ] Run PCA, clustering analysis
   - [ ] Generate before/after jump visualizations

---

## ABSOLUTE PATHS - QUICK COPY-PASTE REFERENCE

```bash
# Core library
/home/user/ndt/src/ndt/core/tracker.py
/home/user/ndt/src/ndt/core/hooks.py
/home/user/ndt/src/ndt/core/estimators.py
/home/user/ndt/src/ndt/core/jump_detector.py

# Architectures
/home/user/ndt/src/ndt/architectures/

# Visualization
/home/user/ndt/src/ndt/visualization/plots.py
/home/user/ndt/src/ndt/visualization/interactive.py

# Examples
/home/user/ndt/examples/01_quickstart_mnist.py
/home/user/ndt/examples/02_cnn_cifar10.py

# Phase 1 Results
/home/user/ndt/experiments/new/results/phase1_full/

# Phase 1 Code Reference
/home/user/ndt/experiments/new/phase1_calibration.py
/home/user/ndt/experiments/new/phase1_analysis.py
/home/user/ndt/experiments/new/phase2_prediction.py
```

