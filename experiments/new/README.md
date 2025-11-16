# TAP Dynamics Experiments

**Complete experimental framework for testing Transition-Adapted Plasticity (TAP) in neural network training**

This directory contains a comprehensive research program to move beyond the initial MNIST + MLP toy experiment and establish whether the TAP framework has genuine predictive power.

## The Brutal Truth (From CLAUDE.md)

The initial work had:
- ✓ High-temporal resolution measurement (every 5 steps)
- ✓ Multi-estimator validation (4 dimensionality metrics)
- ✓ Interesting empirical finding (85:1 activation-to-weight jump ratio)
- ✗ **Single toy experiment** (MNIST + MLP only)
- ✗ **TAP framework is descriptive, not predictive**

## The Real Opportunity

**Can we predict training dynamics from architecture BEFORE training?**

If yes, this has genuine scientific and practical value:
1. **Architecture search**: Detect poor designs before expensive training
2. **Early stopping**: Flag premature saturation
3. **Interpretability**: Understand when capabilities will emerge

## Experimental Design

### Phase 1: Establish the Phenomenon (Proper Scale)

**Goal:** Measure α_empirical for each architecture and find α = f(depth, width, connectivity)

**Specifications:**
- **10+ architectures**: MLPs (varying depth/width), CNNs, ResNets, Transformers
- **5+ datasets**: MNIST, CIFAR-10, Fashion-MNIST, SVHN, CIFAR-100
- **Measurement**: Every 5 steps for 5000+ steps
- **Output**: α parameters for each architecture

**Key equation to fit:**
```
D(t+1) = D(t) + α_arch · ||∇L||_t · D(t) · (1 - D(t)/D_max)
```

### Phase 2: Prediction Experiments

**Goal:** Test if we can predict D(t) from architecture parameters BEFORE training

**Success criterion:** R² > 0.8 between predicted and actual curves

**Process:**
1. Given: New architecture (not in Phase 1 training set)
2. Estimate: α from architectural parameters using fitted model
3. Predict: Dimensionality curve D(t) before training
4. Train: Network and measure actual D(t)
5. Compare: Predicted vs actual (R² metric)

### Phase 3: Real-Time Monitoring Tool

**Goal:** Build practical tool that practitioners can use during training

**Features:**
- Estimates D(t) during training
- Predicts D(t+k) for k=50,100,200 steps ahead
- Flags if expansion is insufficient (α too small)
- Flags if jumps stop occurring (premature saturation)
- Provides actionable recommendations

**This is useful to practitioners** - helps diagnose training issues early.

### Phase 4: Connection to Emergent Capabilities

**Goal:** Test if dimensionality jumps correlate with capability emergence

**Hypothesis:** Dimensionality jumps PRECEDE performance jumps

**Measurement:**
- Track BOTH dimensionality AND task performance every 5 steps
- Detect jumps in both signals
- Compute temporal correlation
- Test: Does ΔD(t) predict Δ(accuracy)(t+k)?

**If successful:** You have an early warning system for capability emergence

## File Structure

```
experiments/new/
├── README.md                           # This file
├── CLAUDE.md                           # Original critical assessment
│
├── phase1_calibration.py               # Phase 1: Multi-architecture calibration
├── phase1_analysis.py                  # Extract α = f(architecture)
│
├── phase2_prediction.py                # Phase 2: Test predictive power
│
├── phase3_realtime_monitor.py          # Phase 3: Practical monitoring tool
│
├── phase4_capability_emergence.py      # Phase 4: Capability correlation
│
├── run_all_experiments.py              # Master runner for all phases
│
└── results/                            # Experimental results
    ├── phase1/                         # Raw calibration data
    ├── phase1_analysis/                # α extraction and models
    ├── phase2/                         # Prediction validation
    ├── phase3/                         # Monitoring demonstrations
    └── phase4/                         # Capability emergence findings
```

## Quick Start

### Option 1: Quick Test (Recommended First)

Run a quick test to ensure everything works:

```bash
python run_all_experiments.py --all --quick-test
```

This runs all 4 phases with:
- Reduced architectures (2 instead of 17)
- Reduced datasets (2 instead of 5)
- Shorter training (500 steps instead of 5000)
- **Runtime:** ~30 minutes on GPU, ~2 hours on CPU

### Option 2: Full Experiments

Run the complete experimental program:

```bash
python run_all_experiments.py --all
```

This runs all 4 phases with:
- 17 architectures across 5 datasets (Phase 1)
- 10+ prediction tests (Phase 2)
- Real-time monitoring demonstration (Phase 3)
- 4 architectures × 2 datasets capability tracking (Phase 4)
- **Runtime:** ~8-12 hours on GPU, ~48 hours on CPU

### Option 3: Individual Phases

Run specific phases:

```bash
# Phase 1 only
python run_all_experiments.py --phase 1

# Phase 2 only (requires Phase 1 first)
python run_all_experiments.py --phase 2

# Phase 3 only (requires Phase 1 first)
python run_all_experiments.py --phase 3

# Phase 4 only
python run_all_experiments.py --phase 4
```

## Detailed Usage

### Phase 1: Calibration

```bash
python phase1_calibration.py \
    --output-dir ./results/phase1 \
    --num-steps 5000 \
    --measurement-interval 5
```

Then analyze:

```bash
python phase1_analysis.py \
    --results-dir ./results/phase1 \
    --output-dir ./results/phase1_analysis
```

**Output:**
- `alpha_summary.csv`: Architecture → α mapping
- `alpha_models.pkl`: Fitted predictive models
- `alpha_relationships.png`: Visualizations
- `phase1_report.md`: Analysis summary

### Phase 2: Prediction

```bash
python phase2_prediction.py \
    --predictor ./results/phase1_analysis/alpha_models.pkl \
    --output-dir ./results/phase2 \
    --num-steps 5000
```

**Output:**
- Individual prediction results (JSON)
- Prediction vs actual plots (PNG)
- `phase2_summary.json`: Aggregate results

### Phase 3: Real-Time Monitoring

```bash
python phase3_realtime_monitor.py \
    --predictor ./results/phase1_analysis/alpha_models.pkl \
    --arch mlp_medium_5 \
    --num-steps 2000 \
    --output-dir ./results/phase3
```

**Output:**
- `monitor_state.json`: Full monitoring state
- `monitor_visualization.png`: Monitoring plots
- Real-time warnings and recommendations (stdout)

### Phase 4: Capability Emergence

```bash
python phase4_capability_emergence.py \
    --output-dir ./results/phase4 \
    --num-steps 3000 \
    --measurement-interval 5
```

**Output:**
- Individual experiment results (JSON)
- Capability emergence visualizations (PNG)
- `phase4_summary.md`: Correlation analysis

## Architecture Catalog

The experiments include the following architectures:

**Group 1: Depth Variation (MLPs)**
- `mlp_shallow_2`: 2 hidden layers
- `mlp_medium_5`: 5 hidden layers
- `mlp_deep_10`: 10 hidden layers
- `mlp_verydeep_15`: 15 hidden layers

**Group 2: Width Variation (MLPs)**
- `mlp_narrow`: 4 layers × 32 units
- `mlp_medium`: 4 layers × 128 units
- `mlp_wide`: 4 layers × 256 units
- `mlp_verywide`: 4 layers × 512 units

**Group 3: CNNs**
- `cnn_shallow`: 2 conv layers
- `cnn_medium`: 3 conv layers
- `cnn_deep`: 4 conv layers

**Group 4: ResNets**
- `resnet18`: ResNet-18 (adapted for 32×32 images)

**Group 5: Transformers**
- `transformer_shallow`: 2 layers
- `transformer_medium`: 4 layers
- `transformer_deep`: 6 layers
- `transformer_narrow`: 64-dim embeddings
- `transformer_wide`: 256-dim embeddings

## Datasets

1. **MNIST**: 28×28 grayscale handwritten digits
2. **Fashion-MNIST**: 28×28 grayscale clothing items
3. **CIFAR-10**: 32×32 RGB natural images (10 classes)
4. **SVHN**: 32×32 RGB street view house numbers
5. **CIFAR-100**: 32×32 RGB natural images (100 classes)

## Expected Results

### Phase 1: Calibration

**Expected findings:**
- α correlates with depth and width
- Best model: Linear or log-linear (R² > 0.7)
- Deeper/wider networks → larger α
- 3-5 discrete jumps per training run

### Phase 2: Prediction

**Success criteria:**
- R² > 0.8 for refined predictions (with gradient history)
- R² > 0.5 for simple predictions (constant gradient)
- 70%+ of experiments meet success criterion

### Phase 3: Monitoring

**Demonstration:**
- Real-time dimensionality tracking
- Accurate k-step predictions
- Correct warning flags (vanishing gradients, saturation)
- Actionable recommendations

### Phase 4: Capability Emergence

**Key test:**
- Do dimensionality jumps precede accuracy jumps?
- Expected: Mean lag of 20-100 steps (dimensionality leads)
- Significant correlation (p < 0.05) in predictive power tests

## Troubleshooting

### Out of Memory

Reduce batch size in loader functions:
```python
train_loader, val_loader, ... = get_mnist_loaders(batch_size=32)  # Default is 64
```

Or reduce `max_batches` in measurement functions.

### Slow Training

Use GPU if available:
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

Or run quick test first:
```bash
python run_all_experiments.py --all --quick-test
```

### Phase 2 Fails (No Predictor Found)

Ensure Phase 1 completed successfully:
```bash
ls -l results/phase1_analysis/alpha_models.pkl
```

If missing, run Phase 1 first.

### Import Errors

Ensure you're in the correct directory:
```bash
cd experiments/new
python run_all_experiments.py --all
```

## Dependencies

All dependencies are already included in the main NDT package:

- torch >= 1.12.0
- numpy >= 1.21.0
- pandas >= 1.3.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- scipy >= 1.7.0
- scikit-learn >= 1.0.0
- tqdm >= 4.62.0

## Interpretation Guide

### What Makes a Successful Result?

**Phase 1:**
- ✓ R² > 0.7 for α = f(architecture) model
- ✓ Consistent α across datasets for same architecture
- ✓ Clear relationship between architecture parameters and α

**Phase 2:**
- ✓ R² > 0.8 for refined predictions
- ✓ 70%+ success rate across experiments
- ✓ Predictions work on unseen architectures

**Phase 3:**
- ✓ Accurate real-time tracking
- ✓ Warnings trigger correctly
- ✓ Predictions have MAE < 10% of D_max

**Phase 4:**
- ✓ Positive temporal correlation (dimensionality leads)
- ✓ Significant predictive power (p < 0.05)
- ✓ Clear phase transitions

### What If Results Don't Meet Criteria?

**If Phase 1 R² < 0.7:**
- The α = f(architecture) relationship may be more complex
- Try adding interaction terms or nonlinear models
- May need more architectural diversity

**If Phase 2 predictions fail:**
- The TAP model may be too simple
- Need to incorporate more training dynamics (LR schedule, optimizer state)
- α may not be constant throughout training

**If Phase 4 shows no correlation:**
- Dimensionality may not be the right metric for capability
- Jumps may not be causally related to performance
- Need finer-grained capability measurements

## Citation

If you use these experiments in your research, please cite:

```bibtex
@software{marin2024tap,
  author = {Marín, Javier},
  title = {Transition-Adapted Plasticity: Predicting Neural Network Training Dynamics from Architecture},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/Javihaus/ndt},
}
```

## Contributing

This experimental framework is designed to be extensible:

1. **Add new architectures**: Edit `create_architecture()` in `phase1_calibration.py`
2. **Add new datasets**: Add loader function to `DATASET_LOADERS` dict
3. **Add new metrics**: Extend `DimensionalityEstimator` class
4. **Add new analyses**: Create new analysis scripts in each phase

## Support

For questions or issues:
1. Check troubleshooting section above
2. Review CLAUDE.md for context on experimental design
3. Open an issue on GitHub

## Acknowledgments

This experimental design is based on the critical assessment in CLAUDE.md, which identified the gap between interesting empirical observations and genuine predictive power. The goal is to move from descriptive to predictive science.

**Key insight:** The novelty isn't the TAP analogy—it's whether we can predict training dynamics from architecture before training begins. These experiments test that hypothesis rigorously.
