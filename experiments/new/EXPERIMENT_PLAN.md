## TAP Dynamics Research Program - Implementation Summary

**Created:** 2024-11-16
**Based on:** CLAUDE.md critical assessment

### What Was Implemented

This directory contains a complete 4-phase research program to test whether the Transition-Adapted Plasticity (TAP) framework has genuine predictive power.

### The Core Scientific Question

**Can we predict neural network training dynamics from architecture BEFORE training?**

If yes → Practical value for architecture search, early stopping, interpretability
If no → TAP is just a descriptive model, not a predictive theory

---

## File Inventory

### Core Experimental Scripts (160+ KB of code)

1. **phase1_calibration.py** (29 KB)
   - Train 17 architectures × 5 datasets
   - Measure dimensionality every 5 steps
   - Extract architectural parameters
   - **Output:** Raw training curves for all experiments

2. **phase1_analysis.py** (19 KB)
   - Fit TAP model: D(t+1) = D(t) + α·||∇L||·D(t)·(1-D/D_max)
   - Extract α for each architecture
   - Fit α = f(depth, width, connectivity)
   - **Output:** Predictive models for α

3. **phase2_prediction.py** (20 KB)
   - Test predictive power on unseen architectures
   - Predict D(t) before training
   - Compare predicted vs actual (R² metric)
   - **Success criterion:** R² > 0.8

4. **phase3_realtime_monitor.py** (22 KB)
   - Real-time training monitoring tool
   - Predict D(t+k) during training
   - Flag issues: vanishing gradients, premature saturation
   - **Output:** Actionable recommendations for practitioners

5. **phase4_capability_emergence.py** (27 KB)
   - Measure dimensionality AND task performance every 5 steps
   - Detect jumps in both signals
   - Test: Do dimensionality jumps precede performance jumps?
   - **Key test:** Temporal correlation analysis

6. **run_all_experiments.py** (16 KB)
   - Master orchestrator for all 4 phases
   - Handles dependencies between phases
   - Generates final summary reports
   - **Usage:** `python run_all_experiments.py --all`

7. **visualization_utils.py** (16 KB)
   - Common plotting functions
   - Consistent style across all phases
   - Dashboard creation utilities
   - **Reusable:** Import in any analysis script

### Documentation

8. **README.md** (12 KB)
   - Complete usage guide
   - Expected results and interpretation
   - Troubleshooting section
   - Architecture and dataset catalogs

9. **CLAUDE.md** (4 KB)
   - Original critical assessment
   - Identifies gap between descriptive and predictive
   - Outlines the 4-phase research program

10. **EXPERIMENT_PLAN.md** (this file)
    - Implementation summary
    - Quick reference guide

---

## Architecture Catalog (17 Total)

### MLPs (8 architectures)
- Depth variation: 2, 5, 10, 15 hidden layers
- Width variation: 32, 128, 256, 512 units per layer

### CNNs (3 architectures)
- Shallow: 2 conv layers
- Medium: 3 conv layers
- Deep: 4 conv layers

### ResNets (1 architecture)
- ResNet-18 (adapted for 32×32 images)

### Transformers (5 architectures)
- Depth variation: 2, 4, 6 encoder layers
- Width variation: 64, 128, 256 embedding dimensions

---

## Datasets (5 Total)

1. **MNIST**: 60K handwritten digits (28×28 grayscale)
2. **Fashion-MNIST**: 60K clothing items (28×28 grayscale)
3. **CIFAR-10**: 60K natural images (32×32 RGB, 10 classes)
4. **SVHN**: 73K street numbers (32×32 RGB)
5. **CIFAR-100**: 60K natural images (32×32 RGB, 100 classes)

---

## Expected Computational Requirements

### Quick Test (--quick-test)
- 2 architectures × 2 datasets
- 500 training steps each
- **Time:** ~30 min GPU, ~2 hours CPU
- **Disk:** ~100 MB results

### Full Experiments (--all)
- Phase 1: 17 arch × 5 datasets = 85 experiments
- Phase 2: 10+ prediction tests
- Phase 3: 1 monitoring demonstration
- Phase 4: 4 arch × 2 datasets = 8 experiments
- **Total experiments:** ~104
- **Time:** 8-12 hours GPU, ~48 hours CPU
- **Disk:** ~2-5 GB results

---

## Experimental Flow

```
Phase 1: Calibration
    ↓
    Train all architectures on all datasets
    Measure D(t) every 5 steps
    ↓
Phase 1 Analysis
    ↓
    Fit TAP model to each curve
    Extract α for each architecture
    Fit α = f(depth, width, connectivity)
    Save predictive models
    ↓
Phase 2: Prediction
    ↓
    Load α models
    For new architecture:
        - Estimate α from parameters
        - Predict D(t) curve
        - Train and measure actual D(t)
        - Compute R² (success if > 0.8)
    ↓
Phase 3: Monitoring
    ↓
    Demonstrate real-time tool
    Show predictions during training
    Test warning system
    ↓
Phase 4: Capability Emergence
    ↓
    Track dimensionality + accuracy every 5 steps
    Detect jumps in both signals
    Compute temporal correlation
    Test: Does ΔD(t) predict Δ(acc)(t+k)?
```

---

## Quick Start Commands

```bash
# Navigate to experiments directory
cd experiments/new

# Quick test (30 min)
python run_all_experiments.py --all --quick-test

# Full experiments (8-12 hours)
python run_all_experiments.py --all

# Individual phases
python run_all_experiments.py --phase 1
python run_all_experiments.py --phase 2
python run_all_experiments.py --phase 3
python run_all_experiments.py --phase 4

# Custom configuration
python run_all_experiments.py --all --num-steps 3000 --output-dir ./my_results
```

---

## Success Criteria

### Phase 1: Calibration
✓ R² > 0.7 for α = f(architecture) model
✓ Consistent α across datasets
✓ Clear depth/width relationship

### Phase 2: Prediction
✓ R² > 0.8 for refined predictions
✓ 70%+ success rate across experiments
✓ Works on unseen architectures

### Phase 3: Monitoring
✓ Accurate real-time tracking
✓ Warnings trigger correctly
✓ MAE < 10% of D_max for predictions

### Phase 4: Capability Emergence
✓ Positive temporal correlation (dim leads)
✓ Significant predictive power (p < 0.05)
✓ Clear phase transitions

---

## Interpreting Results

### If Successful (Criteria Met)

**Scientific Impact:**
- TAP framework has genuine predictive power
- α is a meaningful architectural parameter
- Dimensionality expansion predicts capability emergence

**Practical Applications:**
- Architecture search: Estimate α before training
- Early stopping: Detect saturation early
- Interpretability: Predict when capabilities will emerge

**Publication Angle:**
- "From Descriptive to Predictive: TAP Dynamics in Neural Networks"
- Emphasis on R² > 0.8 prediction accuracy
- Novel connection to capability emergence

### If Unsuccessful (Criteria Not Met)

**Phase 1 fails (R² < 0.7):**
- α relationship more complex than linear
- Need nonlinear models or interaction terms
- May require optimizer-specific parameters

**Phase 2 fails (R² < 0.8):**
- TAP model too simple
- Need to incorporate LR schedules, optimizer state
- α may vary during training

**Phase 4 fails (no correlation):**
- Dimensionality may not drive capability
- Need finer-grained metrics
- Correlation ≠ causation

**Honest conclusion:**
- TAP provides useful framework for thinking
- But limited predictive power
- Dimensionality jumps are interesting but not predictive

---

## Next Steps After Experiments

1. **Review Results**
   - Check `results/EXPERIMENTS_SUMMARY.md`
   - Examine visualizations in each phase directory
   - Validate against success criteria

2. **Write Up Findings**
   - Phase 1: α = f(architecture) relationship
   - Phase 2: Prediction accuracy statistics
   - Phase 4: Capability emergence correlation

3. **Iterate if Needed**
   - Add more architectures (Transformers, ResNets)
   - Test on different tasks (NLP, time-series)
   - Refine TAP model (add nonlinear terms)

4. **Publication Decision**
   - If successful: Full paper with 4 phases
   - If partial: Focus on successful phases
   - If unsuccessful: Honest negative result (still valuable!)

---

## Files Created

```
experiments/new/
├── CLAUDE.md                          (Original critique)
├── README.md                          (Usage guide)
├── EXPERIMENT_PLAN.md                 (This file)
│
├── phase1_calibration.py              (29 KB)
├── phase1_analysis.py                 (19 KB)
├── phase2_prediction.py               (20 KB)
├── phase3_realtime_monitor.py         (22 KB)
├── phase4_capability_emergence.py     (27 KB)
├── run_all_experiments.py             (16 KB)
├── visualization_utils.py             (16 KB)
│
├── tap_experiments_main.py            (Legacy - kept for reference)
│
└── results/                           (Created during experiments)
    ├── phase1/
    ├── phase1_analysis/
    ├── phase2/
    ├── phase3/
    └── phase4/
```

**Total:** ~160 KB of new experimental code + comprehensive documentation

---

## Key Innovations

1. **Proper Scale**: 17 architectures × 5 datasets (vs original 1 × 1)
2. **Predictive Test**: Phase 2 tests if we can predict curves beforehand
3. **Practical Tool**: Phase 3 provides real value to practitioners
4. **Novel Connection**: Phase 4 tests dimensionality → capability link
5. **Reproducible**: Complete scripts, clear success criteria, honest interpretation

---

## Citation

If these experiments prove successful:

```bibtex
@article{marin2024tap,
  title={Transition-Adapted Plasticity: Predicting Neural Network Training Dynamics from Architecture},
  author={Mar{\\'i}n, Javier},
  journal={arXiv preprint},
  year={2024}
}
```

---

## Contact

For questions about these experiments:
- Review README.md for usage questions
- Check CLAUDE.md for design rationale
- Open GitHub issue for bugs/improvements

**Remember:** The goal is not to confirm the hypothesis, but to TEST it rigorously. Negative results are valuable if honestly reported.
