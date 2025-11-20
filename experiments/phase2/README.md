# Phase 2: Mechanistic Interpretability Analysis

This directory contains the implementation and results of Phase 2 mechanistic deep dive analysis.

## Project Overview

**Goal**: Use dimensionality transitions detected in Phase 1 to guide mechanistic interpretability analysis, demonstrating that coarse-grained measurements can efficiently direct fine-grained investigation.

**Paper Title**: "Layer-Wise Representational Dynamics: Using Dimensionality Measurements to Guide Mechanistic Investigation"

---

## Directory Structure

```
phase2/
├── README.md                                          # This file
├── MECHANISTIC_INTERPRETABILITY_DETAILED_PLAN.md     # 50+ page detailed plan (8-12 weeks)
├── QUICK_START_GUIDE.md                              # Fast path to get started (3 weeks)
├── phase2_main.py                                     # Full analysis orchestrator (requires models)
├── phase2_analysis_simple.py                         # Simplified analysis (Phase 1 data only)
└── results/
    ├── PHASE2_SUMMARY.md                             # Executive summary of findings
    ├── PHASE3_PAPER_DRAFT.md                         # Paper structure and draft
    ├── report_*.md                                   # Individual phenomenon reports (3)
    └── analysis_*.json                               # Quantitative results (3)
```

---

## Three Target Phenomena

### 1. CNN Jump Cascades
**Experiment**: cnn_deep/fashion_mnist
**Pattern**: 23 dimensionality jumps, scattered distribution
**Hypothesis**: Each jump = emergence of new feature type (edges → textures → parts → objects)
**Analysis**: Feature visualization, neuron selectivity, hierarchical composition

### 2. Transformer Discrete Transitions
**Experiment**: transformer_shallow/mnist
**Pattern**: 9 jumps, early cascade
**Hypothesis**: Jumps = attention head specialization events
**Analysis**: Attention pattern evolution, head clustering, importance probing

### 3. MLP Smooth Learning
**Experiment**: mlp_narrow/mnist
**Pattern**: 0 jumps, R²=0.934 (excellent TAP fit)
**Hypothesis**: Smooth subspace refinement without discrete transitions
**Analysis**: Subspace evolution, gradient flow, feature disentanglement

---

## Quick Start

### Option 1: Phase 1 Data Analysis Only (No Re-training)

```bash
# Run simplified analysis on existing Phase 1 results
python phase2_analysis_simple.py

# Results saved to: results/
# - PHASE2_SUMMARY.md: Executive summary
# - report_*.md: Individual reports for each phenomenon
# - analysis_*.json: Quantitative metrics
```

**Time**: 5 minutes
**Requirements**: Python 3, no external dependencies

---

### Option 2: Full Mechanistic Analysis (Requires Models)

```bash
# 1. Install dependencies
pip install torch torchvision numpy scikit-learn matplotlib

# 2. Run full analysis with model checkpoints
python phase2_main.py

# 3. Results include:
# - Activation analysis (PCA, clustering)
# - Feature visualizations (when models provided)
# - Detailed mechanistic reports
```

**Time**: Variable (depends on checkpoints available)
**Requirements**: PyTorch, saved model checkpoints

---

### Option 3: Implement Detailed Plan (8-12 weeks)

See `MECHANISTIC_INTERPRETABILITY_DETAILED_PLAN.md` for:
- Week-by-week protocols
- Specific experimental procedures
- Code examples for each analysis
- Expected outputs and validation

---

## Current Status

### ✅ Completed

- [x] Phase 1 analysis (55 experiments, 17,600+ measurements)
- [x] Three target phenomena identified
- [x] Phase 2 analysis modules implemented (`/src/ndt/analysis/`)
- [x] Initial analysis on Phase 1 data
- [x] Phase 3 paper draft structure
- [x] Detailed mechanistic plan (50+ pages)
- [x] Quick start guide

### 🔄 In Progress

- [ ] Re-train experiments with checkpoint saving
- [ ] Feature visualization at critical moments
- [ ] Activation analysis before/after jumps
- [ ] Ablation studies

### 📋 Planned

- [ ] Full mechanistic deep dive (8-12 weeks)
- [ ] Paper writing and figure generation
- [ ] Code release and documentation
- [ ] Submission to ICML/NeurIPS/ICLR

---

## Key Results (Phase 2 Initial Analysis)

| Phenomenon | Jumps | Pattern | R² | Interpretation |
|------------|-------|---------|-----|----------------|
| **CNN Cascades** | 23 | scattered | 0.632 | Hierarchical feature emergence |
| **Transformer Transitions** | 9 | early_cascade | 0.261 | Attention head specialization |
| **MLP Smooth** | 0 | smooth | 0.934 | Gradual TAP-following refinement |

**Key Insight**: Architecture-specific learning patterns require architecture-specific interpretability approaches.

---

## Analysis Modules

Located in `/src/ndt/analysis/`:

### 1. ActivationAnalyzer (`activation_analysis.py`)
**Capabilities**:
- PCA analysis (principal components, explained variance)
- Clustering (k-means, DBSCAN)
- Manifold embedding (t-SNE, UMAP)
- Singular value analysis
- Neuron importance scoring
- Before/after comparisons

**Example**:
```python
from ndt.analysis import ActivationAnalyzer

analyzer = ActivationAnalyzer()

# PCA analysis
pca_results = analyzer.pca_analysis(activations, n_components=50)
print(f"90% variance: {pca_results['n_components_90']} components")

# Clustering
clusters = analyzer.cluster_analysis(activations, method='kmeans', n_clusters=5)

# Comparison
comparison = analyzer.compare_activations(acts_before, acts_after)
print(f"Subspace overlap: {comparison['subspace_overlap']:.3f}")
```

---

### 2. FeatureVisualizer (`feature_visualization.py`)
**Capabilities**:
- Gradient-based Class Activation Mapping (Grad-CAM)
- Saliency maps
- Integrated gradients
- Attention visualization
- Feature maps extraction
- Neuron activation maximization
- Layer conductance

**Example**:
```python
from ndt.analysis import FeatureVisualizer

visualizer = FeatureVisualizer(model)

# Generate Grad-CAM
cam = visualizer.grad_cam(img, target_class=5, target_layer=conv_layer)

# Saliency map
saliency = visualizer.saliency_map(img, target_class=5)

# Integrated gradients
ig = visualizer.integrated_gradients(img, target_class=5, steps=50)
```

---

## Documentation

### For Quick Start
📖 **QUICK_START_GUIDE.md**: 3-week minimal viable analysis
- Day-by-day protocols
- Code templates
- Expected outputs

### For Deep Dive
📚 **MECHANISTIC_INTERPRETABILITY_DETAILED_PLAN.md**: 8-12 week comprehensive plan
- Detailed experimental protocols
- Statistical validation
- Resource requirements
- Success criteria

### For Context
📄 **PHASE2_SUMMARY.md**: Executive summary of current findings
📄 **PHASE3_PAPER_DRAFT.md**: Paper structure and content outline

---

## How to Contribute

### Adding New Analyses

1. Create analysis script in this directory
2. Use existing modules from `/src/ndt/analysis/`
3. Save results to `results/`
4. Update this README

### Adding New Phenomena

1. Identify interesting experiment from Phase 1
2. Characterize jump pattern or dimensionality dynamics
3. Formulate mechanistic hypothesis
4. Add to target list with analysis plan

---

## Resources

### Compute Requirements

**Minimal (Phase 1 data only)**:
- CPU only
- 8GB RAM
- 1GB storage

**Full (with re-training)**:
- 1-2 GPUs
- 32GB RAM
- 50GB storage
- ~800 GPU-hours total

### Timeline Estimates

- **Quick analysis**: 3 weeks (one phenomenon)
- **Full Phase 2**: 8-12 weeks (all phenomena)
- **Paper submission**: 4-6 months (including writing)

---

## Citation

If you use this work, please cite:

```bibtex
@article{phase2-mechanistic-2025,
  title={Layer-Wise Representational Dynamics: Using Dimensionality
         Measurements to Guide Mechanistic Investigation},
  author={[Authors]},
  journal={[Venue]},
  year={2025}
}
```

---

## Questions?

- **Technical**: See `/PHASE2_INFRASTRUCTURE.md` in project root
- **Code examples**: See `/PHASE2_CODE_EXAMPLES.md` in project root
- **Planning**: See `MECHANISTIC_INTERPRETABILITY_DETAILED_PLAN.md`
- **Quick start**: See `QUICK_START_GUIDE.md`

---

## License

[To be determined]

---

**Last Updated**: 2025-11-20
**Version**: 1.0
**Status**: Analysis infrastructure complete, ready for deep dive
