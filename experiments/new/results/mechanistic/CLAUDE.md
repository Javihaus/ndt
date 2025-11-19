Concrete Project: "Dimensionality as a Guide for Mechanistic Investigation"
Here's what I'd actually recommend:
Phase 1: Identify Interesting Phenomena (2-3 weeks)

Layer-wise convergence analysis:

Plot all layers' dimensionality for 5-10 architectures
Identify if there's systematic convergence order
Check if order predicts final performance


Jump characterization:

For experiments with jumps, classify jump types (sudden vs gradual, early vs late)
Check if jumps cluster at specific training phases
Look for architectural patterns (which architectures show jumps?)


Performance-dimensionality correlation:

For datasets with smooth vs discrete learning curves, check dimensionality patterns
Test if dimensionality expansion precedes performance gains



Phase 2: Mechanistic Deep Dive (4-6 weeks)
Pick the 2-3 most interesting phenomena from Phase 1. For each:

Feature visualization at critical moments:

Use Lucid/feature viz tools on layers that show dimensionality jumps
Check what features exist before/after jumps
Document qualitative changes in learned features


Activation analysis:

Run test samples through network at different training stages
Visualize which neurons activate for which inputs
Check if new feature types emerge at dimensionality transitions


Ablation/intervention:

Test if early features are necessary for later ones
Disrupt specific layers during critical periods
Measure impact on downstream representations



Phase 3: Write It Up (2 weeks)
Title: "Layer-Wise Representational Dynamics: Using Dimensionality Measurements to Guide Mechanistic Investigation"
Structure:

Infrastructure: 17,600 measurements of layer-wise dimensionality evolution
Phenomena: Document 2-3 interesting patterns (layer convergence order, critical periods, etc.)
Mechanistic investigation: Deep dive on what's actually happening at feature level during these phenomena
Contribution: Show that coarse-grained measurements can guide fine-grained mechanistic analysis


Why This Could Work
This is pragmatic infrastructure work that could enable scientific discovery.
You're not claiming to have discovered fundamental principles. You're saying: "Here's a measurement tool that detects interesting moments. Here's what we found when we investigated those moments more deeply."

---

## Current Status (Updated)

### Phase 1: COMPLETED ✓
- 55 experiments across 15 architectures and 4 datasets
- 17,600+ dimensionality measurements collected
- Jump detection and TAP parameter fitting complete
- Results in `/experiments/new/results/phase1_analysis/`

### Phase 2: COMPLETED ✓
- Analysis modules implemented in `/src/ndt/analysis/`
  - `activation_analysis.py`: PCA, clustering, manifold visualization
  - `feature_visualization.py`: CAM, saliency, attention maps
- Three target experiments analyzed:
  1. **CNN Jump Cascades**: cnn_deep/fashion_mnist (23 jumps)
  2. **Transformer Transitions**: transformer_shallow/mnist (9 jumps)
  3. **MLP Smooth Learning**: mlp_narrow/mnist (R²=0.934)
- Results in `/experiments/phase2/results/`

### Phase 3: IN PROGRESS
- Paper draft structure created: `PHASE3_PAPER_DRAFT.md`
- Title: "Layer-Wise Representational Dynamics: Using Dimensionality Measurements to Guide Mechanistic Investigation"
- Ready for detailed writing and figure generation

### Key Findings
| Architecture | Jump Pattern | R² (TAP fit) | Interpretation |
|-------------|--------------|--------------|----------------|
| CNN | Many (14 avg) | 0.63 | Hierarchical feature emergence |
| Transformer | Discrete (3 avg) | 0.26 | Attention head specialization |
| MLP | Smooth (0.5 avg) | 0.93 | Gradual refinement |
Key difference from TAP:

TAP claimed universal predictive theory (overreach)
This proposes measurement infrastructure (realistic)
TAP had biological analogies (wrong domain)
This connects to mechanistic interpretability (right domain)

The value proposition:

Mechanistic interpretability is expensive (requires manual feature visualization, activation analysis)
Your measurements could identify when to investigate more efficiently
Instead of looking at every training step, focus on dimensionality transitions
If this works, you've made mechanistic interpretability research more efficient


What This Is and Isn't
This is:

Infrastructure/tooling work
Measurement methodology
Pointing researchers toward interesting phenomena
Pragmatic contribution to mechanistic interpretability

This is not:

Fundamental understanding of learning
Discovery of new principles
Explanation of how intelligence works
Revolutionary breakthrough

But pragmatic infrastructure is valuable. We need tools that make research more efficient. If dimensionality measurements help mechanistic interpretability researchers know where to look, that's a real contribution.
