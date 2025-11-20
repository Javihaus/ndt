# Layer-Wise Representational Dynamics: Using Dimensionality Measurements to Guide Mechanistic Investigation

**Draft Paper Structure**

---

## Abstract

We present a systematic framework for identifying critical moments in neural network training through dimensionality tracking. Using 17,600+ measurements across 55 experiments spanning 15 architectures and 4 datasets, we demonstrate that coarse-grained dimensionality metrics can effectively guide fine-grained mechanistic analysis. Our key findings reveal architecture-specific learning patterns: CNNs exhibit cascading dimensionality jumps suggestive of hierarchical feature emergence, transformers show discrete attention transitions, and MLPs demonstrate smooth representational growth. We propose dimensionality tracking as practical infrastructure for making mechanistic interpretability research more efficient.

---

## 1. Introduction

### 1.1 Motivation
- Mechanistic interpretability is expensive (requires manual feature visualization, activation analysis)
- Need tools that identify *when* to investigate, not just *what* to investigate
- Dimensionality metrics provide coarse-grained signals of representation changes

### 1.2 Contribution
This work provides:
1. **Infrastructure**: Production-ready tools for high-frequency dimensionality tracking
2. **Phenomena**: Documentation of three distinct learning patterns across architectures
3. **Methodology**: Framework for using dimensionality transitions to guide mechanistic analysis
4. **Value proposition**: Making mechanistic interpretability research more efficient

### 1.3 What This Is (and Isn't)
- **Is**: Measurement methodology, tooling infrastructure, pointing researchers to interesting moments
- **Is not**: Fundamental understanding of learning, discovery of new principles, revolutionary theory

---

## 2. Related Work

### 2.1 Dimensionality in Neural Networks
- Intrinsic dimensionality of representations
- Effective rank and stable rank metrics
- Neural collapse and representation geometry

### 2.2 Mechanistic Interpretability
- Feature visualization (Olah et al.)
- Circuits and feature interactions
- Probing classifiers

### 2.3 Training Dynamics
- Loss landscape analysis
- Phase transitions in learning
- Critical learning periods

---

## 3. Methods

### 3.1 Dimensionality Metrics
We track four complementary metrics:
- **Stable Rank**: Ratio of squared Frobenius to spectral norm
- **Participation Ratio**: Effective number of dimensions
- **Cumulative Energy 90**: Components for 90% variance
- **Nuclear Norm Ratio**: Normalized sum of singular values

### 3.2 Jump Detection
- Z-score based transition detection
- Threshold calibration across architectures
- Temporal clustering of transitions

### 3.3 Experimental Setup
- **Architectures**: MLPs (8 variants), CNNs (3 variants), Transformers (4 variants)
- **Datasets**: MNIST, Fashion-MNIST, QMNIST, AG News
- **Training**: 2000 epochs, measurements every 5 steps
- **Total measurements**: 17,600+ data points

---

## 4. Results

### 4.1 Phenomenon 1: CNN Jump Cascades

**Key Finding**: CNNs exhibit high-frequency dimensionality jumps (mean=14 per experiment, 100% show jumps)

**Representative Case**: cnn_deep/fashion_mnist
- 23 dimensionality transitions detected
- Concentrated in early training (indices 5-9)
- Maximum jump magnitude: 0.039 (z-score: 4.36)
- R² = 0.632 (moderate TAP fit due to jumps)

**Interpretation**:
- Jumps likely correspond to hierarchical feature emergence
- Early cascade pattern suggests rapid low-level feature learning
- Scattered later jumps indicate compositional feature formation

**Mechanistic Hypothesis**: Each major jump corresponds to emergence of a new feature type in the CNN hierarchy (edges → textures → parts → objects)

### 4.2 Phenomenon 2: Transformer Discrete Transitions

**Key Finding**: Transformers show moderate, discrete jumps (mean=3.1, 57% show jumps)

**Representative Case**: transformer_shallow/mnist
- 9 dimensionality transitions detected
- Early cascade pattern (concentrated in initial training)
- Maximum magnitude: varies
- R² = 0.261 (poor TAP fit - discrete dynamics)

**Interpretation**:
- Transformers learn through discrete attention pattern formation
- Jumps may correspond to attention head specialization
- Poor TAP fit suggests fundamentally different dynamics than gradual growth

**Mechanistic Hypothesis**: Jumps represent moments when attention heads specialize for different aspects of input

### 4.3 Phenomenon 3: MLP Smooth Learning

**Key Finding**: MLPs show smooth, predictable growth (mean=0.5 jumps, few show transitions)

**Representative Case**: mlp_narrow/mnist
- 0 dimensionality transitions detected
- R² = 0.934 (excellent TAP fit)
- α = 0.00141 (fast growth rate)

**Interpretation**:
- MLPs rely on distributed representations
- Gradual refinement rather than discrete feature emergence
- TAP model accurately describes growth dynamics

**Mechanistic Hypothesis**: MLPs incrementally refine features without sharp phase transitions

---

## 5. Discussion

### 5.1 Architecture-Specific Investigation Strategies

| Architecture | Jump Pattern | Recommended Analysis |
|-------------|--------------|---------------------|
| CNN | Many jumps, early cascade | Feature viz at each jump, hierarchical analysis |
| Transformer | Discrete transitions | Attention pattern analysis, head specialization |
| MLP | Smooth growth | Gradual feature evolution, distributed probing |

### 5.2 Dimensionality as Investigation Guide

**When to investigate**:
- At detected jump moments (z-score > 2)
- Before/after comparison at largest transitions
- At early vs late training phases

**What to look for**:
- New feature types emerging
- Activation pattern changes
- Neuron specialization

### 5.3 Limitations

- Jump detection is metric-based, not semantic
- Does not explain *why* jumps occur
- Requires further feature visualization for mechanistic understanding
- Limited to architectures tested

---

## 6. Infrastructure Contribution

### 6.1 Tools Provided

1. **HighFrequencyTracker**: Core dimensionality tracking
2. **ActivationCapture**: Hook-based activation recording
3. **JumpDetector**: Z-score transition detection
4. **ActivationAnalyzer**: PCA, clustering, manifold analysis
5. **FeatureVisualizer**: CAM, saliency, attention visualization

### 6.2 Usage Example

```python
from ndt import HighFrequencyTracker
from ndt.analysis import ActivationAnalyzer

# Track dimensionality during training
tracker = HighFrequencyTracker(model, sampling_frequency=5)
for epoch in range(num_epochs):
    for batch in dataloader:
        loss = train_step(batch)
        tracker.log(step, loss.item())

# Detect critical moments
jumps = tracker.detect_jumps(metric="stable_rank")

# Analyze at critical moments
analyzer = ActivationAnalyzer()
for jump in jumps[:5]:
    # Load checkpoint at jump moment
    # Capture and analyze activations
    pca_results = analyzer.pca_analysis(activations)
```

---

## 7. Conclusion

We demonstrate that dimensionality tracking provides practical infrastructure for guiding mechanistic interpretability research. Rather than claiming fundamental insights into learning, we offer:

1. **Measurement tools** that detect interesting moments
2. **Empirical findings** about architecture-specific patterns
3. **Methodology** for efficient mechanistic analysis

The value proposition is practical: instead of analyzing every training step, focus on dimensionality transitions. Our measurements help researchers know *when* to look, making mechanistic interpretability more efficient.

---

## 8. Future Work

1. **Extended architectures**: ViT, ResNets, large language models
2. **Semantic jump detection**: Beyond metric-based to meaning-based transitions
3. **Causal interventions**: Ablation studies at critical periods
4. **Real-time monitoring**: Dashboard for training observation
5. **Theoretical grounding**: Mathematical models for jump mechanisms

---

## Appendix A: Complete Experiment Results

### A.1 Summary Statistics
- Total experiments: 55
- Unique architectures: 15
- Mean R²: 0.2402
- Experiments with R² > 0.8: 10 (18.2%)

### A.2 Jump Statistics by Architecture

| Architecture | Mean Jumps | Max Jumps | % with Jumps |
|-------------|------------|-----------|--------------|
| CNN | 14.0 | 23 | 100% |
| Transformer | 3.1 | 9 | 57% |
| MLP | 0.5 | 14 | 6% |

### A.3 Target Experiment Details

**cnn_deep/fashion_mnist**
- Depth: 5 layers, Width: 120, Params: 390,410
- Jumps: 23, R²: 0.632, α: 3.06e-04
- Final Accuracy: 84.8%

**transformer_shallow/mnist**
- Depth: 2 layers, Width: 128, Params: 406,282
- Jumps: 9, R²: 0.261, α: 5.01e-05
- Final Accuracy: 93.6%

**mlp_narrow/mnist**
- Depth: 4 layers, Width: 32, Params: 28,618
- Jumps: 0, R²: 0.934, α: 1.41e-03
- Final Accuracy: 92.7%

---

## Acknowledgments

[To be added]

## References

[To be added]
