# Feature-Level Hypothesis Testing

## Research Question

**Are early features (step 100) qualitatively different from late features (step 2000)?**

---

## Competing Hypotheses

### Hypothesis A: Qualitative Difference (Phase Transitions)
Early and late features are fundamentally different in structure and function.

**Predictions**:
- **CNN filters**:
  - Step 100: Random noise → basic edges
  - Step 1000: Edges → textures and curves
  - Step 2000: Textures → object-specific patterns
  - **Feature diversity increases** across phases
  - **Low similarity** between early and late filters

- **Transformer attention**:
  - Step 100: Diffuse, unstructured attention
  - Step 1000: Emerging attention patterns
  - Step 2000: Specialized, structured attention
  - **Attention entropy decreases** (more focused)
  - **Head specialization emerges**

- **MLP representations**:
  - Step 100: Random, high-dimensional representations
  - Step 1000: Structured representations emerge
  - Step 2000: Refined, lower effective dimensionality
  - **Low similarity** between early and late representations
  - **Representation geometry changes**

---

### Hypothesis B: Refinement Only (No Phases)
Early and late features are similar in structure, just refined over time.

**Predictions**:
- **CNN filters**:
  - Step 100: Already show object-relevant features (blurry)
  - Step 1000: Same features, sharper
  - Step 2000: Same features, slightly better
  - **Feature diversity stable** across phases
  - **High similarity** between all checkpoints

- **Transformer attention**:
  - Step 100: Similar patterns to late, just noisier
  - Step 1000: Same patterns, cleaner
  - Step 2000: Same patterns, minimal improvement
  - **Attention structure similar** throughout
  - **Gradual refinement**, no reorganization

- **MLP representations**:
  - Step 100: Similar structure to late
  - Step 1000: Same structure, better quality
  - Step 2000: Same structure, fine-tuned
  - **High similarity** between all checkpoints
  - **Consistent representation geometry**

---

## Quantitative Metrics

### CNN Filters

**1. Visual Inspection**
- Examine first conv layer filters (most interpretable)
- Look for qualitative changes in structure

**2. Diversity Score**
```python
diversity = count_unique_patterns(filters, threshold=0.8)
```
- **Hypothesis A**: Diversity increases (3 → 7 → 10 unique patterns)
- **Hypothesis B**: Diversity stable (7 → 7 → 7 unique patterns)

**3. Cross-Phase Similarity**
```python
similarity = cosine_similarity(filters_step100, filters_step2000)
```
- **Hypothesis A**: Low similarity (<0.5)
- **Hypothesis B**: High similarity (>0.7)

**4. Silhouette Score (cluster quality)**
- **Hypothesis A**: Increases (better separation)
- **Hypothesis B**: Stable or decreases

---

### Transformer Attention

**1. Attention Pattern Visualization**
- Extract attention weights from each layer/head
- Plot heatmaps showing which positions attend to which

**2. Attention Entropy**
```python
entropy = -sum(p * log(p)) for attention distribution p
```
- **Hypothesis A**: Entropy decreases (more focused attention)
- **Hypothesis B**: Entropy stable

**3. Head Specialization**
- Measure whether different heads develop different functions
- **Hypothesis A**: Specialization emerges (low inter-head correlation)
- **Hypothesis B**: Heads remain similar

---

### MLP Activations

**1. Representation Similarity**
```python
sim_early_mid = cosine_similarity(repr_100, repr_1000)
sim_mid_late = cosine_similarity(repr_1000, repr_2000)
```
- **Hypothesis A**: Early→Mid similarity < Mid→Late similarity
  - Interpretation: Representations change more early than late
- **Hypothesis B**: Early→Mid ≈ Mid→Late
  - Interpretation: Gradual refinement throughout

**2. Activation Sparsity**
```python
sparsity = fraction of activations near zero
```
- **Hypothesis A**: Sparsity increases (more selective neurons)
- **Hypothesis B**: Sparsity stable

**3. Effective Dimensionality**
```python
eff_dim = (sum of eigenvalues)^2 / sum of squared eigenvalues
```
- **Hypothesis A**: Dimensionality decreases (more structured)
- **Hypothesis B**: Dimensionality stable

---

## Expected Outcomes

### If Hypothesis A is True (Qualitative Phases)

**CNN Results**:
```
Step 100 → 1000: Similarity = 0.35 (low)
Step 1000 → 2000: Similarity = 0.68 (high)

Interpretation: Major reorganization early, refinement late
```

**Transformer Results**:
```
Step 100: Entropy = 2.8 (diffuse attention)
Step 1000: Entropy = 1.9 (more focused)
Step 2000: Entropy = 1.4 (highly focused)

Interpretation: Attention becomes more structured
```

**MLP Results**:
```
Step 100 → 1000: Similarity = 0.42 (low)
Step 1000 → 2000: Similarity = 0.81 (high)

Interpretation: Representations reorganize early, stabilize late
```

**Conclusion**: **Early phase is qualitatively different**. Training exhibits phase-like behavior.

---

### If Hypothesis B is True (Refinement Only)

**CNN Results**:
```
Step 100 → 1000: Similarity = 0.78 (high)
Step 1000 → 2000: Similarity = 0.82 (high)

Interpretation: Gradual refinement throughout
```

**Transformer Results**:
```
Step 100: Entropy = 1.8
Step 1000: Entropy = 1.6
Step 2000: Entropy = 1.5

Interpretation: Minor improvements, no reorganization
```

**MLP Results**:
```
Step 100 → 1000: Similarity = 0.84 (high)
Step 1000 → 2000: Similarity = 0.87 (high)

Interpretation: Consistent representations throughout
```

**Conclusion**: **No qualitative phases**. Training is gradual refinement. Original temporal boundary hypothesis not supported.

---

## What This Analysis Actually Tests

### Tests These Claims:
✅ Whether early and late features are structurally different
✅ Whether feature diversity changes across training
✅ Whether representations reorganize or just refine
✅ Cross-architecture consistency of phase patterns

### Does NOT Test:
❌ Why phases occur (if they exist)
❌ Causal mechanisms underlying transitions
❌ Individual "jumps" (magnitudes too small)
❌ Discrete vs continuous transitions

---

## Scientific Contribution

**If Hypothesis A is supported**:
> "Feature-level analysis reveals qualitative differences between early and late training phases, with CNN filters, transformer attention, and MLP representations all showing greater change in the first 50% of training. This provides direct evidence that temporal boundaries identified by dimensionality tracking correspond to structural reorganization of learned features."

**If Hypothesis B is supported**:
> "Despite training dynamics showing apparent temporal boundaries (90% of loss improvement in first 50%), feature-level analysis reveals gradual refinement without qualitative phase transitions. This demonstrates that loss-based metrics can be misleading, and dimensionality tracking does not reliably identify mechanistic transitions."

**Either outcome is scientifically valuable** - we learn something true about how neural networks train.

---

## Honest Framing

### What We Will Know After This Analysis:
- Whether features change qualitatively or just refine
- Specific numerical measures of feature similarity
- Cross-architecture patterns in feature evolution

### What We Still Won't Know:
- Why features evolve the way they do
- Causal mechanisms behind any transitions
- How to predict or control phase transitions
- Whether this generalizes beyond MNIST

### Limitations:
- Only 3 checkpoints per experiment (coarse temporal resolution)
- Only 1 dataset (MNIST)
- Only 1 training run per architecture
- No control experiments
- Correlation, not causation

---

## Analysis Timeline

**Estimated time**: 30-60 minutes
- PyTorch installation: 5-10 minutes
- CNN analysis: 10-15 minutes
- Transformer analysis: 10-15 minutes
- MLP analysis: 10-15 minutes
- Visualization and reporting: 5-10 minutes

**Total**: Under 1 hour for complete feature-level analysis

---

**Status**: Ready to execute once PyTorch is installed
