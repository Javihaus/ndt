# Quick Start Guide: Mechanistic Interpretability Analysis

## Overview

This guide provides a **fast path** to begin mechanistic interpretability analysis on the three target phenomena.

**Full detailed plan**: See `MECHANISTIC_INTERPRETABILITY_DETAILED_PLAN.md`

---

## Three Target Phenomena

| Phenomenon | Experiment | Key Stats | Analysis Focus |
|------------|-----------|-----------|----------------|
| **CNN Cascades** | cnn_deep/fashion_mnist | 23 jumps, R²=0.63 | Feature hierarchy emergence |
| **Transformer Transitions** | transformer_shallow/mnist | 9 jumps, R²=0.26 | Attention head specialization |
| **MLP Smooth** | mlp_narrow/mnist | 0 jumps, R²=0.93 | Gradual subspace refinement |

---

## Quick Start: CNN Jump Cascades (Week 1)

### Day 1: Setup

```bash
# 1. Prepare checkpoint directory
mkdir -p experiments/phase2/checkpoints/cnn_deep_fashion_mnist

# 2. Re-run training with checkpoint saving
python experiments/phase2/train_with_checkpoints.py \
    --arch cnn_deep \
    --dataset fashion_mnist \
    --save-every 5 \
    --save-around-jumps 7,6,8,9,5
```

### Day 2-3: Feature Visualization

```python
from ndt.analysis import FeatureVisualizer

# Load model at jump
model = load_checkpoint("after_jump_7")
visualizer = FeatureVisualizer(model)

# Generate Grad-CAM for all layers
for layer_idx, layer in enumerate(model.conv_layers):
    for class_idx in range(10):
        examples = get_class_examples(class_idx, n=10)

        for img in examples:
            cam = visualizer.grad_cam(img, class_idx, layer)
            save_visualization(cam, f"jump7_layer{layer_idx}_class{class_idx}")
```

**Expected Output**: 500 CAM visualizations (5 layers × 10 classes × 10 examples)

### Day 4-5: Neuron Selectivity

```python
from ndt.analysis import ActivationAnalyzer

analyzer = ActivationAnalyzer()

# Find newly active neurons at jump 7
newly_active = compare_checkpoints("before_jump_7", "after_jump_7")

# Analyze top 50
for neuron_idx in newly_active[:50]:
    selectivity = analyze_neuron_selectivity(model, layer, neuron_idx)
    classify_neuron_type(selectivity)  # edge/texture/part detector?
```

**Expected Output**: Neuron type classifications, preferred stimuli visualizations

---

## Quick Start: Transformer Transitions (Week 2)

### Day 1-2: Attention Pattern Analysis

```python
# Capture attention weights
_, attentions = model(img, return_attention=True)

# For each head
for head_idx in range(model.n_heads):
    head_attn = attentions[:, head_idx]

    # Compute statistics
    entropy = attention_entropy(head_attn)
    sparsity = attention_sparsity(head_attn)

    # Visualize
    plot_attention_heatmap(head_attn.mean(0), f"head_{head_idx}")
```

### Day 3-4: Head Clustering

```python
from sklearn.cluster import KMeans

# Collect attention patterns for all heads
patterns = collect_all_head_patterns(model, test_set)

# Cluster heads
kmeans = KMeans(n_clusters=3)
head_types = kmeans.fit_predict(patterns)

# Visualize clusters
for cluster_id in range(3):
    heads_in_cluster = np.where(head_types == cluster_id)[0]
    plot_representative_pattern(kmeans.cluster_centers_[cluster_id])
```

### Day 5: Head Ablation

```python
# Test importance of each head
baseline_acc = evaluate(model, test_set)

for head_idx in range(model.n_heads):
    model_ablated = ablate_head(model, head_idx)
    ablated_acc = evaluate(model_ablated, test_set)

    print(f"Head {head_idx}: {baseline_acc - ablated_acc:.3f} drop")
```

---

## Quick Start: MLP Smooth Learning (Week 3)

### Day 1-3: Subspace Evolution

```python
from ndt.analysis import ActivationAnalyzer

analyzer = ActivationAnalyzer()

# Sample checkpoints uniformly
checkpoints = [f"checkpoint_{i*100}" for i in range(20)]

subspace_evolution = []
for ckpt in checkpoints:
    model.load_checkpoint(ckpt)

    # Get activations
    acts = collect_activations(model, test_set)

    # PCA
    pca = analyzer.pca_analysis(acts, n_components=50)

    subspace_evolution.append({
        'checkpoint': ckpt,
        'components': pca['components'],
        'variance': pca['explained_variance_ratio']
    })

# Compute subspace overlap
overlaps = compute_pairwise_overlaps(subspace_evolution)
plot_overlap_matrix(overlaps)
```

### Day 4-5: Feature Disentanglement

```python
# Measure class separability over time
separability_over_time = []

for ckpt in checkpoints:
    model.load_checkpoint(ckpt)

    # Get representations
    reps, labels = get_representations(model, test_set)

    # Linear separability
    from sklearn.svm import LinearSVC
    svm = LinearSVC()
    svm.fit(reps, labels)
    acc = svm.score(reps, labels)

    # Silhouette score
    silhouette = silhouette_score(reps, labels)

    separability_over_time.append({
        'checkpoint': ckpt,
        'linear_acc': acc,
        'silhouette': silhouette
    })

plot_separability_evolution(separability_over_time)
```

---

## Key Analysis Techniques (Code Templates)

### 1. Grad-CAM Visualization

```python
def generate_gradcam(model, img, target_class, target_layer):
    """Generate Grad-CAM heatmap."""
    model.eval()

    # Forward pass
    output = model(img)
    score = output[0, target_class]

    # Backward pass
    model.zero_grad()
    score.backward()

    # Get activations and gradients
    activations = target_layer.output
    gradients = target_layer.output.grad

    # Weighted combination
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations).sum(dim=1)

    # ReLU + normalize
    cam = F.relu(cam)
    cam = (cam - cam.min()) / (cam.max() - cam.min())

    return cam
```

### 2. Attention Pattern Analysis

```python
def analyze_attention_pattern(attention_weights):
    """Compute attention statistics."""
    # Entropy: How focused is attention?
    probs = attention_weights.softmax(dim=-1)
    entropy = -(probs * probs.log()).sum(dim=-1).mean()

    # Sparsity: How many tokens attended to?
    top_k_mass = attention_weights.topk(5, dim=-1)[0].sum() / attention_weights.sum()

    # Diagonal dominance: Self-attention vs cross-attention?
    diag_mass = attention_weights.diagonal().sum() / attention_weights.sum()

    return {
        'entropy': entropy.item(),
        'top5_mass': top_k_mass.item(),
        'diagonal_mass': diag_mass.item()
    }
```

### 3. Subspace Overlap Computation

```python
def compute_subspace_overlap(pca1, pca2, n_components=10):
    """Compute overlap between two PCA subspaces."""
    # Get top principal components
    V1 = pca1['components'][:n_components].T
    V2 = pca2['components'][:n_components].T

    # Compute principal angles via SVD
    M = V1.T @ V2
    _, s, _ = np.linalg.svd(M)

    # Overlap = mean cosine of principal angles
    overlap = s.mean()

    return overlap
```

---

## Timeline Summary

### Minimum Viable Analysis (3 weeks)

**Week 1**: CNN feature visualization + neuron selectivity
**Week 2**: Transformer attention analysis + head clustering
**Week 3**: MLP subspace evolution + separability

**Output**:
- 3 phenomenon-specific reports
- 6-8 key figures
- Draft results section

### Full Analysis (8-12 weeks)

See `MECHANISTIC_INTERPRETABILITY_DETAILED_PLAN.md` for complete timeline

---

## Expected Outputs (MVP)

### Figures

1. **CNN Feature Hierarchy**: Grid of CAMs showing edge → texture → part progression
2. **Transformer Head Types**: Clustering dendrogram + representative attention patterns
3. **MLP Subspace Evolution**: PCA component trajectories + overlap heatmap
4. **Unified Comparison**: Three phenomena side-by-side dimensionality curves

### Tables

1. Jump statistics for CNN (timing, magnitude, features)
2. Attention head types for Transformer (cluster assignments, importance)
3. Separability metrics for MLP (linear accuracy, silhouette over time)

### Text Outputs

- 3 individual phenomenon reports (~5 pages each)
- 1 unified comparison (~3 pages)
- Methods section draft (~2 pages)

---

## Tools You'll Need

### Already Implemented

✅ `ActivationAnalyzer` - PCA, clustering, manifold analysis
✅ `FeatureVisualizer` - Grad-CAM, saliency maps
✅ `ActivationCapture` - Hook-based activation recording

### To Implement

⬜ `CheckpointManager` - Automated checkpoint saving around jumps
⬜ `ComparisonAnalyzer` - Before/after comparison utilities
⬜ `VisualizationExporter` - Batch figure generation

---

## Common Pitfalls

### Checkpoint Management
❌ **Wrong**: Save every single step (wastes storage)
✅ **Right**: Save before/after jumps + uniform sampling

### Visualization
❌ **Wrong**: Generate 10,000 individual CAMs
✅ **Right**: Generate 100 representatives + quantitative summaries

### Statistical Testing
❌ **Wrong**: Report single observation as conclusion
✅ **Right**: Compute confidence intervals, test significance

---

## Getting Help

1. **Detailed protocols**: See `MECHANISTIC_INTERPRETABILITY_DETAILED_PLAN.md`
2. **Code examples**: See `PHASE2_CODE_EXAMPLES.md`
3. **Infrastructure**: See `PHASE2_INFRASTRUCTURE.md`

---

## Success Checklist

### Week 1 (CNN)
- [ ] Checkpoints saved around top 5 jumps
- [ ] 500+ CAM visualizations generated
- [ ] Neuron selectivity analysis complete
- [ ] Feature emergence timeline documented

### Week 2 (Transformer)
- [ ] Attention patterns captured for all heads
- [ ] Head clustering complete (3-5 clusters identified)
- [ ] Head importance ranking computed
- [ ] Representative patterns visualized

### Week 3 (MLP)
- [ ] 20+ checkpoints analyzed for subspace evolution
- [ ] Overlap matrix computed and visualized
- [ ] Separability metrics computed over time
- [ ] Comparison with CNN/Transformer documented

### Week 4 (Integration)
- [ ] Unified comparison figures generated
- [ ] Draft results section written
- [ ] All code committed and documented
- [ ] Results validated and reproducible

---

**Ready to start?**

1. Read the relevant section above for your phenomenon
2. Set up checkpoints and environment
3. Run the code templates
4. Iterate and refine

For detailed protocols, see `MECHANISTIC_INTERPRETABILITY_DETAILED_PLAN.md`.
