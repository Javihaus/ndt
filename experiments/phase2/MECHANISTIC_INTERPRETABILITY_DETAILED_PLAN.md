# Detailed Mechanistic Interpretability Plan
## Phase 2 Deep Dive: From Dimensionality Signals to Feature Understanding

**Version**: 1.0
**Date**: 2025-11-20
**Status**: Ready for Implementation

---

## Executive Summary

This document provides a **detailed, actionable plan** for conducting mechanistic interpretability analysis on the three phenomena identified in Phase 1. The plan is organized by phenomenon, with specific experimental protocols, analysis techniques, and expected outputs.

**Timeline**: 8-12 weeks
**Team Size**: 2-4 researchers
**Compute**: 1-2 GPUs for training replay and feature visualization

---

## Table of Contents

1. [Overview and Strategy](#overview-and-strategy)
2. [Phenomenon 1: CNN Jump Cascades](#phenomenon-1-cnn-jump-cascades)
3. [Phenomenon 2: Transformer Transitions](#phenomenon-2-transformer-transitions)
4. [Phenomenon 3: MLP Smooth Learning](#phenomenon-3-mlp-smooth-learning)
5. [Cross-Cutting Analyses](#cross-cutting-analyses)
6. [Implementation Roadmap](#implementation-roadmap)
7. [Validation and Quality Control](#validation-and-quality-control)
8. [Expected Outputs](#expected-outputs)

---

## Overview and Strategy

### Core Research Question

**How do different neural network architectures learn representations, and can dimensionality transitions guide us to the most informative moments for mechanistic analysis?**

### Three-Pronged Investigation Strategy

| Phenomenon | Key Question | Primary Techniques |
|------------|--------------|-------------------|
| **CNN Cascades** | What features emerge at each jump? | Feature viz, neuron activation patterns |
| **Transformer Transitions** | How do attention patterns specialize? | Attention analysis, head probing |
| **MLP Smooth** | What changes during smooth growth? | Subspace evolution, gradient flow |

### Experimental Philosophy

1. **Checkpoint-based analysis**: Save models before/after critical moments
2. **Comparative approach**: Always compare transitions to controls
3. **Multiple scales**: Analyze individual neurons, layers, and full network
4. **Quantitative validation**: Every qualitative observation needs metrics

---

## Phenomenon 1: CNN Jump Cascades

### Target Experiment
**cnn_deep/fashion_mnist**: 23 jumps, indices 4-22, top jumps at [7, 6, 8, 9, 5]

### Hypothesis
Each dimensionality jump corresponds to the emergence of a new level in the feature hierarchy (edges → textures → parts → objects).

### Detailed Experimental Protocol

#### 1.1 Checkpoint Preparation

**Objective**: Capture model states around critical jumps

**Protocol**:
```python
# For each of the top 5 jumps
jump_indices = [7, 6, 8, 9, 5]

for jump_idx in jump_indices:
    checkpoints_to_save = [
        jump_idx - 10,  # Well before
        jump_idx - 2,   # Immediately before
        jump_idx,       # At jump
        jump_idx + 2,   # Immediately after
        jump_idx + 10   # Well after
    ]

    # Save: model weights, optimizer state, activations on test set
```

**Output**: 25 checkpoints (5 per jump) × ~100MB = 2.5GB storage

**Timeline**: 1 day for re-training with checkpoint saving

---

#### 1.2 Feature Visualization Analysis

**Objective**: Identify what visual features emerge at each jump

**Technique 1: Gradient-based Class Activation Mapping (Grad-CAM)**

```python
from ndt.analysis import FeatureVisualizer

visualizer = FeatureVisualizer(model)

# For each convolutional layer
for layer_idx, layer in enumerate(conv_layers):
    # For each class in Fashion-MNIST
    for class_idx in range(10):
        # Get 10 correctly classified examples
        examples = get_examples(class_idx, n=10)

        for img in examples:
            # Generate CAM before jump
            model.load_checkpoint(f"before_jump_{jump_idx}")
            cam_before = visualizer.grad_cam(img, class_idx, layer)

            # Generate CAM after jump
            model.load_checkpoint(f"after_jump_{jump_idx}")
            cam_after = visualizer.grad_cam(img, class_idx, layer)

            # Compute difference
            cam_diff = cam_after - cam_before

            # Save visualization
            save_cam_comparison(cam_before, cam_after, cam_diff,
                              f"jump{jump_idx}_layer{layer_idx}_class{class_idx}")
```

**Analysis Questions**:
- Do early jumps (indices 5-9) show edge/texture emergence?
- Do later jumps show part-based features?
- Is there a consistent progression across classes?

**Expected Output**:
- 5 jumps × 5 layers × 10 classes × 10 examples = 2,500 visualizations
- Summary heatmaps showing spatial attention evolution

**Timeline**: 2-3 days

---

**Technique 2: Feature Map Activation Analysis**

```python
from ndt.analysis import ActivationAnalyzer
from ndt.core.hooks import ActivationCapture

analyzer = ActivationAnalyzer()
capture = ActivationCapture()

# For each jump
for jump_idx in [7, 6, 8, 9, 5]:
    # Capture activations before and after
    for checkpoint in [f"before_{jump_idx}", f"after_{jump_idx}"]:
        model.load_checkpoint(checkpoint)
        capture.register_hooks(model, conv_layers)

        # Forward pass on test set
        all_activations = []
        all_labels = []

        for batch_x, batch_y in test_loader:
            output = model(batch_x)
            activations = capture.get_all_activations()
            all_activations.append(activations)
            all_labels.extend(batch_y.tolist())

        # Analyze each layer
        for layer_name, acts in activations.items():
            # Which neurons became active?
            neuron_importance = analyzer.neuron_importance(acts)

            # How many new active neurons?
            dead_before = set(neuron_importance['dead_neurons'])
            # Compare to after checkpoint...
            newly_active = dead_before - dead_after

            # What do the new neurons respond to?
            analyze_neuron_selectivity(newly_active, acts, labels)
```

**Analysis Questions**:
- How many neurons "wake up" at each jump?
- What visual patterns activate new neurons?
- Do new neurons show class-selectivity?

**Expected Output**:
- Per-jump neuron activation statistics
- Selectivity curves for newly active neurons
- Visualization of preferred stimuli

**Timeline**: 3-4 days

---

#### 1.3 Neuron-Level Selectivity Analysis

**Objective**: Understand what each neuron has learned to detect

**Protocol**:

```python
def analyze_neuron_selectivity(model, layer, neuron_idx, dataset):
    """
    Find images that maximally activate a specific neuron.
    """
    activations = []
    images = []

    for img, label in dataset:
        act = get_neuron_activation(model, layer, neuron_idx, img)
        activations.append(act)
        images.append(img)

    # Get top-k activating images
    top_k_indices = np.argsort(activations)[-20:]
    top_images = [images[i] for i in top_k_indices]

    # Visualize
    plot_neuron_preferred_images(neuron_idx, top_images)

    # Compute receptive field
    receptive_field = compute_effective_receptive_field(
        model, layer, neuron_idx
    )

    return {
        'top_activations': activations[top_k_indices],
        'preferred_images': top_images,
        'receptive_field': receptive_field,
        'selectivity_index': compute_selectivity_index(activations, labels)
    }

# Run for top 50 neurons in each layer at each jump
for jump_idx in top_5_jumps:
    for layer in conv_layers:
        # Get neurons that became active at this jump
        newly_active = get_newly_active_neurons(jump_idx, layer)

        for neuron_idx in newly_active[:50]:
            selectivity = analyze_neuron_selectivity(
                model_after_jump, layer, neuron_idx, test_set
            )

            # Classify neuron type
            neuron_type = classify_neuron(selectivity)
            # e.g., 'edge_detector', 'texture', 'part_detector'
```

**Expected Output**:
- Neuron selectivity profiles for 250+ neurons (50 per jump × 5 jumps)
- Classification of neuron types (edge/texture/part/object)
- Temporal emergence patterns (which types appear at which jumps)

**Timeline**: 4-5 days

---

#### 1.4 Hierarchical Feature Composition Analysis

**Objective**: Test if later layers compose features from earlier layers

**Protocol**:

```python
def test_feature_composition(model, early_layer, late_layer, jump_idx):
    """
    Test if late-layer features are compositions of early-layer features.
    """
    # Get activations for both layers
    capture = ActivationCapture()
    capture.register_hooks(model, [early_layer, late_layer])

    activations_early = []
    activations_late = []

    for img, _ in test_loader:
        model(img)
        acts = capture.get_all_activations()
        activations_early.append(acts['early'])
        activations_late.append(acts['late'])

    # For each late-layer neuron
    for late_neuron_idx in range(activations_late.shape[1]):
        late_acts = activations_late[:, late_neuron_idx]

        # Fit linear model: late = W * early + b
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=0.1)
        model.fit(activations_early, late_acts)

        # Check R²
        r2 = model.score(activations_early, late_acts)

        # If high R², this late neuron is a composition
        if r2 > 0.7:
            # Which early neurons contribute most?
            top_contributors = np.argsort(np.abs(model.coef_))[-10:]

            save_composition_graph(
                late_neuron_idx, top_contributors, model.coef_
            )

# Test composition after each major jump
for jump_idx in [7, 8, 9]:
    model.load_checkpoint(f"after_jump_{jump_idx}")

    # Test layer pairs
    for early, late in [(0, 2), (1, 3), (2, 4)]:
        composition = test_feature_composition(model, early, late, jump_idx)
```

**Expected Output**:
- Feature composition graphs showing which neurons feed into which
- Quantification of hierarchical organization
- Evidence for/against compositional emergence at jumps

**Timeline**: 3-4 days

---

#### 1.5 Ablation Studies

**Objective**: Confirm that jump moments are causally important

**Protocol**:

```python
def ablation_study(base_model, jump_idx):
    """
    Test what happens if we prevent the jump from occurring.
    """
    # Train until just before jump
    model = train_until_step(jump_idx - 2)
    baseline_acc = evaluate(model, test_set)

    # Intervention 1: Freeze weights during jump period
    model_frozen = copy.deepcopy(model)
    freeze_weights(model_frozen)
    train_steps(model_frozen, steps=20)  # Through the jump period
    unfreeze_weights(model_frozen)
    train_to_convergence(model_frozen)
    frozen_acc = evaluate(model_frozen, test_set)

    # Intervention 2: Continue training normally
    model_normal = copy.deepcopy(model)
    train_to_convergence(model_normal)
    normal_acc = evaluate(model_normal, test_set)

    # Intervention 3: Perturb during jump
    model_perturbed = copy.deepcopy(model)
    add_noise_to_gradients(model_perturbed, steps=20)
    train_to_convergence(model_perturbed)
    perturbed_acc = evaluate(model_perturbed, test_set)

    return {
        'baseline': baseline_acc,
        'frozen_through_jump': frozen_acc,
        'normal': normal_acc,
        'perturbed_during_jump': perturbed_acc,
        'jump_importance': normal_acc - frozen_acc
    }

# Test for top 3 jumps
for jump_idx in [7, 8, 9]:
    ablation_results = ablation_study(base_model, jump_idx)
    print(f"Jump {jump_idx} importance: {ablation_results['jump_importance']:.3f}")
```

**Expected Output**:
- Causal importance scores for each jump
- Evidence that jumps are necessary for performance
- Identification of most critical jumps

**Timeline**: 5-7 days (requires retraining)

---

### 1.6 Summary for CNN Cascades

**Total Timeline**: 3-4 weeks

**Key Deliverables**:
1. Feature hierarchy diagram (edges → textures → parts → objects)
2. Quantification of feature emergence timing
3. Neuron selectivity profiles across training
4. Causal evidence for jump importance
5. 3-5 publication-quality figures

**Expected Insights**:
- Precise characterization of what emerges when
- Evidence for hierarchical composition
- Understanding of why CNNs show many jumps

---

## Phenomenon 2: Transformer Transitions

### Target Experiment
**transformer_shallow/mnist**: 9 jumps, early cascade pattern, R²=0.261

### Hypothesis
Jumps correspond to attention head specialization events where heads learn distinct attention patterns.

### Detailed Experimental Protocol

#### 2.1 Attention Pattern Evolution

**Objective**: Track how attention patterns change through training

**Protocol**:

```python
from ndt.analysis import FeatureVisualizer

def analyze_attention_evolution(model, jump_indices):
    """
    Track attention patterns before/after each jump.
    """
    visualizer = FeatureVisualizer(model)

    for jump_idx in jump_indices:
        for checkpoint in ['before', 'at', 'after']:
            model.load_checkpoint(f"{checkpoint}_jump_{jump_idx}")

            # Get attention weights for test examples
            attention_patterns = []

            for img, label in test_loader:
                # Forward pass with attention capture
                output, attentions = model(img, return_attention=True)
                # attentions shape: (batch, n_heads, seq_len, seq_len)

                attention_patterns.append(attentions)

            # Analyze each head
            for head_idx in range(model.n_heads):
                head_attentions = extract_head(attention_patterns, head_idx)

                # Compute attention statistics
                stats = {
                    'entropy': attention_entropy(head_attentions),
                    'sparsity': attention_sparsity(head_attentions),
                    'avg_pattern': head_attentions.mean(axis=0),
                    'class_selectivity': compute_class_selectivity(
                        head_attentions, labels
                    )
                }

                save_attention_stats(jump_idx, checkpoint, head_idx, stats)

                # Visualize average attention pattern
                plot_attention_heatmap(
                    stats['avg_pattern'],
                    f"jump{jump_idx}_{checkpoint}_head{head_idx}"
                )
```

**Analysis Questions**:
- Do heads show low entropy (focused attention) after jumps?
- Does each head specialize for different input regions?
- Are there consistent attention patterns across examples?

**Expected Output**:
- Attention pattern visualizations for each head at each jump
- Entropy/sparsity evolution curves
- Head specialization metrics

**Timeline**: 3-4 days

---

#### 2.2 Attention Head Clustering

**Objective**: Identify groups of heads with similar behaviors

**Protocol**:

```python
def cluster_attention_heads(model, test_set):
    """
    Cluster attention heads by their attention patterns.
    """
    from sklearn.cluster import KMeans
    from ndt.analysis import ActivationAnalyzer

    analyzer = ActivationAnalyzer()

    # Collect attention patterns for all heads
    head_patterns = []  # Will be (n_heads, n_examples, seq_len, seq_len)

    for img, _ in test_set:
        _, attentions = model(img, return_attention=True)
        # attentions: (batch, n_heads, seq_len, seq_len)

        for head_idx in range(model.n_heads):
            head_attn = attentions[0, head_idx].flatten()
            head_patterns.append(head_attn)

    # Reshape for clustering
    head_patterns = np.array(head_patterns)
    # Shape: (n_heads * n_examples, seq_len^2)

    # Cluster heads based on average patterns
    avg_patterns = compute_average_per_head(head_patterns)

    # Apply clustering
    n_clusters = 3  # Hypothesize 3 types of heads
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    head_clusters = kmeans.fit_predict(avg_patterns)

    # Analyze each cluster
    for cluster_id in range(n_clusters):
        heads_in_cluster = np.where(head_clusters == cluster_id)[0]

        print(f"\nCluster {cluster_id}: {len(heads_in_cluster)} heads")

        # Visualize representative attention pattern
        representative = kmeans.cluster_centers_[cluster_id]
        plot_attention_pattern(
            representative.reshape(seq_len, seq_len),
            f"cluster_{cluster_id}_representative"
        )

        # Compute cluster statistics
        cluster_stats = {
            'size': len(heads_in_cluster),
            'avg_entropy': compute_avg_entropy(heads_in_cluster),
            'head_indices': heads_in_cluster.tolist()
        }

# Run clustering before and after major jumps
for jump_idx in [0, 4, 8]:  # Early, middle, late jumps
    model.load_checkpoint(f"after_jump_{jump_idx}")
    clusters = cluster_attention_heads(model, test_set)

    # Track how clustering changes
    track_cluster_evolution(jump_idx, clusters)
```

**Analysis Questions**:
- Do heads start homogeneous and specialize over time?
- Do jumps correspond to cluster splits (1 cluster → 2 clusters)?
- What attention strategies do different clusters use?

**Expected Output**:
- Head clustering dendrograms over training
- Cluster evolution visualization
- Characterization of head "types"

**Timeline**: 2-3 days

---

#### 2.3 Head Importance and Probing

**Objective**: Determine which heads are important for which tasks

**Protocol**:

```python
def probe_head_importance(model, test_set):
    """
    Systematically ablate each head and measure impact.
    """
    baseline_acc = evaluate(model, test_set)
    head_importance = {}

    for layer_idx in range(model.n_layers):
        for head_idx in range(model.n_heads):
            # Ablate this head (zero out its attention)
            model_ablated = ablate_attention_head(
                model, layer_idx, head_idx
            )

            # Evaluate
            ablated_acc = evaluate(model_ablated, test_set)

            # Importance = drop in accuracy
            importance = baseline_acc - ablated_acc

            head_importance[f"L{layer_idx}_H{head_idx}"] = {
                'importance': importance,
                'baseline_acc': baseline_acc,
                'ablated_acc': ablated_acc
            }

    return head_importance

def probe_head_functionality(model, head_id, test_set):
    """
    Use linear probes to determine what information a head captures.
    """
    from sklearn.linear_model import LogisticRegression

    # Extract head outputs for all examples
    head_outputs = []
    labels = []

    for img, label in test_set:
        output = extract_head_output(model, head_id, img)
        head_outputs.append(output.flatten())
        labels.append(label)

    X = np.array(head_outputs)
    y = np.array(labels)

    # Train probe for class prediction
    probe = LogisticRegression(max_iter=1000)
    probe.fit(X, y)
    class_acc = probe.score(X, y)

    # Train probe for position information
    positions = get_patch_positions(test_set)
    probe_pos = LogisticRegression(max_iter=1000)
    probe_pos.fit(X, positions)
    position_acc = probe_pos.score(X, positions)

    return {
        'class_accuracy': class_acc,
        'position_accuracy': position_acc,
        'functionality': classify_head_function(class_acc, position_acc)
    }

# Run probing before and after jumps
for jump_idx in top_5_jumps:
    for checkpoint in ['before', 'after']:
        model.load_checkpoint(f"{checkpoint}_jump_{jump_idx}")

        # Head importance
        importance = probe_head_importance(model, test_set)

        # Head functionality
        functionality = {}
        for head_id in range(model.n_heads * model.n_layers):
            functionality[head_id] = probe_head_functionality(
                model, head_id, test_set
            )

        analyze_head_changes(jump_idx, importance, functionality)
```

**Analysis Questions**:
- Which heads become important at each jump?
- Do heads specialize for content vs. position?
- Can we predict which heads will specialize based on early training?

**Expected Output**:
- Head importance rankings over training
- Functional categorization of heads
- Evidence for sudden vs. gradual specialization

**Timeline**: 4-5 days

---

#### 2.4 Attention Flow Analysis

**Objective**: Understand how information flows through attention layers

**Protocol**:

```python
def analyze_attention_flow(model, example_image):
    """
    Track how attention flows from input to output.
    """
    # Get attention for all layers
    _, attentions = model(example_image, return_attention=True)
    # attentions: [(batch, heads, seq, seq) for each layer]

    # Compute effective attention (product of all layers)
    effective_attention = attentions[0]
    for layer_attn in attentions[1:]:
        # Matrix multiply to get end-to-end attention
        effective_attention = torch.bmm(
            effective_attention.mean(dim=1, keepdim=True),
            layer_attn.mean(dim=1, keepdim=True)
        )

    # Analyze flow patterns
    flow_metrics = {
        'input_diversity': compute_input_diversity(effective_attention),
        'output_concentration': compute_output_concentration(effective_attention),
        'bottleneck_tokens': identify_bottleneck_tokens(effective_attention),
        'flow_entropy': compute_flow_entropy(attentions)
    }

    # Visualize
    plot_attention_flow(attentions, effective_attention)

    return flow_metrics

# Compare flow before and after jumps
for jump_idx in top_5_jumps:
    for checkpoint in ['before', 'after']:
        model.load_checkpoint(f"{checkpoint}_jump_{jump_idx}")

        # Analyze flow on 100 test examples
        flow_metrics_all = []
        for img, _ in test_loader.sample(100):
            metrics = analyze_attention_flow(model, img)
            flow_metrics_all.append(metrics)

        # Aggregate statistics
        avg_flow = aggregate_flow_metrics(flow_metrics_all)

        save_flow_analysis(jump_idx, checkpoint, avg_flow)
```

**Expected Output**:
- Attention flow diagrams showing information pathways
- Quantification of information bottlenecks
- Changes in flow structure at jumps

**Timeline**: 3-4 days

---

### 2.5 Summary for Transformer Transitions

**Total Timeline**: 2-3 weeks

**Key Deliverables**:
1. Attention pattern evolution visualizations
2. Head specialization timeline
3. Functional categorization of attention heads
4. Attention flow diagrams
5. Evidence for discrete vs. gradual specialization
6. 3-4 publication-quality figures

**Expected Insights**:
- Mechanism of attention head specialization
- Whether jumps represent sudden specialization events
- Types of attention strategies learned

---

## Phenomenon 3: MLP Smooth Learning

### Target Experiment
**mlp_narrow/mnist**: 0 jumps, R²=0.934, smooth TAP-following growth

### Hypothesis
MLPs learn through gradual subspace refinement rather than discrete feature emergence.

### Detailed Experimental Protocol

#### 3.1 Subspace Evolution Analysis

**Objective**: Track how the representation subspace evolves smoothly

**Protocol**:

```python
from ndt.analysis import ActivationAnalyzer

def analyze_subspace_evolution(model, checkpoints, test_set):
    """
    Track how principal subspaces evolve during training.
    """
    analyzer = ActivationAnalyzer()

    subspace_evolution = []

    for checkpoint_idx, checkpoint_path in enumerate(checkpoints):
        model.load_checkpoint(checkpoint_path)

        # Collect activations for each layer
        layer_activations = {}

        for img, label in test_set:
            activations = capture_layer_activations(model, img)
            for layer_name, acts in activations.items():
                if layer_name not in layer_activations:
                    layer_activations[layer_name] = []
                layer_activations[layer_name].append(acts)

        # Analyze each layer
        for layer_name, acts_list in layer_activations.items():
            acts = np.vstack(acts_list)

            # PCA analysis
            pca_results = analyzer.pca_analysis(acts, n_components=50)

            # Singular value analysis
            sv_results = analyzer.singular_value_analysis(acts)

            subspace_evolution.append({
                'checkpoint': checkpoint_idx,
                'layer': layer_name,
                'pca_components': pca_results['components'],
                'explained_variance': pca_results['explained_variance_ratio'],
                'singular_values': sv_results['singular_values'],
                'stable_rank': sv_results['stable_rank'],
                'participation_ratio': sv_results['participation_ratio']
            })

    # Compute subspace similarity over time
    subspace_similarities = compute_subspace_overlap(subspace_evolution)

    return subspace_evolution, subspace_similarities

# Sample checkpoints uniformly through training
checkpoints = [f"checkpoint_step_{i}" for i in range(0, 2000, 100)]

evolution, similarities = analyze_subspace_evolution(model, checkpoints, test_set)

# Plot evolution
plot_subspace_evolution(evolution)
plot_subspace_similarity_matrix(similarities)
```

**Analysis Questions**:
- How smoothly do principal components change?
- Is there continuous refinement vs. sudden reorientation?
- Do different layers evolve at different rates?

**Expected Output**:
- Principal component evolution curves
- Subspace similarity matrices over time
- Quantification of smoothness

**Timeline**: 3-4 days

---

#### 3.2 Gradient Flow Analysis

**Objective**: Understand how gradients flow through smooth learning

**Protocol**:

```python
def analyze_gradient_flow(model, train_loader, checkpoints):
    """
    Track gradient magnitudes and directions during training.
    """
    gradient_stats = []

    for checkpoint_path in checkpoints:
        model.load_checkpoint(checkpoint_path)

        # Compute gradients on a batch
        batch_x, batch_y = next(iter(train_loader))
        loss = compute_loss(model, batch_x, batch_y)

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Collect gradient statistics
        layer_grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                layer_grads[name] = {
                    'mean': param.grad.abs().mean().item(),
                    'std': param.grad.std().item(),
                    'max': param.grad.abs().max().item(),
                    'l2_norm': param.grad.norm(2).item(),
                    'sparsity': (param.grad == 0).float().mean().item()
                }

        gradient_stats.append({
            'checkpoint': checkpoint_path,
            'layer_grads': layer_grads,
            'total_grad_norm': compute_total_grad_norm(model)
        })

    # Analyze gradient evolution
    plot_gradient_evolution(gradient_stats)

    return gradient_stats

# Analyze at 50 checkpoints
gradient_evolution = analyze_gradient_flow(model, train_loader, checkpoints)

# Check for gradient-based early warning signs
check_gradient_anomalies(gradient_evolution)
```

**Expected Output**:
- Gradient magnitude evolution curves
- Layer-wise gradient flow patterns
- Evidence for smooth vs. chaotic updates

**Timeline**: 2-3 days

---

#### 3.3 Feature Disentanglement Analysis

**Objective**: Test if MLPs gradually disentangle class features

**Protocol**:

```python
def analyze_feature_disentanglement(model, checkpoints, test_set):
    """
    Measure how separable class representations become over time.
    """
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    disentanglement_metrics = []

    for checkpoint_path in checkpoints:
        model.load_checkpoint(checkpoint_path)

        # Get hidden representations for each layer
        for layer_idx, layer in enumerate(model.layers):
            representations = []
            labels = []

            for img, label in test_set:
                hidden = get_layer_output(model, layer_idx, img)
                representations.append(hidden.flatten())
                labels.append(label)

            X = np.array(representations)
            y = np.array(labels)

            # Measure 1: Linear separability
            from sklearn.svm import LinearSVC
            svm = LinearSVC(max_iter=1000)
            svm.fit(X, y)
            linear_acc = svm.score(X, y)

            # Measure 2: Silhouette score (cluster quality)
            silhouette = silhouette_score(X, y)

            # Measure 3: Between-class vs within-class variance
            between_var, within_var = compute_class_variance(X, y)
            separation_ratio = between_var / (within_var + 1e-8)

            # Measure 4: Dimension of class manifolds
            class_dims = []
            for class_id in range(10):
                class_X = X[y == class_id]
                pca = PCA(n_components=0.95)  # 95% variance
                pca.fit(class_X)
                class_dims.append(pca.n_components_)

            disentanglement_metrics.append({
                'checkpoint': checkpoint_path,
                'layer': layer_idx,
                'linear_separability': linear_acc,
                'silhouette_score': silhouette,
                'separation_ratio': separation_ratio,
                'avg_class_dim': np.mean(class_dims),
                'class_dims': class_dims
            })

    return disentanglement_metrics

# Analyze disentanglement
disentanglement = analyze_feature_disentanglement(model, checkpoints, test_set)

# Plot evolution of separability
plot_disentanglement_evolution(disentanglement)
```

**Analysis Questions**:
- Do classes become linearly separable gradually or suddenly?
- Is there progressive dimensionality reduction per class?
- Which layers show the most disentanglement?

**Expected Output**:
- Separability evolution curves
- Per-class manifold dimensionality over time
- Evidence for gradual vs. sudden disentanglement

**Timeline**: 3-4 days

---

#### 3.4 Comparison with Synthetic Jumps

**Objective**: Compare smooth MLP learning to CNNs with jumps

**Protocol**:

```python
def compare_with_jumping_network(mlp_model, cnn_model):
    """
    Contrast smooth MLP evolution with jumping CNN evolution.
    """
    # MLP checkpoints (uniform sampling)
    mlp_checkpoints = sample_uniformly(mlp_training, n=50)

    # CNN checkpoints (sample around jumps)
    cnn_jump_indices = [5, 6, 7, 8, 9]
    cnn_checkpoints = sample_around_jumps(cnn_training, cnn_jump_indices)

    # Measure smoothness of dimensionality growth
    mlp_dims = [get_stable_rank(ckpt) for ckpt in mlp_checkpoints]
    cnn_dims = [get_stable_rank(ckpt) for ckpt in cnn_checkpoints]

    # Compute derivatives (rate of change)
    mlp_derivatives = np.diff(mlp_dims)
    cnn_derivatives = np.diff(cnn_dims)

    # Statistical tests
    from scipy.stats import levene, kruskal

    # Test for homogeneity of variance (smoothness)
    stat, p_value = levene(mlp_derivatives, cnn_derivatives)

    print(f"MLP variance: {np.var(mlp_derivatives):.4f}")
    print(f"CNN variance: {np.var(cnn_derivatives):.4f}")
    print(f"Levene test p-value: {p_value:.4f}")

    # Visualize comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(mlp_dims, label='MLP (smooth)', linewidth=2)
    axes[0].plot(cnn_dims, label='CNN (jumps)', linewidth=2)
    axes[0].set_xlabel('Training Progress')
    axes[0].set_ylabel('Stable Rank')
    axes[0].legend()
    axes[0].set_title('Dimensionality Evolution')

    axes[1].hist(mlp_derivatives, alpha=0.5, label='MLP', bins=20)
    axes[1].hist(cnn_derivatives, alpha=0.5, label='CNN', bins=20)
    axes[1].set_xlabel('Rate of Change')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    axes[1].set_title('Distribution of Growth Rates')

    plt.tight_layout()
    plt.savefig('mlp_vs_cnn_comparison.png', dpi=150)

compare_with_jumping_network(mlp_model, cnn_model)
```

**Expected Output**:
- Side-by-side comparison plots
- Statistical quantification of smoothness
- Evidence for fundamentally different learning dynamics

**Timeline**: 2-3 days

---

### 3.5 Summary for MLP Smooth Learning

**Total Timeline**: 2 weeks

**Key Deliverables**:
1. Subspace evolution trajectories
2. Gradient flow analysis
3. Feature disentanglement curves
4. Comparison with jumping architectures
5. 2-3 publication-quality figures

**Expected Insights**:
- Mechanism of smooth representation refinement
- Why MLPs don't show jumps
- Alternative learning dynamics to discrete transitions

---

## Cross-Cutting Analyses

### 4.1 Architecture Comparison Framework

**Objective**: Unified comparison across all three phenomena

```python
def unified_comparison_analysis():
    """
    Compare the three target experiments on common metrics.
    """
    experiments = [
        ('cnn_deep', 'fashion_mnist'),
        ('transformer_shallow', 'mnist'),
        ('mlp_narrow', 'mnist')
    ]

    comparison_metrics = {
        'dimensionality_evolution': [],
        'jump_statistics': [],
        'feature_separability': [],
        'training_efficiency': []
    }

    for arch, dataset in experiments:
        model = load_final_model(arch, dataset)
        checkpoints = load_checkpoints(arch, dataset)

        # Metric 1: Dimensionality trajectory
        dim_traj = compute_dimensionality_trajectory(checkpoints)
        comparison_metrics['dimensionality_evolution'].append(dim_traj)

        # Metric 2: Jump characterization
        jumps = detect_jumps(dim_traj)
        jump_stats = characterize_jumps(jumps)
        comparison_metrics['jump_statistics'].append(jump_stats)

        # Metric 3: Feature quality at convergence
        separability = measure_feature_separability(model, test_set)
        comparison_metrics['feature_separability'].append(separability)

        # Metric 4: Sample efficiency
        efficiency = compute_sample_efficiency(checkpoints)
        comparison_metrics['training_efficiency'].append(efficiency)

    # Generate comparison table
    generate_comparison_table(experiments, comparison_metrics)

    # Generate comparison figures
    plot_unified_comparison(comparison_metrics)

unified_comparison_analysis()
```

**Timeline**: 1 week

---

### 4.2 Theoretical Connection Analysis

**Objective**: Connect empirical findings to theoretical frameworks

**Analysis Areas**:

1. **Neural Tangent Kernel (NTK) Analysis**
   - Compute NTK at different training stages
   - Test if jumps correspond to kernel changes
   - Compare NTK evolution for smooth vs. jumping networks

2. **Information Bottleneck Analysis**
   - Measure mutual information I(X; H) and I(H; Y)
   - Test if jumps correspond to information plane transitions
   - Track compression vs. prediction trade-off

3. **Loss Landscape Analysis**
   - Visualize loss landscape around jump moments
   - Test if jumps correspond to barrier crossings
   - Analyze mode connectivity before/after jumps

**Timeline**: 2-3 weeks (advanced analysis)

---

## Implementation Roadmap

### Week-by-Week Timeline

**Weeks 1-2: Infrastructure and Preparation**
- Week 1: Set up checkpoint saving for all experiments, re-run training
- Week 2: Validate checkpoint quality, prepare analysis scripts

**Weeks 3-6: CNN Jump Cascade Analysis**
- Week 3: Feature visualization (Grad-CAM, feature maps)
- Week 4: Neuron selectivity analysis
- Week 5: Hierarchical composition analysis
- Week 6: Ablation studies

**Weeks 7-9: Transformer Transition Analysis**
- Week 7: Attention pattern evolution, head clustering
- Week 8: Head importance probing
- Week 9: Attention flow analysis

**Weeks 10-11: MLP Smooth Learning Analysis**
- Week 10: Subspace evolution, gradient flow
- Week 11: Feature disentanglement, comparisons

**Week 12: Integration and Write-up**
- Unified comparison analysis
- Generate all publication figures
- Draft results section

---

## Validation and Quality Control

### Quality Checks

1. **Reproducibility**: Every analysis must produce consistent results across 3 runs
2. **Statistical Significance**: Use bootstrapping and permutation tests
3. **Visualization Quality**: All figures must be publication-ready
4. **Code Review**: All analysis code peer-reviewed before execution

### Validation Criteria

**For Feature Visualization**:
- [ ] CAMs are interpretable and consistent across examples
- [ ] Saliency maps highlight relevant image regions
- [ ] Before/after comparisons show meaningful differences

**For Quantitative Analyses**:
- [ ] Effect sizes computed with confidence intervals
- [ ] Multiple comparison correction applied
- [ ] Results hold across different random seeds

**For Theoretical Connections**:
- [ ] Predictions made before analysis
- [ ] Null hypotheses clearly stated
- [ ] Alternative explanations considered

---

## Expected Outputs

### Academic Outputs

**1. Main Paper**: "Layer-Wise Representational Dynamics: Using Dimensionality Measurements to Guide Mechanistic Investigation"
- 8-10 pages
- 6-8 main figures
- 3-4 supplementary figures
- Expected venue: ICML, NeurIPS, or ICLR

**2. Supplementary Material**:
- Complete experimental protocols
- Additional visualizations
- Code and data release

**3. Blog Post**: Accessible explanation of findings

### Software Outputs

**1. Analysis Package**: `/src/ndt/analysis/` (already created)
- `activation_analysis.py`
- `feature_visualization.py`
- Additional modules as needed

**2. Experimental Scripts**: `/experiments/phase2/`
- Checkpoint management
- Analysis orchestration
- Visualization generation

**3. Documentation**:
- API documentation
- Tutorial notebooks
- Example analyses

---

## Resource Requirements

### Compute

- **Training**: 2 GPUs × 3 days = 144 GPU-hours
- **Analysis**: 1 GPU × 4 weeks = 672 GPU-hours
- **Total**: ~800 GPU-hours (~$400 on cloud)

### Storage

- **Checkpoints**: 25 checkpoints × 100MB × 3 experiments = 7.5GB
- **Activations**: 10GB per experiment = 30GB
- **Visualizations**: 5GB
- **Total**: ~50GB

### Personnel

- **Ideal**: 2-3 researchers (one per phenomenon)
- **Minimum**: 1 researcher + advisor oversight
- **Time**: 8-12 weeks

---

## Risk Mitigation

### Potential Issues and Solutions

**Issue 1**: Checkpoints too large
- Solution: Save only layer activations, not full optimizer state

**Issue 2**: Analysis takes too long
- Solution: Prioritize top 3 jumps instead of all jumps

**Issue 3**: Results don't match hypotheses
- Solution: Document unexpected findings (these are often most interesting!)

**Issue 4**: Computational budget exceeded
- Solution: Use smaller models or fewer checkpoints for validation

---

## Success Criteria

### Minimum Viable Product (MVP)

- [ ] Complete analysis of 1 phenomenon (CNN cascades)
- [ ] 3-4 publication-quality figures
- [ ] Draft of results section
- [ ] Code released on GitHub

### Full Success

- [ ] Complete analysis of all 3 phenomena
- [ ] 8-10 publication-quality figures
- [ ] Full paper draft
- [ ] Validated theoretical connections
- [ ] Code + documentation + tutorials released

### Stretch Goals

- [ ] Analysis extended to additional architectures (ResNets, ViTs)
- [ ] Real-time monitoring dashboard implemented
- [ ] Interactive visualization tool created
- [ ] Accepted at top-tier ML conference

---

## Conclusion

This plan provides a **concrete, actionable roadmap** for mechanistic interpretability analysis building on the Phase 1 and Phase 2 work. The key innovation is using dimensionality transitions as a guide for when to apply expensive interpretability techniques.

By systematically investigating three distinct phenomena (CNN cascades, transformer transitions, MLP smooth learning), we'll demonstrate the practical value of coarse-grained measurement infrastructure for making mechanistic interpretability research more efficient.

**Next Steps**:
1. Review and approve this plan
2. Set up infrastructure (checkpoints, analysis environment)
3. Begin with CNN cascade analysis (highest expected impact)
4. Iterate based on initial findings

---

**Document Version**: 1.0
**Last Updated**: 2025-11-20
**Status**: Ready for implementation
