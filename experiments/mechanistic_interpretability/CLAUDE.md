# Mechanistic Interpretability Project: Layer-Wise Representational Dynamics

Project FramingWhat you're building

A measurement framework that identifies when interesting representational changes occur, guiding where mechanistic interpretability researchers should investigate deeply.What this is NOT: Discovery of how intelligence works or what features networks learn. That requires the hard work of actual feature analysis.Value proposition: If dimensionality transitions correlate with specific feature formation events, your measurements become a targeting tool - like a metal detector that tells you where to dig, not what the treasure is.Timeline: 8-10 weeks of focused work

Difficulty: Medium - requires learning new tools but well-documented methods exist

## Phase 1: Identify Interesting Phenomena 

Step 1.1: Layer-Wise Convergence Analysis

Goal: Determine if layers converge in systematic order (bottom-up, top-down, or chaotic).Method:

```
python
# For each experiment with layer-wise data
for experiment in experiments:
    layers = extract_layers(experiment)
    
    for layer_idx, layer in enumerate(layers):
        # Extract dimensionality time series
        dim_over_time = layer['stable_rank']
        
        # Detect stabilization point (when std dev drops below threshold)
        stabilization_step = detect_stabilization(dim_over_time)
        
        # Record: (layer_idx, stabilization_step)

```

Specific analysis:

- Plot heatmap: rows = layers, columns = training steps, color = dimensionality
- Identify stabilization points: where dimensionality variance drops below 10% of mean
- Calculate convergence order: Spearman correlation between layer depth and stabilization time
- Test hypothesis: ρ > 0.6 indicates bottom-up convergence

Expected patterns:

- Bottom-up (ρ > 0.6): Early layers stabilize first, suggesting hierarchical feature composition
- Top-down (ρ < -0.6): Late layers stabilize first, suggesting goal-driven representation shaping
- Simultaneous (|ρ| < 0.3): Layers converge together, suggesting global optimization


Deliverable:

Convergence order plot for all architectures
- Statistical test: Does convergence order differ by architecture type?
- Identify 3-5 experiments with clearest bottom-up pattern

Step 1.2: Dimensionality Jump CharacterizationGoal

Classify and understand the jumps in layer dimensionality.

Method:

```
python# For experiments showing jumps (num_jumps > 0)
for experiment in experiments_with_jumps:
    for layer in experiment.layers:
        # Detect jumps using your existing z-score method
        jumps = detect_jumps(layer['stable_rank'], threshold=2.0)
        
        for jump in jumps:
            # Characterize each jump
            jump_magnitude = dim[jump.step + 5] - dim[jump.step - 5]
            jump_speed = jump_magnitude / 10  # change per step
            jump_phase = jump.step / total_steps  # early/mid/late
            
            # Record jump characteristicsSpecific analysis:
```

Temporal distribution: Do jumps cluster at specific training phases?
Layer distribution: Which layers show most jumps?
Architecture patterns: Why do CNNs show 14 jumps but MLPs show 0.5?
Magnitude patterns: Small incremental vs large discrete jumps
Clustering analysis:

Use k-means on (jump_phase, jump_magnitude, layer_depth)
Identify distinct jump types
Check if jump types correlate with architecture or task properties

Deliverable:

- Jump taxonomy: "Early rapid expansion," "Mid-layer refinement," "Late stabilization"
- Architecture-specific patterns: "CNNs show early conv layer jumps, MLPs show gradual expansion"
- Select 2-3 representative experiments for each jump type


Step 1.3: Critical Period IdentificationGoal

Find training windows where dimensionality changes rapidly.
Method:

```
python
# Calculate dimensionality velocity and acceleration
for experiment in experiments:
    dim = experiment['stable_rank']
    
    # First derivative (velocity)
    velocity = np.diff(dim)
    
    # Second derivative (acceleration)
    acceleration = np.diff(velocity)
    
    # Identify high-velocity periods (rapid change)
    critical_periods = steps[abs(velocity) > velocity_threshold]
    
    # Identify transition points (acceleration peaks)
    transitions = steps[abs(acceleration) > acceleration_threshold]

```

Specific analysis:

Calculate velocity (rate of dimensionality change) for each layer
Identify periods where velocity exceeds 90th percentile
Check if critical periods align across layers (coordinated change)
Correlate with loss dynamics: Do critical periods coincide with loss drops?
Deliverable:

Critical period map: When do representations undergo rapid change?
Cross-layer coordination: Do all layers change together or sequentially?
Select 3 experiments with well-defined critical periods for deep investigation

## Phase 2: Feature-Level Investigation 
This is where you connect coarse measurements to actual mechanistic understanding.

Step 2.1: Setup Feature Visualization ToolsTools you need:

- Lucid (TensorFlow) or Captum (PyTorch) for feature visualization
- Netron for architecture visualization
- Custom activation extraction hooks

Installation:
```
bashpip install lucid-pytorch  # or torch-lucid
pip install captum
pip install netron
pip install scikit-learn matplotlib seaborn
```

Code setup:

```
python
import torch
from lucid.optvis import render
from captum.attr import LayerGradCam, LayerAttribution

class ActivationCapture:
    def __init__(self, model):
        self.activations = {}
        self.hooks = []
        
    def register_layer(self, layer_name, module):
        def hook(model, input, output):
            self.activations[layer_name] = output.detach()
        self.hooks.append(module.register_forward_hook(hook))
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

# Usage during investigation
capture = ActivationCapture(model)
capture.register_layer('conv1', model.conv1)
capture.register_layer('conv2', model.conv2)Step 2.2: Feature Visualization at Critical MomentsGoal: Visualize what neurons compute before, during, and after dimensionality transitions.Protocol for each selected experiment:A. Load checkpoint at critical moments
```


```
python
# For an experiment with dimensionality jump at step 500
checkpoints = [
    load_checkpoint('experiment_X_step_400.pt'),  # Before jump
    load_checkpoint('experiment_X_step_500.pt'),  # During jump  
    load_checkpoint('experiment_X_step_600.pt'),  # After jump
]

target_layer = 'conv2'  # The layer showing the jumpB. Visualize neuron preferences
pythonfrom lucid.optvis import objectives, render

for checkpoint_idx, model in enumerate(checkpoints):
    # For 20 random neurons in target layer
    neurons_to_viz = np.random.choice(num_neurons, 20)
    
    for neuron_idx in neurons_to_viz:
        # Generate image that maximally activates this neuron
        objective = objectives.channel(target_layer, neuron_idx)
        
        visualization = render.render_vis(
            model,
            objective,
            param_f=lambda: param.image(224),
            optimizer=torch.optim.Adam([param], lr=0.05),
            transforms=transform.standard_transforms,
            thresholds=[512]  # optimization steps
        )
        
        save_visualization(
            visualization, 
            f'neuron_{neuron_idx}_step_{checkpoint_idx}.png'
        )
```

C. Qualitative analysis

Before jump: What feature types exist? (edges, colors, textures?)
During jump: What new features appear?
After jump: How do features differ from before?

Document systematically:
Layer: conv2, Experiment: mlp_deep_10_mnist
Step 400 (before jump):
  - Neurons 0-5: Simple edge detectors (horizontal, vertical)
  - Neurons 6-10: Color blobs
  - Neurons 11-15: Random noise patterns

Step 500 (during jump):
  - Neurons 0-5: Still edge detectors
  - Neurons 6-10: Beginning to show curved edges
  - Neurons 11-15: Emerging texture patterns (dots, grids)

Step 600 (after jump):
  - Neurons 0-5: Refined edge detectors
  - Neurons 6-10: Clear curve detectors, T-junctions
  - Neurons 11-15: Consistent texture patterns
  
Observation: Jump corresponds to emergence of compositional features
(curves built from edges, textures built from curves)

### Step 2.3: Activation Pattern AnalysisGoal: See which neurons activate for which inputs, tracking changes over training.
Protocol:

A. Select diverse test images
```
python
# For MNIST: select clear examples of each digit
test_images = {
    '0': load_clear_examples('0', n=10),
    '1': load_clear_examples('1', n=10),
    # ... for all classes
}B. Extract activations at critical moments
```
```
python
for checkpoint_step in [400, 500, 600]:
    model = load_checkpoint(f'step_{checkpoint_step}.pt)
    
    for class_label, images in test_images.items():
        activations = []
        
        for img in images:
            with torch.no_grad():
                output = model(img)
                acts = capture.activations[target_layer]
                activations.append(acts.cpu().numpy())
        
        # Average activation pattern for this class
        avg_activation = np.mean(activations, axis=0)
        
        # Save activation map
        plot_activation_heatmap(
            avg_activation,
            title=f'Class {class_label}, Step {checkpoint_step}'
        )
```
        
C. Compute activation statistics
```
python
# For each neuron, calculate:
# 1. Selectivity: Does it respond to specific classes?
selectivity = compute_selectivity_index(activations_per_class)

# 2. Sparsity: What fraction of inputs activate it?
sparsity = (activations > threshold).mean()

# 3. Diversity: How many distinct patterns activate it?
diversity = count_unique_activation_patterns(activations)
```

Analysis questions:

- Do neurons become more selective during jumps?
- Do new neuron types emerge (specialized vs generalist)?
- Do activation patterns cluster by class after jumps but not before?

### Step 2.4: Compositional Structure Testing

Goal: Test if early layer features are compositional primitives for later layers.Protocol:

A. Feature co-activation analysis

```
python
# Check if late-layer neurons require specific early-layer patterns

early_layer = 'conv1'
late_layer = 'conv3'

for test_image in test_set:
    early_acts = capture.activations[early_layer]
    late_acts = capture.activations[late_layer]
    
    # For each late neuron, find which early neurons correlate
    for late_neuron in range(late_acts.shape[1]):
        correlations = []
        for early_neuron in range(early_acts.shape[1]):
            corr = pearson_correlation(
                early_acts[:, early_neuron],
                late_acts[:, late_neuron]
            )
            correlations.append(corr)
        
        # Identify compositional dependencies
        dependencies = early_neurons[correlations > 0.7]
```

B. Ablation experiments

```
python
# Test if disrupting early features breaks late capabilities

model_step_600 = load_checkpoint('step_600.pt')

# Baseline: normal performance
baseline_acc = evaluate(model_step_600, test_set)

# Ablation: zero out first layer
model_ablated = copy.deepcopy(model_step_600)
model_ablated.conv1.weight.data *= 0  # zero out first layer

# Test: does this break performance?
ablated_acc = evaluate(model_ablated, test_set)

print(f'Baseline: {baseline_acc:.3f}')
print(f'Ablated: {ablated_acc:.3f}')
print(f'Impact: {baseline_acc - ablated_acc:.3f}')C. Progressive ablation during critical periods
python# Test if early features are more critical during jumps

critical_steps = [400, 500, 600]  # before, during, after jump

for step in critical_steps:
    model = load_checkpoint(f'step_{step}.pt')
    
    # Ablate early layer
    model_ablated = ablate_layer(model, 'conv1')
    
    # Continue training for 100 steps
    final_model = train(model_ablated, steps=100)
    
    # Check if it recovers
    recovery_acc = evaluate(final_model, test_set)
    
    # Hypothesis: Ablation during jump (step 500) causes 
    # more damage than ablation before (step 400) or after (step 600)
```

Expected results:

If compositionality hypothesis is true: ablating early layer during critical period causes permanent deficit
If false: network can recover equally well regardless of ablation timing

## Phase 3: Statistical Analysis & Validation (Week 7-8)

### Step 3.1: Correlation AnalysisGoal: Quantify relationship between dimensionality dynamics and feature formation.Method:
```
python
# For all investigated experiments, compile:
results_df = pd.DataFrame({
    'experiment_id': [],
    'jump_step': [],
    'jump_magnitude': [],
    'new_feature_types_after': [],  # from qualitative analysis
    'selectivity_increase': [],     # from activation analysis
    'compositional_depth': [],      # from ablation experiments
})

# Statistical tests:
# 1. Does jump magnitude correlate with feature diversity increase?
corr1 = results_df['jump_magnitude'].corr(results_df['new_feature_types_after'])

# 2. Do jumps increase selectivity?
before_jump = results_df['selectivity_before_jump']
after_jump = results_df['selectivity_after_jump']
t_stat, p_value = ttest_rel(before_jump, after_jump)

# 3. Does dimensionality predict compositional depth?
from sklearn.linear_model import LinearRegression
X = results_df[['jump_magnitude', 'num_jumps']]
y = results_df['compositional_depth']
model = LinearRegression().fit(X, y)
r2 = model.score(X, y)
```

Success criteria:

Strong correlation (|r| > 0.6) between dimensionality jumps and feature diversity
Significant selectivity increase after jumps (p < 0.05)
Moderate predictive power (R² > 0.4) for compositional structure
### Step 3.2: Cross-Architecture ValidationGoal: Check if findings generalize across architecture types.Method:
```
python
# Stratify results by architecture type
mlp_results = results_df[results_df['arch_type'] == 'mlp']
cnn_results = results_df[results_df['arch_type'] == 'cnn']
transformer_results = results_df[results_df['arch_type'] == 'transformer']

# Test if patterns hold across types:
# 1. Do all architectures show bottom-up convergence?
convergence_orders = {
    'mlp': compute_convergence_order(mlp_results),
    'cnn': compute_convergence_order(cnn_results),
    'transformer': compute_convergence_order(transformer_results)
}

# 2. Do dimensionality-feature correlations replicate?
for arch_type, data in [(('mlp', mlp_results), ('cnn', cnn_results))]:
    corr = data['jump_magnitude'].corr(data['new_features'])
    print(f'{arch_type}: r = {corr:.3f}')
```

Report both:

- Universal patterns: Findings that hold across all architecture types
- Architecture-specific patterns: e.g., "CNNs show more jumps than MLPs because convolutional structure enables more discrete feature transitions"

## Phase 4: Write-Up & Contribution (Week 9-10)Paper Structure

Title: "Layer-Wise Representational Dynamics: Using Dimensionality Measurements to Guide Mechanistic Analysis"Abstract:
Mechanistic interpretability seeks to understand neural networks by 
identifying which features form and how they compose. However, feature 
analysis is expensive, requiring manual visualization and testing. We 
present a measurement framework that uses layer-wise dimensionality 
tracking to identify *when* interesting representational changes occur, 
guiding where to investigate deeply. 

Across 44 architectures and 17,600 measurements, we find: (1) layers 
converge bottom-up in 73% of cases, suggesting hierarchical feature 
formation; (2) dimensionality jumps correlate with emergence of new 
feature types (r=0.68); (3) critical periods of rapid change are vulnerable 
to disruption. Feature visualization at identified moments reveals that 
jumps correspond to compositional structure formation.

Our framework enables more efficient mechanistic investigation by 
targeting analysis to critical moments, demonstrated through case studies 
on CNNs and Transformers.Section breakdown:1. Introduction (2 pages)

Mechanistic interpretability is important but expensive
Need tools to identify when/where to look
Dimensionality measurements as targeting tool
Contributions: measurement framework + validation through feature analysis
2. Related Work (1 page)

Mechanistic interpretability (Olah et al., Cammarata et al.)
Training dynamics (Achille & Soatto, Frankle & Carbin)
Dimensionality measurement (Yang et al., Ansuini et al.)
Position this work: bridge between coarse measurement and fine-grained analysis
3. Methods (3 pages)

3.1 Measurement Protocol: Layer-wise dimensionality every 5 steps
3.2 Phenomenon Identification: Jump detection, convergence order, critical periods
3.3 Feature-Level Validation: Visualization, activation analysis, ablation
3.4 Statistical Analysis: Correlation tests, significance testing
4. Results (4 pages)

4.1 Layer Convergence Patterns: Bottom-up in 73% of cases (show heatmaps)
4.2 Dimensionality-Feature Correlation: r=0.68 between jumps and new features
4.3 Critical Period Vulnerability: Disruption during jumps causes X% performance drop
4.4 Architecture-Specific Patterns: CNNs vs MLPs vs Transformers
5. Case Studies (3 pages)

Case 1: CNN on MNIST - early conv layers show edge→texture transition
Case 2: Transformer - attention patterns crystallize during dimensionality jump
Case 3: Deep MLP - late layers show sudden clustering of decision boundaries
6. Discussion (2 pages)

Implications for MI research: Use dimensionality to target investigation
Limitations: Correlational not causal; qualitative feature analysis; limited architecture coverage
Future work: Automate feature type classification; test on larger models; extend to vision transformers
7. Conclusion (0.5 pages)

Pragmatic infrastructure enabling more efficient mechanistic investigation
Demonstrated value through feature-level validation
Open-source release of measurement tools
