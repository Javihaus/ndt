# Phase 2 Infrastructure Documentation

## Overview

The Phase 2 infrastructure provides tools for mechanistic analysis of neural network training dynamics. It supports three main investigation types:

1. **Transformer Analysis** - Attention pattern extraction and specialization measurement
2. **CNN Analysis** - Filter visualization and activation analysis
3. **MLP Analysis** - Hidden unit dynamics and selectivity measurement

## Architecture

```
phase2_infrastructure.py
├── Data Structures
│   ├── CriticalMoment - Represents a moment to investigate
│   └── AnalysisResult - Container for analysis results
├── CheckpointManager - Loads data at critical moments
├── MeasurementTools - Quantitative analysis functions
├── VisualizationTools - Plotting and visualization
├── BeforeAfterComparison - Framework for comparing moments
└── Phase2Infrastructure - Main interface
```

## Key Features

### 1. PyTorch-Optional Design
- Works without PyTorch using NumPy (for Phase 1 data analysis)
- Full PyTorch support when model checkpoints are available
- Automatic detection and graceful degradation

### 2. Critical Moment Identification
Automatically identifies interesting moments based on Phase 1 findings:
- Jump locations (before/during/after)
- Critical periods with high coordination
- Moments with strong loss correlation

### 3. Measurement Functions

#### Attention Analysis (Transformers)
```python
# Attention entropy - measures focus vs distribution
entropy = measurements.attention_entropy(attention_weights)

# Head specialization - measures how different heads are
specialization = measurements.attention_specialization(attention_weights)
```

#### Weight/Activation Analysis (All architectures)
```python
# Cosine similarity - measures change between moments
similarity = measurements.cosine_similarity(weights_before, weights_after)

# Weight change magnitude
change_stats = measurements.weight_change_magnitude(weights_before, weights_after)

# Activation sparsity - measures feature selectivity
sparsity = measurements.activation_sparsity(activations)

# Class selectivity - which neurons respond to which classes
selectivity = measurements.activation_selectivity(activations, labels)
```

### 4. Visualization Tools

- `plot_attention_pattern()` - Heatmap of single attention head
- `plot_attention_heads_comparison()` - Before/after comparison
- `plot_filter_visualization()` - CNN filter visualization
- `plot_activation_heatmap()` - Activation patterns
- `plot_entropy_evolution()` - Entropy over training

## Usage

### Basic Usage

```python
from pathlib import Path
from phase2_infrastructure import Phase2Infrastructure

# Initialize
experiments_dir = Path('/home/user/ndt/experiments')
infra = Phase2Infrastructure(experiments_dir)

# Get experiment summary
summary = infra.get_experiment_summary('transformer_deep_mnist')
print(f"Experiment has {summary['num_jumps']} jumps")

# Identify critical moments
moments = infra.identify_moments('transformer_deep_mnist', num_jumps=5)

# Compare two moments
result = infra.compare_representations(moments[0], moments[2])
print(f"Loss changed by: {result.measurements['loss_change']:.4f}")

# Access layer-wise changes
for layer, changes in result.measurements['layer_changes'].items():
    print(f"{layer}: {changes['change']:.3f}")
```

### Advanced: Direct Tool Usage

```python
from phase2_infrastructure import MeasurementTools, VisualizationTools
import numpy as np

measurements = MeasurementTools()
viz = VisualizationTools()

# Analyze attention patterns (example with dummy data)
attention = np.random.rand(32, 8, 64, 64)  # [batch, heads, seq, seq]
entropy = measurements.attention_entropy(attention)
specialization = measurements.attention_specialization(attention)

print(f"Mean entropy: {specialization['mean_entropy']:.3f}")
print(f"Specialization index: {specialization['specialization_index']:.3f}")

# Compare weight changes
weights_before = np.random.randn(64, 128)
weights_after = weights_before + np.random.randn(64, 128) * 0.1
changes = measurements.weight_change_magnitude(weights_before, weights_after)

print(f"Cosine similarity: {changes['cosine_similarity']:.3f}")
print(f"Relative change: {changes['relative_change']:.3f}")
```

## Workflow for Phase 2 Investigations

### Week 3-4: Transformer Deep Dive

```python
# 1. Select experiment
experiment = 'transformer_deep_mnist'

# 2. Identify representative jumps
moments = infra.identify_moments(experiment, num_jumps=5)

# 3. For each jump, analyze before/after
for i in range(0, len(moments), 3):  # Every 3rd is a complete before/during/after
    before = moments[i]
    after = moments[i+2]

    # Compare representations
    result = infra.compare_representations(before, after)

    # Extract dimensionality changes
    layer_changes = result.measurements['layer_changes']

    # When checkpoints are available:
    # - Load model states
    # - Extract attention patterns
    # - Measure entropy change
    # - Visualize attention heads
```

### Week 5-6: CNN vs MLP Comparison

```python
# Compare CNN and MLP
cnn_moments = infra.identify_moments('cnn_deep_mnist', num_jumps=3)
mlp_moments = infra.identify_moments('mlp_narrow_mnist', num_jumps=3)

# Analyze jump discreteness
for cnn_m, mlp_m in zip(cnn_moments, mlp_moments):
    # Load measurements
    cnn_data = infra.checkpoint_manager.load_measurements_at_step(
        cnn_m.experiment_name, cnn_m.step
    )
    mlp_data = infra.checkpoint_manager.load_measurements_at_step(
        mlp_m.experiment_name, mlp_m.step
    )

    # When checkpoints available:
    # - Visualize CNN filters
    # - Measure filter differentiation
    # - Compare to MLP hidden unit evolution
```

## Data Structures

### CriticalMoment
```python
@dataclass
class CriticalMoment:
    experiment_name: str      # e.g., 'transformer_deep_mnist'
    step: int                 # Training step number
    moment_type: str          # 'before_jump', 'during_jump', 'after_jump'
    jump_info: Optional[Dict] # Jump metadata from Phase 1
```

### AnalysisResult
```python
@dataclass
class AnalysisResult:
    experiment: str           # Experiment name
    moment: str               # Description of moment
    measurements: Dict        # Quantitative measurements
    visualizations: Dict      # Generated plots
```

## Measurement Details

### Attention Entropy
- **High entropy** (~4.0): Attention distributed uniformly
- **Low entropy** (~0.0): Attention highly focused
- **Interpretation**: Jump → entropy decrease suggests specialization

### Cosine Similarity
- **Value 1.0**: Identical representations
- **Value 0.0**: Orthogonal representations
- **Value -1.0**: Opposite representations
- **Interpretation**: Low similarity → discrete change

### Specialization Index
- **High value**: Attention heads have different entropy (specialized)
- **Low value**: All heads similar entropy (uniform)
- **Interpretation**: Jump → increased specialization index

## Extending the Infrastructure

### Adding New Measurements

```python
class MeasurementTools:
    @staticmethod
    def my_new_measurement(data: np.ndarray) -> float:
        """
        Docstring describing what this measures.

        Args:
            data: Input data

        Returns:
            Measurement value
        """
        # Convert to numpy if needed
        data = MeasurementTools._to_numpy(data)

        # Your analysis here
        result = np.some_function(data)

        return float(result)
```

### Adding New Visualizations

```python
class VisualizationTools:
    @staticmethod
    def plot_my_visualization(data: np.ndarray,
                             save_path: Optional[Path] = None):
        """
        Docstring describing the visualization.
        """
        # Convert to numpy
        data = MeasurementTools._to_numpy(data) if TORCH_AVAILABLE else data

        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(data)
        plt.title('My Visualization')

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
```

## Limitations and Future Work

### Current Limitations
1. **No Model Checkpoints**: Phase 1 data contains only measurements, not model weights
2. **No Real Attention Patterns**: Transformer analysis currently limited to dimensionality
3. **No Activation Data**: CNN/MLP analysis limited without forward passes

### Required for Full Phase 2
To complete the Phase 2 investigations as planned, you'll need to:

1. **Re-run experiments with checkpoint saving**:
   ```python
   # During training, save model states at critical moments
   if step in critical_steps:
       torch.save(model.state_dict(), f'checkpoint_step_{step}.pt')
   ```

2. **Save activation data**:
   ```python
   # Hook into forward pass to save activations
   activations = {}
   def hook_fn(module, input, output):
       activations[module_name] = output.detach()
   ```

3. **For Transformers, save attention weights**:
   ```python
   # Most transformer implementations expose attention weights
   outputs = model(input, output_attentions=True)
   attention_weights = outputs.attentions  # List of attention tensors
   ```

### Workarounds for Now
Without checkpoints, you can still:
- ✓ Identify critical moments from Phase 1
- ✓ Analyze dimensionality changes
- ✓ Compare loss dynamics
- ✓ Measure cross-layer coordination
- ✓ Test measurement functions with synthetic data
- ✓ Develop analysis protocols

## Testing

The infrastructure includes a test suite in the main block:

```bash
cd /home/user/ndt/experiments/mechanistic_interpretability
python3 phase2_infrastructure.py
```

Expected output:
```
Note: PyTorch not available. Some features will be limited.
Initializing Phase 2 Infrastructure...
Testing with transformer_deep_mnist...
Experiment Summary:
  Jumps: 174
  Mean magnitude: 0.000000
Identifying critical moments...
Identified 9 critical moments
Comparing first two moments...
Loss change: 0.1298
Layers analyzed: 14
✓ Infrastructure initialized and tested successfully!
```

## Integration with Phase 1

The infrastructure automatically loads Phase 1 results:
- Jump characterization data from `step1_2/all_jumps_detailed.csv`
- Critical periods from `step1_3/critical_periods_detailed.csv`
- Uses this data to identify the most interesting moments to investigate

## Support and Troubleshooting

### Common Issues

**Issue**: "PyTorch not available"
- **Solution**: This is expected. Infrastructure works without PyTorch for Phase 1 data.
- **Action**: Install PyTorch only when working with model checkpoints

**Issue**: "Jump data not found"
- **Solution**: Ensure Phase 1 analysis was completed
- **Action**: Run step1_2 and step1_3 scripts first

**Issue**: "Checkpoint not available"
- **Solution**: Phase 1 data doesn't include checkpoints
- **Action**: Re-run experiments with checkpoint saving, or work with dimensionality data only

### Contact
For questions or issues, refer to the main CLAUDE.md in the experiments directory.

## Example: Complete Analysis Workflow

```python
#!/usr/bin/env python3
"""
Example: Analyzing transformer attention dynamics during jumps
"""

from pathlib import Path
from phase2_infrastructure import Phase2Infrastructure

def analyze_transformer_jumps():
    # Initialize
    infra = Phase2Infrastructure(Path('/home/user/ndt/experiments'))

    # Select experiment
    experiment = 'transformer_deep_mnist'

    # Get summary
    summary = infra.get_experiment_summary(experiment)
    print(f"Analyzing {experiment}")
    print(f"Total jumps: {summary['num_jumps']}")

    # Identify top 5 jumps
    moments = infra.identify_moments(experiment, num_jumps=5)

    # Analyze each jump
    results = []
    for i in range(0, len(moments), 3):
        if i+2 >= len(moments):
            break

        before = moments[i]
        after = moments[i+2]

        print(f"\nJump at step {moments[i+1].step}")

        # Compare
        result = infra.compare_representations(before, after)

        if result:
            print(f"  Loss change: {result.measurements['loss_change']:.4f}")

            # Count layers with significant change
            significant = sum(
                1 for changes in result.measurements['layer_changes'].values()
                if abs(changes['relative_change']) > 0.01
            )
            print(f"  Layers changed significantly: {significant}")

            results.append(result)

    return results

if __name__ == '__main__':
    results = analyze_transformer_jumps()
    print(f"\n✓ Analyzed {len(results)} jumps")
```

## Files Generated

When using the infrastructure, expect these files:
- `*_attention_pattern.png` - Attention heatmaps
- `*_filter_viz.png` - CNN filter visualizations
- `*_entropy_evolution.png` - Entropy over time
- `analysis_results.json` - Quantitative measurements
- `comparison_report.md` - Before/after comparison summary

## Next Steps

1. **Test infrastructure with Phase 1 data** ✓
2. **Design checkpoint saving strategy** - Determine which steps to save
3. **Re-run key experiments** - With checkpoint saving enabled
4. **Begin Week 3-4 analysis** - Transformer attention specialization
5. **Implement additional measurements** - As hypotheses emerge
