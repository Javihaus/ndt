"""
Checkpoint Planning: Jump-Targeted Strategy

Based on Phase 1 jump analysis, this script identifies exact checkpoint
steps for Phase 2 investigations.

Strategy: Save 3 checkpoints (before/during/after) for 5 representative jumps
Total: 15 checkpoints per experiment × 3 experiments = 45 checkpoints

Selection Criteria:
- 2 Type 5 jumps (early training, highest magnitude)
- 2 Type 4 jumps (late training, high magnitude)
- 1 Type 3 jump (mid training, moderate magnitude)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Load Phase 1 jump data
phase1_dir = Path('/home/user/ndt/experiments/mechanistic_interpretability/results')
jump_data_file = phase1_dir / 'step1_2' / 'all_jumps_detailed.csv'

print("=" * 80)
print("CHECKPOINT PLANNING: Jump-Targeted Strategy")
print("=" * 80)

# Load jump data
df_jumps = pd.read_csv(jump_data_file)

print(f"\nLoaded {len(df_jumps)} jumps from Phase 1 analysis")


def identify_representative_jumps(experiment_name: str, df: pd.DataFrame,
                                  num_jumps: int = 5) -> List[Dict]:
    """
    Identify representative jumps for an experiment.

    Returns list of jump dictionaries with checkpoint steps.
    """
    exp_jumps = df[df['experiment'] == experiment_name].copy()

    if len(exp_jumps) == 0:
        print(f"  WARNING: No jumps found for {experiment_name}")
        return []

    print(f"\n  Total jumps: {len(exp_jumps)}")
    print(f"  Phase distribution: {exp_jumps['phase_category'].value_counts().to_dict()}")
    print(f"  Cluster distribution: {exp_jumps['cluster'].value_counts().to_dict()}")

    selected_jumps = []

    # Try to get jumps from different clusters
    # Clusters 0-4 identified in Phase 1
    # Let's select based on cluster AND phase

    # Type 5: Cluster 4 (largest, early)
    type5_candidates = exp_jumps[exp_jumps['cluster'] == 4].nlargest(2, 'magnitude')
    if len(type5_candidates) < 2:
        # Fallback: earliest 2 jumps
        type5_candidates = exp_jumps.nsmallest(2, 'phase')

    # Type 4: Cluster 3 (late training)
    type4_candidates = exp_jumps[exp_jumps['cluster'] == 3].nlargest(2, 'magnitude')
    if len(type4_candidates) < 2:
        # Fallback: latest 2 jumps
        type4_candidates = exp_jumps.nlargest(2, 'phase')

    # Type 3: Cluster 2 (mid training)
    type3_candidates = exp_jumps[exp_jumps['cluster'] == 2].sample(min(1, len(exp_jumps[exp_jumps['cluster'] == 2])))
    if len(type3_candidates) == 0:
        # Fallback: middle phase jump
        mid_phase = exp_jumps['phase'].median()
        type3_candidates = exp_jumps.iloc[[(exp_jumps['phase'] - mid_phase).abs().argmin()]]

    # Combine selections
    for idx, row in type5_candidates.iterrows():
        selected_jumps.append({
            'type': 'Type 5 (early, large)',
            'cluster': int(row['cluster']),
            'step': int(row['step']),
            'phase': float(row['phase']),
            'magnitude': float(row['magnitude']),
            'layer': row['layer'],
            'checkpoints': [
                max(0, int(row['step']) - 10),
                int(row['step']),
                int(row['step']) + 10
            ]
        })

    for idx, row in type4_candidates.iterrows():
        selected_jumps.append({
            'type': 'Type 4 (late, large)',
            'cluster': int(row['cluster']),
            'step': int(row['step']),
            'phase': float(row['phase']),
            'magnitude': float(row['magnitude']),
            'layer': row['layer'],
            'checkpoints': [
                max(0, int(row['step']) - 10),
                int(row['step']),
                int(row['step']) + 10
            ]
        })

    for idx, row in type3_candidates.iterrows():
        selected_jumps.append({
            'type': 'Type 3 (mid, moderate)',
            'cluster': int(row['cluster']),
            'step': int(row['step']),
            'phase': float(row['phase']),
            'magnitude': float(row['magnitude']),
            'layer': row['layer'],
            'checkpoints': [
                max(0, int(row['step']) - 10),
                int(row['step']),
                int(row['step']) + 10
            ]
        })

    # Sort by step
    selected_jumps = sorted(selected_jumps, key=lambda x: x['step'])

    # Limit to num_jumps
    selected_jumps = selected_jumps[:num_jumps]

    return selected_jumps


# ============================================================================
# EXPERIMENT 1: Transformer Deep
# ============================================================================

print("\n" + "=" * 80)
print("EXPERIMENT 1: transformer_deep_mnist")
print("=" * 80)

transformer_jumps = identify_representative_jumps('transformer_deep_mnist', df_jumps)

print("\nSelected Jumps:")
for i, jump in enumerate(transformer_jumps, 1):
    print(f"\n  Jump {i}: {jump['type']}")
    print(f"    Step: {jump['step']} (phase={jump['phase']:.2f})")
    print(f"    Layer: {jump['layer']}")
    print(f"    Magnitude: {jump['magnitude']:.6f}")
    print(f"    Checkpoints: {jump['checkpoints']}")

transformer_checkpoints = sorted(set(sum([j['checkpoints'] for j in transformer_jumps], [])))
print(f"\n  Total unique checkpoint steps: {len(transformer_checkpoints)}")
print(f"  Steps: {transformer_checkpoints}")


# ============================================================================
# EXPERIMENT 2: CNN Deep
# ============================================================================

print("\n" + "=" * 80)
print("EXPERIMENT 2: cnn_deep_mnist")
print("=" * 80)

cnn_jumps = identify_representative_jumps('cnn_deep_mnist', df_jumps)

print("\nSelected Jumps:")
for i, jump in enumerate(cnn_jumps, 1):
    print(f"\n  Jump {i}: {jump['type']}")
    print(f"    Step: {jump['step']} (phase={jump['phase']:.2f})")
    print(f"    Layer: {jump['layer']}")
    print(f"    Magnitude: {jump['magnitude']:.6f}")
    print(f"    Checkpoints: {jump['checkpoints']}")

cnn_checkpoints = sorted(set(sum([j['checkpoints'] for j in cnn_jumps], [])))
print(f"\n  Total unique checkpoint steps: {len(cnn_checkpoints)}")
print(f"  Steps: {cnn_checkpoints}")


# ============================================================================
# EXPERIMENT 3: MLP Narrow
# ============================================================================

print("\n" + "=" * 80)
print("EXPERIMENT 3: mlp_narrow_mnist")
print("=" * 80)

mlp_jumps = identify_representative_jumps('mlp_narrow_mnist', df_jumps)

print("\nSelected Jumps:")
for i, jump in enumerate(mlp_jumps, 1):
    print(f"\n  Jump {i}: {jump['type']}")
    print(f"    Step: {jump['step']} (phase={jump['phase']:.2f})")
    print(f"    Layer: {jump['layer']}")
    print(f"    Magnitude: {jump['magnitude']:.6f}")
    print(f"    Checkpoints: {jump['checkpoints']}")

mlp_checkpoints = sorted(set(sum([j['checkpoints'] for j in mlp_jumps], [])))
print(f"\n  Total unique checkpoint steps: {len(mlp_checkpoints)}")
print(f"  Steps: {mlp_checkpoints}")


# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("CHECKPOINT PLAN SUMMARY")
print("=" * 80)

total_checkpoints = len(transformer_checkpoints) + len(cnn_checkpoints) + len(mlp_checkpoints)

print(f"\nTotal checkpoints across 3 experiments: {total_checkpoints}")
print(f"  transformer_deep_mnist: {len(transformer_checkpoints)} checkpoints")
print(f"  cnn_deep_mnist: {len(cnn_checkpoints)} checkpoints")
print(f"  mlp_narrow_mnist: {len(mlp_checkpoints)} checkpoints")

# Save checkpoint plan
checkpoint_plan = {
    'transformer_deep_mnist': {
        'jumps': transformer_jumps,
        'checkpoint_steps': transformer_checkpoints
    },
    'cnn_deep_mnist': {
        'jumps': cnn_jumps,
        'checkpoint_steps': cnn_checkpoints
    },
    'mlp_narrow_mnist': {
        'jumps': mlp_jumps,
        'checkpoint_steps': mlp_checkpoints
    }
}

output_file = phase1_dir.parent / 'checkpoint_plan.json'
with open(output_file, 'w') as f:
    json.dump(checkpoint_plan, f, indent=2)

print(f"\nCheckpoint plan saved to: {output_file}")


# ============================================================================
# INTEGRATION CODE SNIPPETS
# ============================================================================

print("\n" + "=" * 80)
print("INTEGRATION CODE SNIPPETS")
print("=" * 80)

print("\n1. Add to your training loop:")
print("""
```python
import torch
from pathlib import Path

# Load checkpoint plan
with open('checkpoint_plan.json', 'r') as f:
    checkpoint_plan = json.load(f)

experiment_name = 'transformer_deep_mnist'  # or cnn_deep_mnist, mlp_narrow_mnist
checkpoint_steps = set(checkpoint_plan[experiment_name]['checkpoint_steps'])
checkpoint_dir = Path(f'checkpoints/{experiment_name}')
checkpoint_dir.mkdir(parents=True, exist_ok=True)

# In training loop
for step in range(num_steps):
    # ... training code ...

    if step in checkpoint_steps:
        checkpoint_path = checkpoint_dir / f'step_{step:05d}.pt'
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
            'metrics': metrics  # any other info
        }, checkpoint_path)
        print(f'Saved checkpoint at step {step}')
```
""")

print("\n2. Load checkpoint for analysis:")
print("""
```python
def load_checkpoint(experiment_name, step):
    checkpoint_path = Path(f'checkpoints/{experiment_name}/step_{step:05d}.pt')
    checkpoint = torch.load(checkpoint_path)

    # Reconstruct model
    model = YourModelClass()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, checkpoint

# Example usage
model_before, info_before = load_checkpoint('transformer_deep_mnist', 40)
model_after, info_after = load_checkpoint('transformer_deep_mnist', 60)

# Now extract attention, filters, activations, etc.
```
""")

print("\n3. Extract attention patterns (Transformer):")
print("""
```python
def extract_attention_patterns(model, dataloader, device='cuda'):
    \"\"\"Extract attention weights from transformer.\"\"\"
    attention_patterns = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input_ids'].to(device)
            outputs = model(inputs, output_attentions=True)

            # outputs.attentions is tuple of attention tensors
            # Each: [batch, num_heads, seq_len, seq_len]
            attention_patterns.append([
                attn.cpu().numpy() for attn in outputs.attentions
            ])

    return attention_patterns

# Usage
attn_before = extract_attention_patterns(model_before, test_loader)
attn_after = extract_attention_patterns(model_after, test_loader)

# Measure change
from phase2_infrastructure import MeasurementTools
measurements = MeasurementTools()

entropy_before = measurements.attention_entropy(attn_before[0][0])  # First layer, first batch
entropy_after = measurements.attention_entropy(attn_after[0][0])

print(f"Entropy change: {entropy_after - entropy_before:.3f}")
# Negative = more focused (specialized)
```
""")

print("\n4. Extract CNN filters:")
print("""
```python
def extract_conv_filters(model):
    \"\"\"Extract all convolutional layer filters.\"\"\"
    filters = {}

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            filters[name] = module.weight.data.clone()

    return filters

# Usage
filters_before = extract_conv_filters(model_before)
filters_after = extract_conv_filters(model_after)

# Measure change
from phase2_infrastructure import MeasurementTools
measurements = MeasurementTools()

for layer_name in filters_before.keys():
    change = measurements.weight_change_magnitude(
        filters_before[layer_name],
        filters_after[layer_name]
    )
    print(f"{layer_name}:")
    print(f"  Cosine similarity: {change['cosine_similarity']:.3f}")
    print(f"  Relative change: {change['relative_change']:.3f}")
```
""")

print("\n5. Extract MLP activations:")
print("""
```python
def extract_activations(model, dataloader, layer_names, device='cuda'):
    \"\"\"Hook into layers to extract activations.\"\"\"
    activations = {name: [] for name in layer_names}

    def get_activation(name):
        def hook(model, input, output):
            activations[name].append(output.detach().cpu())
        return hook

    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if name in layer_names:
            hooks.append(module.register_forward_hook(get_activation(name)))

    # Run inference
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input'].to(device)
            _ = model(inputs)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Concatenate batches
    for name in activations:
        activations[name] = torch.cat(activations[name], dim=0).numpy()

    return activations

# Usage
layer_names = ['network.0', 'network.4', 'network.8']
acts_before = extract_activations(model_before, test_loader, layer_names)
acts_after = extract_activations(model_after, test_loader, layer_names)

# Measure selectivity change
labels = test_dataset.targets.numpy()
for layer in layer_names:
    sel_before = measurements.activation_selectivity(acts_before[layer], labels)
    sel_after = measurements.activation_selectivity(acts_after[layer], labels)

    # Compute change in selectivity
    selectivity_change = np.linalg.norm(sel_after - sel_before)
    print(f"{layer}: Selectivity change = {selectivity_change:.3f}")
```
""")

print("\n" + "=" * 80)
print("✓ Checkpoint plan generated successfully!")
print("=" * 80)
print("\nNext steps:")
print("1. Review checkpoint_plan.json for exact steps")
print("2. Integrate checkpoint saving into training scripts")
print("3. Re-run 3 experiments with checkpoint saving")
print("4. Use phase2_infrastructure.py to analyze checkpoints")
