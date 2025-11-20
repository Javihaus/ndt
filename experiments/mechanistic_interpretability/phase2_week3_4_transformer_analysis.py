"""
Phase 2: Week 3-4 - Transformer Deep Dive

Hypothesis: Transformer jumps represent attention head specialization

Investigation:
1. Analyze dimensionality dynamics around identified jumps
2. Measure cross-layer coordination during jumps
3. Compare early vs late transformer jumps
4. When checkpoints available: Extract attention patterns and measure entropy

Target: transformer_deep_mnist with 5 representative jumps
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Academic plot style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'figure.titlesize': 16,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})
sns.set_palette("colorblind")

# Load infrastructure
from phase2_infrastructure import Phase2Infrastructure, MeasurementTools

print("=" * 80)
print("PHASE 2: TRANSFORMER DEEP DIVE (Week 3-4)")
print("=" * 80)
print("\nHypothesis: Jumps represent attention head specialization")
print("Target: transformer_deep_mnist")
print()

# ============================================================================
# SETUP
# ============================================================================

experiments_dir = Path('/home/user/ndt/experiments')
results_dir = Path('/home/user/ndt/experiments/mechanistic_interpretability/results/phase2_week3_4')
results_dir.mkdir(parents=True, exist_ok=True)

# Load checkpoint plan
checkpoint_plan_file = Path('/home/user/ndt/experiments/mechanistic_interpretability/checkpoint_plan.json')
with open(checkpoint_plan_file, 'r') as f:
    checkpoint_plan = json.load(f)

transformer_plan = checkpoint_plan['transformer_deep_mnist']
print(f"Loaded checkpoint plan: {len(transformer_plan['jumps'])} representative jumps identified")
print()

# Load Phase 1 data for transformer
phase1_data_file = Path('/home/user/ndt/experiments/new/results/phase1_full/transformer_deep_mnist.json')
with open(phase1_data_file, 'r') as f:
    transformer_data = json.load(f)

measurements = transformer_data['measurements']
print(f"Loaded Phase 1 data: {len(measurements)} measurement steps")
print()

# ============================================================================
# ANALYSIS 1: Jump Characteristics
# ============================================================================

print("=" * 80)
print("ANALYSIS 1: Characterizing Transformer Jumps")
print("=" * 80)
print()

jump_analysis = []

for i, jump in enumerate(transformer_plan['jumps'], 1):
    print(f"\nJump {i}: {jump['type']}")
    print(f"  Step: {jump['step']} (phase {jump['phase']:.3f})")
    print(f"  Layer: {jump['layer']}")
    print(f"  Cluster: {jump['cluster']}")
    print(f"  Magnitude: {jump['magnitude']:.2e}")

    # Extract dimensionality around this jump
    checkpoint_steps = jump['checkpoints']
    step_before = checkpoint_steps[0]
    step_during = checkpoint_steps[1]
    step_after = checkpoint_steps[2]

    # Find corresponding measurements (measurements are every 5 steps)
    idx_before = step_before // 5
    idx_during = step_during // 5
    idx_after = step_after // 5

    if idx_after < len(measurements):
        # Extract layer metrics
        layer_name = jump['layer']

        dim_before = measurements[idx_before]['layer_metrics'][layer_name]['stable_rank']
        dim_during = measurements[idx_during]['layer_metrics'][layer_name]['stable_rank']
        dim_after = measurements[idx_after]['layer_metrics'][layer_name]['stable_rank']

        loss_before = measurements[idx_before]['loss']
        loss_after = measurements[idx_after]['loss']

        print(f"  Dimensionality: {dim_before:.3f} → {dim_during:.3f} → {dim_after:.3f}")
        print(f"  Loss: {loss_before:.4f} → {loss_after:.4f} (Δ={loss_after-loss_before:.4f})")

        # Check other layers at same time
        all_layers = list(measurements[idx_during]['layer_metrics'].keys())
        layer_changes = []

        for layer in all_layers:
            dim_b = measurements[idx_before]['layer_metrics'][layer]['stable_rank']
            dim_a = measurements[idx_after]['layer_metrics'][layer]['stable_rank']
            change = abs(dim_a - dim_b)
            layer_changes.append((layer, change))

        # Sort by change magnitude
        layer_changes.sort(key=lambda x: x[1], reverse=True)

        print(f"  Top 3 changing layers:")
        for layer, change in layer_changes[:3]:
            print(f"    {layer}: Δ={change:.2e}")

        jump_analysis.append({
            'jump_num': i,
            'type': jump['type'],
            'step': jump['step'],
            'phase': jump['phase'],
            'cluster': jump['cluster'],
            'target_layer': layer_name,
            'dim_before': dim_before,
            'dim_during': dim_during,
            'dim_after': dim_after,
            'dim_change': abs(dim_after - dim_before),
            'loss_before': loss_before,
            'loss_after': loss_after,
            'loss_change': loss_after - loss_before,
            'num_coordinated_layers': sum(1 for _, ch in layer_changes if ch > 0.01)
        })

df_jumps = pd.DataFrame(jump_analysis)

# ============================================================================
# ANALYSIS 2: Cross-Layer Coordination
# ============================================================================

print("\n" + "=" * 80)
print("ANALYSIS 2: Cross-Layer Coordination During Jumps")
print("=" * 80)
print()

# For each jump, measure how many layers change together
coordination_analysis = []

for i, jump in enumerate(transformer_plan['jumps'], 1):
    step_before_idx = jump['checkpoints'][0] // 5
    step_after_idx = jump['checkpoints'][2] // 5

    if step_after_idx < len(measurements):
        all_layers = list(measurements[step_before_idx]['layer_metrics'].keys())

        # Count how many layers show significant change
        significant_changes = 0
        total_change = 0

        for layer in all_layers:
            dim_b = measurements[step_before_idx]['layer_metrics'][layer]['stable_rank']
            dim_a = measurements[step_after_idx]['layer_metrics'][layer]['stable_rank']
            change = abs(dim_a - dim_b)
            total_change += change

            if change > 1e-9:  # Very small threshold given magnitudes
                significant_changes += 1

        coordination_score = significant_changes / len(all_layers)

        print(f"Jump {i} ({jump['type']}):")
        print(f"  Layers with change: {significant_changes}/{len(all_layers)} ({coordination_score:.1%})")
        print(f"  Mean change magnitude: {total_change/len(all_layers):.2e}")

        coordination_analysis.append({
            'jump_num': i,
            'coordination_score': coordination_score,
            'num_layers_changing': significant_changes,
            'total_layers': len(all_layers),
            'mean_change': total_change / len(all_layers)
        })

df_coordination = pd.DataFrame(coordination_analysis)

# ============================================================================
# ANALYSIS 3: Early vs Late Jumps
# ============================================================================

print("\n" + "=" * 80)
print("ANALYSIS 3: Early vs Late Jump Comparison")
print("=" * 80)
print()

early_jumps = df_jumps[df_jumps['type'].str.contains('Type 5')]
late_jumps = df_jumps[df_jumps['type'].str.contains('Type 4')]
mid_jumps = df_jumps[df_jumps['type'].str.contains('Type 3')]

print("Early Jumps (Type 5):")
print(f"  Count: {len(early_jumps)}")
print(f"  Mean phase: {early_jumps['phase'].mean():.3f}")
print(f"  Mean loss change: {early_jumps['loss_change'].mean():.4f}")
print(f"  Mean coordination: {df_coordination[df_coordination['jump_num'].isin(early_jumps['jump_num'])]['coordination_score'].mean():.2%}")

print("\nLate Jumps (Type 4):")
print(f"  Count: {len(late_jumps)}")
print(f"  Mean phase: {late_jumps['phase'].mean():.3f}")
print(f"  Mean loss change: {late_jumps['loss_change'].mean():.4f}")
print(f"  Mean coordination: {df_coordination[df_coordination['jump_num'].isin(late_jumps['jump_num'])]['coordination_score'].mean():.2%}")

print("\nMid Jumps (Type 3):")
print(f"  Count: {len(mid_jumps)}")
print(f"  Mean phase: {mid_jumps['phase'].mean():.3f}")
print(f"  Mean loss change: {mid_jumps['loss_change'].mean():.4f}")

# ============================================================================
# ANALYSIS 4: Layer-Specific Patterns
# ============================================================================

print("\n" + "=" * 80)
print("ANALYSIS 4: Which Transformer Layers Jump Most?")
print("=" * 80)
print()

# Count jumps per layer
layer_jump_counts = {}
for jump in transformer_plan['jumps']:
    layer = jump['layer']
    layer_jump_counts[layer] = layer_jump_counts.get(layer, 0) + 1

print("Jumps per layer:")
for layer, count in sorted(layer_jump_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {layer}: {count} jumps")

# Check if linear2 layers (attention output projections) are more active
linear2_jumps = sum(1 for jump in transformer_plan['jumps'] if 'linear2' in jump['layer'])
linear1_jumps = sum(1 for jump in transformer_plan['jumps'] if 'linear1' in jump['layer'])

print(f"\nAttention pattern:")
print(f"  linear1 (attention MLP first layer): {linear1_jumps} jumps")
print(f"  linear2 (attention MLP output): {linear2_jumps} jumps")
print(f"  Ratio (linear2/linear1): {linear2_jumps/max(linear1_jumps, 1):.2f}x")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n" + "=" * 80)
print("CREATING VISUALIZATIONS")
print("=" * 80)
print()

# Plot 1: Jump timeline with coordination
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Subplot 1: Loss evolution with jump markers
steps = [m['step'] for m in measurements]
losses = [m['loss'] for m in measurements]

axes[0].plot(steps, losses, 'b-', linewidth=1, alpha=0.7)
for jump in transformer_plan['jumps']:
    axes[0].axvline(jump['step'], color='red', linestyle='--', alpha=0.5, linewidth=1)
    axes[0].text(jump['step'], max(losses)*0.95, f"J{transformer_plan['jumps'].index(jump)+1}",
                ha='center', fontsize=8, color='red')

axes[0].set_xlabel('Training Step')
axes[0].set_ylabel('Loss')
axes[0].set_title('Transformer Training: Loss Evolution with Jump Locations')
axes[0].grid(True, alpha=0.3)

# Subplot 2: Dimensionality evolution for jump layers
for jump in transformer_plan['jumps']:
    layer = jump['layer']
    layer_dims = [m['layer_metrics'][layer]['stable_rank'] for m in measurements]
    label = f"{layer.split('.')[-1]} (J{transformer_plan['jumps'].index(jump)+1})"
    axes[1].plot(steps, layer_dims, linewidth=1, alpha=0.7, label=label)

axes[1].set_xlabel('Training Step')
axes[1].set_ylabel('Stable Rank (Dimensionality)')
axes[1].set_title('Dimensionality Evolution of Jumping Layers')
axes[1].legend(loc='best', fontsize=8, ncol=2)
axes[1].grid(True, alpha=0.3)

# Subplot 3: Coordination scores
jump_steps = [jump['step'] for jump in transformer_plan['jumps']]
coord_scores = df_coordination['coordination_score'].values

axes[2].bar(range(1, len(jump_steps)+1), coord_scores, color='coral', edgecolor='black', alpha=0.7)
axes[2].set_xlabel('Jump Number')
axes[2].set_ylabel('Coordination Score')
axes[2].set_title('Cross-Layer Coordination During Each Jump')
axes[2].set_xticks(range(1, len(jump_steps)+1))
axes[2].set_xticklabels([f"J{i}\n{s}" for i, s in enumerate(jump_steps, 1)], fontsize=8)
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(results_dir / 'transformer_jump_overview.png')
plt.close()
print("✓ Saved transformer_jump_overview.png")

# Plot 2: Early vs Late comparison
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Loss change comparison
jump_types = ['Type 5\n(early)', 'Type 3\n(mid)', 'Type 4\n(late)']
loss_changes = [
    early_jumps['loss_change'].mean() if len(early_jumps) > 0 else 0,
    mid_jumps['loss_change'].mean() if len(mid_jumps) > 0 else 0,
    late_jumps['loss_change'].mean() if len(late_jumps) > 0 else 0
]
axes[0, 0].bar(jump_types, loss_changes, color=['steelblue', 'seagreen', 'coral'], edgecolor='black')
axes[0, 0].set_ylabel('Mean Loss Change')
axes[0, 0].set_title('Loss Impact by Jump Type')
axes[0, 0].axhline(0, color='black', linestyle='-', linewidth=0.5)
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Coordination comparison
coord_early = df_coordination[df_coordination['jump_num'].isin(early_jumps['jump_num'])]['coordination_score'].mean() if len(early_jumps) > 0 else 0
coord_mid = df_coordination[df_coordination['jump_num'].isin(mid_jumps['jump_num'])]['coordination_score'].mean() if len(mid_jumps) > 0 else 0
coord_late = df_coordination[df_coordination['jump_num'].isin(late_jumps['jump_num'])]['coordination_score'].mean() if len(late_jumps) > 0 else 0

coord_scores_by_type = [coord_early, coord_mid, coord_late]
axes[0, 1].bar(jump_types, coord_scores_by_type, color=['steelblue', 'seagreen', 'coral'], edgecolor='black')
axes[0, 1].set_ylabel('Mean Coordination Score')
axes[0, 1].set_title('Cross-Layer Coordination by Jump Type')
axes[0, 1].set_ylim([0, 1])
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Phase distribution
phases_by_type = [
    early_jumps['phase'].values if len(early_jumps) > 0 else np.array([]),
    mid_jumps['phase'].values if len(mid_jumps) > 0 else np.array([]),
    late_jumps['phase'].values if len(late_jumps) > 0 else np.array([])
]
bp = axes[1, 0].boxplot(phases_by_type, labels=jump_types, patch_artist=True)
for patch, color in zip(bp['boxes'], ['steelblue', 'seagreen', 'coral']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[1, 0].set_ylabel('Training Phase')
axes[1, 0].set_title('Training Phase Distribution by Jump Type')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Scatter: coordination vs loss change
axes[1, 1].scatter(df_coordination['coordination_score'], df_jumps['loss_change'],
                  c=df_jumps['phase'], cmap='viridis', s=100, alpha=0.7, edgecolors='black')
axes[1, 1].set_xlabel('Coordination Score')
axes[1, 1].set_ylabel('Loss Change')
axes[1, 1].set_title('Coordination vs Loss Impact')
axes[1, 1].grid(True, alpha=0.3)
cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1], label='Training Phase')

plt.tight_layout()
plt.savefig(results_dir / 'transformer_early_vs_late.png')
plt.close()
print("✓ Saved transformer_early_vs_late.png")

# ============================================================================
# SAVE RESULTS
# ============================================================================

results = {
    'summary': {
        'num_jumps_analyzed': len(transformer_plan['jumps']),
        'mean_coordination': float(df_coordination['coordination_score'].mean()),
        'mean_loss_change': float(df_jumps['loss_change'].mean()),
        'early_jumps': {
            'count': int(len(early_jumps)),
            'mean_phase': float(early_jumps['phase'].mean()) if len(early_jumps) > 0 else 0,
            'mean_loss_change': float(early_jumps['loss_change'].mean()) if len(early_jumps) > 0 else 0,
            'mean_coordination': float(coord_early)
        },
        'late_jumps': {
            'count': int(len(late_jumps)),
            'mean_phase': float(late_jumps['phase'].mean()) if len(late_jumps) > 0 else 0,
            'mean_loss_change': float(late_jumps['loss_change'].mean()) if len(late_jumps) > 0 else 0,
            'mean_coordination': float(coord_late)
        },
        'layer_patterns': {
            'linear2_jumps': linear2_jumps,
            'linear1_jumps': linear1_jumps,
            'ratio': float(linear2_jumps / max(linear1_jumps, 1))
        }
    },
    'jump_details': jump_analysis,
    'coordination_details': coordination_analysis
}

with open(results_dir / 'transformer_analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Save CSV
df_jumps.to_csv(results_dir / 'transformer_jumps_detailed.csv', index=False)
df_coordination.to_csv(results_dir / 'transformer_coordination_detailed.csv', index=False)

print("✓ Saved transformer_analysis_results.json")
print("✓ Saved transformer_jumps_detailed.csv")
print("✓ Saved transformer_coordination_detailed.csv")

# ============================================================================
# SUMMARY AND NEXT STEPS
# ============================================================================

print("\n" + "=" * 80)
print("WEEK 3-4 SUMMARY: Transformer Jump Analysis")
print("=" * 80)
print()

print("Key Findings from Dimensionality Analysis:")
print(f"  • Analyzed {len(transformer_plan['jumps'])} representative jumps")
print(f"  • Mean coordination score: {df_coordination['coordination_score'].mean():.1%}")
print(f"    (On average, {df_coordination['coordination_score'].mean()*14:.1f} out of 14 layers change together)")
print(f"  • Mean loss change during jumps: {df_jumps['loss_change'].mean():.4f}")
print()

print("Early vs Late Jumps:")
print(f"  • Early jumps (Type 5): phase {early_jumps['phase'].mean():.3f}, coordination {coord_early:.1%}")
print(f"  • Late jumps (Type 4): phase {late_jumps['phase'].mean():.3f}, coordination {coord_late:.1%}")
if coord_early > coord_late:
    print(f"  → Early jumps show {coord_early/max(coord_late, 0.01):.2f}x MORE coordination")
else:
    print(f"  → Late jumps show {coord_late/max(coord_early, 0.01):.2f}x MORE coordination")
print()

print("Layer-Specific Patterns:")
print(f"  • Attention output layers (linear2): {linear2_jumps} jumps")
print(f"  • Attention MLP first layer (linear1): {linear1_jumps} jumps")
if linear2_jumps > linear1_jumps:
    print(f"  → Output projections show {linear2_jumps/max(linear1_jumps, 1):.2f}x MORE jumps")
    print("  → Suggests representational changes occur at output rather than intermediate processing")
print()

print("=" * 80)
print("NEXT STEPS")
print("=" * 80)
print()
print("To complete the transformer investigation:")
print()
print("1. Re-run transformer_deep_mnist with checkpoint saving:")
print("   - Load checkpoint_plan.json")
print("   - Save model states at steps:", transformer_plan['checkpoint_steps'])
print()
print("2. Extract attention patterns from checkpoints:")
print("   ```python")
print("   model.eval()")
print("   with torch.no_grad():")
print("       outputs = model(inputs, output_attentions=True)")
print("       attention_weights = outputs.attentions  # List of [batch, heads, seq, seq]")
print("   ```")
print()
print("3. Measure attention head specialization:")
print("   ```python")
print("   from phase2_infrastructure import MeasurementTools")
print("   measurements = MeasurementTools()")
print("   ")
print("   # For each jump")
print("   attn_before = load_attention(step_before)")
print("   attn_after = load_attention(step_after)")
print("   ")
print("   entropy_before = measurements.attention_entropy(attn_before)")
print("   entropy_after = measurements.attention_entropy(attn_after)")
print("   specialization = measurements.attention_specialization(attn_after)")
print("   ")
print("   print(f'Entropy change: {entropy_after - entropy_before:.3f}')")
print("   print(f'Specialization index: {specialization[\"specialization_index\"]:.3f}')")
print("   ```")
print()
print("4. Test hypothesis:")
print("   - If entropy DECREASES → heads becoming more focused (specialized)")
print("   - If specialization index INCREASES → heads diverging in behavior")
print("   - Early jumps should show MORE specialization than late jumps")
print()
print("=" * 80)
print("✓ Week 3-4 Analysis Complete!")
print("=" * 80)
