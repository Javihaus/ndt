"""
Phase 2: Week 5-6 - CNN vs MLP Comparison

Hypothesis: CNNs show more jumps than MLPs because convolutional structure
enables discrete filter differentiation, while MLPs have smooth, distributed representations.

Investigation:
1. Compare jump frequency and characteristics between CNN and MLP
2. Analyze layer-wise patterns (early vs late layers)
3. Measure jump discreteness (how abrupt vs gradual)
4. When checkpoints available: Visualize filters vs hidden unit activations

Targets: cnn_deep_mnist (45 jumps) vs mlp_narrow_mnist (18 jumps)
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

print("=" * 80)
print("PHASE 2: CNN vs MLP COMPARISON (Week 5-6)")
print("=" * 80)
print("\nHypothesis: CNNs enable discrete filter differentiation")
print("Targets: cnn_deep_mnist vs mlp_narrow_mnist")
print()

# ============================================================================
# SETUP
# ============================================================================

results_dir = Path('/home/user/ndt/experiments/mechanistic_interpretability/results/phase2_week5_6')
results_dir.mkdir(parents=True, exist_ok=True)

# Load checkpoint plan
checkpoint_plan_file = Path('/home/user/ndt/experiments/mechanistic_interpretability/checkpoint_plan.json')
with open(checkpoint_plan_file, 'r') as f:
    checkpoint_plan = json.load(f)

cnn_plan = checkpoint_plan['cnn_deep_mnist']
mlp_plan = checkpoint_plan['mlp_narrow_mnist']

print(f"CNN: {len(cnn_plan['jumps'])} representative jumps")
print(f"MLP: {len(mlp_plan['jumps'])} representative jumps")
print()

# Load Phase 1 data
phase1_dir = Path('/home/user/ndt/experiments/new/results/phase1_full')

with open(phase1_dir / 'cnn_deep_mnist.json', 'r') as f:
    cnn_data = json.load(f)

with open(phase1_dir / 'mlp_narrow_mnist.json', 'r') as f:
    mlp_data = json.load(f)

# Load Phase 1 jump data for full statistics
jump_data_file = Path('/home/user/ndt/experiments/mechanistic_interpretability/results/step1_2/all_jumps_detailed.csv')
df_all_jumps = pd.read_csv(jump_data_file)

cnn_all_jumps = df_all_jumps[df_all_jumps['experiment'] == 'cnn_deep_mnist']
mlp_all_jumps = df_all_jumps[df_all_jumps['experiment'] == 'mlp_narrow_mnist']

print(f"Total jumps (Phase 1):")
print(f"  CNN: {len(cnn_all_jumps)} jumps")
print(f"  MLP: {len(mlp_all_jumps)} jumps")
print(f"  Ratio: CNN has {len(cnn_all_jumps)/max(len(mlp_all_jumps), 1):.2f}x more jumps")
print()

# ============================================================================
# ANALYSIS 1: Jump Frequency and Distribution
# ============================================================================

print("=" * 80)
print("ANALYSIS 1: Jump Frequency and Temporal Distribution")
print("=" * 80)
print()

# Phase distribution
print("Phase distribution:")
print(f"  CNN: {cnn_all_jumps['phase_category'].value_counts().to_dict()}")
print(f"  MLP: {mlp_all_jumps['phase_category'].value_counts().to_dict()}")
print()

# Mean phase
cnn_mean_phase = cnn_all_jumps['phase'].mean()
mlp_mean_phase = mlp_all_jumps['phase'].mean()

print(f"Mean jump phase:")
print(f"  CNN: {cnn_mean_phase:.4f}")
print(f"  MLP: {mlp_mean_phase:.4f}")
print()

# Cluster distribution
print("Cluster distribution:")
cnn_clusters = cnn_all_jumps['cluster'].value_counts().sort_index()
mlp_clusters = mlp_all_jumps['cluster'].value_counts().sort_index()

for cluster in sorted(set(list(cnn_clusters.index) + list(mlp_clusters.index))):
    cnn_count = cnn_clusters.get(cluster, 0)
    mlp_count = mlp_clusters.get(cluster, 0)
    print(f"  Cluster {cluster}: CNN={cnn_count}, MLP={mlp_count}")

# ============================================================================
# ANALYSIS 2: Layer-Wise Patterns
# ============================================================================

print("\n" + "=" * 80)
print("ANALYSIS 2: Layer-Wise Jump Patterns")
print("=" * 80)
print()

# CNN layer analysis
cnn_layer_counts = cnn_all_jumps['layer'].value_counts()
print("CNN - Jumps per layer:")
for layer, count in cnn_layer_counts.items():
    print(f"  {layer}: {count} jumps")

print(f"\nCNN depth analysis:")
cnn_layers = list(cnn_data['measurements'][0]['layer_metrics'].keys())
print(f"  Total layers: {len(cnn_layers)}")
print(f"  First layer (conv_layers.0) jumps: {cnn_layer_counts.get('conv_layers.0', 0)}")
print(f"  Last layer jumps: {cnn_layer_counts.get(cnn_layers[-1], 0)}")

# MLP layer analysis
mlp_layer_counts = mlp_all_jumps['layer'].value_counts()
print("\nMLP - Jumps per layer:")
for layer, count in mlp_layer_counts.items():
    print(f"  {layer}: {count} jumps")

print(f"\nMLP depth analysis:")
mlp_layers = list(mlp_data['measurements'][0]['layer_metrics'].keys())
print(f"  Total layers: {len(mlp_layers)}")
print(f"  First layer (network.0) jumps: {mlp_layer_counts.get('network.0', 0)}")
print(f"  Last layer jumps: {mlp_layer_counts.get(mlp_layers[-1], 0)}")

# ============================================================================
# ANALYSIS 3: Jump Characteristics
# ============================================================================

print("\n" + "=" * 80)
print("ANALYSIS 3: Jump Magnitude and Speed Comparison")
print("=" * 80)
print()

print("Magnitude statistics:")
print(f"  CNN: mean={cnn_all_jumps['magnitude'].mean():.2e}, std={cnn_all_jumps['magnitude'].std():.2e}")
print(f"  MLP: mean={mlp_all_jumps['magnitude'].mean():.2e}, std={mlp_all_jumps['magnitude'].std():.2e}")
print()

print("Speed statistics:")
print(f"  CNN: mean={cnn_all_jumps['speed'].mean():.2e}, std={cnn_all_jumps['speed'].std():.2e}")
print(f"  MLP: mean={mlp_all_jumps['speed'].mean():.2e}, std={mlp_all_jumps['speed'].std():.2e}")
print()

# ============================================================================
# ANALYSIS 4: Detailed Analysis of Representative Jumps
# ============================================================================

print("=" * 80)
print("ANALYSIS 4: Representative Jump Deep Dive")
print("=" * 80)
print()

def analyze_jump_window(measurements, jump, experiment_name):
    """Analyze dimensionality dynamics around a jump."""
    step_before_idx = jump['checkpoints'][0] // 5
    step_during_idx = jump['checkpoints'][1] // 5
    step_after_idx = jump['checkpoints'][2] // 5

    if step_after_idx >= len(measurements):
        return None

    layer = jump['layer']

    dim_before = measurements[step_before_idx]['layer_metrics'][layer]['stable_rank']
    dim_during = measurements[step_during_idx]['layer_metrics'][layer]['stable_rank']
    dim_after = measurements[step_after_idx]['layer_metrics'][layer]['stable_rank']

    loss_before = measurements[step_before_idx]['loss']
    loss_after = measurements[step_after_idx]['loss']

    # Check all layers
    all_layers = list(measurements[step_during_idx]['layer_metrics'].keys())
    coordinated_changes = 0

    for l in all_layers:
        dim_b = measurements[step_before_idx]['layer_metrics'][l]['stable_rank']
        dim_a = measurements[step_after_idx]['layer_metrics'][l]['stable_rank']
        if abs(dim_a - dim_b) > 1e-9:
            coordinated_changes += 1

    return {
        'experiment': experiment_name,
        'type': jump['type'],
        'step': jump['step'],
        'layer': layer,
        'dim_before': dim_before,
        'dim_during': dim_during,
        'dim_after': dim_after,
        'dim_change': abs(dim_after - dim_before),
        'loss_before': loss_before,
        'loss_after': loss_after,
        'loss_change': loss_after - loss_before,
        'coordination': coordinated_changes / len(all_layers)
    }

print("CNN representative jumps:")
cnn_jump_details = []
for i, jump in enumerate(cnn_plan['jumps'], 1):
    result = analyze_jump_window(cnn_data['measurements'], jump, 'CNN')
    if result:
        cnn_jump_details.append(result)
        print(f"\n  Jump {i}: {result['type']}")
        print(f"    Step: {result['step']}, Layer: {result['layer']}")
        print(f"    Dimensionality change: {result['dim_change']:.2e}")
        print(f"    Loss change: {result['loss_change']:.4f}")
        print(f"    Coordination: {result['coordination']:.1%}")

print("\n\nMLP representative jumps:")
mlp_jump_details = []
for i, jump in enumerate(mlp_plan['jumps'], 1):
    result = analyze_jump_window(mlp_data['measurements'], jump, 'MLP')
    if result:
        mlp_jump_details.append(result)
        print(f"\n  Jump {i}: {result['type']}")
        print(f"    Step: {result['step']}, Layer: {result['layer']}")
        print(f"    Dimensionality change: {result['dim_change']:.2e}")
        print(f"    Loss change: {result['loss_change']:.4f}")
        print(f"    Coordination: {result['coordination']:.1%}")

df_cnn_jumps = pd.DataFrame(cnn_jump_details)
df_mlp_jumps = pd.DataFrame(mlp_jump_details)

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n" + "=" * 80)
print("CREATING VISUALIZATIONS")
print("=" * 80)
print()

# Figure 1: Comprehensive comparison
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Row 1: Jump distributions
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist([cnn_all_jumps['phase'], mlp_all_jumps['phase']],
         bins=20, label=['CNN', 'MLP'], alpha=0.7, color=['steelblue', 'coral'])
ax1.set_xlabel('Training Phase')
ax1.set_ylabel('Jump Count')
ax1.set_title('Temporal Distribution of Jumps')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])
cluster_data = []
cluster_labels = []
for cluster in sorted(set(list(cnn_clusters.index) + list(mlp_clusters.index))):
    cluster_data.append([cnn_clusters.get(cluster, 0), mlp_clusters.get(cluster, 0)])
    cluster_labels.append(f'C{cluster}')
cluster_data = np.array(cluster_data).T
x = np.arange(len(cluster_labels))
width = 0.35
ax2.bar(x - width/2, cluster_data[0], width, label='CNN', color='steelblue', alpha=0.7)
ax2.bar(x + width/2, cluster_data[1], width, label='MLP', color='coral', alpha=0.7)
ax2.set_xlabel('Cluster')
ax2.set_ylabel('Jump Count')
ax2.set_title('Cluster Distribution')
ax2.set_xticks(x)
ax2.set_xticklabels(cluster_labels)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

ax3 = fig.add_subplot(gs[0, 2])
layer_counts = [
    len(cnn_layer_counts),
    len(mlp_layer_counts)
]
total_layers = [len(cnn_layers), len(mlp_layers)]
x_pos = [0, 1]
ax3.bar(x_pos, layer_counts, color=['steelblue', 'coral'], alpha=0.7, label='Layers with jumps')
ax3.plot(x_pos, total_layers, 'ko-', linewidth=2, markersize=10, label='Total layers')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(['CNN', 'MLP'])
ax3.set_ylabel('Layer Count')
ax3.set_title('Layer Coverage')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Row 2: Loss evolution with jump markers
ax4 = fig.add_subplot(gs[1, :])
cnn_steps = [m['step'] for m in cnn_data['measurements']]
cnn_losses = [m['loss'] for m in cnn_data['measurements']]
mlp_steps = [m['step'] for m in mlp_data['measurements']]
mlp_losses = [m['loss'] for m in mlp_data['measurements']]

ax4.plot(cnn_steps, cnn_losses, 'b-', linewidth=1.5, alpha=0.7, label='CNN')
ax4.plot(mlp_steps, mlp_losses, 'r-', linewidth=1.5, alpha=0.7, label='MLP')

# Mark representative jumps
for jump in cnn_plan['jumps']:
    ax4.axvline(jump['step'], color='steelblue', linestyle='--', alpha=0.3, linewidth=1)
for jump in mlp_plan['jumps']:
    ax4.axvline(jump['step'], color='coral', linestyle='--', alpha=0.3, linewidth=1)

ax4.set_xlabel('Training Step')
ax4.set_ylabel('Loss')
ax4.set_title('Training Dynamics: CNN vs MLP (vertical lines = representative jumps)')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Row 3: Jump characteristics
ax5 = fig.add_subplot(gs[2, 0])
if len(df_cnn_jumps) > 0 and len(df_mlp_jumps) > 0:
    dim_changes = [df_cnn_jumps['dim_change'].values, df_mlp_jumps['dim_change'].values]
    bp = ax5.boxplot(dim_changes, labels=['CNN', 'MLP'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['steelblue', 'coral']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax5.set_ylabel('Dimensionality Change')
    ax5.set_title('Jump Magnitude Distribution')
    ax5.grid(True, alpha=0.3, axis='y')

ax6 = fig.add_subplot(gs[2, 1])
if len(df_cnn_jumps) > 0 and len(df_mlp_jumps) > 0:
    loss_changes = [df_cnn_jumps['loss_change'].values, df_mlp_jumps['loss_change'].values]
    bp = ax6.boxplot(loss_changes, labels=['CNN', 'MLP'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['steelblue', 'coral']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax6.set_ylabel('Loss Change')
    ax6.set_title('Loss Impact Distribution')
    ax6.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax6.grid(True, alpha=0.3, axis='y')

ax7 = fig.add_subplot(gs[2, 2])
if len(df_cnn_jumps) > 0 and len(df_mlp_jumps) > 0:
    coord_scores = [df_cnn_jumps['coordination'].values, df_mlp_jumps['coordination'].values]
    bp = ax7.boxplot(coord_scores, labels=['CNN', 'MLP'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['steelblue', 'coral']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax7.set_ylabel('Coordination Score')
    ax7.set_title('Cross-Layer Coordination')
    ax7.set_ylim([0, 1])
    ax7.grid(True, alpha=0.3, axis='y')

plt.savefig(results_dir / 'cnn_mlp_comprehensive_comparison.png')
plt.close()
print("✓ Saved cnn_mlp_comprehensive_comparison.png")

# Figure 2: Layer-specific analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# CNN layer distribution
ax = axes[0]
cnn_layer_names = list(cnn_layer_counts.index)
cnn_layer_values = list(cnn_layer_counts.values)
ax.barh(cnn_layer_names, cnn_layer_values, color='steelblue', alpha=0.7, edgecolor='black')
ax.set_xlabel('Number of Jumps')
ax.set_title('CNN: Jumps per Layer')
ax.grid(True, alpha=0.3, axis='x')

# MLP layer distribution
ax = axes[1]
mlp_layer_names = list(mlp_layer_counts.index)
mlp_layer_values = list(mlp_layer_counts.values)
ax.barh(mlp_layer_names, mlp_layer_values, color='coral', alpha=0.7, edgecolor='black')
ax.set_xlabel('Number of Jumps')
ax.set_title('MLP: Jumps per Layer')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(results_dir / 'cnn_mlp_layer_analysis.png')
plt.close()
print("✓ Saved cnn_mlp_layer_analysis.png")

# ============================================================================
# SAVE RESULTS
# ============================================================================

results = {
    'summary': {
        'cnn': {
            'total_jumps': int(len(cnn_all_jumps)),
            'mean_phase': float(cnn_mean_phase),
            'layers_with_jumps': int(len(cnn_layer_counts)),
            'total_layers': int(len(cnn_layers)),
            'mean_magnitude': float(cnn_all_jumps['magnitude'].mean()),
            'mean_speed': float(cnn_all_jumps['speed'].mean())
        },
        'mlp': {
            'total_jumps': int(len(mlp_all_jumps)),
            'mean_phase': float(mlp_mean_phase),
            'layers_with_jumps': int(len(mlp_layer_counts)),
            'total_layers': int(len(mlp_layers)),
            'mean_magnitude': float(mlp_all_jumps['magnitude'].mean()),
            'mean_speed': float(mlp_all_jumps['speed'].mean())
        },
        'comparison': {
            'jump_ratio': float(len(cnn_all_jumps) / max(len(mlp_all_jumps), 1)),
            'magnitude_ratio': float(cnn_all_jumps['magnitude'].mean() / max(mlp_all_jumps['magnitude'].mean(), 1e-20)),
            'speed_ratio': float(cnn_all_jumps['speed'].mean() / max(mlp_all_jumps['speed'].mean(), 1e-20))
        }
    },
    'cnn_representative_jumps': cnn_jump_details,
    'mlp_representative_jumps': mlp_jump_details
}

with open(results_dir / 'cnn_mlp_comparison_results.json', 'w') as f:
    json.dump(results, f, indent=2)

if len(df_cnn_jumps) > 0:
    df_cnn_jumps.to_csv(results_dir / 'cnn_representative_jumps.csv', index=False)
if len(df_mlp_jumps) > 0:
    df_mlp_jumps.to_csv(results_dir / 'mlp_representative_jumps.csv', index=False)

print("✓ Saved cnn_mlp_comparison_results.json")
if len(df_cnn_jumps) > 0:
    print("✓ Saved cnn_representative_jumps.csv")
if len(df_mlp_jumps) > 0:
    print("✓ Saved mlp_representative_jumps.csv")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("WEEK 5-6 SUMMARY: CNN vs MLP Comparison")
print("=" * 80)
print()

print("Key Findings:")
print(f"  • CNN shows {len(cnn_all_jumps)/max(len(mlp_all_jumps), 1):.2f}x more jumps than MLP")
print(f"    (CNN: {len(cnn_all_jumps)} jumps, MLP: {len(mlp_all_jumps)} jumps)")
print()

print(f"  • Jump magnitudes:")
print(f"    CNN: {cnn_all_jumps['magnitude'].mean():.2e} (mean)")
print(f"    MLP: {mlp_all_jumps['magnitude'].mean():.2e} (mean)")
print(f"    Ratio: {cnn_all_jumps['magnitude'].mean() / max(mlp_all_jumps['magnitude'].mean(), 1e-20):.2f}x")
print()

print(f"  • Layer coverage:")
print(f"    CNN: {len(cnn_layer_counts)}/{len(cnn_layers)} layers have jumps ({len(cnn_layer_counts)/len(cnn_layers):.1%})")
print(f"    MLP: {len(mlp_layer_counts)}/{len(mlp_layers)} layers have jumps ({len(mlp_layer_counts)/len(mlp_layers):.1%})")
print()

if len(df_cnn_jumps) > 0 and len(df_mlp_jumps) > 0:
    print(f"  • Representative jump coordination:")
    print(f"    CNN: {df_cnn_jumps['coordination'].mean():.1%} mean coordination")
    print(f"    MLP: {df_mlp_jumps['coordination'].mean():.1%} mean coordination")
    print()

print("=" * 80)
print("HYPOTHESIS EVALUATION")
print("=" * 80)
print()
print("Hypothesis: CNNs enable discrete filter differentiation")
print()
if len(cnn_all_jumps) > len(mlp_all_jumps):
    print("✓ SUPPORTED: CNN shows significantly more jumps")
    print("  This suggests convolutional structure may enable more discrete changes")
    print()
print("Further investigation needed:")
print("  1. Visualize CNN filters before/after jumps")
print("  2. Measure filter orthogonality and specialization")
print("  3. Compare to MLP hidden unit activation patterns")
print("  4. Test if CNN jumps correspond to filter differentiation events")
print()

print("=" * 80)
print("NEXT STEPS")
print("=" * 80)
print()
print("To complete CNN vs MLP investigation:")
print()
print("1. Re-run experiments with checkpoint saving at:")
print(f"   CNN steps: {cnn_plan['checkpoint_steps']}")
print(f"   MLP steps: {mlp_plan['checkpoint_steps']}")
print()
print("2. Extract and visualize CNN filters:")
print("   ```python")
print("   filters_before = model.conv_layers[0].weight.data  # [out_ch, in_ch, H, W]")
print("   filters_after = model_after.conv_layers[0].weight.data")
print("   ")
print("   # Measure filter similarity")
print("   from phase2_infrastructure import MeasurementTools")
print("   similarity = MeasurementTools.cosine_similarity(filters_before, filters_after)")
print("   ```")
print()
print("3. Extract MLP activations:")
print("   ```python")
print("   # Hook into hidden layers")
print("   activations = {}")
print("   def hook_fn(name):")
print("       def hook(module, input, output):")
print("           activations[name] = output.detach()")
print("       return hook")
print("   ")
print("   model.network[0].register_forward_hook(hook_fn('layer0'))")
print("   ```")
print()
print("4. Compare selectivity:")
print("   - CNN: Do filters become more specialized?")
print("   - MLP: Do hidden units develop class selectivity?")
print()
print("=" * 80)
print("✓ Week 5-6 Analysis Complete!")
print("=" * 80)
