"""
Modified Phase 2: Early vs Late Feature Analysis
================================================

Honest approach: Test if temporal patterns (83.3% early) correspond to
qualitative differences in learned features.

NOT investigating: Individual "jumps" (magnitudes too small)
NOT claiming: Discrete transitions, mechanistic understanding
YES testing: Whether early features differ qualitatively from late features

Timeline: 2 weeks
Checkpoints: 9 total (early/mid/late × 3 experiments)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Check if PyTorch available
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    print("Note: PyTorch not available. Running in placeholder mode.")
    print("Install PyTorch when ready to analyze actual checkpoints.")

# Academic plot style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

print("=" * 80)
print("MODIFIED PHASE 2: EARLY vs LATE FEATURE ANALYSIS")
print("=" * 80)
print()

# ============================================================================
# SETUP
# ============================================================================

base_dir = Path('/home/user/ndt/experiments/mechanistic_interpretability')
results_dir = base_dir / 'results' / 'modified_phase2'
results_dir.mkdir(parents=True, exist_ok=True)

# Load modified checkpoint plan
plan_file = base_dir / 'checkpoint_plan_modified.json'
with open(plan_file, 'r') as f:
    checkpoint_plan = json.load(f)

print("Loaded modified checkpoint plan:")
print(f"  Total checkpoints: {checkpoint_plan['metadata']['total_checkpoints']}")
print(f"  Experiments: {checkpoint_plan['metadata']['experiments']}")
print(f"  Timeline: {checkpoint_plan['metadata']['timeline_weeks']} weeks")
print()

# ============================================================================
# CHECKPOINT LOADING UTILITIES
# ============================================================================

def load_checkpoint(experiment_name: str, step: int, base_path: Path = None):
    """Load a checkpoint and reconstruct the model."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available. Cannot load checkpoints.")

    if base_path is None:
        base_path = Path('/home/user/ndt/experiments/checkpoints')

    checkpoint_path = base_path / experiment_name / f'checkpoint_step_{step:05d}.pt'

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if torch is not None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        return checkpoint
    return None


def check_checkpoints_exist(experiment_name: str, steps: List[int]) -> Dict[str, bool]:
    """Check which checkpoints exist for an experiment."""
    checkpoint_dir = Path('/home/user/ndt/experiments/checkpoints') / experiment_name

    status = {}
    for step in steps:
        checkpoint_path = checkpoint_dir / f'checkpoint_step_{step:05d}.pt'
        status[f'step_{step}'] = checkpoint_path.exists()

    return status


# ============================================================================
# FEATURE EXTRACTION (Architecture-Specific)
# ============================================================================

def extract_transformer_features(model, dataloader, device='cpu'):
    """
    Extract attention patterns from transformer.

    Returns averaged attention weights across heads and layers.
    """
    if not TORCH_AVAILABLE:
        return None

    model.eval()
    model.to(device)

    all_attention_patterns = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)

            # Forward pass with attention extraction
            # This assumes model returns attention when available
            try:
                outputs = model(inputs, output_attentions=True)
                if hasattr(outputs, 'attentions'):
                    attentions = outputs.attentions
                    all_attention_patterns.append([attn.cpu().numpy() for attn in attentions])
            except:
                # Fallback: extract from attention weights directly
                attention_weights = []
                for name, module in model.named_modules():
                    if 'attention' in name.lower() and hasattr(module, 'attention_weights'):
                        attention_weights.append(module.attention_weights.cpu().numpy())

                if attention_weights:
                    all_attention_patterns.append(attention_weights)

            # Limit samples for efficiency
            if batch_idx >= 20:
                break

    return all_attention_patterns


def extract_cnn_features(model, layer_name='conv_layers.0', device='cpu'):
    """
    Extract convolutional filters from CNN.

    Returns filter weights for visualization.
    """
    if not TORCH_AVAILABLE or torch is None:
        return None

    model.eval()
    model.to(device)

    filters = {}
    for name, module in model.named_modules():
        if torch is not None and isinstance(module, nn.Conv2d):
            filters[name] = module.weight.data.cpu().numpy()

    return filters


def extract_mlp_features(model, dataloader, layer_names=None, device='cpu'):
    """
    Extract activation patterns from MLP hidden layers.

    Returns activations for specified layers.
    """
    if not TORCH_AVAILABLE:
        return None

    if layer_names is None:
        layer_names = ['network.0', 'network.4', 'network.8']

    model.eval()
    model.to(device)

    activations = {name: [] for name in layer_names}

    def get_activation(name):
        def hook(module, input, output):
            activations[name].append(output.detach().cpu().numpy())
        return hook

    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if name in layer_names:
            hooks.append(module.register_forward_hook(get_activation(name)))

    # Collect activations
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            _ = model(inputs)

            if batch_idx >= 20:
                break

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Concatenate batches
    for name in activations:
        if activations[name]:
            activations[name] = np.concatenate(activations[name], axis=0)

    return activations


# ============================================================================
# FEATURE COMPARISON METRICS
# ============================================================================

def cosine_similarity(features1, features2):
    """Compute cosine similarity between two feature sets."""
    f1_flat = features1.flatten()
    f2_flat = features2.flatten()

    dot_product = np.dot(f1_flat, f2_flat)
    norm1 = np.linalg.norm(f1_flat)
    norm2 = np.linalg.norm(f2_flat)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def measure_diversity(features, num_clusters=10):
    """
    Measure feature diversity using k-means clustering.

    Returns number of distinct patterns and cluster quality.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    # Reshape features for clustering
    if features.ndim > 2:
        features_flat = features.reshape(features.shape[0], -1)
    else:
        features_flat = features

    # Cluster
    kmeans = KMeans(n_clusters=min(num_clusters, len(features_flat)), random_state=42)
    labels = kmeans.fit_predict(features_flat)

    # Measure quality
    if len(np.unique(labels)) > 1:
        silhouette = silhouette_score(features_flat, labels)
    else:
        silhouette = 0.0

    diversity = {
        'num_clusters': len(np.unique(labels)),
        'silhouette_score': silhouette,
        'cluster_sizes': [int(np.sum(labels == i)) for i in range(num_clusters)]
    }

    return diversity


def compare_feature_sets(early_features, mid_features, late_features):
    """
    Compare three feature sets (early/mid/late).

    Returns similarity matrix and diversity measures.
    """
    # Similarity matrix
    similarity = {
        'early_vs_mid': cosine_similarity(early_features, mid_features),
        'early_vs_late': cosine_similarity(early_features, late_features),
        'mid_vs_late': cosine_similarity(mid_features, late_features)
    }

    # Diversity measures
    diversity = {
        'early': measure_diversity(early_features),
        'mid': measure_diversity(mid_features),
        'late': measure_diversity(late_features)
    }

    return similarity, diversity


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_cnn_filters(filters_early, filters_mid, filters_late, save_path):
    """Visualize CNN filter evolution across training phases."""

    fig, axes = plt.subplots(3, 8, figsize=(16, 6))

    for phase_idx, (filters, phase_name) in enumerate([
        (filters_early, 'Early (5%)'),
        (filters_mid, 'Mid (50%)'),
        (filters_late, 'Late (100%)')
    ]):
        # Show first 8 filters
        for i in range(8):
            if i < filters.shape[0]:
                # Get filter (assuming [out_channels, in_channels, H, W])
                filt = filters[i, 0] if filters.ndim == 4 else filters[i]

                axes[phase_idx, i].imshow(filt, cmap='RdBu', vmin=-filt.std()*2, vmax=filt.std()*2)
                axes[phase_idx, i].axis('off')

                if i == 0:
                    axes[phase_idx, i].set_ylabel(phase_name, fontsize=12, fontweight='bold')

    plt.suptitle('CNN Filter Evolution: Early vs Mid vs Late', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def visualize_similarity_matrix(similarity_dict, save_path):
    """Visualize similarity between training phases."""

    # Create matrix
    phases = ['Early', 'Mid', 'Late']
    matrix = np.zeros((3, 3))

    matrix[0, 1] = matrix[1, 0] = similarity_dict['early_vs_mid']
    matrix[0, 2] = matrix[2, 0] = similarity_dict['early_vs_late']
    matrix[1, 2] = matrix[2, 1] = similarity_dict['mid_vs_late']
    matrix[0, 0] = matrix[1, 1] = matrix[2, 2] = 1.0

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1)

    # Labels
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(phases)
    ax.set_yticklabels(phases)

    # Annotate with values
    for i in range(3):
        for j in range(3):
            text = ax.text(j, i, f'{matrix[i, j]:.3f}',
                          ha='center', va='center', color='black', fontsize=12)

    ax.set_title('Feature Similarity Across Training Phases', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Cosine Similarity')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ============================================================================
# MAIN ANALYSIS WORKFLOW
# ============================================================================

def analyze_experiment(experiment_name: str, checkpoints_available: bool = False):
    """
    Analyze one experiment comparing early/mid/late features.

    If checkpoints_available=False, generates placeholder analysis.
    """

    print(f"\nAnalyzing: {experiment_name}")
    print("-" * 80)

    exp_config = checkpoint_plan['experiments'][experiment_name]
    checkpoint_steps = exp_config['checkpoint_steps']

    # Check checkpoint availability
    if checkpoints_available and TORCH_AVAILABLE:
        status = check_checkpoints_exist(experiment_name, checkpoint_steps)
        all_available = all(status.values())

        if not all_available:
            print("⚠ Checkpoints not yet available:")
            for step, exists in status.items():
                symbol = "✓" if exists else "✗"
                print(f"  {symbol} {step}")
            print()
            print("Generating placeholder analysis...")
            checkpoints_available = False

    # Actual analysis or placeholder
    if checkpoints_available and TORCH_AVAILABLE:
        # Load checkpoints
        print("Loading checkpoints...")
        checkpoint_early = load_checkpoint(experiment_name, checkpoint_steps[0])
        checkpoint_mid = load_checkpoint(experiment_name, checkpoint_steps[1])
        checkpoint_late = load_checkpoint(experiment_name, checkpoint_steps[2])

        # Extract features (architecture-specific)
        if 'transformer' in experiment_name:
            print("Extracting transformer attention patterns...")
            # Would need dataloader - placeholder for now
            features_early = None  # extract_transformer_features(model_early, dataloader)
            features_mid = None
            features_late = None

        elif 'cnn' in experiment_name:
            print("Extracting CNN filters...")
            # Would need model reconstruction - placeholder for now
            features_early = None  # extract_cnn_features(model_early)
            features_mid = None
            features_late = None

        else:  # MLP
            print("Extracting MLP activations...")
            features_early = None  # extract_mlp_features(model_early, dataloader)
            features_mid = None
            features_late = None

    else:
        print("Generating placeholder analysis (checkpoints not available)")

        # Simulate expected analysis structure
        results = {
            'experiment': experiment_name,
            'checkpoints': checkpoint_steps,
            'analysis': 'placeholder',
            'message': 'Run experiments to generate actual checkpoints',
            'expected_outputs': {
                'similarity_matrix': 'Cosine similarity between early/mid/late features',
                'diversity_measures': 'Feature diversity at each phase',
                'visualizations': 'Filter/attention/activation visualizations'
            },
            'hypothesis_test': {
                'question': 'Are early features qualitatively different from late features?',
                'metric': 'early_vs_late similarity < mid_vs_late similarity',
                'interpretation': 'If true, supports temporal boundary hypothesis'
            }
        }

        # Save placeholder
        output_file = results_dir / f'{experiment_name}_placeholder.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"✓ Saved placeholder: {output_file.name}")

    return results


# ============================================================================
# RUN ANALYSIS
# ============================================================================

print("=" * 80)
print("CHECKPOINT STATUS CHECK")
print("=" * 80)
print()

all_results = {}

for experiment_name in checkpoint_plan['experiments'].keys():
    results = analyze_experiment(experiment_name, checkpoints_available=False)
    all_results[experiment_name] = results

print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print()

print("Generated placeholder analysis for:")
for exp_name in all_results.keys():
    print(f"  ✓ {exp_name}")

print()
print("=" * 80)
print("NEXT STEPS")
print("=" * 80)
print()

print("To complete the analysis:")
print()
print("1. Re-run 3 experiments with checkpoint saving:")
print("   See checkpoint_plan_modified.json for exact steps")
print()
print("2. Checkpoints to save per experiment:")
for exp_name, exp_config in checkpoint_plan['experiments'].items():
    steps_str = ', '.join(str(s) for s in exp_config['checkpoint_steps'])
    print(f"   {exp_name}: [{steps_str}]")
print()
print("3. Then re-run this script with checkpoints_available=True")
print()
print("4. Expected analysis:")
print("   - Early vs late feature comparison")
print("   - Similarity matrices")
print("   - Diversity measures")
print("   - Architecture-specific visualizations")
print()

print("=" * 80)
print("HONEST FRAMING")
print("=" * 80)
print()

print("What this analysis will test:")
print("  ✓ Are early features qualitatively different from late features?")
print("  ✓ Do mid and late features look similar (refinement)?")
print("  ✓ Does temporal boundary align across architectures?")
print()

print("What this analysis will NOT claim:")
print("  ✗ Discrete phase transitions (magnitudes too small)")
print("  ✗ Mechanistic understanding (features ≠ mechanisms)")
print("  ✗ Causal relationships (correlation only)")
print()

print("Expected contribution:")
print("  'Dimensionality tracking identifies temporal boundaries where")
print("  representations may differ qualitatively'")
print()

print("=" * 80)
print("✓ Placeholder analysis complete")
print("=" * 80)
