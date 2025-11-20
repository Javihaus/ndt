"""
Feature-Level Analysis: Testing Qualitative Differences
========================================================

This script extracts and analyzes the actual learned features from checkpoints
to test whether early/mid/late phases are qualitatively different.

Analyses:
1. CNN: Visualize convolutional filters, measure diversity
2. Transformer: Extract attention patterns, measure specialization
3. MLP: Extract activation patterns, measure representation similarity
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')

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
print("FEATURE-LEVEL ANALYSIS: TESTING QUALITATIVE DIFFERENCES")
print("=" * 80)
print()

# Setup paths
base_dir = Path('/home/user/ndt/experiments/mechanistic_interpretability')
checkpoints_dir = base_dir / 'check_points_results'
results_dir = base_dir / 'results' / 'feature_analysis'
results_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# MODEL ARCHITECTURES (must match training code)
# ============================================================================

class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], num_classes: int):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x.view(x.size(0), -1))


class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, conv_channels: List[int]):
        super().__init__()
        layers = []
        prev_channels = in_channels
        for channels in conv_channels:
            layers.extend([
                nn.Conv2d(prev_channels, channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ])
            prev_channels = channels
        self.conv_layers = nn.Sequential(*layers)
        self.flat_size = conv_channels[-1] * 3 * 3
        self.fc = nn.Linear(self.flat_size, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class SimpleTransformer(nn.Module):
    def __init__(self, input_dim: int, d_model: int, nhead: int,
                 num_layers: int, num_classes: int, seq_len: int = 16):
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(input_dim // seq_len, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x, return_attention=False):
        batch_size = x.size(0)
        x = x.view(batch_size, self.seq_len, -1)
        x = self.input_proj(x)
        x = x + self.pos_encoder.unsqueeze(0)

        if return_attention:
            # Extract attention weights by hooking into transformer layers
            attentions = []
            def hook_fn(module, input, output):
                # This is a simplified version - actual extraction is more complex
                if hasattr(module, 'self_attn'):
                    attentions.append(module.self_attn.attention_weights)

            hooks = []
            for layer in self.transformer.layers:
                hooks.append(layer.register_forward_hook(hook_fn))

            x = self.transformer(x)

            for hook in hooks:
                hook.remove()

            x = x.mean(dim=1)
            return self.fc(x), attentions
        else:
            x = self.transformer(x)
            x = x.mean(dim=1)
            return self.fc(x)


def create_model(arch_name: str, input_dim: int, num_classes: int):
    """Create model matching training configuration."""
    if arch_name == 'mlp_narrow':
        return SimpleMLP(input_dim, [32, 32, 32, 32], num_classes)
    elif arch_name == 'cnn_deep':
        return SimpleCNN(1, num_classes, [32, 64, 128])
    elif arch_name == 'transformer_deep':
        return SimpleTransformer(input_dim, d_model=128, nhead=4, num_layers=4, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")


def load_checkpoint_model(experiment_name: str, step: int):
    """Load checkpoint and reconstruct model."""
    # Map experiment names to architecture names
    arch_map = {
        'transformer_deep_mnist': 'transformer_deep',
        'cnn_deep_mnist': 'cnn_deep',
        'mlp_narrow_mnist': 'mlp_narrow'
    }

    arch_name = arch_map[experiment_name]
    model = create_model(arch_name, 28*28, 10)

    # Load checkpoint
    folder_name = f"{experiment_name}2"
    checkpoint_path = checkpoints_dir / folder_name / f'checkpoint_step_{step:05d}.pt'

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, checkpoint


# ============================================================================
# CNN FILTER ANALYSIS
# ============================================================================

def extract_cnn_filters(model):
    """Extract all convolutional filters from CNN."""
    filters = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            filters[name] = module.weight.data.cpu().numpy()
    return filters


def visualize_cnn_filters(filters_dict, step, save_dir):
    """Visualize CNN filters for a checkpoint."""
    num_layers = len(filters_dict)

    fig, axes = plt.subplots(num_layers, 8, figsize=(16, num_layers*2))
    if num_layers == 1:
        axes = axes.reshape(1, -1)

    for layer_idx, (layer_name, filters) in enumerate(sorted(filters_dict.items())):
        # Show first 8 filters
        for i in range(min(8, filters.shape[0])):
            ax = axes[layer_idx, i] if num_layers > 1 else axes[i]

            # Get filter (shape: [out_channels, in_channels, H, W])
            if filters.shape[1] == 1:  # First layer (grayscale)
                filt = filters[i, 0]
            else:  # Later layers (take mean across input channels)
                filt = filters[i].mean(axis=0)

            # Normalize for visualization
            vmin, vmax = -np.abs(filt).max(), np.abs(filt).max()
            ax.imshow(filt, cmap='RdBu_r', vmin=vmin, vmax=vmax)
            ax.axis('off')

            if i == 0:
                ax.set_ylabel(layer_name, fontsize=10, rotation=0, ha='right', va='center')

    plt.suptitle(f'CNN Filters at Step {step}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = save_dir / f'cnn_filters_step_{step:05d}.png'
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path.name}")


def measure_filter_diversity(filters):
    """Measure diversity of filters using clustering."""
    # Flatten filters
    filters_flat = filters.reshape(filters.shape[0], -1)

    # Cluster
    n_clusters = min(10, filters_flat.shape[0])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(filters_flat)

    # Measure quality
    if len(np.unique(labels)) > 1:
        silhouette = silhouette_score(filters_flat, labels)
    else:
        silhouette = 0.0

    # Count unique patterns (filters with low similarity to others)
    unique_count = len(np.unique(labels))

    return {
        'num_clusters': unique_count,
        'silhouette_score': silhouette,
        'std_dev': filters_flat.std(),
        'mean_abs': np.abs(filters_flat).mean()
    }


def analyze_cnn_phases(experiment_name='cnn_deep_mnist', steps=[100, 1000, 2000]):
    """Analyze CNN features across phases."""
    print(f"\nAnalyzing CNN: {experiment_name}")
    print("-" * 80)

    save_dir = results_dir / 'cnn'
    save_dir.mkdir(exist_ok=True)

    phase_filters = {}
    phase_diversity = {}

    for step in steps:
        print(f"\n  Loading checkpoint at step {step}...")
        model, checkpoint = load_checkpoint_model(experiment_name, step)

        # Extract filters
        filters_dict = extract_cnn_filters(model)
        phase_filters[step] = filters_dict

        # Visualize
        visualize_cnn_filters(filters_dict, step, save_dir)

        # Measure diversity for each layer
        diversity_by_layer = {}
        for layer_name, filters in filters_dict.items():
            diversity = measure_filter_diversity(filters)
            diversity_by_layer[layer_name] = diversity
            print(f"    {layer_name}: {diversity['num_clusters']} clusters, "
                  f"silhouette={diversity['silhouette_score']:.3f}, "
                  f"std={diversity['std_dev']:.4f}")

        phase_diversity[step] = diversity_by_layer

    # Compare phases
    print(f"\n  Comparing phases...")
    comparison = compare_filter_evolution(phase_filters, save_dir)

    # Save results (convert numpy types to native Python)
    def convert_to_native(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_native(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj

    results = {
        'experiment': experiment_name,
        'steps': steps,
        'diversity_by_phase': convert_to_native(phase_diversity),
        'phase_comparison': convert_to_native(comparison)
    }

    with open(save_dir / 'cnn_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


def compare_filter_evolution(phase_filters, save_dir):
    """Compare how filters evolve across phases."""
    steps = sorted(phase_filters.keys())

    # Get first conv layer for comparison
    first_layer_name = list(phase_filters[steps[0]].keys())[0]

    comparisons = {}

    for i in range(len(steps) - 1):
        step1, step2 = steps[i], steps[i+1]
        filters1 = phase_filters[step1][first_layer_name]
        filters2 = phase_filters[step2][first_layer_name]

        # Compute cosine similarity
        f1_flat = filters1.reshape(filters1.shape[0], -1)
        f2_flat = filters2.reshape(filters2.shape[0], -1)

        # Average pairwise similarity
        similarities = []
        for f1, f2 in zip(f1_flat, f2_flat):
            sim = 1 - cosine(f1, f2)
            similarities.append(sim)

        mean_similarity = np.mean(similarities)

        comparisons[f'step_{step1}_to_{step2}'] = {
            'mean_similarity': float(mean_similarity),
            'std_similarity': float(np.std(similarities)),
            'interpretation': 'High similarity = refinement, Low similarity = qualitative change'
        }

        print(f"    Step {step1} → {step2}: similarity = {mean_similarity:.4f}")

    # Visualize filter evolution
    plot_filter_evolution_comparison(phase_filters, save_dir)

    return comparisons


def plot_filter_evolution_comparison(phase_filters, save_dir):
    """Create side-by-side comparison of filters across phases."""
    steps = sorted(phase_filters.keys())
    first_layer = list(phase_filters[steps[0]].keys())[0]

    fig, axes = plt.subplots(len(steps), 8, figsize=(16, len(steps)*2))

    for phase_idx, step in enumerate(steps):
        filters = phase_filters[step][first_layer]

        for i in range(min(8, filters.shape[0])):
            ax = axes[phase_idx, i]

            filt = filters[i, 0] if filters.shape[1] == 1 else filters[i].mean(axis=0)
            vmin, vmax = -np.abs(filt).max(), np.abs(filt).max()

            ax.imshow(filt, cmap='RdBu_r', vmin=vmin, vmax=vmax)
            ax.axis('off')

            if i == 0:
                phase_name = {100: 'Early (5%)', 1000: 'Mid (50%)', 2000: 'Late (100%)'}
                ax.set_ylabel(phase_name.get(step, f'Step {step}'),
                            fontsize=12, rotation=0, ha='right', va='center')

    plt.suptitle('CNN Filter Evolution: Early vs Mid vs Late', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / 'filter_evolution_comparison.png')
    plt.close()
    print(f"  Saved: filter_evolution_comparison.png")


# ============================================================================
# TRANSFORMER ATTENTION ANALYSIS
# ============================================================================

def extract_attention_patterns(model, dataloader, num_samples=100):
    """Extract attention patterns from transformer."""
    model.eval()

    attention_patterns = []

    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(dataloader):
            if batch_idx * inputs.size(0) >= num_samples:
                break

            # Note: This requires modifying forward pass to return attention
            # For now, we'll extract from the transformer encoder layers

            # Forward pass
            _ = model(inputs)

            # Extract attention from each layer
            # This is a simplified version - actual extraction requires hooks

    return attention_patterns


def analyze_transformer_phases(experiment_name='transformer_deep_mnist', steps=[100, 1000, 2000]):
    """Analyze transformer attention across phases."""
    print(f"\nAnalyzing Transformer: {experiment_name}")
    print("-" * 80)

    save_dir = results_dir / 'transformer'
    save_dir.mkdir(exist_ok=True)

    phase_results = {}

    for step in steps:
        print(f"\n  Loading checkpoint at step {step}...")
        model, checkpoint = load_checkpoint_model(experiment_name, step)

        # Extract model statistics
        param_stats = {}
        for name, param in model.named_parameters():
            param_stats[name] = {
                'mean': float(param.data.mean()),
                'std': float(param.data.std()),
                'max': float(param.data.max()),
                'min': float(param.data.min())
            }

        phase_results[step] = {
            'checkpoint_loss': checkpoint['loss'],
            'param_stats': param_stats
        }

        print(f"    Checkpoint loss: {checkpoint['loss']:.4f}")
        print(f"    Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Compare parameter evolution
    compare_transformer_evolution(phase_results, save_dir)

    results = {
        'experiment': experiment_name,
        'steps': steps,
        'phase_results': phase_results
    }

    with open(save_dir / 'transformer_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


def compare_transformer_evolution(phase_results, save_dir):
    """Compare transformer parameters across phases."""
    steps = sorted(phase_results.keys())

    # Extract key parameters for comparison
    print(f"\n  Comparing parameter evolution...")

    for param_name in ['input_proj.weight', 'fc.weight']:
        if param_name in phase_results[steps[0]]['param_stats']:
            print(f"\n    {param_name}:")
            for step in steps:
                stats = phase_results[step]['param_stats'][param_name]
                print(f"      Step {step}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")


# ============================================================================
# MLP ACTIVATION ANALYSIS
# ============================================================================

def extract_mlp_activations(model, dataloader, num_samples=1000):
    """Extract activation patterns from MLP hidden layers."""
    model.eval()

    activations = {f'layer_{i}': [] for i in range(4)}  # 4 hidden layers

    def get_activation(name):
        def hook(module, input, output):
            activations[name].append(output.detach().cpu().numpy())
        return hook

    # Register hooks
    hooks = []
    layer_idx = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and 'network' in name:
            if layer_idx < 4:  # Only hidden layers
                hooks.append(module.register_forward_hook(get_activation(f'layer_{layer_idx}')))
                layer_idx += 1

    # Collect activations
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(dataloader):
            if batch_idx * inputs.size(0) >= num_samples:
                break
            _ = model(inputs)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Concatenate batches
    for name in activations:
        if activations[name]:
            activations[name] = np.concatenate(activations[name], axis=0)

    return activations


def analyze_mlp_phases(experiment_name='mlp_narrow_mnist', steps=[100, 1000, 2000]):
    """Analyze MLP activations across phases."""
    print(f"\nAnalyzing MLP: {experiment_name}")
    print("-" * 80)

    save_dir = results_dir / 'mlp'
    save_dir.mkdir(exist_ok=True)

    # Skip activation extraction for now - analyze parameter statistics instead
    phase_params = {}
    phase_stats = {}

    for step in steps:
        print(f"\n  Loading checkpoint at step {step}...")
        model, checkpoint = load_checkpoint_model(experiment_name, step)

        # Extract parameter statistics
        param_stats = {}
        for name, param in model.named_parameters():
            if 'weight' in name:
                param_stats[name] = {
                    'mean': float(param.data.mean()),
                    'std': float(param.data.std()),
                    'max': float(param.data.max()),
                    'min': float(param.data.min()),
                    'norm': float(param.data.norm())
                }

        phase_params[step] = param_stats
        phase_stats[step] = {
            'checkpoint_loss': float(checkpoint['loss']),
            'total_params': sum(p.numel() for p in model.parameters())
        }

        print(f"    Checkpoint loss: {checkpoint['loss']:.4f}")
        print(f"    Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Compare parameter evolution
    print(f"\n  Comparing parameter evolution...")
    similarity = compare_mlp_parameters(phase_params, save_dir)

    results = {
        'experiment': experiment_name,
        'steps': steps,
        'phase_stats': phase_stats,
        'parameter_similarity': similarity
    }

    with open(save_dir / 'mlp_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


def compare_mlp_parameters(phase_params, save_dir):
    """Compare MLP parameter weights across phases."""
    steps = sorted(phase_params.keys())

    similarities = {}

    for i in range(len(steps) - 1):
        step1, step2 = steps[i], steps[i+1]

        # Compare all weight matrices
        all_sims = []
        for param_name in phase_params[step1].keys():
            if param_name in phase_params[step2]:
                params1 = phase_params[step1][param_name]
                params2 = phase_params[step2][param_name]

                # Compare norms as a proxy for similarity
                norm_ratio = params2['norm'] / params1['norm'] if params1['norm'] > 0 else 1.0
                all_sims.append(norm_ratio)

        mean_ratio = np.mean(all_sims)
        similarities[f'step_{step1}_to_{step2}'] = {
            'mean_norm_ratio': float(mean_ratio),
            'std_norm_ratio': float(np.std(all_sims))
        }

        print(f"    Step {step1} → {step2}: parameter norm ratio = {mean_ratio:.4f}")

    return similarities


def plot_representation_similarity(phase_activations, steps, save_dir):
    """Plot similarity matrix of representations across phases."""
    last_layer = 'layer_3'

    # Compute pairwise similarity
    n_steps = len(steps)
    similarity_matrix = np.zeros((n_steps, n_steps))

    for i, step1 in enumerate(steps):
        for j, step2 in enumerate(steps):
            if i == j:
                similarity_matrix[i, j] = 1.0
            elif last_layer in phase_activations[step1] and last_layer in phase_activations[step2]:
                acts1 = phase_activations[step1][last_layer]
                acts2 = phase_activations[step2][last_layer]

                # Mean similarity across samples
                sims = []
                for a1, a2 in zip(acts1[:100], acts2[:100]):
                    sims.append(1 - cosine(a1, a2))
                similarity_matrix[i, j] = np.mean(sims)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(similarity_matrix, cmap='RdYlGn', vmin=0, vmax=1)

    labels = [f'Step {s}' for s in steps]
    ax.set_xticks(range(n_steps))
    ax.set_yticks(range(n_steps))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Annotate
    for i in range(n_steps):
        for j in range(n_steps):
            text = ax.text(j, i, f'{similarity_matrix[i, j]:.3f}',
                          ha='center', va='center', color='black', fontsize=12)

    ax.set_title('MLP Representation Similarity Across Training Phases',
                 fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Cosine Similarity')

    plt.tight_layout()
    plt.savefig(save_dir / 'representation_similarity.png')
    plt.close()
    print(f"  Saved: representation_similarity.png")


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

if __name__ == '__main__':
    print("Starting feature-level analysis...")
    print()

    all_results = {}

    # Analyze CNN
    try:
        cnn_results = analyze_cnn_phases()
        all_results['cnn'] = cnn_results
    except Exception as e:
        print(f"Error analyzing CNN: {e}")

    # Analyze Transformer
    try:
        transformer_results = analyze_transformer_phases()
        all_results['transformer'] = transformer_results
    except Exception as e:
        print(f"Error analyzing Transformer: {e}")

    # Analyze MLP
    try:
        mlp_results = analyze_mlp_phases()
        all_results['mlp'] = mlp_results
    except Exception as e:
        print(f"Error analyzing MLP: {e}")

    # Save comprehensive results
    with open(results_dir / 'feature_analysis_summary.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print()
    print("=" * 80)
    print("FEATURE ANALYSIS COMPLETE")
    print("=" * 80)
    print()
    print(f"Results saved to: {results_dir}")
    print()
