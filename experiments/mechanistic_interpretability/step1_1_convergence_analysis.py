"""
Step 1.1: Layer-Wise Convergence Analysis

Goal: Determine if layers converge in systematic order (bottom-up, top-down, or chaotic).

Analysis:
- Extract layer-wise dimensionality time series
- Detect stabilization points (where variance drops below threshold)
- Calculate convergence order using Spearman correlation
- Generate heatmaps showing dimensionality evolution
- Identify experiments with clear convergence patterns
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr
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


def load_experiment_data(file_path: Path) -> Dict:
    """Load experiment data from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def extract_layer_wise_dimensionality(measurements: List[Dict]) -> Dict[str, np.ndarray]:
    """
    Extract dimensionality time series for each layer.

    Returns:
        Dictionary mapping layer name to dimensionality array over time
    """
    # Get all layer names from first measurement
    layer_names = sorted(measurements[0]['layer_metrics'].keys())

    # Initialize arrays
    layer_timeseries = {layer: [] for layer in layer_names}

    # Extract stable_rank for each layer at each timestep
    for measurement in measurements:
        for layer_name in layer_names:
            if layer_name in measurement['layer_metrics']:
                stable_rank = measurement['layer_metrics'][layer_name]['stable_rank']
                layer_timeseries[layer_name].append(stable_rank)

    # Convert to numpy arrays
    layer_timeseries = {k: np.array(v) for k, v in layer_timeseries.items()}

    return layer_timeseries


def detect_stabilization_point(dimensionality: np.ndarray,
                               window_size: int = 20,
                               variance_threshold: float = 0.1) -> int:
    """
    Detect when dimensionality stabilizes.

    Stabilization = when rolling variance drops below threshold

    Args:
        dimensionality: Time series of dimensionality values
        window_size: Size of rolling window for variance calculation
        variance_threshold: Threshold as fraction of mean dimensionality

    Returns:
        Step index where stabilization occurs (-1 if never stabilizes)
    """
    if len(dimensionality) < window_size:
        return -1

    mean_dim = np.mean(dimensionality)
    threshold = variance_threshold * mean_dim

    # Calculate rolling variance
    for i in range(window_size, len(dimensionality)):
        window = dimensionality[i-window_size:i]
        variance = np.var(window)

        if variance < threshold ** 2:
            return i - window_size  # Return start of stable window

    return -1  # Never stabilizes


def extract_layer_index(layer_name: str) -> int:
    """
    Extract layer index from layer name.

    Handles various formats:
    - "network.0" -> 0
    - "conv_layers.3" -> 3
    - "fc" -> 999 (assign high index for final layers)
    """
    if '.' in layer_name:
        parts = layer_name.split('.')
        try:
            return int(parts[-1])
        except ValueError:
            # If last part is not a number (e.g., "fc"), use a high index
            return 999
    else:
        # Extract any digits from the name
        digits = ''.join(filter(str.isdigit, layer_name))
        if digits:
            return int(digits)
        else:
            # No digits found, assign high index
            return 999


def calculate_convergence_order(stabilization_points: Dict[str, int]) -> Tuple[float, float]:
    """
    Calculate convergence order using Spearman correlation.

    Tests if layer depth correlates with stabilization time:
    - ρ > 0.6: bottom-up convergence (early layers first)
    - ρ < -0.6: top-down convergence (late layers first)
    - |ρ| < 0.3: simultaneous convergence

    Returns:
        (correlation, p-value)
    """
    # Extract layer indices and stabilization times
    layers = []
    stab_times = []

    for layer_name, stab_time in stabilization_points.items():
        if stab_time >= 0:  # Only include layers that stabilized
            layer_idx = extract_layer_index(layer_name)
            layers.append(layer_idx)
            stab_times.append(stab_time)

    if len(layers) < 3:
        return 0.0, 1.0

    # Spearman correlation
    correlation, p_value = spearmanr(layers, stab_times)

    return correlation, p_value


def analyze_experiment_convergence(experiment_data: Dict) -> Dict:
    """
    Analyze convergence patterns for a single experiment.

    Returns:
        Dictionary with convergence analysis results
    """
    measurements = experiment_data['measurements']

    # Extract layer-wise dimensionality
    layer_timeseries = extract_layer_wise_dimensionality(measurements)

    # Detect stabilization points for each layer
    stabilization_points = {}
    for layer_name, dim_series in layer_timeseries.items():
        stab_point = detect_stabilization_point(dim_series)
        stabilization_points[layer_name] = stab_point

    # Calculate convergence order
    correlation, p_value = calculate_convergence_order(stabilization_points)

    # Classify convergence pattern
    if correlation > 0.6 and p_value < 0.05:
        pattern = "bottom-up"
    elif correlation < -0.6 and p_value < 0.05:
        pattern = "top-down"
    elif abs(correlation) < 0.3:
        pattern = "simultaneous"
    else:
        pattern = "unclear"

    return {
        'layer_timeseries': layer_timeseries,
        'stabilization_points': stabilization_points,
        'convergence_correlation': correlation,
        'convergence_p_value': p_value,
        'convergence_pattern': pattern,
        'num_stabilized_layers': sum(1 for v in stabilization_points.values() if v >= 0),
        'total_layers': len(layer_timeseries)
    }


def create_convergence_heatmap(layer_timeseries: Dict[str, np.ndarray],
                               stabilization_points: Dict[str, int],
                               experiment_name: str,
                               save_path: Path):
    """
    Create heatmap showing dimensionality evolution across layers.

    Rows = layers, Columns = training steps, Color = dimensionality
    """
    # Prepare data matrix
    layer_names = sorted(layer_timeseries.keys(), key=extract_layer_index)

    # Create matrix: rows = layers, cols = time steps
    matrix = np.array([layer_timeseries[layer] for layer in layer_names])

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot heatmap
    im = ax.imshow(matrix, aspect='auto', cmap='viridis', interpolation='nearest')

    # Mark stabilization points
    for i, layer_name in enumerate(layer_names):
        stab_point = stabilization_points[layer_name]
        if stab_point >= 0:
            ax.plot(stab_point, i, 'r*', markersize=10, markeredgecolor='white', markeredgewidth=0.5)

    # Labels
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Layer Index')
    ax.set_title(f'Layer-Wise Dimensionality Evolution\n{experiment_name}')

    # Y-axis: show layer names (simplified)
    layer_labels = [name.replace('conv_layers.', 'C').replace('network.', 'L').replace('fc', 'FC')
                   for name in layer_names]
    ax.set_yticks(range(len(layer_labels)))
    ax.set_yticklabels(layer_labels)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='Stable Rank (Dimensionality)')

    # Legend for stabilization markers
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='*', color='w',
                             markerfacecolor='r', markersize=10,
                             markeredgecolor='white', markeredgewidth=0.5,
                             label='Stabilization Point')]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def run_convergence_analysis():
    """
    Run convergence analysis on all experiments.
    """
    # Setup paths
    data_dir = Path('/home/user/ndt/experiments/new/results/phase1_full')
    output_dir = Path('/home/user/ndt/experiments/mechanistic_interpretability/results/step1_1')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all experiment files
    experiment_files = sorted(data_dir.glob('*.json'))

    print(f"Found {len(experiment_files)} experiments to analyze")
    print("=" * 80)

    # Store results
    all_results = []

    for i, exp_file in enumerate(experiment_files, 1):
        print(f"\n[{i}/{len(experiment_files)}] Analyzing: {exp_file.stem}")

        # Load data
        exp_data = load_experiment_data(exp_file)

        # Run analysis
        results = analyze_experiment_convergence(exp_data)

        # Add metadata
        results['experiment_name'] = exp_file.stem
        results['arch_name'] = exp_data['arch_name']
        results['dataset_name'] = exp_data['dataset_name']
        results['num_layers'] = len(results['layer_timeseries'])

        # Print summary
        print(f"  Pattern: {results['convergence_pattern']}")
        print(f"  Correlation: {results['convergence_correlation']:.3f} (p={results['convergence_p_value']:.3f})")
        print(f"  Stabilized: {results['num_stabilized_layers']}/{results['total_layers']} layers")

        # Create heatmap
        heatmap_path = output_dir / f"{exp_file.stem}_heatmap.png"
        create_convergence_heatmap(
            results['layer_timeseries'],
            results['stabilization_points'],
            exp_file.stem,
            heatmap_path
        )

        # Remove layer_timeseries from results (too large for JSON)
        results_summary = {k: v for k, v in results.items() if k != 'layer_timeseries'}
        all_results.append(results_summary)

    # Create summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    # Count patterns
    patterns = [r['convergence_pattern'] for r in all_results]
    pattern_counts = pd.Series(patterns).value_counts()

    print("\nConvergence Patterns:")
    for pattern, count in pattern_counts.items():
        percentage = 100 * count / len(all_results)
        print(f"  {pattern}: {count} ({percentage:.1f}%)")

    # Architecture-specific patterns
    df = pd.DataFrame(all_results)

    print("\nPatterns by Architecture Type:")
    for arch_type in ['mlp', 'cnn', 'transformer']:
        arch_data = df[df['arch_name'].str.contains(arch_type)]
        if len(arch_data) > 0:
            arch_patterns = arch_data['convergence_pattern'].value_counts()
            print(f"\n  {arch_type.upper()}:")
            for pattern, count in arch_patterns.items():
                percentage = 100 * count / len(arch_data)
                print(f"    {pattern}: {count} ({percentage:.1f}%)")

    # Correlation statistics
    correlations = [r['convergence_correlation'] for r in all_results]
    print(f"\nCorrelation Statistics:")
    print(f"  Mean: {np.mean(correlations):.3f}")
    print(f"  Median: {np.median(correlations):.3f}")
    print(f"  Std: {np.std(correlations):.3f}")

    # Save results
    results_file = output_dir / 'convergence_analysis_summary.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Create summary plot: distribution of convergence patterns
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Pattern distribution
    pattern_counts.plot(kind='bar', ax=axes[0], color='steelblue')
    axes[0].set_title('Distribution of Convergence Patterns')
    axes[0].set_xlabel('Pattern')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)

    # Plot 2: Correlation distribution by pattern
    pattern_colors = {'bottom-up': 'green', 'top-down': 'red',
                     'simultaneous': 'blue', 'unclear': 'gray'}
    for pattern in df['convergence_pattern'].unique():
        pattern_data = df[df['convergence_pattern'] == pattern]['convergence_correlation']
        axes[1].hist(pattern_data, alpha=0.6, label=pattern,
                    color=pattern_colors.get(pattern, 'gray'), bins=15)

    axes[1].set_title('Convergence Correlation Distribution')
    axes[1].set_xlabel('Spearman Correlation')
    axes[1].set_ylabel('Count')
    axes[1].legend()
    axes[1].axvline(x=0.6, color='green', linestyle='--', alpha=0.5, label='Bottom-up threshold')
    axes[1].axvline(x=-0.6, color='red', linestyle='--', alpha=0.5, label='Top-down threshold')

    plt.tight_layout()
    summary_plot_path = output_dir / 'convergence_summary.png'
    plt.savefig(summary_plot_path)
    plt.close()

    print(f"Summary plot saved to: {summary_plot_path}")

    # Identify best examples for each pattern
    print("\n" + "=" * 80)
    print("TOP EXAMPLES FOR EACH PATTERN")
    print("=" * 80)

    for pattern in ['bottom-up', 'top-down', 'simultaneous']:
        pattern_data = df[df['convergence_pattern'] == pattern]
        if len(pattern_data) > 0:
            # For bottom-up/top-down: highest absolute correlation
            # For simultaneous: lowest absolute correlation
            if pattern == 'simultaneous':
                best = pattern_data.nsmallest(3, 'convergence_correlation', keep='all')
            else:
                best = pattern_data.nlargest(3, 'convergence_correlation', keep='all')

            print(f"\n{pattern.upper()}:")
            for _, row in best.iterrows():
                print(f"  - {row['experiment_name']}")
                print(f"    Correlation: {row['convergence_correlation']:.3f}, "
                      f"Stabilized: {row['num_stabilized_layers']}/{row['total_layers']}")

    return all_results


if __name__ == '__main__':
    print("Step 1.1: Layer-Wise Convergence Analysis")
    print("=" * 80)
    results = run_convergence_analysis()
    print("\n✓ Analysis complete!")
