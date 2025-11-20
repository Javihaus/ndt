"""
Step 1.2: Dimensionality Jump Characterization

Goal: Classify and understand jumps in layer dimensionality.

Analysis:
- Characterize jumps by magnitude, speed, and training phase
- Analyze temporal and layer distribution
- Use clustering to identify distinct jump types
- Identify architecture-specific patterns
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
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


def detect_jumps_in_timeseries(values: np.ndarray, threshold: float = 2.0) -> List[Dict]:
    """
    Detect significant jumps in a time series using z-score method.

    A jump is detected when the derivative exceeds threshold standard deviations.

    Returns:
        List of jump dictionaries with step, magnitude, and speed
    """
    if len(values) < 3:
        return []

    # Calculate first derivative (velocity)
    velocity = np.diff(values)

    # Calculate z-scores of velocity
    velocity_mean = np.mean(velocity)
    velocity_std = np.std(velocity)

    if velocity_std == 0:
        return []

    z_scores = np.abs((velocity - velocity_mean) / velocity_std)

    # Detect jumps where z-score exceeds threshold
    jump_indices = np.where(z_scores > threshold)[0]

    # Characterize each jump
    jumps = []
    for idx in jump_indices:
        # Calculate jump characteristics
        step = idx
        magnitude = abs(velocity[idx])

        # Calculate speed (rate of change over small window)
        window = 5
        start = max(0, idx - window//2)
        end = min(len(values), idx + window//2 + 1)
        speed = abs(values[end-1] - values[start]) / (end - start)

        jumps.append({
            'step': int(step),
            'magnitude': float(magnitude),
            'speed': float(speed),
            'direction': 'increase' if velocity[idx] > 0 else 'decrease'
        })

    return jumps


def analyze_layer_jumps(measurements: List[Dict], total_steps: int) -> Dict:
    """
    Analyze jumps for each layer in an experiment.

    Returns:
        Dictionary with layer-wise jump analysis
    """
    # Extract layer names
    layer_names = sorted(measurements[0]['layer_metrics'].keys())

    layer_jump_analysis = {}

    for layer_name in layer_names:
        # Extract dimensionality time series for this layer
        dim_series = np.array([
            m['layer_metrics'][layer_name]['stable_rank']
            for m in measurements
        ])

        # Detect jumps
        jumps = detect_jumps_in_timeseries(dim_series, threshold=2.0)

        # Add training phase info
        for jump in jumps:
            jump['phase'] = jump['step'] / total_steps
            jump['layer'] = layer_name

        layer_jump_analysis[layer_name] = {
            'num_jumps': len(jumps),
            'jumps': jumps
        }

    return layer_jump_analysis


def classify_jump_phase(phase: float) -> str:
    """Classify jump into early/mid/late training phase."""
    if phase < 0.33:
        return 'early'
    elif phase < 0.67:
        return 'mid'
    else:
        return 'late'


def run_jump_characterization():
    """
    Run jump characterization analysis on all experiments.
    """
    # Setup paths
    data_dir = Path('/home/user/ndt/experiments/new/results/phase1_full')
    output_dir = Path('/home/user/ndt/experiments/mechanistic_interpretability/results/step1_2')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all experiment files
    experiment_files = sorted(data_dir.glob('*.json'))

    print(f"Found {len(experiment_files)} experiments to analyze")
    print("=" * 80)

    # Collect all jumps across experiments
    all_jumps = []
    experiment_summaries = []

    for i, exp_file in enumerate(experiment_files, 1):
        print(f"\n[{i}/{len(experiment_files)}] Analyzing: {exp_file.stem}")

        # Load data
        exp_data = load_experiment_data(exp_file)
        measurements = exp_data['measurements']
        total_steps = exp_data['num_steps']

        # Analyze layer-wise jumps
        layer_analysis = analyze_layer_jumps(measurements, total_steps)

        # Collect all jumps for this experiment
        exp_jumps = []
        for layer_name, analysis in layer_analysis.items():
            exp_jumps.extend(analysis['jumps'])

        # Add metadata to jumps
        for jump in exp_jumps:
            jump['experiment'] = exp_file.stem
            jump['arch_name'] = exp_data['arch_name']
            jump['dataset'] = exp_data['dataset_name']
            jump['phase_category'] = classify_jump_phase(jump['phase'])

        all_jumps.extend(exp_jumps)

        # Summary for this experiment
        total_jumps = sum(a['num_jumps'] for a in layer_analysis.values())
        print(f"  Total jumps: {total_jumps}")
        print(f"  Jumps per layer: {total_jumps / len(layer_analysis):.1f}")

        # Identify layers with most jumps
        if total_jumps > 0:
            max_jump_layer = max(layer_analysis.items(), key=lambda x: x[1]['num_jumps'])
            print(f"  Most active layer: {max_jump_layer[0]} ({max_jump_layer[1]['num_jumps']} jumps)")

        experiment_summaries.append({
            'experiment': exp_file.stem,
            'arch_name': exp_data['arch_name'],
            'dataset': exp_data['dataset_name'],
            'total_jumps': total_jumps,
            'num_layers': len(layer_analysis),
            'jumps_per_layer': total_jumps / len(layer_analysis),
            'layer_analysis': {k: v['num_jumps'] for k, v in layer_analysis.items()}
        })

    print("\n" + "=" * 80)
    print("JUMP CHARACTERIZATION SUMMARY")
    print("=" * 80)

    # Overall statistics
    print(f"\nTotal jumps detected: {len(all_jumps)}")
    print(f"Experiments with jumps: {sum(1 for s in experiment_summaries if s['total_jumps'] > 0)}")

    if len(all_jumps) == 0:
        print("\nNo jumps detected in any experiment!")
        print("This suggests stable, smooth dimensionality evolution.")

        # Save empty results
        results = {
            'summary': 'No dimensionality jumps detected',
            'total_jumps': 0,
            'experiment_summaries': experiment_summaries
        }

        with open(output_dir / 'jump_characterization_summary.json', 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_dir / 'jump_characterization_summary.json'}")
        return results

    # Convert to DataFrame for analysis
    df_jumps = pd.DataFrame(all_jumps)

    # Phase distribution
    print("\nJumps by Training Phase:")
    phase_counts = df_jumps['phase_category'].value_counts()
    for phase, count in phase_counts.items():
        percentage = 100 * count / len(df_jumps)
        print(f"  {phase}: {count} ({percentage:.1f}%)")

    # Architecture analysis
    print("\nJumps by Architecture:")
    for arch_type in ['mlp', 'cnn', 'transformer']:
        arch_jumps = df_jumps[df_jumps['arch_name'].str.contains(arch_type)]
        if len(arch_jumps) > 0:
            print(f"  {arch_type.upper()}: {len(arch_jumps)} jumps")
            print(f"    Mean magnitude: {arch_jumps['magnitude'].mean():.3f}")
            print(f"    Mean speed: {arch_jumps['speed'].mean():.3f}")

    # Clustering analysis
    print("\nClustering jumps into types...")

    # Prepare features for clustering
    features = df_jumps[['phase', 'magnitude', 'speed']].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # K-means clustering (try different k values)
    n_clusters = min(5, len(df_jumps) // 10 + 1)  # Adaptive number of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_jumps['cluster'] = kmeans.fit_predict(features_scaled)

    print(f"\nIdentified {n_clusters} jump types:")
    for cluster_id in range(n_clusters):
        cluster_jumps = df_jumps[df_jumps['cluster'] == cluster_id]
        print(f"\n  Type {cluster_id + 1}: {len(cluster_jumps)} jumps")
        print(f"    Phase: {cluster_jumps['phase'].mean():.2f} ± {cluster_jumps['phase'].std():.2f}")
        print(f"    Magnitude: {cluster_jumps['magnitude'].mean():.3f} ± {cluster_jumps['magnitude'].std():.3f}")
        print(f"    Speed: {cluster_jumps['speed'].mean():.3f} ± {cluster_jumps['speed'].std():.3f}")
        print(f"    Dominant phase: {cluster_jumps['phase_category'].mode()[0] if len(cluster_jumps) > 0 else 'N/A'}")

    # Create visualizations
    print("\nCreating visualizations...")

    # Plot 1: Jump distribution over training
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1a: Histogram of jump phases
    axes[0, 0].hist(df_jumps['phase'], bins=30, color='steelblue', edgecolor='black')
    axes[0, 0].set_xlabel('Training Phase')
    axes[0, 0].set_ylabel('Number of Jumps')
    axes[0, 0].set_title('Temporal Distribution of Jumps')
    axes[0, 0].axvline(x=0.33, color='r', linestyle='--', alpha=0.5, label='Phase boundaries')
    axes[0, 0].axvline(x=0.67, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].legend()

    # 1b: Magnitude vs Phase scatter
    scatter = axes[0, 1].scatter(df_jumps['phase'], df_jumps['magnitude'],
                                 c=df_jumps['cluster'], cmap='viridis',
                                 alpha=0.6, s=50)
    axes[0, 1].set_xlabel('Training Phase')
    axes[0, 1].set_ylabel('Jump Magnitude')
    axes[0, 1].set_title('Jump Magnitude vs Training Phase')
    plt.colorbar(scatter, ax=axes[0, 1], label='Cluster')

    # 1c: Jump counts by architecture
    arch_counts = df_jumps.groupby('arch_name').size().sort_values(ascending=False).head(10)
    axes[1, 0].barh(range(len(arch_counts)), arch_counts.values, color='coral')
    axes[1, 0].set_yticks(range(len(arch_counts)))
    axes[1, 0].set_yticklabels(arch_counts.index, fontsize=9)
    axes[1, 0].set_xlabel('Number of Jumps')
    axes[1, 0].set_title('Top 10 Architectures by Jump Count')

    # 1d: Box plot of magnitude by phase category
    phase_order = ['early', 'mid', 'late']
    df_jumps_plot = df_jumps[df_jumps['phase_category'].isin(phase_order)]
    if len(df_jumps_plot) > 0:
        sns.boxplot(data=df_jumps_plot, x='phase_category', y='magnitude',
                   order=phase_order, ax=axes[1, 1], palette='Set2')
        axes[1, 1].set_xlabel('Training Phase')
        axes[1, 1].set_ylabel('Jump Magnitude')
        axes[1, 1].set_title('Jump Magnitude by Training Phase')

    plt.tight_layout()
    plt.savefig(output_dir / 'jump_characterization_overview.png')
    plt.close()

    # Plot 2: Cluster visualization
    fig = plt.figure(figsize=(12, 5))

    # 2a: 3D scatter of clusters
    ax1 = fig.add_subplot(121, projection='3d')
    scatter = ax1.scatter(df_jumps['phase'], df_jumps['magnitude'], df_jumps['speed'],
                         c=df_jumps['cluster'], cmap='viridis', s=50, alpha=0.6)
    ax1.set_xlabel('Training Phase')
    ax1.set_ylabel('Magnitude')
    ax1.set_zlabel('Speed')
    ax1.set_title('Jump Clusters (3D)')
    plt.colorbar(scatter, ax=ax1, label='Cluster', shrink=0.5)

    # 2b: Cluster characteristics
    ax2 = fig.add_subplot(122)
    cluster_summary = df_jumps.groupby('cluster').agg({
        'phase': 'mean',
        'magnitude': 'mean',
        'speed': 'mean'
    })
    cluster_summary_norm = (cluster_summary - cluster_summary.min()) / (cluster_summary.max() - cluster_summary.min())
    cluster_summary_norm.T.plot(kind='bar', ax=ax2, colormap='viridis')
    ax2.set_xlabel('Feature')
    ax2.set_ylabel('Normalized Value')
    ax2.set_title('Cluster Characteristics (Normalized)')
    ax2.legend(title='Cluster', labels=[f'Type {i+1}' for i in range(n_clusters)])
    ax2.set_xticklabels(['Phase', 'Magnitude', 'Speed'], rotation=0)

    plt.tight_layout()
    plt.savefig(output_dir / 'jump_clusters.png')
    plt.close()

    # Save results
    results = {
        'total_jumps': len(all_jumps),
        'experiments_with_jumps': sum(1 for s in experiment_summaries if s['total_jumps'] > 0),
        'phase_distribution': phase_counts.to_dict(),
        'cluster_info': {
            'n_clusters': n_clusters,
            'cluster_sizes': df_jumps['cluster'].value_counts().to_dict()
        },
        'experiment_summaries': experiment_summaries
    }

    with open(output_dir / 'jump_characterization_summary.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save detailed jump data
    df_jumps.to_csv(output_dir / 'all_jumps_detailed.csv', index=False)

    print(f"\nResults saved to: {output_dir}")
    print(f"  - jump_characterization_summary.json")
    print(f"  - all_jumps_detailed.csv")
    print(f"  - jump_characterization_overview.png")
    print(f"  - jump_clusters.png")

    return results


if __name__ == '__main__':
    print("Step 1.2: Dimensionality Jump Characterization")
    print("=" * 80)
    results = run_jump_characterization()
    print("\n✓ Analysis complete!")
