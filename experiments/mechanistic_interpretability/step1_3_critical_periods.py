"""
Step 1.3: Critical Period Identification

Goal: Find training windows where dimensionality changes rapidly.

Analysis:
- Calculate dimensionality velocity and acceleration
- Identify high-velocity periods (rapid change)
- Identify transition points (acceleration peaks)
- Check cross-layer coordination
- Correlate with loss dynamics
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


def load_experiment_data(file_path: Path) -> Dict:
    """Load experiment data from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def calculate_velocity_acceleration(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate velocity (first derivative) and acceleration (second derivative).

    Returns:
        (velocity, acceleration) arrays
    """
    # First derivative (velocity)
    velocity = np.diff(values)

    # Second derivative (acceleration)
    acceleration = np.diff(velocity)

    return velocity, acceleration


def identify_critical_periods(values: np.ndarray, percentile: float = 90) -> List[int]:
    """
    Identify periods of rapid change (high velocity).

    Args:
        values: Time series of dimensionality
        percentile: Threshold percentile for velocity

    Returns:
        List of step indices where velocity exceeds threshold
    """
    velocity, _ = calculate_velocity_acceleration(values)

    # Calculate threshold
    threshold = np.percentile(np.abs(velocity), percentile)

    # Identify steps exceeding threshold
    critical_steps = np.where(np.abs(velocity) > threshold)[0]

    return critical_steps.tolist()


def identify_transitions(values: np.ndarray, percentile: float = 90) -> List[int]:
    """
    Identify transition points (acceleration peaks).

    Args:
        values: Time series of dimensionality
        percentile: Threshold percentile for acceleration

    Returns:
        List of step indices where acceleration peaks occur
    """
    _, acceleration = calculate_velocity_acceleration(values)

    # Calculate threshold
    threshold = np.percentile(np.abs(acceleration), percentile)

    # Identify steps exceeding threshold
    transition_steps = np.where(np.abs(acceleration) > threshold)[0]

    return transition_steps.tolist()


def analyze_cross_layer_coordination(measurements: List[Dict], critical_periods: Dict[str, List[int]]) -> Dict:
    """
    Check if critical periods align across layers.

    Returns:
        Dictionary with coordination analysis
    """
    layer_names = list(critical_periods.keys())

    # For each step, count how many layers are in critical period
    max_steps = len(measurements)
    coordination_score = np.zeros(max_steps)

    for layer_name, critical_steps in critical_periods.items():
        for step in critical_steps:
            if step < max_steps:
                coordination_score[step] += 1

    # Normalize by number of layers
    coordination_score = coordination_score / len(layer_names)

    # Identify coordinated periods (where many layers change together)
    coordinated_threshold = 0.5  # At least 50% of layers active
    coordinated_periods = np.where(coordination_score >= coordinated_threshold)[0]

    return {
        'coordination_score': coordination_score.tolist(),
        'coordinated_periods': coordinated_periods.tolist(),
        'max_coordination': float(coordination_score.max()),
        'mean_coordination': float(coordination_score.mean())
    }


def correlate_with_loss(measurements: List[Dict], critical_periods: Dict[str, List[int]]) -> Dict:
    """
    Correlate critical periods with loss dynamics.

    Returns:
        Dictionary with correlation analysis
    """
    # Extract loss curve
    losses = np.array([m['loss'] for m in measurements])

    # Calculate loss velocity
    loss_velocity, _ = calculate_velocity_acceleration(losses)

    # For each layer's critical periods, check if they coincide with loss drops
    layer_loss_correlation = {}

    for layer_name, critical_steps in critical_periods.items():
        if len(critical_steps) == 0:
            layer_loss_correlation[layer_name] = {
                'mean_loss_velocity': 0.0,
                'loss_drop_fraction': 0.0
            }
            continue

        # Get loss velocity at critical steps
        valid_steps = [s for s in critical_steps if s < len(loss_velocity)]
        if len(valid_steps) == 0:
            layer_loss_correlation[layer_name] = {
                'mean_loss_velocity': 0.0,
                'loss_drop_fraction': 0.0
            }
            continue

        loss_velocities_at_critical = [loss_velocity[s] for s in valid_steps]
        mean_loss_velocity = np.mean(loss_velocities_at_critical)

        # Fraction of critical periods with loss drops (negative velocity)
        loss_drop_fraction = np.mean([v < 0 for v in loss_velocities_at_critical])

        layer_loss_correlation[layer_name] = {
            'mean_loss_velocity': float(mean_loss_velocity),
            'loss_drop_fraction': float(loss_drop_fraction)
        }

    return {
        'layer_correlations': layer_loss_correlation,
        'overall_correlation': np.mean([v['loss_drop_fraction'] for v in layer_loss_correlation.values()])
    }


def run_critical_period_analysis():
    """
    Run critical period analysis on all experiments.
    """
    # Setup paths
    data_dir = Path('/home/user/ndt/experiments/new/results/phase1_full')
    output_dir = Path('/home/user/ndt/experiments/mechanistic_interpretability/results/step1_3')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all experiment files
    experiment_files = sorted(data_dir.glob('*.json'))

    print(f"Found {len(experiment_files)} experiments to analyze")
    print("=" * 80)

    # Collect results
    all_results = []

    for i, exp_file in enumerate(experiment_files, 1):
        print(f"\n[{i}/{len(experiment_files)}] Analyzing: {exp_file.stem}")

        # Load data
        exp_data = load_experiment_data(exp_file)
        measurements = exp_data['measurements']

        # Extract layer names
        layer_names = sorted(measurements[0]['layer_metrics'].keys())

        # Analyze each layer
        layer_critical_periods = {}
        layer_transitions = {}

        for layer_name in layer_names:
            # Extract dimensionality time series
            dim_series = np.array([
                m['layer_metrics'][layer_name]['stable_rank']
                for m in measurements
            ])

            # Identify critical periods and transitions
            critical_steps = identify_critical_periods(dim_series, percentile=90)
            transition_steps = identify_transitions(dim_series, percentile=90)

            layer_critical_periods[layer_name] = critical_steps
            layer_transitions[layer_name] = transition_steps

        # Cross-layer coordination analysis
        coordination = analyze_cross_layer_coordination(measurements, layer_critical_periods)

        # Correlation with loss
        loss_correlation = correlate_with_loss(measurements, layer_critical_periods)

        # Summary
        total_critical_steps = sum(len(steps) for steps in layer_critical_periods.values())
        total_transitions = sum(len(steps) for steps in layer_transitions.values())

        print(f"  Critical periods: {total_critical_steps} total")
        print(f"  Transitions: {total_transitions} total")
        print(f"  Max coordination: {coordination['max_coordination']:.2f}")
        print(f"  Loss correlation: {loss_correlation['overall_correlation']:.2f}")

        # Store results
        all_results.append({
            'experiment': exp_file.stem,
            'arch_name': exp_data['arch_name'],
            'dataset': exp_data['dataset_name'],
            'total_critical_steps': total_critical_steps,
            'total_transitions': total_transitions,
            'num_coordinated_periods': len(coordination['coordinated_periods']),
            'max_coordination': coordination['max_coordination'],
            'mean_coordination': coordination['mean_coordination'],
            'loss_correlation': loss_correlation['overall_correlation']
        })

    print("\n" + "=" * 80)
    print("CRITICAL PERIOD SUMMARY")
    print("=" * 80)

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Statistics
    print(f"\nOverall Statistics:")
    print(f"  Mean critical periods per experiment: {df['total_critical_steps'].mean():.1f}")
    print(f"  Mean transitions per experiment: {df['total_transitions'].mean():.1f}")
    print(f"  Mean coordination: {df['mean_coordination'].mean():.3f}")
    print(f"  Mean loss correlation: {df['loss_correlation'].mean():.3f}")

    # Architecture comparison
    print("\nBy Architecture:")
    for arch_type in ['mlp', 'cnn', 'transformer']:
        arch_data = df[df['arch_name'].str.contains(arch_type)]
        if len(arch_data) > 0:
            print(f"\n  {arch_type.upper()}:")
            print(f"    Critical periods: {arch_data['total_critical_steps'].mean():.1f}")
            print(f"    Coordination: {arch_data['mean_coordination'].mean():.3f}")
            print(f"    Loss correlation: {arch_data['loss_correlation'].mean():.3f}")

    # Identify experiments with well-defined critical periods
    print("\n" + "=" * 80)
    print("EXPERIMENTS WITH WELL-DEFINED CRITICAL PERIODS")
    print("=" * 80)

    # High coordination and high loss correlation
    strong_critical = df[(df['max_coordination'] > 0.3) & (df['loss_correlation'] > 0.3)]
    if len(strong_critical) > 0:
        print(f"\nFound {len(strong_critical)} experiments with strong critical periods:")
        for _, row in strong_critical.head(10).iterrows():
            print(f"  - {row['experiment']}")
            print(f"    Coordination: {row['max_coordination']:.3f}, Loss corr: {row['loss_correlation']:.3f}")
    else:
        print("\nNo experiments found with strong coordinated critical periods.")
        print("This suggests smooth, distributed learning without distinct critical moments.")

    # Create visualizations
    print("\nCreating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Distribution of critical periods
    axes[0, 0].hist(df['total_critical_steps'], bins=20, color='steelblue', edgecolor='black')
    axes[0, 0].set_xlabel('Number of Critical Periods')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Distribution of Critical Periods')

    # Plot 2: Coordination scores
    axes[0, 1].hist(df['mean_coordination'], bins=20, color='coral', edgecolor='black')
    axes[0, 1].set_xlabel('Mean Coordination Score')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Cross-Layer Coordination')

    # Plot 3: Loss correlation
    axes[1, 0].hist(df['loss_correlation'], bins=20, color='seagreen', edgecolor='black')
    axes[1, 0].set_xlabel('Loss Correlation')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Correlation with Loss Drops')

    # Plot 4: Scatter - coordination vs loss correlation
    scatter = axes[1, 1].scatter(df['mean_coordination'], df['loss_correlation'],
                                c=df['total_critical_steps'], cmap='viridis',
                                s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    axes[1, 1].set_xlabel('Mean Coordination')
    axes[1, 1].set_ylabel('Loss Correlation')
    axes[1, 1].set_title('Coordination vs Loss Correlation')
    plt.colorbar(scatter, ax=axes[1, 1], label='Total Critical Periods')

    plt.tight_layout()
    plt.savefig(output_dir / 'critical_periods_overview.png')
    plt.close()

    # Architecture comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, arch_type in enumerate(['mlp', 'cnn', 'transformer']):
        arch_data = df[df['arch_name'].str.contains(arch_type)]
        if len(arch_data) > 0:
            metrics = ['total_critical_steps', 'mean_coordination', 'loss_correlation']
            values = [
                arch_data['total_critical_steps'].mean(),
                arch_data['mean_coordination'].mean() * 100,  # Scale for visibility
                arch_data['loss_correlation'].mean() * 100   # Scale for visibility
            ]
            axes[idx].bar(range(len(metrics)), values, color=['steelblue', 'coral', 'seagreen'])
            axes[idx].set_xticks(range(len(metrics)))
            axes[idx].set_xticklabels(['Critical\nPeriods', 'Coordination\n(×100)', 'Loss Corr\n(×100)'],
                                     fontsize=9)
            axes[idx].set_title(f'{arch_type.upper()}')
            axes[idx].set_ylabel('Value')

    plt.tight_layout()
    plt.savefig(output_dir / 'architecture_comparison.png')
    plt.close()

    # Save results
    results = {
        'summary': {
            'mean_critical_periods': float(df['total_critical_steps'].mean()),
            'mean_coordination': float(df['mean_coordination'].mean()),
            'mean_loss_correlation': float(df['loss_correlation'].mean()),
            'experiments_with_strong_periods': len(strong_critical)
        },
        'experiment_results': all_results
    }

    with open(output_dir / 'critical_periods_summary.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save detailed DataFrame
    df.to_csv(output_dir / 'critical_periods_detailed.csv', index=False)

    print(f"\nResults saved to: {output_dir}")
    print(f"  - critical_periods_summary.json")
    print(f"  - critical_periods_detailed.csv")
    print(f"  - critical_periods_overview.png")
    print(f"  - architecture_comparison.png")

    return results


if __name__ == '__main__':
    print("Step 1.3: Critical Period Identification")
    print("=" * 80)
    results = run_critical_period_analysis()
    print("\n✓ Analysis complete!")
