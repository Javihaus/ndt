"""
Visualization Utilities

Common visualization functions used across all phases.
Provides consistent plotting style and reusable components.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json


# ============================================================================
# STYLE CONFIGURATION
# ============================================================================

def setup_plot_style():
    """Set consistent plotting style across all visualizations."""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16


# ============================================================================
# DIMENSIONALITY PLOTTING
# ============================================================================

def plot_dimensionality_curve(steps: List[int],
                              dimensionality: List[float],
                              title: str = "Dimensionality Evolution",
                              xlabel: str = "Training Step",
                              ylabel: str = "Dimensionality (Stable Rank)",
                              jumps: Optional[List[int]] = None,
                              predictions: Optional[Dict] = None,
                              output_file: Optional[str] = None):
    """
    Plot dimensionality curve with optional jump markers and predictions.

    Args:
        steps: Training step indices
        dimensionality: Dimensionality values
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        jumps: Optional list of step indices where jumps occurred
        predictions: Optional dict with 'steps' and 'values' for predicted curve
        output_file: Optional path to save figure
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot actual curve
    ax.plot(steps, dimensionality, 'o-', label='Actual', linewidth=2,
           markersize=4, alpha=0.7)

    # Plot predictions if provided
    if predictions is not None:
        ax.plot(predictions['steps'], predictions['values'],
               's--', label='Predicted', linewidth=2,
               markersize=4, alpha=0.7)

    # Mark jumps if provided
    if jumps is not None:
        for jump_step in jumps:
            ax.axvline(jump_step, color='red', alpha=0.3,
                      linestyle='--', linewidth=1)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_multi_layer_dimensionality(measurements: List[Dict],
                                    metric: str = 'stable_rank',
                                    output_file: Optional[str] = None):
    """
    Plot dimensionality evolution for multiple layers.

    Args:
        measurements: List of measurement dictionaries with 'step' and 'layer_metrics'
        metric: Which metric to plot
        output_file: Optional path to save figure
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(14, 7))

    # Extract layer names from first measurement
    if measurements and 'layer_metrics' in measurements[0]:
        layer_names = list(measurements[0]['layer_metrics'].keys())

        for layer_name in layer_names:
            steps = []
            values = []

            for m in measurements:
                if 'layer_metrics' in m and layer_name in m['layer_metrics']:
                    steps.append(m['step'])
                    values.append(m['layer_metrics'][layer_name][metric])

            ax.plot(steps, values, 'o-', label=layer_name,
                   linewidth=2, markersize=3, alpha=0.7)

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel(f'{metric.replace("_", " ").title()}', fontsize=12)
    ax.set_title(f'Layer-wise {metric.replace("_", " ").title()} Evolution', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ============================================================================
# COMPARISON PLOTS
# ============================================================================

def plot_architecture_comparison(summary_df,
                                x_col: str = 'depth',
                                y_col: str = 'alpha',
                                hue_col: Optional[str] = None,
                                title: str = "Architecture Comparison",
                                output_file: Optional[str] = None):
    """
    Create scatter plot comparing architectural parameters.

    Args:
        summary_df: Pandas DataFrame with architecture data
        x_col: Column for x-axis
        y_col: Column for y-axis
        hue_col: Optional column for color coding
        title: Plot title
        output_file: Optional path to save figure
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    if hue_col:
        for hue_value in summary_df[hue_col].unique():
            subset = summary_df[summary_df[hue_col] == hue_value]
            ax.scatter(subset[x_col], subset[y_col],
                      label=hue_value, alpha=0.6, s=100)
    else:
        ax.scatter(summary_df[x_col], summary_df[y_col],
                  alpha=0.6, s=100)

    ax.set_xlabel(x_col.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel(y_col.replace('_', ' ').title(), fontsize=12)
    ax.set_title(title, fontsize=14)

    if hue_col:
        ax.legend()

    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ============================================================================
# CORRELATION PLOTS
# ============================================================================

def plot_prediction_vs_actual(actual: np.ndarray,
                              predicted: np.ndarray,
                              r2: float,
                              title: str = "Predicted vs Actual",
                              output_file: Optional[str] = None):
    """
    Plot predicted vs actual values with R² annotation.

    Args:
        actual: Actual values
        predicted: Predicted values
        r2: R² score
        title: Plot title
        output_file: Optional path to save figure
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(8, 8))

    # Scatter plot
    ax.scatter(actual, predicted, alpha=0.6, s=50)

    # Perfect prediction line
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    ax.plot([min_val, max_val], [min_val, max_val],
           'r--', linewidth=2, label='Perfect prediction')

    # Annotations
    ax.text(0.05, 0.95, f'R² = {r2:.4f}',
           transform=ax.transAxes,
           fontsize=14, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Actual', fontsize=12)
    ax.set_ylabel('Predicted', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_dual_axis_correlation(x: np.ndarray,
                               y1: np.ndarray,
                               y2: np.ndarray,
                               x_label: str = "X",
                               y1_label: str = "Y1",
                               y2_label: str = "Y2",
                               title: str = "Correlation Plot",
                               y1_jumps: Optional[List] = None,
                               y2_jumps: Optional[List] = None,
                               output_file: Optional[str] = None):
    """
    Plot two time series on dual y-axes.

    Args:
        x: X-axis values (e.g., training steps)
        y1: First time series
        y2: Second time series
        x_label: X-axis label
        y1_label: First y-axis label
        y2_label: Second y-axis label
        title: Plot title
        y1_jumps: Optional jump indices for first series
        y2_jumps: Optional jump indices for second series
        output_file: Optional path to save figure
    """
    setup_plot_style()

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()

    # Plot first series
    line1 = ax1.plot(x, y1, 'b-', linewidth=2, alpha=0.7, label=y1_label)

    # Plot second series
    line2 = ax2.plot(x, y2, 'r-', linewidth=2, alpha=0.7, label=y2_label)

    # Mark jumps
    if y1_jumps is not None:
        for jump in y1_jumps:
            ax1.axvline(jump, color='blue', alpha=0.2, linestyle='--')

    if y2_jumps is not None:
        for jump in y2_jumps:
            ax1.axvline(jump, color='red', alpha=0.2, linestyle='--')

    # Labels
    ax1.set_xlabel(x_label, fontsize=12)
    ax1.set_ylabel(y1_label, color='b', fontsize=12)
    ax2.set_ylabel(y2_label, color='r', fontsize=12)
    ax1.set_title(title, fontsize=14)

    # Legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    ax1.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ============================================================================
# SUMMARY DASHBOARDS
# ============================================================================

def create_training_dashboard(measurements: List[Dict],
                              arch_params: Dict,
                              analysis: Optional[Dict] = None,
                              output_file: Optional[str] = None):
    """
    Create comprehensive training dashboard.

    Args:
        measurements: List of measurement dicts
        arch_params: Architecture parameters
        analysis: Optional analysis results (jumps, phases, etc.)
        output_file: Optional path to save figure
    """
    setup_plot_style()

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    steps = [m['step'] for m in measurements]
    dim = [m.get('stable_rank', 0) for m in measurements]
    loss = [m['loss'] for m in measurements]
    grad_norm = [m['grad_norm'] for m in measurements]

    # Plot 1: Dimensionality
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(steps, dim, 'o-', linewidth=2, markersize=3)
    if analysis and 'dimensionality_jumps' in analysis:
        for jump in analysis['dimensionality_jumps']:
            ax1.axvline(jump, color='red', alpha=0.3, linestyle='--')
    ax1.set_title('Dimensionality Evolution')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Stable Rank')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Loss
    ax2 = fig.add_subplot(gs[1, :2])
    ax2.plot(steps, loss, 'o-', color='green', linewidth=2, markersize=3)
    ax2.set_title('Loss Evolution')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Loss')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Gradient Norm
    ax3 = fig.add_subplot(gs[2, :2])
    ax3.plot(steps, grad_norm, 'o-', color='orange', linewidth=2, markersize=3)
    ax3.set_title('Gradient Norm Evolution')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Gradient Norm')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Architecture Info
    ax4 = fig.add_subplot(gs[0, 2])
    ax4.axis('off')
    info_text = "Architecture\n\n"
    for k, v in arch_params.items():
        if isinstance(v, float):
            info_text += f"{k}: {v:.2f}\n"
        elif isinstance(v, int):
            info_text += f"{k}: {v:,}\n"
        else:
            info_text += f"{k}: {v}\n"
    ax4.text(0.1, 0.9, info_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', family='monospace')

    # Plot 5: Statistics
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    stats_text = "Statistics\n\n"
    stats_text += f"Final D: {dim[-1]:.2f}\n"
    stats_text += f"Final Loss: {loss[-1]:.4f}\n"
    stats_text += f"Min Loss: {min(loss):.4f}\n"
    if analysis:
        stats_text += f"\nJumps: {analysis.get('num_dim_jumps', 0)}\n"
        if 'phases' in analysis:
            stats_text += f"Phases: {len(analysis['phases'].get('phases', []))}\n"
    ax5.text(0.1, 0.9, stats_text, transform=ax5.transAxes,
            fontsize=10, verticalalignment='top', family='monospace')

    # Plot 6: Growth Rate
    ax6 = fig.add_subplot(gs[2, 2])
    if len(dim) > 1:
        growth = [dim[i] - dim[i-1] for i in range(1, len(dim))]
        ax6.plot(steps[1:], growth, 'o-', color='purple',
                linewidth=2, markersize=3)
        ax6.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax6.set_title('Growth Rate')
        ax6.set_xlabel('Step')
        ax6.set_ylabel('ΔD')
        ax6.grid(True, alpha=0.3)

    plt.suptitle('Training Dashboard', fontsize=16, y=0.995)

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ============================================================================
# EXPORT UTILITIES
# ============================================================================

def save_experiment_summary(measurements: List[Dict],
                           arch_params: Dict,
                           output_file: str,
                           metadata: Optional[Dict] = None):
    """
    Save experiment summary to JSON.

    Args:
        measurements: List of measurement dicts
        arch_params: Architecture parameters
        output_file: Output file path
        metadata: Optional additional metadata
    """
    summary = {
        'architecture_params': arch_params,
        'num_measurements': len(measurements),
        'measurements': measurements
    }

    if metadata:
        summary['metadata'] = metadata

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved: {output_file}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Create synthetic data and visualize
    import numpy as np

    steps = np.arange(0, 1000, 10)
    dimensionality = 10 + 40 * (1 - np.exp(-steps / 300))
    dimensionality += np.random.randn(len(dimensionality)) * 0.5

    # Simple curve
    plot_dimensionality_curve(
        steps, dimensionality,
        title="Example Dimensionality Curve",
        jumps=[200, 500],
        output_file='example_dim_curve.png'
    )

    # Dashboard
    measurements = [
        {
            'step': s,
            'stable_rank': d,
            'loss': 2.0 * np.exp(-s / 200),
            'grad_norm': 1.0 * np.exp(-s / 300)
        }
        for s, d in zip(steps, dimensionality)
    ]

    arch_params = {
        'depth': 5,
        'width': 128,
        'num_params': 50000
    }

    create_training_dashboard(
        measurements, arch_params,
        output_file='example_dashboard.png'
    )

    print("Example visualizations created!")
