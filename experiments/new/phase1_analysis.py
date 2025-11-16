"""
Phase 1 Analysis: Extract α parameters from experimental results

Goal: Fit the TAP growth model to empirical data and extract α = f(depth, width, connectivity)

The TAP growth model:
D(t+1) = D(t) + α_arch · ||∇L||_t · D(t) · (1 - D(t)/D_max)

We'll fit this to our measurements and extract α_arch for each architecture.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# TAP MODEL FITTING
# ============================================================================

def logistic_growth(t, D_max, alpha, t0):
    """
    Logistic growth model: D(t) = D_max / (1 + exp(-alpha * (t - t0)))

    This is the analytical solution to the TAP differential equation
    under constant gradient assumptions.
    """
    return D_max / (1 + np.exp(-alpha * (t - t0)))


def constrained_growth_with_gradient(t, grad_norms, D_max, alpha):
    """
    Numerical integration of TAP model with actual gradient norms.

    D(t+1) = D(t) + α · ||∇L||_t · D(t) · (1 - D(t)/D_max)
    """
    D = np.zeros(len(t))
    D[0] = 0.1  # Initial dimensionality

    for i in range(len(t) - 1):
        dt = t[i+1] - t[i]
        growth_rate = alpha * grad_norms[i] * D[i] * (1 - D[i] / D_max)
        D[i+1] = D[i] + growth_rate * dt
        D[i+1] = np.clip(D[i+1], 0, D_max)

    return D


def fit_tap_model_simple(steps, dimensionality, method='logistic'):
    """
    Fit TAP model to dimensionality curve.

    Returns:
        - alpha: growth rate parameter
        - D_max: maximum dimensionality
        - R2: goodness of fit
        - predictions: fitted curve
    """
    try:
        # Normalize steps to [0, 1] for numerical stability
        t_norm = np.array(steps) / max(steps)
        D = np.array(dimensionality)

        # Estimate initial parameters
        D_max_init = max(D) * 1.2
        alpha_init = 5.0
        t0_init = 0.5

        # Fit logistic curve
        popt, pcov = curve_fit(
            logistic_growth,
            t_norm, D,
            p0=[D_max_init, alpha_init, t0_init],
            bounds=([max(D), 0.1, 0], [max(D) * 2, 50, 1]),
            maxfev=10000
        )

        D_max, alpha, t0 = popt
        predictions = logistic_growth(t_norm, D_max, alpha, t0)

        # Calculate R²
        ss_res = np.sum((D - predictions) ** 2)
        ss_tot = np.sum((D - np.mean(D)) ** 2)
        R2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Denormalize alpha (since we normalized time)
        alpha_denorm = alpha / max(steps)

        return {
            'alpha': alpha_denorm,
            'D_max': D_max,
            'R2': R2,
            't0': t0 * max(steps),
            'predictions': predictions.tolist(),
            'method': 'logistic'
        }

    except Exception as e:
        print(f"Fitting failed: {e}")
        return {
            'alpha': 0,
            'D_max': 0,
            'R2': 0,
            't0': 0,
            'predictions': [],
            'method': 'failed'
        }


def detect_jumps(dimensionality, window=20, threshold=2.0):
    """
    Detect discrete jumps in dimensionality using local gradient analysis.

    Returns list of jump indices and magnitudes.
    """
    D = np.array(dimensionality)

    # Compute local gradients
    gradients = np.gradient(D)

    # Smooth gradients
    from scipy.ndimage import gaussian_filter1d
    gradients_smooth = gaussian_filter1d(gradients, sigma=3)

    # Detect jumps as outliers in gradient
    mean_grad = np.mean(gradients_smooth)
    std_grad = np.std(gradients_smooth)

    jumps = []
    for i in range(len(gradients_smooth)):
        z_score = (gradients_smooth[i] - mean_grad) / (std_grad + 1e-10)
        if z_score > threshold:
            jumps.append({
                'index': i,
                'magnitude': gradients_smooth[i],
                'z_score': z_score
            })

    return jumps


# ============================================================================
# RESULT ANALYSIS
# ============================================================================

def analyze_single_experiment(result: Dict, metric: str = 'stable_rank') -> Dict:
    """
    Analyze a single experiment result.

    Extracts dimensionality curves, fits TAP model, detects jumps.
    """
    measurements = result['measurements']
    arch_params = result['architecture_params']

    # Extract aggregate dimensionality (average across all layers)
    steps = []
    dimensionalities = []
    grad_norms = []
    losses = []

    for m in measurements:
        steps.append(m['step'])
        grad_norms.append(m['grad_norm'])
        losses.append(m['loss'])

        # Average metric across all layers
        layer_metrics = m['layer_metrics']
        if layer_metrics:
            avg_dim = np.mean([layer[metric] for layer in layer_metrics.values()])
            dimensionalities.append(avg_dim)
        else:
            dimensionalities.append(0)

    # Fit TAP model
    tap_fit = fit_tap_model_simple(steps, dimensionalities)

    # Detect jumps
    jumps = detect_jumps(dimensionalities)

    return {
        'arch_name': result['arch_name'],
        'dataset_name': result['dataset_name'],
        'architecture_params': arch_params,
        'tap_parameters': tap_fit,
        'jumps': jumps,
        'num_jumps': len(jumps),
        'final_accuracy': result['final_accuracy'],
        'curves': {
            'steps': steps,
            'dimensionality': dimensionalities,
            'grad_norms': grad_norms,
            'losses': losses
        }
    }


def analyze_phase1_results(results_dir: str = './experiments/new/results/phase1',
                           output_dir: str = './experiments/new/results/phase1_analysis',
                           metric: str = 'stable_rank') -> pd.DataFrame:
    """
    Analyze all Phase 1 results and extract α parameters.

    Returns DataFrame with architecture parameters and fitted α values.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results_dir = Path(results_dir)
    result_files = list(results_dir.glob('*.json'))

    print(f"Found {len(result_files)} result files")

    analyses = []

    for result_file in result_files:
        if result_file.name == 'phase1_all_results.json':
            continue

        print(f"Analyzing {result_file.name}...")

        with open(result_file, 'r') as f:
            result = json.load(f)

        analysis = analyze_single_experiment(result, metric=metric)
        analyses.append(analysis)

        # Save individual analysis
        output_file = output_dir / f"analysis_{result_file.stem}.json"
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)

    # Create summary DataFrame
    summary_data = []

    for analysis in analyses:
        arch_params = analysis['architecture_params']
        tap_params = analysis['tap_parameters']

        summary_data.append({
            'arch_name': analysis['arch_name'],
            'dataset_name': analysis['dataset_name'],
            'depth': arch_params['depth'],
            'width': arch_params['width'],
            'num_params': arch_params['num_params'],
            'connectivity': arch_params.get('connectivity', 0),
            'alpha': tap_params['alpha'],
            'D_max': tap_params['D_max'],
            'R2': tap_params['R2'],
            't0': tap_params['t0'],
            'num_jumps': analysis['num_jumps'],
            'final_accuracy': analysis['final_accuracy']
        })

    df = pd.DataFrame(summary_data)

    # Save summary
    df.to_csv(output_dir / 'alpha_summary.csv', index=False)

    print(f"\nAnalysis complete. Summary saved to {output_dir / 'alpha_summary.csv'}")
    print(f"Total experiments analyzed: {len(df)}")

    return df


# ============================================================================
# ALPHA RELATIONSHIP MODELING
# ============================================================================

def fit_alpha_relationship(df: pd.DataFrame) -> Dict:
    """
    Fit α = f(depth, width, connectivity) relationship.

    Tests multiple functional forms and returns best fit.
    """
    # Average alpha across datasets for each architecture
    arch_alphas = df.groupby('arch_name').agg({
        'depth': 'first',
        'width': 'first',
        'num_params': 'first',
        'connectivity': 'first',
        'alpha': 'mean',
        'R2': 'mean'
    }).reset_index()

    # Filter out poor fits
    arch_alphas = arch_alphas[arch_alphas['R2'] > 0.5]

    print(f"\nFitting α relationship ({len(arch_alphas)} architectures)")

    # Model 1: Linear combination
    from sklearn.linear_model import LinearRegression

    X = arch_alphas[['depth', 'width', 'connectivity']].values
    y = arch_alphas['alpha'].values

    model_linear = LinearRegression()
    model_linear.fit(X, y)
    y_pred_linear = model_linear.predict(X)

    r2_linear = 1 - np.sum((y - y_pred_linear)**2) / np.sum((y - np.mean(y))**2)

    print(f"Linear model R²: {r2_linear:.4f}")
    print(f"Coefficients: depth={model_linear.coef_[0]:.6f}, "
          f"width={model_linear.coef_[1]:.6f}, "
          f"connectivity={model_linear.coef_[2]:.6f}")

    # Model 2: Log-linear
    X_log = np.log1p(X)
    model_log = LinearRegression()
    model_log.fit(X_log, y)
    y_pred_log = model_log.predict(X_log)

    r2_log = 1 - np.sum((y - y_pred_log)**2) / np.sum((y - np.mean(y))**2)

    print(f"Log-linear model R²: {r2_log:.4f}")

    # Model 3: Product form α = a * depth^b * width^c
    X_loglog = np.log1p(X)
    y_log = np.log1p(y)
    model_power = LinearRegression()
    model_power.fit(X_loglog, y_log)

    y_pred_power = np.exp(model_power.predict(X_loglog)) - 1
    r2_power = 1 - np.sum((y - y_pred_power)**2) / np.sum((y - np.mean(y))**2)

    print(f"Power-law model R²: {r2_power:.4f}")

    return {
        'linear': {
            'model': model_linear,
            'R2': r2_linear,
            'coefficients': {
                'depth': model_linear.coef_[0],
                'width': model_linear.coef_[1],
                'connectivity': model_linear.coef_[2],
                'intercept': model_linear.intercept_
            }
        },
        'log_linear': {
            'model': model_log,
            'R2': r2_log
        },
        'power_law': {
            'model': model_power,
            'R2': r2_power
        },
        'data': arch_alphas
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_analysis(df: pd.DataFrame,
                       alpha_models: Dict,
                       output_dir: str = './experiments/new/results/phase1_analysis'):
    """Create comprehensive visualization of Phase 1 results."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)

    # 1. Alpha vs Architecture Parameters
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Average alpha per architecture
    arch_data = df.groupby('arch_name').agg({
        'depth': 'first',
        'width': 'first',
        'num_params': 'first',
        'alpha': 'mean',
        'R2': 'mean'
    }).reset_index()

    # Filter good fits
    arch_data = arch_data[arch_data['R2'] > 0.5]

    # Alpha vs Depth
    axes[0, 0].scatter(arch_data['depth'], arch_data['alpha'], s=100, alpha=0.6)
    axes[0, 0].set_xlabel('Depth (# layers)', fontsize=12)
    axes[0, 0].set_ylabel('α (growth rate)', fontsize=12)
    axes[0, 0].set_title('α vs Network Depth', fontsize=14)

    # Alpha vs Width
    axes[0, 1].scatter(arch_data['width'], arch_data['alpha'], s=100, alpha=0.6)
    axes[0, 1].set_xlabel('Width (avg # units)', fontsize=12)
    axes[0, 1].set_ylabel('α (growth rate)', fontsize=12)
    axes[0, 1].set_title('α vs Network Width', fontsize=14)

    # Alpha vs Total Parameters
    axes[1, 0].scatter(arch_data['num_params'], arch_data['alpha'], s=100, alpha=0.6)
    axes[1, 0].set_xlabel('Total Parameters', fontsize=12)
    axes[1, 0].set_ylabel('α (growth rate)', fontsize=12)
    axes[1, 0].set_title('α vs Model Size', fontsize=14)
    axes[1, 0].set_xscale('log')

    # R² distribution
    axes[1, 1].hist(df['R2'], bins=30, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('R² (fit quality)', fontsize=12)
    axes[1, 1].set_ylabel('Count', fontsize=12)
    axes[1, 1].set_title('TAP Model Fit Quality', fontsize=14)
    axes[1, 1].axvline(0.8, color='red', linestyle='--', label='R² = 0.8')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'alpha_relationships.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'alpha_relationships.png'}")
    plt.close()

    # 2. Dataset Consistency
    fig, ax = plt.subplots(figsize=(12, 6))

    # Group by dataset
    for dataset in df['dataset_name'].unique():
        data = df[df['dataset_name'] == dataset]
        arch_avg = data.groupby('arch_name')['alpha'].mean()
        ax.scatter(range(len(arch_avg)), arch_avg.values,
                  label=dataset, alpha=0.6, s=100)

    ax.set_xlabel('Architecture Index', fontsize=12)
    ax.set_ylabel('α (growth rate)', fontsize=12)
    ax.set_title('α Consistency Across Datasets', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'alpha_consistency.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'alpha_consistency.png'}")
    plt.close()

    # 3. Jumps Analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Jumps vs Depth
    arch_jumps = df.groupby('arch_name').agg({
        'depth': 'first',
        'width': 'first',
        'num_jumps': 'mean'
    }).reset_index()

    axes[0].scatter(arch_jumps['depth'], arch_jumps['num_jumps'], s=100, alpha=0.6)
    axes[0].set_xlabel('Depth (# layers)', fontsize=12)
    axes[0].set_ylabel('Average # of Jumps', fontsize=12)
    axes[0].set_title('Phase Transitions vs Depth', fontsize=14)

    # Jumps vs Width
    axes[1].scatter(arch_jumps['width'], arch_jumps['num_jumps'], s=100, alpha=0.6)
    axes[1].set_xlabel('Width (avg # units)', fontsize=12)
    axes[1].set_ylabel('Average # of Jumps', fontsize=12)
    axes[1].set_title('Phase Transitions vs Width', fontsize=14)

    plt.tight_layout()
    plt.savefig(output_dir / 'jumps_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'jumps_analysis.png'}")
    plt.close()


def create_phase1_report(df: pd.DataFrame,
                         alpha_models: Dict,
                         output_dir: str = './experiments/new/results/phase1_analysis'):
    """Generate comprehensive Phase 1 report."""

    output_dir = Path(output_dir)

    report = []
    report.append("# Phase 1 Calibration: Analysis Report\n")
    report.append(f"Generated: {pd.Timestamp.now()}\n\n")

    report.append("## Summary Statistics\n")
    report.append(f"- Total experiments: {len(df)}\n")
    report.append(f"- Unique architectures: {df['arch_name'].nunique()}\n")
    report.append(f"- Datasets: {df['dataset_name'].nunique()}\n")
    report.append(f"- Mean R²: {df['R2'].mean():.4f}\n")
    report.append(f"- Experiments with R² > 0.8: {(df['R2'] > 0.8).sum()} ({(df['R2'] > 0.8).mean()*100:.1f}%)\n\n")

    report.append("## α = f(architecture) Relationship\n\n")

    for model_name, model_data in alpha_models.items():
        if model_name == 'data':
            continue
        report.append(f"### {model_name.replace('_', ' ').title()} Model\n")
        report.append(f"- R²: {model_data['R2']:.4f}\n")
        if 'coefficients' in model_data:
            report.append("- Coefficients:\n")
            for k, v in model_data['coefficients'].items():
                report.append(f"  - {k}: {v:.6f}\n")
        report.append("\n")

    report.append("## Key Findings\n\n")

    # Find best model
    best_model = max(
        [(k, v['R2']) for k, v in alpha_models.items() if k != 'data'],
        key=lambda x: x[1]
    )
    report.append(f"1. **Best predictive model**: {best_model[0]} (R² = {best_model[1]:.4f})\n")

    # Architecture insights
    arch_data = alpha_models['data']
    report.append(f"2. **α range**: [{arch_data['alpha'].min():.6f}, {arch_data['alpha'].max():.6f}]\n")

    corr_depth = arch_data[['depth', 'alpha']].corr().iloc[0, 1]
    corr_width = arch_data[['width', 'alpha']].corr().iloc[0, 1]
    report.append(f"3. **Correlation with depth**: {corr_depth:.3f}\n")
    report.append(f"4. **Correlation with width**: {corr_width:.3f}\n")

    # Jump statistics
    report.append(f"\n## Phase Transitions (Jumps)\n\n")
    report.append(f"- Mean jumps per experiment: {df['num_jumps'].mean():.2f}\n")
    report.append(f"- Range: [{df['num_jumps'].min():.0f}, {df['num_jumps'].max():.0f}]\n")

    # Save report
    with open(output_dir / 'phase1_report.md', 'w') as f:
        f.writelines(report)

    print(f"\nReport saved: {output_dir / 'phase1_report.md'}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Phase 1 Analysis')
    parser.add_argument('--results-dir', type=str,
                       default='./experiments/new/results/phase1',
                       help='Directory containing Phase 1 results')
    parser.add_argument('--output-dir', type=str,
                       default='./experiments/new/results/phase1_analysis',
                       help='Output directory for analysis')
    parser.add_argument('--metric', type=str, default='stable_rank',
                       choices=['stable_rank', 'participation_ratio',
                               'effective_rank_90', 'nuclear_norm_ratio'],
                       help='Dimensionality metric to analyze')

    args = parser.parse_args()

    # Analyze results
    print("Analyzing Phase 1 results...")
    df = analyze_phase1_results(args.results_dir, args.output_dir, args.metric)

    # Fit α relationship
    print("\nFitting α = f(architecture) relationship...")
    alpha_models = fit_alpha_relationship(df)

    # Save models
    import pickle
    with open(Path(args.output_dir) / 'alpha_models.pkl', 'wb') as f:
        pickle.dump(alpha_models, f)

    # Visualize
    print("\nGenerating visualizations...")
    visualize_analysis(df, alpha_models, args.output_dir)

    # Create report
    print("\nGenerating report...")
    create_phase1_report(df, alpha_models, args.output_dir)

    print("\n" + "="*70)
    print("Phase 1 Analysis Complete!")
    print("="*70)
    print(f"\nResults saved to: {args.output_dir}")
    print("\nNext steps:")
    print("1. Review phase1_report.md")
    print("2. Check visualizations")
    print("3. Run Phase 2 prediction experiments")
