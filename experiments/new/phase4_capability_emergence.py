"""
Phase 4: Connection to Emergent Capabilities

Goal: Test if dimensionality jumps correlate with capability emergence

Hypothesis:
- Dimensionality jumps precede performance jumps
- This provides an early warning system for capability emergence

Key Question:
- Do dimensionality jumps at step t predict performance improvements at step t+k?

Experiments:
1. Measure BOTH dimensionality AND task performance every 5 steps
2. Detect jumps in both signals
3. Compute temporal correlation
4. Test: Does ΔD(t) predict Δ(accuracy)(t+k)?
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

# Import from previous phases
import sys
sys.path.append(str(Path(__file__).parent))
from phase1_calibration import (
    create_architecture, DATASET_LOADERS,
    measure_network_dimensionality, DimensionalityEstimator
)


# ============================================================================
# CAPABILITY MEASUREMENT
# ============================================================================

class CapabilityTracker:
    """
    Tracks multiple capability metrics during training.

    Capabilities measured:
    1. Overall accuracy
    2. Per-class accuracy
    3. Confidence calibration
    4. Representation quality (linear separability)
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def measure_capabilities(self,
                            model: nn.Module,
                            dataloader: DataLoader,
                            device: torch.device,
                            max_batches: int = 20) -> Dict:
        """
        Comprehensive capability measurement.

        Returns metrics including:
        - Overall accuracy
        - Per-class accuracy
        - Mean confidence
        - Calibration error
        - Linear separability (estimate)
        """
        model.eval()

        all_preds = []
        all_labels = []
        all_confidences = []
        all_correct = []

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break

                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                # Predictions and confidences
                probs = torch.softmax(outputs, dim=1)
                confidences, preds = probs.max(1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
                all_correct.extend((preds == labels).cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_confidences = np.array(all_confidences)
        all_correct = np.array(all_correct)

        # Overall accuracy
        accuracy = np.mean(all_correct)

        # Per-class accuracy
        per_class_acc = {}
        for c in range(self.num_classes):
            mask = all_labels == c
            if mask.sum() > 0:
                per_class_acc[c] = np.mean(all_correct[mask])
            else:
                per_class_acc[c] = 0.0

        # Mean confidence
        mean_confidence = np.mean(all_confidences)

        # Calibration error (Expected Calibration Error - ECE)
        # Bin predictions by confidence and measure accuracy vs confidence
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            in_bin = (all_confidences >= bin_edges[i]) & (all_confidences < bin_edges[i+1])
            if in_bin.sum() > 0:
                bin_accuracy = np.mean(all_correct[in_bin])
                bin_confidence = np.mean(all_confidences[in_bin])
                ece += (in_bin.sum() / len(all_correct)) * abs(bin_accuracy - bin_confidence)

        # Weakest classes (for capability emergence tracking)
        weakest_classes = sorted(per_class_acc.items(), key=lambda x: x[1])[:3]

        return {
            'overall_accuracy': accuracy,
            'per_class_accuracy': per_class_acc,
            'mean_confidence': mean_confidence,
            'calibration_error': ece,
            'weakest_classes': [c for c, _ in weakest_classes],
            'weakest_class_scores': [score for _, score in weakest_classes]
        }


# ============================================================================
# JOINT TRACKING
# ============================================================================

def train_with_capability_tracking(model: nn.Module,
                                   train_loader: DataLoader,
                                   val_loader: DataLoader,
                                   device: torch.device,
                                   num_steps: int = 5000,
                                   measurement_interval: int = 5,
                                   num_classes: int = 10) -> Dict:
    """
    Train model while tracking BOTH dimensionality AND capabilities.

    This is the core Phase 4 experiment.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    capability_tracker = CapabilityTracker(num_classes)

    measurements = []
    step = 0
    epoch = 0

    arch_params = model.get_architecture_params()
    print(f"\nTraining {arch_params['depth']} layer network")
    print(f"Tracking dimensionality + capabilities every {measurement_interval} steps\n")

    pbar = tqdm(total=num_steps, desc="Training")

    while step < num_steps:
        epoch += 1

        for inputs, labels in train_loader:
            if step >= num_steps:
                break

            inputs, labels = inputs.to(device), labels.to(device)

            # Training step
            model.train()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient norm
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.norm().item() ** 2
            grad_norm = grad_norm ** 0.5

            optimizer.step()

            # Measure dimensionality AND capabilities
            if step % measurement_interval == 0:
                # Dimensionality
                dim_metrics = measure_network_dimensionality(
                    model, val_loader, device, max_batches=5
                )

                if dim_metrics:
                    avg_stable_rank = np.mean([m['stable_rank'] for m in dim_metrics.values()])
                    avg_participation_ratio = np.mean([m['participation_ratio'] for m in dim_metrics.values()])
                else:
                    avg_stable_rank = 0
                    avg_participation_ratio = 0

                # Capabilities
                capabilities = capability_tracker.measure_capabilities(
                    model, val_loader, device, max_batches=10
                )

                measurements.append({
                    'step': step,
                    'epoch': epoch,
                    'loss': loss.item(),
                    'grad_norm': grad_norm,
                    # Dimensionality metrics
                    'stable_rank': avg_stable_rank,
                    'participation_ratio': avg_participation_ratio,
                    # Capability metrics
                    'accuracy': capabilities['overall_accuracy'],
                    'mean_confidence': capabilities['mean_confidence'],
                    'calibration_error': capabilities['calibration_error'],
                    'per_class_accuracy': capabilities['per_class_accuracy'],
                    'weakest_classes': capabilities['weakest_classes'],
                    'weakest_class_scores': capabilities['weakest_class_scores']
                })

                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{capabilities['overall_accuracy']:.3f}",
                    'D': f"{avg_stable_rank:.1f}"
                })

            step += 1
            pbar.update(1)

    pbar.close()

    return {
        'measurements': measurements,
        'architecture_params': arch_params,
        'num_classes': num_classes
    }


# ============================================================================
# JUMP CORRELATION ANALYSIS
# ============================================================================

def detect_jumps_in_signal(signal: np.ndarray,
                          window: int = 20,
                          threshold: float = 2.0) -> List[int]:
    """
    Detect discrete jumps in a time series.

    Returns indices where jumps occur.
    """
    # Compute gradients
    gradients = np.gradient(signal)

    # Smooth gradients
    from scipy.ndimage import gaussian_filter1d
    gradients_smooth = gaussian_filter1d(gradients, sigma=2)

    # Detect jumps as outliers
    mean_grad = np.mean(gradients_smooth)
    std_grad = np.std(gradients_smooth)

    jumps = []
    for i in range(len(gradients_smooth)):
        z_score = (gradients_smooth[i] - mean_grad) / (std_grad + 1e-10)
        if z_score > threshold:
            jumps.append(i)

    return jumps


def analyze_jump_correlation(measurements: List[Dict],
                            dimensionality_key: str = 'stable_rank',
                            capability_key: str = 'accuracy') -> Dict:
    """
    Analyze correlation between dimensionality jumps and capability jumps.

    Key analysis:
    1. Detect jumps in both signals
    2. Compute temporal correlation
    3. Test if dimensionality jumps predict capability jumps
    """
    steps = np.array([m['step'] for m in measurements])
    dimensionality = np.array([m[dimensionality_key] for m in measurements])
    capability = np.array([m[capability_key] for m in measurements])

    # Detect jumps
    dim_jumps = detect_jumps_in_signal(dimensionality, threshold=2.0)
    cap_jumps = detect_jumps_in_signal(capability, threshold=1.5)

    print(f"\nJump Detection:")
    print(f"  Dimensionality jumps: {len(dim_jumps)}")
    print(f"  Capability jumps: {len(cap_jumps)}")

    # Temporal correlation analysis
    # For each dimensionality jump, find nearest capability jump
    correlations = []

    for dim_jump_idx in dim_jumps:
        # Find nearest capability jump
        if cap_jumps:
            nearest_cap_jump = min(cap_jumps, key=lambda x: abs(x - dim_jump_idx))
            time_diff = nearest_cap_jump - dim_jump_idx
            correlations.append(time_diff)

    # Statistical analysis
    if correlations:
        mean_lag = np.mean(correlations)
        std_lag = np.std(correlations)

        print(f"\nTemporal Correlation:")
        print(f"  Mean lag: {mean_lag:.1f} steps")
        print(f"  Std lag: {std_lag:.1f} steps")

        # Test if dimensionality jumps precede capability jumps
        positive_lags = sum(1 for lag in correlations if lag > 0)
        print(f"  Dimensionality jumps BEFORE capability: "
              f"{positive_lags}/{len(correlations)} "
              f"({positive_lags/len(correlations)*100:.1f}%)")
    else:
        mean_lag = 0
        std_lag = 0

    # Granger causality test (simplified)
    # Does past dimensionality predict future capability?
    lag_steps = [5, 10, 20, 50]
    predictive_power = {}

    for lag in lag_steps:
        if len(dimensionality) > lag:
            # Correlation between D(t) and Acc(t+lag)
            dim_lagged = dimensionality[:-lag]
            cap_future = capability[lag:]

            if len(dim_lagged) > 0 and len(cap_future) > 0:
                corr, p_value = stats.pearsonr(dim_lagged, cap_future)
                predictive_power[lag] = {
                    'correlation': corr,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }

    print(f"\nPredictive Power (D(t) → Acc(t+k)):")
    for lag, result in predictive_power.items():
        sig = "✓" if result['significant'] else "✗"
        print(f"  k={lag}: r={result['correlation']:.3f}, "
              f"p={result['p_value']:.4f} {sig}")

    # Phase transition analysis
    # Identify distinct phases based on dimensionality
    phases = identify_training_phases(dimensionality)

    return {
        'dimensionality_jumps': [steps[i] for i in dim_jumps],
        'capability_jumps': [steps[i] for i in cap_jumps],
        'num_dim_jumps': len(dim_jumps),
        'num_cap_jumps': len(cap_jumps),
        'temporal_correlation': {
            'mean_lag': mean_lag,
            'std_lag': std_lag,
            'all_lags': correlations
        },
        'predictive_power': predictive_power,
        'phases': phases
    }


def identify_training_phases(dimensionality: np.ndarray,
                            n_phases: int = 3) -> Dict:
    """
    Identify distinct training phases using clustering.

    Returns phase boundaries and characteristics.
    """
    from sklearn.cluster import KMeans

    # Reshape for clustering
    X = dimensionality.reshape(-1, 1)

    # Cluster
    kmeans = KMeans(n_clusters=n_phases, random_state=42)
    labels = kmeans.fit_predict(X)

    # Identify phase transitions
    transitions = []
    for i in range(1, len(labels)):
        if labels[i] != labels[i-1]:
            transitions.append(i)

    # Phase characteristics
    phases_info = []
    for phase_id in range(n_phases):
        mask = labels == phase_id
        indices = np.where(mask)[0]

        if len(indices) > 0:
            phases_info.append({
                'phase_id': phase_id,
                'start_idx': indices[0],
                'end_idx': indices[-1],
                'mean_dimensionality': np.mean(dimensionality[mask]),
                'duration': len(indices)
            })

    # Sort by start index
    phases_info.sort(key=lambda x: x['start_idx'])

    return {
        'transitions': transitions,
        'phases': phases_info,
        'labels': labels.tolist()
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_capability_emergence(measurements: List[Dict],
                                   analysis: Dict,
                                   output_file: str):
    """
    Create comprehensive visualization of dimensionality-capability relationship.
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))

    steps = np.array([m['step'] for m in measurements])
    dimensionality = np.array([m['stable_rank'] for m in measurements])
    accuracy = np.array([m['accuracy'] for m in measurements])
    confidence = np.array([m['mean_confidence'] for m in measurements])
    loss = np.array([m['loss'] for m in measurements])

    # Plot 1: Dimensionality and Accuracy (dual axis)
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()

    line1 = ax1.plot(steps, dimensionality, 'b-', label='Dimensionality', linewidth=2, alpha=0.7)
    line2 = ax1_twin.plot(steps, accuracy, 'r-', label='Accuracy', linewidth=2, alpha=0.7)

    # Mark jumps
    for jump_step in analysis['dimensionality_jumps']:
        ax1.axvline(jump_step, color='blue', alpha=0.2, linestyle='--')
    for jump_step in analysis['capability_jumps']:
        ax1.axvline(jump_step, color='red', alpha=0.2, linestyle='--')

    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Dimensionality (Stable Rank)', color='b', fontsize=12)
    ax1_twin.set_ylabel('Accuracy', color='r', fontsize=12)
    ax1.set_title('Dimensionality vs Capability Evolution', fontsize=14)

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Phase transitions
    ax2 = axes[0, 1]
    phases = analysis['phases']

    # Color by phase
    phase_labels = np.array(phases['labels'])
    for phase_info in phases['phases']:
        phase_id = phase_info['phase_id']
        start = phase_info['start_idx']
        end = phase_info['end_idx']

        ax2.axvspan(steps[start], steps[end],
                   alpha=0.3, label=f"Phase {phase_id}")

    ax2.plot(steps, dimensionality, 'k-', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('Dimensionality', fontsize=12)
    ax2.set_title('Training Phases', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Scatter - Dimensionality vs Accuracy
    ax3 = axes[1, 0]
    scatter = ax3.scatter(dimensionality, accuracy, c=steps,
                         cmap='viridis', alpha=0.6, s=50)
    ax3.set_xlabel('Dimensionality', fontsize=12)
    ax3.set_ylabel('Accuracy', fontsize=12)
    ax3.set_title('Dimensionality-Accuracy Relationship', fontsize=14)
    plt.colorbar(scatter, ax=ax3, label='Training Step')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Lagged correlation
    ax4 = axes[1, 1]
    predictive_power = analysis['predictive_power']

    lags = list(predictive_power.keys())
    correlations = [predictive_power[k]['correlation'] for k in lags]
    significant = [predictive_power[k]['significant'] for k in lags]

    colors = ['green' if sig else 'gray' for sig in significant]
    ax4.bar(lags, correlations, color=colors, alpha=0.7, edgecolor='black')
    ax4.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_xlabel('Lag (steps)', fontsize=12)
    ax4.set_ylabel('Correlation', fontsize=12)
    ax4.set_title('Predictive Power: D(t) → Acc(t+k)', fontsize=14)
    ax4.grid(True, alpha=0.3, axis='y')

    # Plot 5: Confidence and Calibration
    ax5 = axes[2, 0]
    ax5_twin = ax5.twinx()

    ax5.plot(steps, confidence, 'g-', label='Mean Confidence', linewidth=2, alpha=0.7)
    calibration_error = np.array([m['calibration_error'] for m in measurements])
    ax5_twin.plot(steps, calibration_error, 'm-', label='Calibration Error', linewidth=2, alpha=0.7)

    ax5.set_xlabel('Training Step', fontsize=12)
    ax5.set_ylabel('Mean Confidence', color='g', fontsize=12)
    ax5_twin.set_ylabel('Calibration Error', color='m', fontsize=12)
    ax5.set_title('Confidence and Calibration', fontsize=14)
    ax5.grid(True, alpha=0.3)

    # Plot 6: Summary statistics
    ax6 = axes[2, 1]
    ax6.axis('off')

    summary_text = f"""
    SUMMARY STATISTICS

    Dimensionality Jumps: {analysis['num_dim_jumps']}
    Capability Jumps: {analysis['num_cap_jumps']}

    Temporal Correlation:
      Mean lag: {analysis['temporal_correlation']['mean_lag']:.1f} steps
      Std lag: {analysis['temporal_correlation']['std_lag']:.1f} steps

    Strongest Predictive Power:
    """

    if predictive_power:
        best_lag = max(predictive_power.items(),
                      key=lambda x: abs(x[1]['correlation']))
        summary_text += f"      k={best_lag[0]}: r={best_lag[1]['correlation']:.3f}\n"
        summary_text += f"      {'(significant)' if best_lag[1]['significant'] else '(not significant)'}\n"

    summary_text += f"\n    Phases Detected: {len(phases['phases'])}"

    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
            fontsize=11, verticalalignment='top', family='monospace')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nVisualization saved: {output_file}")


# ============================================================================
# PHASE 4 EXPERIMENTAL PIPELINE
# ============================================================================

def run_phase4_experiments(architectures: Optional[List[str]] = None,
                          datasets: Optional[List[str]] = None,
                          output_dir: str = './experiments/new/results/phase4',
                          num_steps: int = 5000,
                          measurement_interval: int = 5):
    """
    Run Phase 4 experiments across multiple architectures and datasets.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if architectures is None:
        architectures = ['mlp_shallow_2', 'mlp_medium_5', 'cnn_shallow', 'transformer_shallow']

    if datasets is None:
        datasets = ['mnist', 'cifar10']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*70}")
    print("Phase 4: Capability Emergence Experiments")
    print(f"{'='*70}")
    print(f"Architectures: {len(architectures)}")
    print(f"Datasets: {len(datasets)}")
    print(f"Total experiments: {len(architectures) * len(datasets)}")
    print(f"{'='*70}\n")

    all_results = []

    for dataset in datasets:
        for arch in architectures:
            try:
                print(f"\n{'='*70}")
                print(f"Experiment: {arch} on {dataset}")
                print(f"{'='*70}")

                # Load data
                loader_fn = DATASET_LOADERS[dataset]
                train_loader, val_loader, input_dim, num_classes, in_channels = loader_fn(
                    batch_size=64, subset_size=10000
                )

                # Create model
                model = create_architecture(arch, input_dim, num_classes, in_channels)

                # Train with capability tracking
                result = train_with_capability_tracking(
                    model, train_loader, val_loader, device,
                    num_steps=num_steps,
                    measurement_interval=measurement_interval,
                    num_classes=num_classes
                )

                # Analyze jumps and correlations
                analysis = analyze_jump_correlation(result['measurements'])

                # Save
                combined_result = {
                    'arch_name': arch,
                    'dataset_name': dataset,
                    'architecture_params': result['architecture_params'],
                    'measurements': result['measurements'],
                    'analysis': analysis
                }

                output_file = Path(output_dir) / f'{arch}_{dataset}.json'
                with open(output_file, 'w') as f:
                    json.dump(combined_result, f, indent=2)

                # Visualize
                viz_file = Path(output_dir) / f'{arch}_{dataset}_visualization.png'
                visualize_capability_emergence(
                    result['measurements'],
                    analysis,
                    str(viz_file)
                )

                all_results.append(combined_result)

                print(f"\n✓ Completed {arch} on {dataset}")

            except Exception as e:
                print(f"\n✗ Failed {arch} on {dataset}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Summary analysis across all experiments
    create_phase4_summary(all_results, output_dir)

    print(f"\n{'='*70}")
    print("Phase 4 Complete!")
    print(f"{'='*70}")
    print(f"Results saved to: {output_dir}")

    return all_results


def create_phase4_summary(results: List[Dict], output_dir: str):
    """Create summary report for Phase 4."""

    output_dir = Path(output_dir)

    # Aggregate statistics
    total_dim_jumps = sum(r['analysis']['num_dim_jumps'] for r in results)
    total_cap_jumps = sum(r['analysis']['num_cap_jumps'] for r in results)

    avg_lag = np.mean([
        r['analysis']['temporal_correlation']['mean_lag']
        for r in results
    ])

    # Count significant predictive relationships
    significant_count = 0
    total_tests = 0

    for r in results:
        for lag, data in r['analysis']['predictive_power'].items():
            total_tests += 1
            if data['significant']:
                significant_count += 1

    report = []
    report.append("# Phase 4: Capability Emergence Summary\n\n")
    report.append(f"Total experiments: {len(results)}\n\n")

    report.append("## Jump Statistics\n")
    report.append(f"- Total dimensionality jumps: {total_dim_jumps}\n")
    report.append(f"- Total capability jumps: {total_cap_jumps}\n")
    report.append(f"- Average temporal lag: {avg_lag:.1f} steps\n\n")

    report.append("## Predictive Power\n")
    report.append(f"- Significant correlations: {significant_count}/{total_tests} "
                 f"({significant_count/total_tests*100:.1f}%)\n\n")

    report.append("## Key Finding\n")
    if avg_lag > 0:
        report.append(f"**Dimensionality jumps PRECEDE capability jumps by ~{avg_lag:.0f} steps on average.**\n")
        report.append("This supports the hypothesis that dimensionality expansion predicts capability emergence.\n\n")
    else:
        report.append("No clear temporal relationship found between dimensionality and capability jumps.\n\n")

    # Save report
    with open(output_dir / 'phase4_summary.md', 'w') as f:
        f.writelines(report)

    print(f"\nPhase 4 summary saved: {output_dir / 'phase4_summary.md'}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Phase 4: Capability Emergence')
    parser.add_argument('--output-dir', type=str,
                       default='./experiments/new/results/phase4',
                       help='Output directory')
    parser.add_argument('--num-steps', type=int, default=3000,
                       help='Training steps per experiment')
    parser.add_argument('--measurement-interval', type=int, default=5,
                       help='Measurement frequency')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test')

    args = parser.parse_args()

    if args.quick_test:
        archs = ['mlp_shallow_2']
        datasets = ['mnist']
        num_steps = 500
    else:
        archs = ['mlp_shallow_2', 'mlp_medium_5', 'cnn_shallow', 'transformer_shallow']
        datasets = ['mnist', 'cifar10']
        num_steps = args.num_steps

    results = run_phase4_experiments(
        archs, datasets,
        args.output_dir,
        num_steps=num_steps,
        measurement_interval=args.measurement_interval
    )

    print("\nPhase 4 experiments complete!")
