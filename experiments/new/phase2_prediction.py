"""
Phase 2: Prediction Experiments

Goal: Test if we can predict dimensionality curves D(t) from architecture parameters BEFORE training

Success Criterion: R² > 0.8 between predicted and actual curves

Process:
1. Load α models from Phase 1
2. For new architecture (not in training set):
   - Estimate α from design parameters using fitted model
   - Predict dimensionality curve D(t) before training
3. Train the network and measure actual D(t)
4. Compare predicted vs actual (R² metric)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import from phase1
import sys
sys.path.append(str(Path(__file__).parent))
from phase1_calibration import (
    create_architecture, DATASET_LOADERS,
    measure_network_dimensionality, DimensionalityEstimator
)


# ============================================================================
# PREDICTION ENGINE
# ============================================================================

class DimensionalityPredictor:
    """Predicts dimensionality curves from architecture parameters."""

    def __init__(self, alpha_model_path: str):
        """Load fitted α models from Phase 1."""
        with open(alpha_model_path, 'rb') as f:
            self.models = pickle.load(f)

        self.linear_model = self.models['linear']['model']
        self.best_model_name = max(
            [(k, v['R2']) for k, v in self.models.items() if k != 'data'],
            key=lambda x: x[1]
        )[0]

        print(f"Loaded predictor. Best model: {self.best_model_name} "
              f"(R² = {self.models[self.best_model_name]['R2']:.4f})")

    def estimate_alpha(self, arch_params: Dict) -> float:
        """
        Estimate α from architecture parameters.

        Uses the best-fitting model from Phase 1.
        """
        depth = arch_params['depth']
        width = arch_params['width']
        connectivity = arch_params.get('connectivity', 0)

        X = np.array([[depth, width, connectivity]])

        if self.best_model_name == 'linear':
            alpha = self.linear_model.predict(X)[0]
        elif self.best_model_name == 'log_linear':
            X_log = np.log1p(X)
            alpha = self.models['log_linear']['model'].predict(X_log)[0]
        elif self.best_model_name == 'power_law':
            X_log = np.log1p(X)
            log_alpha = self.models['power_law']['model'].predict(X_log)[0]
            alpha = np.exp(log_alpha) - 1
        else:
            alpha = self.linear_model.predict(X)[0]

        return max(alpha, 1e-6)  # Ensure positive

    def predict_curve(self,
                     arch_params: Dict,
                     num_steps: int,
                     D_max: Optional[float] = None,
                     grad_norm_avg: float = 1.0) -> np.ndarray:
        """
        Predict dimensionality curve D(t) using TAP model.

        D(t+1) = D(t) + α · ||∇L||_t · D(t) · (1 - D(t)/D_max)

        Args:
            arch_params: Architecture parameters (depth, width, etc.)
            num_steps: Number of training steps to predict
            D_max: Maximum dimensionality (estimated if None)
            grad_norm_avg: Average gradient norm (assumed constant)

        Returns:
            Array of predicted dimensionality values
        """
        alpha = self.estimate_alpha(arch_params)

        # Estimate D_max from architecture if not provided
        if D_max is None:
            # Heuristic: D_max ≈ sqrt(width) * depth
            D_max = np.sqrt(arch_params['width']) * arch_params['depth']

        # Numerical integration of TAP model
        D = np.zeros(num_steps)
        D[0] = 0.1  # Initial dimensionality

        for t in range(num_steps - 1):
            growth_rate = alpha * grad_norm_avg * D[t] * (1 - D[t] / D_max)
            D[t+1] = D[t] + growth_rate
            D[t+1] = np.clip(D[t+1], 0, D_max)

        return D

    def predict_with_gradient_history(self,
                                      arch_params: Dict,
                                      grad_norms: List[float],
                                      D_max: Optional[float] = None) -> np.ndarray:
        """
        Predict dimensionality using actual gradient history.

        More accurate than constant gradient assumption.
        """
        alpha = self.estimate_alpha(arch_params)

        if D_max is None:
            D_max = np.sqrt(arch_params['width']) * arch_params['depth']

        num_steps = len(grad_norms)
        D = np.zeros(num_steps)
        D[0] = 0.1

        for t in range(num_steps - 1):
            growth_rate = alpha * grad_norms[t] * D[t] * (1 - D[t] / D_max)
            D[t+1] = D[t] + growth_rate
            D[t+1] = np.clip(D[t+1], 0, D_max)

        return D


# ============================================================================
# VALIDATION EXPERIMENT
# ============================================================================

def run_prediction_validation(predictor: DimensionalityPredictor,
                              arch_name: str,
                              dataset_name: str,
                              num_steps: int = 5000,
                              measurement_interval: int = 5,
                              device: Optional[torch.device] = None) -> Dict:
    """
    Run prediction validation for a single architecture.

    1. Predict D(t) before training
    2. Train and measure actual D(t)
    3. Compare and compute R²
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*70}")
    print(f"Prediction Validation: {arch_name} on {dataset_name}")
    print(f"{'='*70}")

    # Load data
    loader_fn = DATASET_LOADERS[dataset_name]
    train_loader, val_loader, input_dim, num_classes, in_channels = loader_fn(
        batch_size=64, subset_size=10000
    )

    # Create model
    model = create_architecture(arch_name, input_dim, num_classes, in_channels)
    arch_params = model.get_architecture_params()

    print(f"\nArchitecture parameters:")
    for k, v in arch_params.items():
        print(f"  {k}: {v}")

    # Step 1: Make prediction
    print("\n1. Predicting dimensionality curve...")
    alpha_predicted = predictor.estimate_alpha(arch_params)
    print(f"   Predicted α: {alpha_predicted:.6f}")

    # Initial prediction (with constant gradient assumption)
    steps_array = np.arange(0, num_steps, measurement_interval)
    D_predicted_simple = predictor.predict_curve(arch_params, len(steps_array))

    # Step 2: Train and measure actual
    print("\n2. Training and measuring actual dimensionality...")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    measurements = []
    step = 0
    epoch = 0

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

            # Measure
            if step % measurement_interval == 0:
                dim_metrics = measure_network_dimensionality(
                    model, val_loader, device, max_batches=5
                )

                # Average across layers
                if dim_metrics:
                    avg_stable_rank = np.mean([
                        m['stable_rank'] for m in dim_metrics.values()
                    ])
                else:
                    avg_stable_rank = 0

                measurements.append({
                    'step': step,
                    'dimensionality': avg_stable_rank,
                    'grad_norm': grad_norm,
                    'loss': loss.item()
                })

            step += 1
            pbar.update(1)

    pbar.close()

    # Extract actual dimensionality
    actual_steps = [m['step'] for m in measurements]
    actual_D = [m['dimensionality'] for m in measurements]
    actual_grads = [m['grad_norm'] for m in measurements]

    # Step 3: Refined prediction using actual gradient history
    print("\n3. Refining prediction with actual gradient history...")
    D_predicted_refined = predictor.predict_with_gradient_history(
        arch_params, actual_grads
    )

    # Step 4: Compare predictions
    print("\n4. Evaluating predictions...")

    # Compute R² for simple prediction
    ss_res_simple = np.sum((np.array(actual_D) - D_predicted_simple[:len(actual_D)]) ** 2)
    ss_tot = np.sum((np.array(actual_D) - np.mean(actual_D)) ** 2)
    r2_simple = 1 - (ss_res_simple / ss_tot) if ss_tot > 0 else 0

    # Compute R² for refined prediction
    ss_res_refined = np.sum((np.array(actual_D) - D_predicted_refined[:len(actual_D)]) ** 2)
    r2_refined = 1 - (ss_res_refined / ss_tot) if ss_tot > 0 else 0

    print(f"\n   Simple prediction R²: {r2_simple:.4f}")
    print(f"   Refined prediction R²: {r2_refined:.4f}")

    # Success criterion
    success = r2_refined > 0.8
    print(f"\n   {'✓ SUCCESS' if success else '✗ FAILED'} "
          f"(criterion: R² > 0.8)")

    return {
        'arch_name': arch_name,
        'dataset_name': dataset_name,
        'architecture_params': arch_params,
        'alpha_predicted': alpha_predicted,
        'predictions': {
            'simple': {
                'curve': D_predicted_simple.tolist(),
                'R2': r2_simple
            },
            'refined': {
                'curve': D_predicted_refined.tolist(),
                'R2': r2_refined
            }
        },
        'actual': {
            'steps': actual_steps,
            'dimensionality': actual_D,
            'grad_norms': actual_grads
        },
        'success': success,
        'success_criterion': 'R² > 0.8'
    }


# ============================================================================
# PHASE 2 EXPERIMENTAL PIPELINE
# ============================================================================

def run_phase2_experiments(predictor_path: str,
                          test_architectures: Optional[List[str]] = None,
                          test_datasets: Optional[List[str]] = None,
                          output_dir: str = './experiments/new/results/phase2',
                          num_steps: int = 5000):
    """
    Run Phase 2 prediction validation experiments.

    Tests prediction accuracy on architectures NOT in Phase 1 training set.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load predictor
    predictor = DimensionalityPredictor(predictor_path)

    # Define test architectures (should be different from Phase 1)
    if test_architectures is None:
        test_architectures = [
            'mlp_shallow_2',  # Included in Phase 1 as control
            'mlp_medium_5',   # Included in Phase 1 as control
            # New architectures not in Phase 1 training
            'mlp_deep_7',
            'mlp_wide_3layer',
            'cnn_custom',
            'transformer_test'
        ]

    if test_datasets is None:
        test_datasets = ['mnist', 'cifar10']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"\nPhase 2 Prediction Validation:")
    print(f"  Test architectures: {len(test_architectures)}")
    print(f"  Test datasets: {len(test_datasets)}")
    print(f"  Total experiments: {len(test_architectures) * len(test_datasets)}")

    results = []

    for dataset in test_datasets:
        for arch in test_architectures:
            try:
                result = run_prediction_validation(
                    predictor, arch, dataset,
                    num_steps=num_steps,
                    device=device
                )
                results.append(result)

                # Save individual result
                output_file = Path(output_dir) / f'{arch}_{dataset}.json'
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)

                print(f"✓ Saved: {output_file}")

            except Exception as e:
                print(f"✗ Failed {arch} on {dataset}: {e}")
                continue

    # Save summary
    summary_file = Path(output_dir) / 'phase2_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Statistics
    successes = sum(1 for r in results if r['success'])
    avg_r2_simple = np.mean([r['predictions']['simple']['R2'] for r in results])
    avg_r2_refined = np.mean([r['predictions']['refined']['R2'] for r in results])

    print(f"\n{'='*70}")
    print("Phase 2 Summary:")
    print(f"  Total experiments: {len(results)}")
    print(f"  Successful predictions (R² > 0.8): {successes}/{len(results)} "
          f"({successes/len(results)*100:.1f}%)")
    print(f"  Average R² (simple): {avg_r2_simple:.4f}")
    print(f"  Average R² (refined): {avg_r2_refined:.4f}")
    print(f"{'='*70}")

    return results


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_predictions(results: List[Dict],
                         output_dir: str = './experiments/new/results/phase2'):
    """Create visualizations comparing predicted vs actual curves."""

    output_dir = Path(output_dir)
    sns.set_style("whitegrid")

    # 1. Individual prediction plots
    for result in results:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        arch_name = result['arch_name']
        dataset = result['dataset_name']

        actual_steps = result['actual']['steps']
        actual_D = result['actual']['dimensionality']
        pred_simple = result['predictions']['simple']['curve']
        pred_refined = result['predictions']['refined']['curve']

        # Plot 1: Predictions vs Actual
        axes[0, 0].plot(actual_steps, actual_D, 'o-', label='Actual', alpha=0.7, linewidth=2)
        axes[0, 0].plot(actual_steps, pred_simple[:len(actual_steps)],
                       's--', label='Predicted (simple)', alpha=0.7)
        axes[0, 0].plot(actual_steps, pred_refined[:len(actual_steps)],
                       '^--', label='Predicted (refined)', alpha=0.7)
        axes[0, 0].set_xlabel('Training Step')
        axes[0, 0].set_ylabel('Dimensionality (Stable Rank)')
        axes[0, 0].set_title(f'{arch_name} on {dataset}')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Residuals (Refined)
        residuals = np.array(actual_D) - np.array(pred_refined[:len(actual_D)])
        axes[0, 1].plot(actual_steps, residuals, 'o-', alpha=0.6)
        axes[0, 1].axhline(0, color='red', linestyle='--', alpha=0.5)
        axes[0, 1].set_xlabel('Training Step')
        axes[0, 1].set_ylabel('Residual (Actual - Predicted)')
        axes[0, 1].set_title('Prediction Residuals')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Gradient norms
        grad_norms = result['actual']['grad_norms']
        axes[1, 0].plot(actual_steps, grad_norms, 'o-', alpha=0.6)
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Gradient Norm')
        axes[1, 0].set_title('Gradient Evolution')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Metrics
        r2_simple = result['predictions']['simple']['R2']
        r2_refined = result['predictions']['refined']['R2']

        metrics_data = [r2_simple, r2_refined]
        labels = ['Simple', 'Refined']
        colors = ['lightblue', 'lightgreen']

        bars = axes[1, 1].bar(labels, metrics_data, color=colors, alpha=0.7, edgecolor='black')
        axes[1, 1].axhline(0.8, color='red', linestyle='--', label='Success threshold')
        axes[1, 1].set_ylabel('R² Score')
        axes[1, 1].set_title('Prediction Quality')
        axes[1, 1].set_ylim([0, 1])
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        # Add R² values on bars
        for bar, value in zip(bars, metrics_data):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                          f'{value:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(output_dir / f'prediction_{arch_name}_{dataset}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

    # 2. Summary plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    r2_simple = [r['predictions']['simple']['R2'] for r in results]
    r2_refined = [r['predictions']['refined']['R2'] for r in results]
    labels = [f"{r['arch_name']}\n{r['dataset_name']}" for r in results]

    x = np.arange(len(results))
    width = 0.35

    axes[0].bar(x - width/2, r2_simple, width, label='Simple', alpha=0.7)
    axes[0].bar(x + width/2, r2_refined, width, label='Refined', alpha=0.7)
    axes[0].axhline(0.8, color='red', linestyle='--', label='Threshold')
    axes[0].set_xlabel('Experiment')
    axes[0].set_ylabel('R² Score')
    axes[0].set_title('Prediction Accuracy Across Experiments')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # Histogram of R² values
    axes[1].hist(r2_refined, bins=15, alpha=0.7, edgecolor='black')
    axes[1].axvline(0.8, color='red', linestyle='--', linewidth=2, label='Success threshold')
    axes[1].set_xlabel('R² Score (Refined)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Distribution of Prediction Quality')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'phase2_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nVisualizations saved to {output_dir}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Phase 2: Prediction Experiments')
    parser.add_argument('--predictor', type=str,
                       default='./experiments/new/results/phase1_analysis/alpha_models.pkl',
                       help='Path to fitted α models from Phase 1')
    parser.add_argument('--output-dir', type=str,
                       default='./experiments/new/results/phase2',
                       help='Output directory')
    parser.add_argument('--num-steps', type=int, default=5000,
                       help='Training steps per experiment')

    args = parser.parse_args()

    print("="*70)
    print("Phase 2: Prediction Experiments")
    print("="*70)

    # Run experiments
    results = run_phase2_experiments(
        args.predictor,
        output_dir=args.output_dir,
        num_steps=args.num_steps
    )

    # Visualize
    print("\nGenerating visualizations...")
    visualize_predictions(results, args.output_dir)

    print("\n" + "="*70)
    print("Phase 2 Complete!")
    print("="*70)
    print(f"Results saved to: {args.output_dir}")
    print("\nNext: Run Phase 3 (real-time monitoring tool)")
