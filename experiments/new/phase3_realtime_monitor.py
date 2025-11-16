"""
Phase 3: Real-Time Interpretability Tool

Goal: Build a real-time monitor that practitioners can use during training

Features:
1. Estimates D(t) during training (every N steps)
2. Predicts D(t+k) for k=50,100,200 steps ahead
3. Flags if expansion is insufficient (α too small)
4. Flags if jumps stop occurring (premature saturation)
5. Provides actionable recommendations

This tool is USEFUL to practitioners - it helps diagnose training issues early.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Callable
from pathlib import Path
import json
import time
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings('ignore')

# Import predictor from Phase 2
import sys
sys.path.append(str(Path(__file__).parent))
from phase2_prediction import DimensionalityPredictor
from phase1_calibration import measure_network_dimensionality, DimensionalityEstimator


# ============================================================================
# REAL-TIME MONITOR
# ============================================================================

class TrainingMonitor:
    """
    Real-time monitoring tool for neural network training dynamics.

    Tracks dimensionality, predicts future evolution, and flags issues.
    """

    def __init__(self,
                 model: nn.Module,
                 predictor: DimensionalityPredictor,
                 measurement_interval: int = 10,
                 prediction_horizons: List[int] = [50, 100, 200],
                 history_window: int = 500):
        """
        Initialize monitor.

        Args:
            model: PyTorch model to monitor
            predictor: Fitted predictor from Phase 1/2
            measurement_interval: How often to measure (in steps)
            prediction_horizons: Steps ahead to predict (k values)
            history_window: How many measurements to keep
        """
        self.model = model
        self.predictor = predictor
        self.measurement_interval = measurement_interval
        self.prediction_horizons = prediction_horizons
        self.history_window = history_window

        # Architecture parameters
        self.arch_params = model.get_architecture_params()
        self.alpha_estimated = predictor.estimate_alpha(self.arch_params)

        # History
        self.history = {
            'steps': deque(maxlen=history_window),
            'dimensionality': deque(maxlen=history_window),
            'grad_norms': deque(maxlen=history_window),
            'losses': deque(maxlen=history_window),
            'predictions': deque(maxlen=history_window)
        }

        # State tracking
        self.current_step = 0
        self.last_jump_step = 0
        self.jump_count = 0
        self.warnings = []
        self.recommendations = []

        # Thresholds
        self.alpha_min = 1e-5  # Minimum acceptable growth rate
        self.saturation_threshold = 0.95  # When to flag premature saturation
        self.jump_detection_window = 50
        self.jump_z_threshold = 2.0

        print(f"\n{'='*70}")
        print("Training Monitor Initialized")
        print(f"{'='*70}")
        print(f"Architecture: {self.arch_params['depth']} layers, "
              f"{self.arch_params['num_params']:,} parameters")
        print(f"Estimated α: {self.alpha_estimated:.6f}")
        print(f"Measurement interval: {self.measurement_interval} steps")
        print(f"Prediction horizons: {self.prediction_horizons}")
        print(f"{'='*70}\n")

    def update(self,
               step: int,
               loss: float,
               grad_norm: float,
               dataloader: Optional[torch.utils.data.DataLoader] = None,
               device: Optional[torch.device] = None) -> Dict:
        """
        Update monitor with current training state.

        Args:
            step: Current training step
            loss: Current loss value
            grad_norm: Current gradient norm
            dataloader: Optional dataloader for activation measurement
            device: Device for computation

        Returns:
            Dictionary with current state, predictions, and warnings
        """
        self.current_step = step

        # Measure dimensionality
        if dataloader is not None and device is not None:
            dim_metrics = measure_network_dimensionality(
                self.model, dataloader, device, max_batches=3
            )
            if dim_metrics:
                current_D = np.mean([m['stable_rank'] for m in dim_metrics.values()])
            else:
                current_D = 0
        else:
            # Fallback: estimate from history
            current_D = self.history['dimensionality'][-1] if self.history['dimensionality'] else 0.1

        # Update history
        self.history['steps'].append(step)
        self.history['dimensionality'].append(current_D)
        self.history['grad_norms'].append(grad_norm)
        self.history['losses'].append(loss)

        # Make predictions
        predictions = self._make_predictions(current_D, grad_norm)
        self.history['predictions'].append(predictions)

        # Detect issues
        warnings = self._check_for_issues(current_D, grad_norm)
        self.warnings.extend(warnings)

        # Generate recommendations
        recommendations = self._generate_recommendations()

        # Detect jumps
        jump_detected = self._detect_jump()

        return {
            'step': step,
            'current_dimensionality': current_D,
            'loss': loss,
            'grad_norm': grad_norm,
            'predictions': predictions,
            'warnings': warnings,
            'recommendations': recommendations,
            'jump_detected': jump_detected,
            'alpha_estimated': self.alpha_estimated
        }

    def _make_predictions(self, current_D: float, current_grad: float) -> Dict:
        """Predict future dimensionality."""

        predictions = {}

        # Estimate D_max
        D_max = np.sqrt(self.arch_params['width']) * self.arch_params['depth']

        for horizon in self.prediction_horizons:
            # Simple prediction: constant gradient
            future_D = current_D
            for _ in range(horizon):
                growth = self.alpha_estimated * current_grad * future_D * (1 - future_D / D_max)
                future_D += growth
                future_D = np.clip(future_D, 0, D_max)

            predictions[f'D_plus_{horizon}'] = future_D

        return predictions

    def _check_for_issues(self, current_D: float, grad_norm: float) -> List[str]:
        """Check for training issues."""

        warnings = []

        # 1. Check if α is too small
        if self.alpha_estimated < self.alpha_min:
            warnings.append(
                f"⚠️  WARNING: α is very small ({self.alpha_estimated:.6e}). "
                f"Network may be too shallow/narrow for effective learning."
            )

        # 2. Check for premature saturation
        if len(self.history['dimensionality']) > 20:
            recent_D = list(self.history['dimensionality'])[-20:]
            D_max_est = np.sqrt(self.arch_params['width']) * self.arch_params['depth']

            if np.mean(recent_D) > self.saturation_threshold * D_max_est:
                if self.current_step < 1000:  # Saturated too early
                    warnings.append(
                        f"⚠️  WARNING: Premature saturation detected at step {self.current_step}. "
                        f"Dimensionality has reached {np.mean(recent_D):.1f} / {D_max_est:.1f}"
                    )

        # 3. Check gradient vanishing
        if len(self.history['grad_norms']) > 10:
            recent_grads = list(self.history['grad_norms'])[-10:]
            if np.mean(recent_grads) < 1e-5:
                warnings.append(
                    f"⚠️  WARNING: Gradient vanishing detected. "
                    f"Mean gradient norm: {np.mean(recent_grads):.6e}"
                )

        # 4. Check if jumps stopped (stagnation)
        if self.current_step - self.last_jump_step > 2000 and self.current_step > 1000:
            warnings.append(
                f"⚠️  WARNING: No jumps detected for {self.current_step - self.last_jump_step} steps. "
                f"Possible training stagnation."
            )

        # 5. Check declining dimensionality
        if len(self.history['dimensionality']) > 50:
            recent_trend = np.polyfit(
                range(50),
                list(self.history['dimensionality'])[-50:],
                1
            )[0]

            if recent_trend < -0.01:
                warnings.append(
                    f"⚠️  WARNING: Dimensionality is declining. "
                    f"This may indicate representation collapse."
                )

        return warnings

    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations."""

        recommendations = []

        # Based on α
        if self.alpha_estimated < 1e-4:
            recommendations.append(
                "💡 RECOMMENDATION: Consider increasing network depth or width to allow "
                "for larger α and better capacity growth."
            )

        # Based on saturation
        if len(self.history['dimensionality']) > 20:
            recent_D = list(self.history['dimensionality'])[-20:]
            D_max_est = np.sqrt(self.arch_params['width']) * self.arch_params['depth']

            if np.mean(recent_D) > 0.9 * D_max_est:
                recommendations.append(
                    "💡 RECOMMENDATION: Network approaching capacity limits. "
                    "Consider early stopping or increasing width."
                )

        # Based on gradient norms
        if len(self.history['grad_norms']) > 10:
            recent_grads = list(self.history['grad_norms'])[-10:]
            if np.mean(recent_grads) < 1e-5:
                recommendations.append(
                    "💡 RECOMMENDATION: Gradients vanishing. Try: "
                    "(1) Reduce depth, (2) Use residual connections, "
                    "(3) Increase learning rate, (4) Check initialization."
                )

        # Based on jump patterns
        if self.jump_count == 0 and self.current_step > 500:
            recommendations.append(
                "💡 RECOMMENDATION: No phase transitions detected. "
                "Network may be in collapse phase. Check loss convergence."
            )

        return recommendations

    def _detect_jump(self) -> bool:
        """Detect discrete jumps in dimensionality."""

        if len(self.history['dimensionality']) < self.jump_detection_window:
            return False

        recent_D = list(self.history['dimensionality'])[-self.jump_detection_window:]

        # Compute gradient
        gradients = np.gradient(recent_D)

        # Check for outlier gradient (jump)
        mean_grad = np.mean(gradients[:-5])  # Exclude most recent
        std_grad = np.std(gradients[:-5])

        latest_grad = gradients[-1]
        z_score = (latest_grad - mean_grad) / (std_grad + 1e-10)

        if z_score > self.jump_z_threshold:
            self.jump_count += 1
            self.last_jump_step = self.current_step
            print(f"\n{'*'*70}")
            print(f"🚀 JUMP DETECTED at step {self.current_step}")
            print(f"   Magnitude: {latest_grad:.4f} (Z-score: {z_score:.2f})")
            print(f"   Total jumps: {self.jump_count}")
            print(f"{'*'*70}\n")
            return True

        return False

    def get_status_summary(self) -> str:
        """Get human-readable status summary."""

        if not self.history['dimensionality']:
            return "No measurements yet."

        current_D = self.history['dimensionality'][-1]
        current_loss = self.history['losses'][-1]
        D_max_est = np.sqrt(self.arch_params['width']) * self.arch_params['depth']

        summary = []
        summary.append(f"\n{'='*70}")
        summary.append(f"Training Status at Step {self.current_step}")
        summary.append(f"{'='*70}")
        summary.append(f"Current Dimensionality: {current_D:.2f} / {D_max_est:.2f} "
                      f"({current_D/D_max_est*100:.1f}%)")
        summary.append(f"Loss: {current_loss:.4f}")
        summary.append(f"Jumps detected: {self.jump_count}")
        summary.append(f"Steps since last jump: {self.current_step - self.last_jump_step}")

        if self.history['predictions']:
            latest_pred = self.history['predictions'][-1]
            summary.append(f"\nPredictions:")
            for k, v in latest_pred.items():
                summary.append(f"  {k}: {v:.2f}")

        if self.warnings:
            summary.append(f"\nRecent Warnings ({len(self.warnings)}):")
            for w in self.warnings[-3:]:
                summary.append(f"  {w}")

        if self.recommendations:
            summary.append(f"\nRecommendations:")
            for r in self.recommendations[:3]:
                summary.append(f"  {r}")

        summary.append(f"{'='*70}\n")

        return "\n".join(summary)

    def save_state(self, filepath: str):
        """Save monitor state to file."""

        state = {
            'arch_params': self.arch_params,
            'alpha_estimated': self.alpha_estimated,
            'current_step': self.current_step,
            'jump_count': self.jump_count,
            'history': {
                'steps': list(self.history['steps']),
                'dimensionality': list(self.history['dimensionality']),
                'grad_norms': list(self.history['grad_norms']),
                'losses': list(self.history['losses'])
            },
            'warnings': self.warnings,
            'recommendations': self.recommendations
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

        print(f"Monitor state saved to {filepath}")


# ============================================================================
# TRAINING WITH MONITORING
# ============================================================================

def train_with_realtime_monitoring(model: nn.Module,
                                   train_loader: torch.utils.data.DataLoader,
                                   val_loader: torch.utils.data.DataLoader,
                                   predictor_path: str,
                                   num_steps: int = 5000,
                                   measurement_interval: int = 10,
                                   status_interval: int = 500,
                                   device: Optional[torch.device] = None) -> Dict:
    """
    Train model with real-time monitoring and early warnings.

    This demonstrates the Phase 3 tool in action.
    """

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load predictor
    predictor = DimensionalityPredictor(predictor_path)

    # Initialize monitor
    monitor = TrainingMonitor(
        model, predictor,
        measurement_interval=measurement_interval,
        prediction_horizons=[50, 100, 200]
    )

    # Setup training
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    step = 0
    epoch = 0

    print("\nStarting training with real-time monitoring...\n")

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

            # Monitor update
            if step % measurement_interval == 0:
                monitor_state = monitor.update(
                    step, loss.item(), grad_norm,
                    val_loader, device
                )

                # Print warnings immediately
                if monitor_state['warnings']:
                    for warning in monitor_state['warnings']:
                        print(warning)

            # Periodic status
            if step % status_interval == 0:
                print(monitor.get_status_summary())

            step += 1

    # Final status
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(monitor.get_status_summary())

    return {
        'monitor': monitor,
        'final_step': step
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_monitoring_results(monitor: TrainingMonitor,
                            output_file: str = './monitor_results.png'):
    """Visualize monitoring results."""

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))

    steps = list(monitor.history['steps'])
    D = list(monitor.history['dimensionality'])
    grad_norms = list(monitor.history['grad_norms'])
    losses = list(monitor.history['losses'])

    # Plot 1: Dimensionality evolution
    axes[0, 0].plot(steps, D, 'o-', alpha=0.7, linewidth=2)
    axes[0, 0].set_xlabel('Training Step')
    axes[0, 0].set_ylabel('Dimensionality (Stable Rank)')
    axes[0, 0].set_title('Dimensionality Evolution')
    axes[0, 0].grid(True, alpha=0.3)

    # Mark jumps
    jump_steps = []
    for i in range(1, len(D)):
        grad = D[i] - D[i-1]
        mean_grad = np.mean([D[j] - D[j-1] for j in range(1, max(1, i-20), 1)])
        std_grad = np.std([D[j] - D[j-1] for j in range(1, max(1, i-20), 1)])
        z = (grad - mean_grad) / (std_grad + 1e-10)
        if z > 2.0:
            jump_steps.append(steps[i])

    for js in jump_steps:
        axes[0, 0].axvline(js, color='red', alpha=0.3, linestyle='--')

    # Plot 2: Gradient norms
    axes[0, 1].plot(steps, grad_norms, 'o-', alpha=0.7, color='orange')
    axes[0, 1].set_xlabel('Training Step')
    axes[0, 1].set_ylabel('Gradient Norm')
    axes[0, 1].set_title('Gradient Evolution')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Loss
    axes[1, 0].plot(steps, losses, 'o-', alpha=0.7, color='green')
    axes[1, 0].set_xlabel('Training Step')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Loss Evolution')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Growth rate
    if len(D) > 1:
        growth_rates = [D[i] - D[i-1] for i in range(1, len(D))]
        axes[1, 1].plot(steps[1:], growth_rates, 'o-', alpha=0.7, color='purple')
        axes[1, 1].axhline(0, color='red', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Training Step')
        axes[1, 1].set_ylabel('Growth Rate (ΔD)')
        axes[1, 1].set_title('Dimensionality Growth Rate')
        axes[1, 1].grid(True, alpha=0.3)

    # Plot 5: Capacity utilization
    D_max = np.sqrt(monitor.arch_params['width']) * monitor.arch_params['depth']
    utilization = [d / D_max * 100 for d in D]
    axes[2, 0].plot(steps, utilization, 'o-', alpha=0.7, color='brown')
    axes[2, 0].axhline(90, color='red', linestyle='--', label='90% capacity')
    axes[2, 0].set_xlabel('Training Step')
    axes[2, 0].set_ylabel('Capacity Utilization (%)')
    axes[2, 0].set_title('Network Capacity Utilization')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    # Plot 6: Statistics
    axes[2, 1].text(0.1, 0.9, f"Total jumps: {monitor.jump_count}", transform=axes[2, 1].transAxes)
    axes[2, 1].text(0.1, 0.8, f"Final D: {D[-1]:.2f} / {D_max:.2f}",
                   transform=axes[2, 1].transAxes)
    axes[2, 1].text(0.1, 0.7, f"α estimated: {monitor.alpha_estimated:.6f}",
                   transform=axes[2, 1].transAxes)
    axes[2, 1].text(0.1, 0.6, f"Warnings: {len(monitor.warnings)}",
                   transform=axes[2, 1].transAxes)
    axes[2, 1].set_title('Summary Statistics')
    axes[2, 1].axis('off')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Monitoring results saved to {output_file}")
    plt.close()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import argparse
    from phase1_calibration import create_architecture, get_mnist_loaders

    parser = argparse.ArgumentParser(description='Phase 3: Real-Time Monitoring')
    parser.add_argument('--predictor', type=str,
                       default='./experiments/new/results/phase1_analysis/alpha_models.pkl',
                       help='Path to predictor from Phase 1')
    parser.add_argument('--arch', type=str, default='mlp_medium_5',
                       help='Architecture to train')
    parser.add_argument('--num-steps', type=int, default=2000,
                       help='Training steps')
    parser.add_argument('--output-dir', type=str,
                       default='./experiments/new/results/phase3',
                       help='Output directory')

    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, input_dim, num_classes, _ = get_mnist_loaders(
        batch_size=64, subset_size=5000
    )

    model = create_architecture(args.arch, input_dim, num_classes, 1)

    # Train with monitoring
    result = train_with_realtime_monitoring(
        model, train_loader, val_loader,
        args.predictor,
        num_steps=args.num_steps,
        measurement_interval=10,
        status_interval=500,
        device=device
    )

    # Save results
    monitor = result['monitor']
    monitor.save_state(Path(args.output_dir) / 'monitor_state.json')

    # Visualize
    plot_monitoring_results(
        monitor,
        Path(args.output_dir) / 'monitor_visualization.png'
    )

    print("\nPhase 3 demonstration complete!")
    print(f"Results saved to {args.output_dir}")
