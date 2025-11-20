"""
Phase 2 Infrastructure: Tools for Mechanistic Analysis

This module provides utilities for:
1. Checkpoint loading at critical moments
2. Attention pattern extraction and analysis (Transformers)
3. Filter/activation visualization (CNNs)
4. Hidden unit analysis (MLPs)
5. Measurement functions (entropy, similarity, specialization)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Optional PyTorch import (needed when working with actual model checkpoints)
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Note: PyTorch not available. Some features will be limited.")
    print("Install PyTorch for full functionality when working with model checkpoints.")


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class CriticalMoment:
    """Represents a critical moment for investigation."""
    experiment_name: str
    step: int
    moment_type: str  # 'before_jump', 'during_jump', 'after_jump', 'critical_period'
    jump_info: Optional[Dict] = None


@dataclass
class AnalysisResult:
    """Container for analysis results."""
    experiment: str
    moment: str
    measurements: Dict[str, Any]
    visualizations: Dict[str, Any]


# ============================================================================
# CHECKPOINT UTILITIES
# ============================================================================

class CheckpointManager:
    """
    Manages loading checkpoints at critical moments.

    Note: Currently experiments only have raw measurements, not saved model checkpoints.
    This class provides the interface for when checkpoints become available.
    """

    def __init__(self, experiments_dir: Path):
        self.experiments_dir = experiments_dir
        self.measurements_dir = experiments_dir / 'new' / 'results' / 'phase1_full'

    def identify_critical_moments(self, experiment_name: str,
                                  jump_data: Dict,
                                  num_moments: int = 5) -> List[CriticalMoment]:
        """
        Identify critical moments for a given experiment based on jump analysis.

        Args:
            experiment_name: Name of experiment
            jump_data: Jump characterization data from Phase 1
            num_moments: Number of moments to identify

        Returns:
            List of CriticalMoment objects
        """
        moments = []

        # Find jumps for this experiment
        exp_jumps = [j for j in jump_data if j.get('experiment') == experiment_name]

        if not exp_jumps:
            print(f"No jumps found for {experiment_name}")
            return moments

        # Sort by magnitude (largest first)
        exp_jumps.sort(key=lambda x: x.get('magnitude', 0), reverse=True)

        # Select top jumps
        for i, jump in enumerate(exp_jumps[:num_moments]):
            step = jump['step']

            # Create before/during/after moments for each jump
            moments.extend([
                CriticalMoment(
                    experiment_name=experiment_name,
                    step=max(0, step - 10),
                    moment_type='before_jump',
                    jump_info=jump
                ),
                CriticalMoment(
                    experiment_name=experiment_name,
                    step=step,
                    moment_type='during_jump',
                    jump_info=jump
                ),
                CriticalMoment(
                    experiment_name=experiment_name,
                    step=step + 10,
                    moment_type='after_jump',
                    jump_info=jump
                )
            ])

        return moments

    def load_measurements_at_step(self, experiment_name: str, step: int) -> Optional[Dict]:
        """
        Load measurements at a specific training step.

        Args:
            experiment_name: Name of experiment
            step: Training step

        Returns:
            Measurement dictionary or None if not found
        """
        exp_file = self.measurements_dir / f"{experiment_name}.json"

        if not exp_file.exists():
            print(f"Experiment file not found: {exp_file}")
            return None

        with open(exp_file, 'r') as f:
            data = json.load(f)

        # Find measurement closest to requested step
        measurements = data['measurements']
        closest_idx = min(range(len(measurements)),
                         key=lambda i: abs(measurements[i]['step'] - step))

        return measurements[closest_idx]

    def get_checkpoint_path(self, experiment_name: str, step: int) -> Optional[Path]:
        """
        Get path to model checkpoint (placeholder for when checkpoints exist).

        Args:
            experiment_name: Name of experiment
            step: Training step

        Returns:
            Path to checkpoint file or None
        """
        # Placeholder - checkpoints don't exist yet
        checkpoint_path = self.experiments_dir / 'checkpoints' / experiment_name / f'step_{step}.pt'

        if checkpoint_path.exists():
            return checkpoint_path
        else:
            print(f"Note: Checkpoint not available at {checkpoint_path}")
            print("Phase 2 will require re-running experiments with checkpoint saving enabled.")
            return None


# ============================================================================
# MEASUREMENT FUNCTIONS
# ============================================================================

class MeasurementTools:
    """Tools for quantifying representational changes."""

    @staticmethod
    def _to_numpy(x: Union[np.ndarray, 'torch.Tensor']) -> np.ndarray:
        """Convert tensor to numpy array."""
        if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    @staticmethod
    def attention_entropy(attention_weights: Union[np.ndarray, 'torch.Tensor']) -> float:
        """
        Calculate entropy of attention patterns.

        High entropy = distributed attention
        Low entropy = focused attention

        Args:
            attention_weights: [batch, heads, seq_len, seq_len] (numpy or torch)

        Returns:
            Mean entropy across batch and heads
        """
        if TORCH_AVAILABLE and isinstance(attention_weights, torch.Tensor):
            # PyTorch implementation
            probs = torch.softmax(attention_weights, dim=-1)
            epsilon = 1e-10
            entropy = -torch.sum(probs * torch.log(probs + epsilon), dim=-1)
            return entropy.mean().item()
        else:
            # NumPy implementation
            attention_weights = np.asarray(attention_weights)
            # Softmax
            exp_weights = np.exp(attention_weights - np.max(attention_weights, axis=-1, keepdims=True))
            probs = exp_weights / np.sum(exp_weights, axis=-1, keepdims=True)
            # Entropy
            epsilon = 1e-10
            entropy = -np.sum(probs * np.log(probs + epsilon), axis=-1)
            return float(np.mean(entropy))

    @staticmethod
    def attention_specialization(attention_weights: Union[np.ndarray, 'torch.Tensor']) -> Dict[str, float]:
        """
        Measure how specialized attention heads are.

        Args:
            attention_weights: [batch, heads, seq_len, seq_len]

        Returns:
            Dictionary with specialization metrics
        """
        attention_weights = MeasurementTools._to_numpy(attention_weights)

        # Softmax
        exp_weights = np.exp(attention_weights - np.max(attention_weights, axis=-1, keepdims=True))
        probs = exp_weights / np.sum(exp_weights, axis=-1, keepdims=True)

        # Per-head entropy
        epsilon = 1e-10
        head_entropies = -np.sum(probs * np.log(probs + epsilon), axis=-1)
        head_entropies = head_entropies.mean(axis=(0, 2))  # Average over batch and sequence

        return {
            'mean_entropy': float(head_entropies.mean()),
            'min_entropy': float(head_entropies.min()),
            'max_entropy': float(head_entropies.max()),
            'entropy_std': float(head_entropies.std()),
            'specialization_index': float(head_entropies.max() - head_entropies.min())
        }

    @staticmethod
    def cosine_similarity(tensor1: Union[np.ndarray, 'torch.Tensor'],
                         tensor2: Union[np.ndarray, 'torch.Tensor']) -> float:
        """
        Calculate cosine similarity between two tensors.

        Args:
            tensor1: First tensor
            tensor2: Second tensor (same shape)

        Returns:
            Cosine similarity (-1 to 1)
        """
        arr1 = MeasurementTools._to_numpy(tensor1).flatten()
        arr2 = MeasurementTools._to_numpy(tensor2).flatten()

        # Cosine similarity
        dot_product = np.dot(arr1, arr2)
        norm1 = np.linalg.norm(arr1)
        norm2 = np.linalg.norm(arr2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    @staticmethod
    def weight_change_magnitude(weights_before: Union[np.ndarray, 'torch.Tensor'],
                               weights_after: Union[np.ndarray, 'torch.Tensor']) -> Dict[str, float]:
        """
        Measure magnitude of weight changes.

        Args:
            weights_before: Weights before jump
            weights_after: Weights after jump

        Returns:
            Dictionary with change metrics
        """
        w_before = MeasurementTools._to_numpy(weights_before)
        w_after = MeasurementTools._to_numpy(weights_after)
        delta = w_after - w_before

        return {
            'l2_norm': float(np.linalg.norm(delta)),
            'relative_change': float(np.linalg.norm(delta) / np.linalg.norm(w_before)),
            'max_change': float(np.abs(delta).max()),
            'mean_change': float(np.abs(delta).mean()),
            'cosine_similarity': MeasurementTools.cosine_similarity(w_before, w_after)
        }

    @staticmethod
    def activation_sparsity(activations: Union[np.ndarray, 'torch.Tensor'],
                           threshold: float = 0.1) -> float:
        """
        Measure sparsity of activations.

        Args:
            activations: Activation tensor [batch, features]
            threshold: Threshold for considering a unit active

        Returns:
            Sparsity (fraction of units below threshold)
        """
        activations = MeasurementTools._to_numpy(activations)
        active = (np.abs(activations) > threshold).astype(float)
        sparsity = 1.0 - float(active.mean())

        return sparsity

    @staticmethod
    def activation_selectivity(activations: Union[np.ndarray, 'torch.Tensor'],
                               labels: Union[np.ndarray, 'torch.Tensor']) -> np.ndarray:
        """
        Measure selectivity of each neuron for different classes.

        Args:
            activations: [batch, features]
            labels: [batch] class labels

        Returns:
            Selectivity matrix [features, num_classes]
        """
        activations = MeasurementTools._to_numpy(activations)
        labels = MeasurementTools._to_numpy(labels).astype(int)

        num_features = activations.shape[1]
        num_classes = int(labels.max()) + 1

        selectivity = np.zeros((num_features, num_classes))

        for c in range(num_classes):
            class_mask = (labels == c)
            if class_mask.sum() > 0:
                class_activations = activations[class_mask]
                selectivity[:, c] = class_activations.mean(axis=0)

        return selectivity


# ============================================================================
# VISUALIZATION TOOLS
# ============================================================================

class VisualizationTools:
    """Tools for visualizing representations."""

    @staticmethod
    def plot_attention_pattern(attention_weights: Union[np.ndarray, 'torch.Tensor'],
                              tokens: List[str],
                              head_idx: int = 0,
                              save_path: Optional[Path] = None):
        """
        Visualize attention pattern for a specific head.

        Args:
            attention_weights: [batch, heads, seq_len, seq_len]
            tokens: List of token strings
            head_idx: Which attention head to visualize
            save_path: Where to save plot
        """
        # Convert to numpy
        if TORCH_AVAILABLE and isinstance(attention_weights, torch.Tensor):
            attn = attention_weights[:, head_idx, :, :].mean(dim=0).cpu().numpy()
        else:
            attention_weights = np.asarray(attention_weights)
            attn = attention_weights[:, head_idx, :, :].mean(axis=0)

        plt.figure(figsize=(10, 8))
        sns.heatmap(attn, xticklabels=tokens, yticklabels=tokens,
                   cmap='viridis', cbar_kws={'label': 'Attention Weight'})
        plt.xlabel('Key Tokens')
        plt.ylabel('Query Tokens')
        plt.title(f'Attention Pattern - Head {head_idx}')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_attention_heads_comparison(attention_before: Union[np.ndarray, 'torch.Tensor'],
                                       attention_after: Union[np.ndarray, 'torch.Tensor'],
                                       num_heads: int = 4,
                                       save_path: Optional[Path] = None):
        """
        Compare attention patterns before and after a jump.

        Args:
            attention_before: Attention weights before jump
            attention_after: Attention weights after jump
            num_heads: Number of heads to visualize
            save_path: Where to save plot
        """
        # Convert to numpy
        if TORCH_AVAILABLE and isinstance(attention_before, torch.Tensor):
            attn_before = attention_before.cpu().numpy()
        else:
            attn_before = np.asarray(attention_before)

        if TORCH_AVAILABLE and isinstance(attention_after, torch.Tensor):
            attn_after = attention_after.cpu().numpy()
        else:
            attn_after = np.asarray(attention_after)

        fig, axes = plt.subplots(2, num_heads, figsize=(4*num_heads, 8))

        for head_idx in range(num_heads):
            # Before
            attn_b = attn_before[:, head_idx, :, :].mean(axis=0)
            axes[0, head_idx].imshow(attn_b, cmap='viridis')
            axes[0, head_idx].set_title(f'Head {head_idx} - Before')
            axes[0, head_idx].axis('off')

            # After
            attn_a = attn_after[:, head_idx, :, :].mean(axis=0)
            axes[1, head_idx].imshow(attn_a, cmap='viridis')
            axes[1, head_idx].set_title(f'Head {head_idx} - After')
            axes[1, head_idx].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_filter_visualization(filters: Union[np.ndarray, 'torch.Tensor'],
                                 num_filters: int = 16,
                                 save_path: Optional[Path] = None):
        """
        Visualize CNN filters.

        Args:
            filters: Conv layer weights [out_channels, in_channels, H, W]
            num_filters: Number of filters to show
            save_path: Where to save plot
        """
        # Convert to numpy
        if TORCH_AVAILABLE and isinstance(filters, torch.Tensor):
            filters = filters.cpu().numpy()
        else:
            filters = np.asarray(filters)

        num_filters = min(num_filters, filters.shape[0])
        grid_size = int(np.ceil(np.sqrt(num_filters)))

        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        axes = axes.flatten()

        for i in range(num_filters):
            # Take first input channel if multiple
            filter_img = filters[i, 0, :, :]

            axes[i].imshow(filter_img, cmap='viridis')
            axes[i].set_title(f'Filter {i}')
            axes[i].axis('off')

        # Hide unused subplots
        for i in range(num_filters, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_activation_heatmap(activations: np.ndarray,
                               labels: Optional[List[str]] = None,
                               save_path: Optional[Path] = None):
        """
        Plot heatmap of activations.

        Args:
            activations: [samples, features]
            labels: Sample labels
            save_path: Where to save plot
        """
        plt.figure(figsize=(12, 8))

        sns.heatmap(activations.T, cmap='viridis',
                   xticklabels=labels if labels else False,
                   yticklabels=False,
                   cbar_kws={'label': 'Activation'})

        plt.xlabel('Samples')
        plt.ylabel('Features')
        plt.title('Activation Patterns')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_entropy_evolution(entropies: List[float],
                              steps: List[int],
                              jump_steps: List[int] = None,
                              save_path: Optional[Path] = None):
        """
        Plot evolution of attention entropy over training.

        Args:
            entropies: List of entropy values
            steps: Training steps
            jump_steps: Steps where jumps occurred (optional)
            save_path: Where to save plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(steps, entropies, linewidth=2, label='Attention Entropy')

        if jump_steps:
            for step in jump_steps:
                plt.axvline(x=step, color='r', linestyle='--', alpha=0.5, label='Jump')

        plt.xlabel('Training Step')
        plt.ylabel('Entropy')
        plt.title('Attention Entropy Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


# ============================================================================
# COMPARISON FRAMEWORK
# ============================================================================

class BeforeAfterComparison:
    """Framework for comparing representations before and after critical moments."""

    def __init__(self, checkpoint_manager: CheckpointManager):
        self.checkpoint_manager = checkpoint_manager
        self.measurements = MeasurementTools()
        self.visualizations = VisualizationTools()

    def compare_moments(self, moment_before: CriticalMoment,
                       moment_after: CriticalMoment,
                       analysis_type: str = 'attention') -> AnalysisResult:
        """
        Compare representations at two moments.

        Args:
            moment_before: Earlier moment
            moment_after: Later moment
            analysis_type: Type of analysis ('attention', 'filter', 'activation')

        Returns:
            AnalysisResult with measurements and visualizations
        """
        # Load measurements
        data_before = self.checkpoint_manager.load_measurements_at_step(
            moment_before.experiment_name, moment_before.step
        )
        data_after = self.checkpoint_manager.load_measurements_at_step(
            moment_after.experiment_name, moment_after.step
        )

        if data_before is None or data_after is None:
            print("Could not load measurements")
            return None

        # Calculate differences in dimensionality
        measurements = {
            'step_before': data_before['step'],
            'step_after': data_after['step'],
            'loss_before': data_before['loss'],
            'loss_after': data_after['loss'],
            'loss_change': data_after['loss'] - data_before['loss'],
        }

        # Layer-wise dimensionality changes
        layer_changes = {}
        for layer_name in data_before['layer_metrics'].keys():
            dim_before = data_before['layer_metrics'][layer_name]['stable_rank']
            dim_after = data_after['layer_metrics'][layer_name]['stable_rank']
            layer_changes[layer_name] = {
                'before': dim_before,
                'after': dim_after,
                'change': dim_after - dim_before,
                'relative_change': (dim_after - dim_before) / (dim_before + 1e-10)
            }

        measurements['layer_changes'] = layer_changes

        return AnalysisResult(
            experiment=moment_before.experiment_name,
            moment=f"{moment_before.moment_type}_to_{moment_after.moment_type}",
            measurements=measurements,
            visualizations={}
        )


# ============================================================================
# MAIN INTERFACE
# ============================================================================

class Phase2Infrastructure:
    """
    Main interface for Phase 2 mechanistic analysis.

    Usage:
        infra = Phase2Infrastructure(experiments_dir)

        # Identify critical moments
        moments = infra.identify_moments('transformer_deep_mnist', num_jumps=5)

        # Compare before/after
        result = infra.compare_representations(moments[0], moments[1])

        # Analyze specific aspects
        entropy_change = infra.measure_attention_change(moments[0], moments[1])
    """

    def __init__(self, experiments_dir: Path):
        self.checkpoint_manager = CheckpointManager(experiments_dir)
        self.measurements = MeasurementTools()
        self.visualizations = VisualizationTools()
        self.comparisons = BeforeAfterComparison(self.checkpoint_manager)

        # Load Phase 1 results
        self.phase1_dir = experiments_dir / 'mechanistic_interpretability' / 'results'
        self.jump_data = self._load_jump_data()
        self.critical_periods = self._load_critical_periods()

    def _load_jump_data(self) -> List[Dict]:
        """Load jump characterization from Phase 1."""
        import pandas as pd
        jump_file = self.phase1_dir / 'step1_2' / 'all_jumps_detailed.csv'

        if jump_file.exists():
            df = pd.read_csv(jump_file)
            return df.to_dict('records')
        else:
            print(f"Jump data not found at {jump_file}")
            return []

    def _load_critical_periods(self) -> pd.DataFrame:
        """Load critical period data from Phase 1."""
        import pandas as pd
        periods_file = self.phase1_dir / 'step1_3' / 'critical_periods_detailed.csv'

        if periods_file.exists():
            return pd.read_csv(periods_file)
        else:
            print(f"Critical periods not found at {periods_file}")
            return pd.DataFrame()

    def identify_moments(self, experiment_name: str,
                        num_jumps: int = 5) -> List[CriticalMoment]:
        """
        Identify critical moments for investigation.

        Args:
            experiment_name: Name of experiment
            num_jumps: Number of top jumps to analyze

        Returns:
            List of CriticalMoment objects
        """
        return self.checkpoint_manager.identify_critical_moments(
            experiment_name, self.jump_data, num_jumps
        )

    def compare_representations(self, moment_before: CriticalMoment,
                               moment_after: CriticalMoment) -> AnalysisResult:
        """
        Compare representations at two moments.

        Args:
            moment_before: Earlier moment
            moment_after: Later moment

        Returns:
            AnalysisResult with findings
        """
        return self.comparisons.compare_moments(moment_before, moment_after)

    def get_experiment_summary(self, experiment_name: str) -> Dict:
        """
        Get summary of an experiment from Phase 1 results.

        Args:
            experiment_name: Name of experiment

        Returns:
            Dictionary with experiment summary
        """
        # Jump statistics
        exp_jumps = [j for j in self.jump_data if j.get('experiment') == experiment_name]

        # Critical period statistics
        if not self.critical_periods.empty:
            exp_periods = self.critical_periods[
                self.critical_periods['experiment'] == experiment_name
            ]
            period_stats = exp_periods.to_dict('records')[0] if len(exp_periods) > 0 else {}
        else:
            period_stats = {}

        return {
            'experiment': experiment_name,
            'num_jumps': len(exp_jumps),
            'jump_summary': {
                'mean_magnitude': np.mean([j['magnitude'] for j in exp_jumps]) if exp_jumps else 0,
                'mean_speed': np.mean([j['speed'] for j in exp_jumps]) if exp_jumps else 0,
                'phases': [j['phase_category'] for j in exp_jumps]
            },
            'critical_periods': period_stats
        }


if __name__ == '__main__':
    # Example usage
    experiments_dir = Path('/home/user/ndt/experiments')

    print("Initializing Phase 2 Infrastructure...")
    infra = Phase2Infrastructure(experiments_dir)

    print("\nTesting with transformer_deep_mnist...")
    summary = infra.get_experiment_summary('transformer_deep_mnist')
    print(f"Experiment Summary:")
    print(f"  Jumps: {summary['num_jumps']}")
    print(f"  Mean magnitude: {summary['jump_summary']['mean_magnitude']:.6f}")

    print("\nIdentifying critical moments...")
    moments = infra.identify_moments('transformer_deep_mnist', num_jumps=3)
    print(f"Identified {len(moments)} critical moments")

    if len(moments) >= 2:
        print("\nComparing first two moments...")
        result = infra.compare_representations(moments[0], moments[1])
        if result:
            print(f"Loss change: {result.measurements['loss_change']:.4f}")
            print(f"Layers analyzed: {len(result.measurements['layer_changes'])}")

    print("\n✓ Infrastructure initialized and tested successfully!")
