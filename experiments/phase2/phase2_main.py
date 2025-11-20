"""Phase 2: Mechanistic Deep Dive - Main Orchestrator

This script performs mechanistic analysis on the three most interesting
phenomena identified in Phase 1:
1. CNN Jump Cascades (cnn_deep/fashion_mnist - 23 jumps)
2. Transformer Transitions (transformer_shallow/mnist - 9 jumps)
3. MLP Smooth Learning (mlp_narrow/mnist - R²=0.934)

The analysis includes:
- Feature visualization at critical moments
- Activation analysis (PCA, clustering)
- Comparison before/after dimensionality transitions
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ndt.analysis.activation_analysis import ActivationAnalyzer
from src.ndt.analysis.feature_visualization import FeatureVisualizer
from src.ndt.core.hooks import ActivationCapture


class Phase2Analyzer:
    """Orchestrates Phase 2 mechanistic analysis experiments.

    Attributes:
        results_dir: Directory to save results
        phase1_dir: Directory containing Phase 1 results
        device: Torch device
    """

    def __init__(
        self,
        results_dir: str = "results",
        phase1_dir: str = "../new/results/phase1_analysis"
    ) -> None:
        """Initialize the Phase 2 analyzer.

        Args:
            results_dir: Directory to save Phase 2 results
            phase1_dir: Directory containing Phase 1 results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.phase1_dir = Path(phase1_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.activation_analyzer = ActivationAnalyzer()
        self.target_experiments = [
            ("cnn_deep", "fashion_mnist"),      # 23 jumps - CNN cascade
            ("transformer_shallow", "mnist"),   # 9 jumps - Transformer transitions
            ("mlp_narrow", "mnist")             # 0 jumps, R²=0.934 - Smooth learning
        ]

    def load_phase1_results(self, arch_name: str, dataset_name: str) -> Dict[str, Any]:
        """Load Phase 1 analysis results for an experiment.

        Args:
            arch_name: Architecture name
            dataset_name: Dataset name

        Returns:
            Phase 1 results dictionary
        """
        filename = f"analysis_{arch_name}_{dataset_name}.json"
        filepath = self.phase1_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Phase 1 results not found: {filepath}")

        with open(filepath) as f:
            return json.load(f)

    def identify_critical_moments(
        self,
        phase1_results: Dict[str, Any],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Identify the most significant critical moments from jumps.

        Args:
            phase1_results: Phase 1 results dictionary
            top_k: Number of critical moments to return

        Returns:
            List of critical moment dictionaries with index and magnitude
        """
        jumps = phase1_results.get('jumps', [])

        if not jumps:
            # For experiments without jumps, sample evenly
            n_points = len(phase1_results.get('tap_parameters', {}).get('predictions', []))
            if n_points > 0:
                indices = np.linspace(0, n_points - 1, top_k, dtype=int)
                return [{'index': int(i), 'magnitude': 0, 'z_score': 0} for i in indices]
            return []

        # Sort by magnitude and return top-k
        sorted_jumps = sorted(jumps, key=lambda x: x['magnitude'], reverse=True)
        return sorted_jumps[:top_k]

    def analyze_experiment(
        self,
        arch_name: str,
        dataset_name: str,
        model: Optional[nn.Module] = None,
        dataloader: Optional[DataLoader] = None
    ) -> Dict[str, Any]:
        """Run full mechanistic analysis on an experiment.

        Args:
            arch_name: Architecture name
            dataset_name: Dataset name
            model: Optional pre-loaded model
            dataloader: Optional dataloader for the dataset

        Returns:
            Dictionary containing all analysis results
        """
        print(f"\n{'='*60}")
        print(f"Analyzing: {arch_name}/{dataset_name}")
        print(f"{'='*60}")

        # Load Phase 1 results
        phase1_results = self.load_phase1_results(arch_name, dataset_name)

        # Identify critical moments
        critical_moments = self.identify_critical_moments(phase1_results)
        print(f"Identified {len(critical_moments)} critical moments")

        # Initialize results
        results = {
            'arch_name': arch_name,
            'dataset_name': dataset_name,
            'phase1_summary': {
                'num_jumps': phase1_results.get('num_jumps', 0),
                'R2': phase1_results['tap_parameters']['R2'],
                'alpha': phase1_results['tap_parameters']['alpha'],
                'final_accuracy': phase1_results.get('final_accuracy', None)
            },
            'critical_moments': critical_moments,
            'analyses': {}
        }

        # If we have model and data, run detailed analysis
        if model is not None and dataloader is not None:
            print("Running activation analysis...")

            # Capture activations for a batch
            capture = ActivationCapture()
            layers = self._get_analysis_layers(model)
            layer_names = [f"layer_{i}" for i in range(len(layers))]
            capture.register_hooks(model, layers, layer_names)

            # Get a batch of data
            batch_x, batch_y = next(iter(dataloader))
            batch_x = batch_x.to(self.device)

            # Forward pass to capture activations
            model.eval()
            with torch.no_grad():
                _ = model(batch_x)

            # Analyze each layer
            for layer_name in layer_names:
                activation = capture.get_activation(layer_name)
                if activation is not None:
                    flat_act = self.activation_analyzer.flatten_activation(activation)

                    # PCA analysis
                    pca_results = self.activation_analyzer.pca_analysis(flat_act)

                    # Singular value analysis
                    sv_results = self.activation_analyzer.singular_value_analysis(flat_act)

                    # Neuron importance
                    importance = self.activation_analyzer.neuron_importance(flat_act)

                    # Clustering
                    if flat_act.shape[0] > 5:  # Need enough samples
                        cluster_results = self.activation_analyzer.cluster_analysis(flat_act)
                    else:
                        cluster_results = None

                    results['analyses'][layer_name] = {
                        'pca': {
                            'n_components_90': pca_results['n_components_90'],
                            'n_components_95': pca_results['n_components_95'],
                            'top_5_variance': pca_results['explained_variance_ratio'][:5].tolist()
                        },
                        'sv': {
                            'stable_rank': sv_results['stable_rank'],
                            'participation_ratio': sv_results['participation_ratio'],
                            'spectral_entropy': sv_results['spectral_entropy']
                        },
                        'importance': {
                            'n_dead': importance['n_dead'],
                            'top_10': importance['top_10_indices'].tolist()
                        },
                        'clustering': cluster_results
                    }

            capture.remove_hooks()

        return results

    def _get_analysis_layers(self, model: nn.Module) -> List[nn.Module]:
        """Get layers to analyze from a model.

        Args:
            model: The model

        Returns:
            List of layer modules
        """
        layers = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                layers.append(module)
        return layers

    def generate_report(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> str:
        """Generate a markdown report from analysis results.

        Args:
            results: Analysis results dictionary
            save_path: Optional path to save report

        Returns:
            Report as markdown string
        """
        report = []
        report.append(f"# Phase 2 Analysis Report: {results['arch_name']}/{results['dataset_name']}")
        report.append("")

        # Phase 1 summary
        report.append("## Phase 1 Summary")
        report.append("")
        p1 = results['phase1_summary']
        report.append(f"- **Number of jumps**: {p1['num_jumps']}")
        report.append(f"- **R²**: {p1['R2']:.4f}")
        report.append(f"- **α (growth rate)**: {p1['alpha']:.6f}")
        if p1['final_accuracy']:
            report.append(f"- **Final accuracy**: {p1['final_accuracy']:.4f}")
        report.append("")

        # Critical moments
        report.append("## Critical Moments Identified")
        report.append("")
        for i, cm in enumerate(results['critical_moments']):
            report.append(f"{i+1}. Index {cm['index']}: magnitude={cm['magnitude']:.4f}, z-score={cm['z_score']:.2f}")
        report.append("")

        # Layer analyses
        if results.get('analyses'):
            report.append("## Layer-wise Analysis")
            report.append("")

            for layer_name, analysis in results['analyses'].items():
                report.append(f"### {layer_name}")
                report.append("")

                # PCA
                pca = analysis['pca']
                report.append(f"**PCA Analysis:**")
                report.append(f"- Components for 90% variance: {pca['n_components_90']}")
                report.append(f"- Components for 95% variance: {pca['n_components_95']}")
                report.append(f"- Top 5 variance ratios: {[f'{v:.3f}' for v in pca['top_5_variance']]}")
                report.append("")

                # Singular values
                sv = analysis['sv']
                report.append(f"**Singular Value Analysis:**")
                report.append(f"- Stable rank: {sv['stable_rank']:.2f}")
                report.append(f"- Participation ratio: {sv['participation_ratio']:.2f}")
                report.append(f"- Spectral entropy: {sv['spectral_entropy']:.3f}")
                report.append("")

                # Importance
                imp = analysis['importance']
                report.append(f"**Neuron Importance:**")
                report.append(f"- Dead neurons: {imp['n_dead']}")
                report.append(f"- Top 10 indices: {imp['top_10']}")
                report.append("")

        report_str = "\n".join(report)

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_str)
            print(f"Report saved to: {save_path}")

        return report_str

    def run_all_analyses(self) -> Dict[str, Dict[str, Any]]:
        """Run analysis on all target experiments.

        Returns:
            Dictionary mapping experiment names to results
        """
        all_results = {}

        for arch_name, dataset_name in self.target_experiments:
            exp_name = f"{arch_name}_{dataset_name}"
            try:
                results = self.analyze_experiment(arch_name, dataset_name)
                all_results[exp_name] = results

                # Generate report
                report_path = self.results_dir / f"report_{exp_name}.md"
                self.generate_report(results, str(report_path))

                # Save JSON results
                json_path = self.results_dir / f"analysis_{exp_name}.json"
                with open(json_path, 'w') as f:
                    # Convert numpy arrays for JSON serialization
                    json_safe = self._make_json_safe(results)
                    json.dump(json_safe, f, indent=2)

            except Exception as e:
                print(f"Error analyzing {exp_name}: {e}")
                all_results[exp_name] = {'error': str(e)}

        return all_results

    def _make_json_safe(self, obj: Any) -> Any:
        """Convert numpy arrays to lists for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_safe(v) for v in obj]
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj

    def plot_jump_analysis(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> None:
        """Plot analysis of jumps and critical moments.

        Args:
            results: Analysis results
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        critical_moments = results['critical_moments']

        if critical_moments:
            # Plot 1: Jump magnitudes
            indices = [cm['index'] for cm in critical_moments]
            magnitudes = [cm['magnitude'] for cm in critical_moments]

            axes[0].bar(range(len(magnitudes)), magnitudes, color='steelblue')
            axes[0].set_xlabel('Jump Rank')
            axes[0].set_ylabel('Magnitude')
            axes[0].set_title(f"Top {len(magnitudes)} Jump Magnitudes")

            # Plot 2: Jump positions over training
            axes[1].scatter(indices, magnitudes, s=100, c='steelblue')
            axes[1].set_xlabel('Training Step Index')
            axes[1].set_ylabel('Jump Magnitude')
            axes[1].set_title('Jump Positions During Training')

        else:
            axes[0].text(0.5, 0.5, 'No jumps detected', ha='center', va='center')
            axes[1].text(0.5, 0.5, 'Smooth learning curve', ha='center', va='center')

        plt.suptitle(f"{results['arch_name']}/{results['dataset_name']}")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        plt.close()


def main():
    """Main entry point for Phase 2 analysis."""
    print("="*60)
    print("PHASE 2: MECHANISTIC DEEP DIVE")
    print("="*60)

    # Initialize analyzer
    analyzer = Phase2Analyzer(
        results_dir="results",
        phase1_dir="../new/results/phase1_analysis"
    )

    # Run all analyses
    all_results = analyzer.run_all_analyses()

    # Summary
    print("\n" + "="*60)
    print("PHASE 2 SUMMARY")
    print("="*60)

    for exp_name, results in all_results.items():
        if 'error' not in results:
            p1 = results['phase1_summary']
            print(f"\n{exp_name}:")
            print(f"  Jumps: {p1['num_jumps']}, R²: {p1['R2']:.3f}")
            print(f"  Critical moments: {len(results['critical_moments'])}")

            # Generate plot
            plot_path = analyzer.results_dir / f"jumps_{exp_name}.png"
            analyzer.plot_jump_analysis(results, str(plot_path))

    print("\n" + "="*60)
    print(f"Results saved to: {analyzer.results_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
