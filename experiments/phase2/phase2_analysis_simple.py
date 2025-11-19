#!/usr/bin/env python3
"""Phase 2: Simplified Mechanistic Analysis using Phase 1 Data

Analyzes the three target experiments using only Phase 1 JSON data.
No external dependencies required (uses standard library only).

Target experiments:
1. CNN Jump Cascades (cnn_deep/fashion_mnist - 23 jumps)
2. Transformer Transitions (transformer_shallow/mnist - 9 jumps)
3. MLP Smooth Learning (mlp_narrow/mnist - R²=0.934)
"""

import json
import os
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple


class Phase2Analyzer:
    """Analyzes Phase 1 results for mechanistic insights."""

    def __init__(self, phase1_dir: str, results_dir: str):
        self.phase1_dir = Path(phase1_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.targets = [
            ("cnn_deep", "fashion_mnist"),      # 23 jumps
            ("transformer_shallow", "mnist"),   # 9 jumps
            ("mlp_narrow", "mnist")             # 0 jumps, high R²
        ]

    def load_experiment(self, arch: str, dataset: str) -> Dict[str, Any]:
        """Load Phase 1 results for an experiment."""
        path = self.phase1_dir / f"analysis_{arch}_{dataset}.json"
        with open(path) as f:
            return json.load(f)

    def analyze_jumps(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze jump patterns and characteristics."""
        jumps = data.get('jumps', [])

        if not jumps:
            return {
                'num_jumps': 0,
                'pattern': 'smooth',
                'characteristics': 'No dimensionality transitions detected'
            }

        # Sort by index to see temporal pattern
        sorted_jumps = sorted(jumps, key=lambda x: x['index'])

        # Analyze jump distribution
        indices = [j['index'] for j in sorted_jumps]
        magnitudes = [j['magnitude'] for j in sorted_jumps]
        z_scores = [j['z_score'] for j in sorted_jumps]

        # Early vs late jumps
        mid_point = max(indices) / 2 if indices else 0
        early_jumps = [j for j in sorted_jumps if j['index'] < mid_point]
        late_jumps = [j for j in sorted_jumps if j['index'] >= mid_point]

        # Jump clustering (are jumps close together?)
        if len(indices) > 1:
            gaps = [indices[i+1] - indices[i] for i in range(len(indices)-1)]
            avg_gap = sum(gaps) / len(gaps)
            std_gap = math.sqrt(sum((g - avg_gap)**2 for g in gaps) / len(gaps)) if len(gaps) > 0 else 0
        else:
            avg_gap = 0
            std_gap = 0

        # Characterize pattern
        if len(early_jumps) > len(late_jumps) * 2:
            pattern = 'early_cascade'
            description = 'Concentrated burst of jumps in early training'
        elif len(late_jumps) > len(early_jumps) * 2:
            pattern = 'late_emergence'
            description = 'Features emerge primarily in late training'
        elif std_gap < avg_gap * 0.5:
            pattern = 'periodic'
            description = 'Regularly spaced jumps throughout training'
        else:
            pattern = 'scattered'
            description = 'Irregular distribution of jumps'

        return {
            'num_jumps': len(jumps),
            'pattern': pattern,
            'characteristics': description,
            'early_jumps': len(early_jumps),
            'late_jumps': len(late_jumps),
            'avg_magnitude': sum(magnitudes) / len(magnitudes),
            'max_magnitude': max(magnitudes),
            'avg_z_score': sum(z_scores) / len(z_scores),
            'avg_gap': avg_gap,
            'top_5_jumps': sorted(jumps, key=lambda x: x['magnitude'], reverse=True)[:5]
        }

    def analyze_dimensionality_growth(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how dimensionality grows during training."""
        tap = data['tap_parameters']

        alpha = tap['alpha']
        D_max = tap['D_max']
        R2 = tap['R2']
        t0 = tap['t0']

        # Growth rate characterization
        if alpha > 0.001:
            growth_type = 'fast'
            growth_desc = 'Rapid dimensionality expansion'
        elif alpha > 0.0005:
            growth_type = 'moderate'
            growth_desc = 'Moderate growth rate'
        elif alpha > 0.0001:
            growth_type = 'slow'
            growth_desc = 'Gradual dimensionality increase'
        else:
            growth_type = 'very_slow'
            growth_desc = 'Very slow expansion, possible bottleneck'

        # Fit quality
        if R2 > 0.9:
            fit_quality = 'excellent'
            fit_desc = 'TAP model fits extremely well - smooth, predictable growth'
        elif R2 > 0.7:
            fit_quality = 'good'
            fit_desc = 'TAP model fits reasonably well'
        elif R2 > 0.3:
            fit_quality = 'moderate'
            fit_desc = 'Some deviation from TAP predictions'
        elif R2 > 0:
            fit_quality = 'poor'
            fit_desc = 'Significant deviation from TAP model - likely jumps/transitions'
        else:
            fit_quality = 'negative'
            fit_desc = 'TAP model does not fit - complex non-monotonic dynamics'

        return {
            'alpha': alpha,
            'D_max': D_max,
            'R2': R2,
            't0': t0,
            'growth_type': growth_type,
            'growth_description': growth_desc,
            'fit_quality': fit_quality,
            'fit_description': fit_desc
        }

    def mechanistic_interpretation(
        self,
        arch: str,
        dataset: str,
        jump_analysis: Dict[str, Any],
        growth_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate mechanistic interpretation of the learning dynamics."""

        interpretations = []
        hypotheses = []

        # CNN-specific interpretations
        if 'cnn' in arch:
            if jump_analysis['num_jumps'] > 10:
                interpretations.append(
                    "The high number of jumps suggests hierarchical feature learning "
                    "where different levels of abstraction emerge at distinct training phases."
                )
                hypotheses.append(
                    "Each jump may correspond to the emergence of a new feature type "
                    "(edges → textures → parts → objects)"
                )

            if jump_analysis.get('pattern') == 'early_cascade':
                interpretations.append(
                    "Early cascade pattern indicates rapid initial learning of "
                    "low-level features followed by stabilization."
                )
                hypotheses.append(
                    "Early jumps likely correspond to edge detector formation, "
                    "while later jumps represent compositional features."
                )

        # Transformer-specific interpretations
        elif 'transformer' in arch:
            if jump_analysis['num_jumps'] > 0:
                interpretations.append(
                    "Transformers show discrete attention pattern transitions "
                    "rather than gradual feature refinement."
                )
                hypotheses.append(
                    "Jumps may correspond to attention heads specializing "
                    "for different aspects of the input."
                )

        # MLP-specific interpretations
        elif 'mlp' in arch:
            if jump_analysis['num_jumps'] == 0 and growth_analysis['R2'] > 0.8:
                interpretations.append(
                    "Smooth, predictable growth suggests MLPs learn features "
                    "gradually without sharp phase transitions."
                )
                hypotheses.append(
                    "MLPs may rely on distributed representations that "
                    "incrementally refine rather than discrete feature emergence."
                )

        # General interpretations based on R²
        if growth_analysis['R2'] < 0:
            interpretations.append(
                "Negative R² indicates complex, non-monotonic learning dynamics "
                "that cannot be captured by simple growth models."
            )

        return {
            'interpretations': interpretations,
            'hypotheses': hypotheses,
            'suggested_investigations': [
                f"Feature visualization at jump indices: {[j['index'] for j in jump_analysis.get('top_5_jumps', [])]}",
                "Compare activation patterns before/after largest jumps",
                "Analyze which neurons become active at each transition"
            ]
        }

    def generate_report(self, arch: str, dataset: str, results: Dict[str, Any]) -> str:
        """Generate markdown report for an experiment."""
        report = []
        exp_name = f"{arch}/{dataset}"

        report.append(f"# Phase 2 Mechanistic Analysis: {exp_name}")
        report.append(f"\nGenerated: {datetime.now().isoformat()}")
        report.append("")

        # Executive Summary
        report.append("## Executive Summary")
        report.append("")
        jump = results['jump_analysis']
        growth = results['growth_analysis']

        if jump['num_jumps'] == 0:
            report.append(f"**Pattern**: Smooth learning with {growth['fit_quality']} TAP fit (R²={growth['R2']:.3f})")
        else:
            report.append(f"**Pattern**: {jump['pattern'].replace('_', ' ').title()} with {jump['num_jumps']} dimensionality transitions")

        report.append(f"\n**Growth Rate**: {growth['growth_type']} (α={growth['alpha']:.6f})")
        report.append(f"\n**Max Dimensionality**: {growth['D_max']:.2f}")
        report.append("")

        # Jump Analysis
        report.append("## Dimensionality Transitions (Jumps)")
        report.append("")

        if jump['num_jumps'] > 0:
            report.append(f"- **Total jumps**: {jump['num_jumps']}")
            report.append(f"- **Pattern**: {jump['characteristics']}")
            report.append(f"- **Early training jumps**: {jump['early_jumps']}")
            report.append(f"- **Late training jumps**: {jump['late_jumps']}")
            report.append(f"- **Average magnitude**: {jump['avg_magnitude']:.4f}")
            report.append(f"- **Maximum magnitude**: {jump['max_magnitude']:.4f}")
            report.append(f"- **Average z-score**: {jump['avg_z_score']:.2f}")
            report.append("")

            report.append("### Top 5 Jumps by Magnitude")
            report.append("")
            report.append("| Rank | Index | Magnitude | Z-Score |")
            report.append("|------|-------|-----------|---------|")
            for i, j in enumerate(jump['top_5_jumps'], 1):
                report.append(f"| {i} | {j['index']} | {j['magnitude']:.4f} | {j['z_score']:.2f} |")
        else:
            report.append("No dimensionality transitions detected. Learning follows smooth TAP dynamics.")

        report.append("")

        # Growth Analysis
        report.append("## Dimensionality Growth Dynamics")
        report.append("")
        report.append(f"- **Growth rate (α)**: {growth['alpha']:.6f} - {growth['growth_description']}")
        report.append(f"- **Maximum dimensionality (D_max)**: {growth['D_max']:.2f}")
        report.append(f"- **TAP fit quality (R²)**: {growth['R2']:.4f} - {growth['fit_description']}")
        report.append(f"- **Onset time (t0)**: {growth['t0']:.2e}")
        report.append("")

        # Mechanistic Interpretation
        report.append("## Mechanistic Interpretation")
        report.append("")
        mech = results['mechanistic_interpretation']

        report.append("### Key Findings")
        report.append("")
        for interp in mech['interpretations']:
            report.append(f"- {interp}")
        report.append("")

        report.append("### Hypotheses for Investigation")
        report.append("")
        for hyp in mech['hypotheses']:
            report.append(f"- {hyp}")
        report.append("")

        report.append("### Suggested Next Steps")
        report.append("")
        for inv in mech['suggested_investigations']:
            report.append(f"1. {inv}")
        report.append("")

        # Architecture context
        report.append("## Architecture Context")
        report.append("")
        arch_params = results['raw_data']['architecture_params']
        report.append(f"- **Depth**: {arch_params['depth']} layers")
        report.append(f"- **Width**: {arch_params['width']}")
        report.append(f"- **Parameters**: {arch_params['num_params']:,}")
        report.append(f"- **Connectivity**: {arch_params['connectivity']:.2f}")

        if results['raw_data'].get('final_accuracy'):
            report.append(f"- **Final Accuracy**: {results['raw_data']['final_accuracy']:.4f}")
        report.append("")

        return "\n".join(report)

    def run(self) -> Dict[str, Any]:
        """Run analysis on all target experiments."""
        all_results = {}

        print("="*60)
        print("PHASE 2: MECHANISTIC DEEP DIVE")
        print("="*60)

        for arch, dataset in self.targets:
            exp_name = f"{arch}_{dataset}"
            print(f"\nAnalyzing: {arch}/{dataset}")

            try:
                # Load data
                data = self.load_experiment(arch, dataset)

                # Run analyses
                jump_analysis = self.analyze_jumps(data)
                growth_analysis = self.analyze_dimensionality_growth(data)
                mech_interp = self.mechanistic_interpretation(
                    arch, dataset, jump_analysis, growth_analysis
                )

                results = {
                    'arch': arch,
                    'dataset': dataset,
                    'jump_analysis': jump_analysis,
                    'growth_analysis': growth_analysis,
                    'mechanistic_interpretation': mech_interp,
                    'raw_data': data
                }

                # Generate report
                report = self.generate_report(arch, dataset, results)
                report_path = self.results_dir / f"report_{exp_name}.md"
                with open(report_path, 'w') as f:
                    f.write(report)
                print(f"  Report: {report_path}")

                # Save JSON results (exclude raw_data to keep it small)
                json_results = {k: v for k, v in results.items() if k != 'raw_data'}
                json_path = self.results_dir / f"analysis_{exp_name}.json"
                with open(json_path, 'w') as f:
                    json.dump(json_results, f, indent=2)

                all_results[exp_name] = results

                # Print summary
                print(f"  Jumps: {jump_analysis['num_jumps']}, R²: {growth_analysis['R2']:.3f}")
                print(f"  Pattern: {jump_analysis['pattern']}")

            except Exception as e:
                print(f"  ERROR: {e}")
                all_results[exp_name] = {'error': str(e)}

        # Generate summary report
        self._generate_summary(all_results)

        return all_results

    def _generate_summary(self, all_results: Dict[str, Any]) -> None:
        """Generate summary comparing all experiments."""
        summary = []
        summary.append("# Phase 2 Summary: Comparative Analysis")
        summary.append(f"\nGenerated: {datetime.now().isoformat()}")
        summary.append("")

        summary.append("## Overview")
        summary.append("")
        summary.append("| Experiment | Jumps | Pattern | R² | α |")
        summary.append("|------------|-------|---------|-----|---|")

        for exp_name, results in all_results.items():
            if 'error' not in results:
                j = results['jump_analysis']
                g = results['growth_analysis']
                summary.append(f"| {exp_name} | {j['num_jumps']} | {j['pattern']} | {g['R2']:.3f} | {g['alpha']:.2e} |")

        summary.append("")

        # Key findings
        summary.append("## Key Findings")
        summary.append("")
        summary.append("### 1. Architecture-Specific Learning Patterns")
        summary.append("")
        summary.append("- **CNNs**: Show cascading dimensionality jumps, suggesting hierarchical feature emergence")
        summary.append("- **Transformers**: Exhibit discrete transitions, possibly attention head specialization")
        summary.append("- **MLPs**: Demonstrate smooth, predictable growth following TAP dynamics")
        summary.append("")

        summary.append("### 2. Implications for Mechanistic Interpretability")
        summary.append("")
        summary.append("- **Jump detection as investigation guide**: Dimensionality transitions identify when to apply feature visualization")
        summary.append("- **Architecture-dependent analysis**: Different architectures require different interpretability approaches")
        summary.append("- **Training phase awareness**: Early vs late jumps suggest different types of feature learning")
        summary.append("")

        summary.append("## Next Steps for Phase 3")
        summary.append("")
        summary.append("1. Deep dive feature visualization on top jump moments")
        summary.append("2. Activation pattern comparison before/after transitions")
        summary.append("3. Ablation studies at critical training phases")
        summary.append("4. Write up findings for publication")
        summary.append("")

        # Save summary
        summary_path = self.results_dir / "PHASE2_SUMMARY.md"
        with open(summary_path, 'w') as f:
            f.write("\n".join(summary))
        print(f"\nSummary saved: {summary_path}")


def main():
    # Get paths relative to this script
    script_dir = Path(__file__).parent
    phase1_dir = script_dir.parent / "new" / "results" / "phase1_analysis"
    results_dir = script_dir / "results"

    analyzer = Phase2Analyzer(str(phase1_dir), str(results_dir))
    results = analyzer.run()

    print("\n" + "="*60)
    print("PHASE 2 COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {results_dir}")


if __name__ == "__main__":
    main()
