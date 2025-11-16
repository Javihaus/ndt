"""
Master Experiment Runner

This script orchestrates all 4 phases of the TAP experiments:

Phase 1: Calibration - Establish α = f(architecture) relationship
Phase 2: Prediction - Test if we can predict D(t) before training
Phase 3: Real-time Monitoring - Practical tool for training diagnosis
Phase 4: Capability Emergence - Test dimensionality-capability correlation

Usage:
    # Run all phases
    python run_all_experiments.py --all

    # Run specific phase
    python run_all_experiments.py --phase 1

    # Quick test (small scale)
    python run_all_experiments.py --quick-test

    # Custom configuration
    python run_all_experiments.py --phase 1 --num-steps 3000
"""

import argparse
import sys
from pathlib import Path
import subprocess
import json
import time
from typing import Dict, List

# Add current directory to path
sys.path.append(str(Path(__file__).parent))


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

class ExperimentRunner:
    """Orchestrates all TAP experiments."""

    def __init__(self, base_dir: str = './experiments/new/results'):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.phase_scripts = {
            1: 'phase1_calibration.py',
            2: 'phase2_prediction.py',
            3: 'phase3_realtime_monitor.py',
            4: 'phase4_capability_emergence.py'
        }

        self.phase_analysis = {
            1: 'phase1_analysis.py'
        }

    def run_phase1(self, num_steps: int = 5000, quick_test: bool = False):
        """
        Phase 1: Calibration
        Train multiple architectures on multiple datasets to establish α relationship.
        """
        print("\n" + "="*80)
        print("PHASE 1: CALIBRATION")
        print("="*80)
        print("Goal: Establish α = f(depth, width, connectivity)")
        print("="*80 + "\n")

        args = [
            sys.executable,
            str(Path(__file__).parent / self.phase_scripts[1]),
            '--output-dir', str(self.base_dir / 'phase1'),
            '--num-steps', str(num_steps),
            '--measurement-interval', '5'
        ]

        if quick_test:
            args.append('--quick-test')

        start_time = time.time()
        result = subprocess.run(args, capture_output=False)
        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"\n✓ Phase 1 completed successfully in {elapsed/60:.1f} minutes")

            # Run analysis
            print("\nRunning Phase 1 analysis...")
            analysis_args = [
                sys.executable,
                str(Path(__file__).parent / self.phase_analysis[1]),
                '--results-dir', str(self.base_dir / 'phase1'),
                '--output-dir', str(self.base_dir / 'phase1_analysis')
            ]

            analysis_result = subprocess.run(analysis_args, capture_output=False)

            if analysis_result.returncode == 0:
                print("✓ Phase 1 analysis completed")
            else:
                print("✗ Phase 1 analysis failed")
                return False
        else:
            print(f"✗ Phase 1 failed with return code {result.returncode}")
            return False

        return True

    def run_phase2(self, num_steps: int = 5000):
        """
        Phase 2: Prediction
        Test if we can predict dimensionality curves from architecture parameters.
        """
        print("\n" + "="*80)
        print("PHASE 2: PREDICTION")
        print("="*80)
        print("Goal: Predict D(t) from architecture before training (R² > 0.8)")
        print("="*80 + "\n")

        # Check if Phase 1 analysis exists
        predictor_path = self.base_dir / 'phase1_analysis' / 'alpha_models.pkl'
        if not predictor_path.exists():
            print(f"✗ Phase 1 analysis not found at {predictor_path}")
            print("  Please run Phase 1 first.")
            return False

        args = [
            sys.executable,
            str(Path(__file__).parent / self.phase_scripts[2]),
            '--predictor', str(predictor_path),
            '--output-dir', str(self.base_dir / 'phase2'),
            '--num-steps', str(num_steps)
        ]

        start_time = time.time()
        result = subprocess.run(args, capture_output=False)
        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"\n✓ Phase 2 completed successfully in {elapsed/60:.1f} minutes")
        else:
            print(f"✗ Phase 2 failed with return code {result.returncode}")
            return False

        return True

    def run_phase3(self, num_steps: int = 2000):
        """
        Phase 3: Real-time Monitoring
        Demonstrate the practical monitoring tool.
        """
        print("\n" + "="*80)
        print("PHASE 3: REAL-TIME MONITORING")
        print("="*80)
        print("Goal: Build practical tool for training diagnosis")
        print("="*80 + "\n")

        # Check if predictor exists
        predictor_path = self.base_dir / 'phase1_analysis' / 'alpha_models.pkl'
        if not predictor_path.exists():
            print(f"✗ Phase 1 analysis not found at {predictor_path}")
            print("  Please run Phase 1 first.")
            return False

        args = [
            sys.executable,
            str(Path(__file__).parent / self.phase_scripts[3]),
            '--predictor', str(predictor_path),
            '--output-dir', str(self.base_dir / 'phase3'),
            '--num-steps', str(num_steps),
            '--arch', 'mlp_medium_5'
        ]

        start_time = time.time()
        result = subprocess.run(args, capture_output=False)
        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"\n✓ Phase 3 completed successfully in {elapsed/60:.1f} minutes")
        else:
            print(f"✗ Phase 3 failed with return code {result.returncode}")
            return False

        return True

    def run_phase4(self, num_steps: int = 3000, quick_test: bool = False):
        """
        Phase 4: Capability Emergence
        Test if dimensionality jumps predict capability emergence.
        """
        print("\n" + "="*80)
        print("PHASE 4: CAPABILITY EMERGENCE")
        print("="*80)
        print("Goal: Test if dimensionality jumps predict performance jumps")
        print("="*80 + "\n")

        args = [
            sys.executable,
            str(Path(__file__).parent / self.phase_scripts[4]),
            '--output-dir', str(self.base_dir / 'phase4'),
            '--num-steps', str(num_steps),
            '--measurement-interval', '5'
        ]

        if quick_test:
            args.append('--quick-test')

        start_time = time.time()
        result = subprocess.run(args, capture_output=False)
        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"\n✓ Phase 4 completed successfully in {elapsed/60:.1f} minutes")
        else:
            print(f"✗ Phase 4 failed with return code {result.returncode}")
            return False

        return True

    def run_all(self, num_steps: int = 5000, quick_test: bool = False):
        """Run all 4 phases in sequence."""

        print("\n" + "="*80)
        print("TAP DYNAMICS EXPERIMENTS - ALL PHASES")
        print("="*80)
        print("This will run all 4 experimental phases:")
        print("  1. Calibration (establish α relationship)")
        print("  2. Prediction (test predictive power)")
        print("  3. Real-time Monitoring (practical tool)")
        print("  4. Capability Emergence (dimensionality-performance link)")
        print("="*80)

        if quick_test:
            print("\nQUICK TEST MODE: Using reduced datasets and shorter training")
            num_steps = 500

        total_start = time.time()

        # Phase 1
        phase1_success = self.run_phase1(num_steps, quick_test)
        if not phase1_success:
            print("\n✗ Aborting: Phase 1 failed")
            return False

        # Phase 2
        phase2_success = self.run_phase2(num_steps)
        if not phase2_success:
            print("\n⚠ Warning: Phase 2 failed, continuing...")

        # Phase 3
        phase3_success = self.run_phase3(min(num_steps, 2000))
        if not phase3_success:
            print("\n⚠ Warning: Phase 3 failed, continuing...")

        # Phase 4
        phase4_success = self.run_phase4(num_steps, quick_test)
        if not phase4_success:
            print("\n⚠ Warning: Phase 4 failed")

        total_elapsed = time.time() - total_start

        # Summary
        print("\n" + "="*80)
        print("ALL PHASES COMPLETE")
        print("="*80)
        print(f"Total time: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.2f} hours)")
        print(f"\nResults location: {self.base_dir}")
        print("\nPhase Status:")
        print(f"  Phase 1 (Calibration): {'✓ Success' if phase1_success else '✗ Failed'}")
        print(f"  Phase 2 (Prediction): {'✓ Success' if phase2_success else '✗ Failed'}")
        print(f"  Phase 3 (Monitoring): {'✓ Success' if phase3_success else '✗ Failed'}")
        print(f"  Phase 4 (Capability): {'✓ Success' if phase4_success else '✗ Failed'}")
        print("="*80)

        # Create final summary
        self.create_final_summary()

        return True

    def create_final_summary(self):
        """Create comprehensive summary of all experiments."""

        summary_file = self.base_dir / 'EXPERIMENTS_SUMMARY.md'

        summary = []
        summary.append("# TAP Dynamics Experiments - Complete Summary\n\n")
        summary.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        summary.append("## Overview\n\n")
        summary.append("This repository contains the complete experimental validation of the ")
        summary.append("Transition-Adapted Plasticity (TAP) framework for neural network training dynamics.\n\n")

        summary.append("## Experimental Phases\n\n")

        # Phase 1
        summary.append("### Phase 1: Calibration\n")
        summary.append("**Goal:** Establish α = f(depth, width, connectivity)\n\n")

        phase1_report = self.base_dir / 'phase1_analysis' / 'phase1_report.md'
        if phase1_report.exists():
            summary.append("✓ Completed\n")
            summary.append(f"- Results: `{phase1_report.relative_to(self.base_dir)}`\n\n")
        else:
            summary.append("⚠ Not completed\n\n")

        # Phase 2
        summary.append("### Phase 2: Prediction\n")
        summary.append("**Goal:** Test if α can predict dimensionality curves (R² > 0.8)\n\n")

        phase2_summary = self.base_dir / 'phase2' / 'phase2_summary.json'
        if phase2_summary.exists():
            summary.append("✓ Completed\n")
            with open(phase2_summary) as f:
                data = json.load(f)
                successes = sum(1 for r in data if r.get('success', False))
                summary.append(f"- Successful predictions: {successes}/{len(data)}\n\n")
        else:
            summary.append("⚠ Not completed\n\n")

        # Phase 3
        summary.append("### Phase 3: Real-Time Monitoring\n")
        summary.append("**Goal:** Build practical tool for training diagnosis\n\n")

        phase3_monitor = self.base_dir / 'phase3' / 'monitor_state.json'
        if phase3_monitor.exists():
            summary.append("✓ Completed\n")
            summary.append("- Demonstration saved\n\n")
        else:
            summary.append("⚠ Not completed\n\n")

        # Phase 4
        summary.append("### Phase 4: Capability Emergence\n")
        summary.append("**Goal:** Test if dimensionality jumps predict performance jumps\n\n")

        phase4_summary = self.base_dir / 'phase4' / 'phase4_summary.md'
        if phase4_summary.exists():
            summary.append("✓ Completed\n")
            summary.append(f"- Results: `{phase4_summary.relative_to(self.base_dir)}`\n\n")
        else:
            summary.append("⚠ Not completed\n\n")

        summary.append("## Key Files\n\n")
        summary.append("```\n")
        summary.append("results/\n")
        summary.append("├── phase1/                    # Raw calibration data\n")
        summary.append("├── phase1_analysis/           # α parameter extraction\n")
        summary.append("│   ├── alpha_summary.csv      # Architecture → α mapping\n")
        summary.append("│   ├── alpha_models.pkl       # Fitted predictive models\n")
        summary.append("│   └── phase1_report.md       # Analysis summary\n")
        summary.append("├── phase2/                    # Prediction experiments\n")
        summary.append("│   └── phase2_summary.json    # Prediction accuracy\n")
        summary.append("├── phase3/                    # Monitoring demonstration\n")
        summary.append("│   └── monitor_state.json     # Monitor state\n")
        summary.append("└── phase4/                    # Capability emergence\n")
        summary.append("    └── phase4_summary.md      # Correlation analysis\n")
        summary.append("```\n\n")

        summary.append("## Next Steps\n\n")
        summary.append("1. Review individual phase reports\n")
        summary.append("2. Examine visualizations in each phase directory\n")
        summary.append("3. Check alpha_summary.csv for architecture-α relationships\n")
        summary.append("4. Validate predictions in phase2_summary.json\n")
        summary.append("5. Analyze capability emergence findings in phase4_summary.md\n")

        with open(summary_file, 'w') as f:
            f.writelines(summary)

        print(f"\nFinal summary saved: {summary_file}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='TAP Dynamics Experiments - Master Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all phases (full experiments)
  python run_all_experiments.py --all

  # Quick test (small scale)
  python run_all_experiments.py --all --quick-test

  # Run specific phase
  python run_all_experiments.py --phase 1
  python run_all_experiments.py --phase 2 --num-steps 3000

  # Custom output directory
  python run_all_experiments.py --all --output-dir ./my_results
        """
    )

    parser.add_argument('--all', action='store_true',
                       help='Run all 4 phases')
    parser.add_argument('--phase', type=int, choices=[1, 2, 3, 4],
                       help='Run specific phase (1-4)')
    parser.add_argument('--num-steps', type=int, default=5000,
                       help='Training steps per experiment (default: 5000)')
    parser.add_argument('--output-dir', type=str,
                       default='./experiments/new/results',
                       help='Base output directory')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with reduced scale')

    args = parser.parse_args()

    # Create runner
    runner = ExperimentRunner(args.output_dir)

    if args.all:
        # Run all phases
        runner.run_all(args.num_steps, args.quick_test)

    elif args.phase:
        # Run specific phase
        if args.phase == 1:
            runner.run_phase1(args.num_steps, args.quick_test)
        elif args.phase == 2:
            runner.run_phase2(args.num_steps)
        elif args.phase == 3:
            runner.run_phase3(min(args.num_steps, 2000))
        elif args.phase == 4:
            runner.run_phase4(args.num_steps, args.quick_test)

    else:
        parser.print_help()
        print("\nPlease specify --all or --phase <1-4>")


if __name__ == "__main__":
    main()
