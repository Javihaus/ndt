"""
Sequential Experiment Runner (Memory-Constrained)
==================================================

Runs experiments one at a time with memory cleanup between each run.
Designed for memory-constrained environments.
"""

import os
import json
import time
import gc
import torch
from phase1_calibration import run_single_experiment
import argparse


# Full Architecture-Dataset Mapping
# Original plan: 17 architectures × 5 datasets = 85 experiments
# Current: 17 architectures × 4 datasets = 68 experiments
# Modalities: Vision (MNIST, Fashion-MNIST, CIFAR-10) + Text (AG News)

# All 17 architectures from phase1_calibration.py
ARCHITECTURES = [
    # Depth variation (MLPs)
    'mlp_shallow_2', 'mlp_medium_5', 'mlp_deep_10', 'mlp_verydeep_15',
    # Width variation (MLPs)
    'mlp_narrow', 'mlp_medium', 'mlp_wide', 'mlp_verywide',
    # CNNs
    'cnn_shallow', 'cnn_medium', 'cnn_deep',
    # ResNet
    'resnet18',
    # Transformers
    'transformer_shallow', 'transformer_medium', 'transformer_deep',
    'transformer_narrow', 'transformer_wide',
]

# Available datasets: 3 vision + 1 text (multimodal validation of TAP framework)
DATASETS = ['mnist', 'fashion_mnist', 'cifar10', 'ag_news']

# Generate all 68 combinations
EXPERIMENT_PLAN = [(arch, dataset) for arch in ARCHITECTURES for dataset in DATASETS]


def run_sequential_experiments(num_steps=2000, output_dir='./results/phase1_full'):
    """
    Run experiments sequentially with memory cleanup.

    Args:
        num_steps: Steps per experiment (reduced from 5000 for speed)
        output_dir: Output directory
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("=" * 70)
    print("SEQUENTIAL EXPERIMENT RUNNER (Memory-Constrained)")
    print("=" * 70)
    print("Total experiments: {}".format(len(EXPERIMENT_PLAN)))
    print("Steps per experiment: {}".format(num_steps))
    print("Output directory: {}".format(output_dir))
    print("=" * 70)
    print()

    results_summary = []
    completed = 0
    skipped = 0
    failed = 0

    start_time = time.time()

    for i, (arch_name, dataset_name) in enumerate(EXPERIMENT_PLAN, 1):
        output_file = os.path.join(output_dir, '{}_{}.json'.format(arch_name, dataset_name))

        # Skip if already exists
        if os.path.exists(output_file):
            print("[{}/{}] [SKIP] {} x {} (already exists)".format(
                i, len(EXPERIMENT_PLAN), arch_name, dataset_name))
            skipped += 1
            continue

        try:
            print("\n[{}/{}] Running {} on {}...".format(i, len(EXPERIMENT_PLAN), arch_name, dataset_name))

            exp_start = time.time()
            result = run_single_experiment(
                arch_name=arch_name,
                dataset_name=dataset_name,
                num_steps=num_steps,
                measurement_interval=5
            )
            exp_elapsed = time.time() - exp_start

            # Save result immediately
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)

            print("[OK] Completed {} x {} in {:.1f} min".format(
                arch_name, dataset_name, exp_elapsed/60))
            print("     Accuracy: {:.2f}%".format(result['final_accuracy'] * 100))

            completed += 1
            results_summary.append({
                'arch': arch_name,
                'dataset': dataset_name,
                'time': exp_elapsed,
                'accuracy': result['final_accuracy']
            })

        except Exception as e:
            print("[FAIL] Failed {} x {}: {}".format(arch_name, dataset_name, str(e)[:100]))
            failed += 1
            continue

        finally:
            # Aggressive memory cleanup between experiments
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # Brief pause to allow system to stabilize
            time.sleep(2)

    elapsed = time.time() - start_time

    # Final summary
    print()
    print("=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print("Total time: {:.1f} minutes ({:.1f} hours)".format(elapsed/60, elapsed/3600))
    print("Completed: {}".format(completed))
    print("Skipped: {}".format(skipped))
    print("Failed: {}".format(failed))
    print("=" * 70)

    # Save summary
    if results_summary:
        summary_file = os.path.join(output_dir, 'experiment_summary.json')
        with open(summary_file, 'w') as f:
            json.dump({
                'total_time': elapsed,
                'completed': completed,
                'skipped': skipped,
                'failed': failed,
                'results': results_summary
            }, f, indent=2)
        print("\nSummary saved to: {}".format(summary_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiments sequentially')
    parser.add_argument('--num-steps', type=int, default=2000,
                        help='Number of training steps per experiment (default: 2000)')
    parser.add_argument('--output-dir', type=str, default='./results/phase1_full',
                        help='Output directory for results')

    args = parser.parse_args()

    run_sequential_experiments(
        num_steps=args.num_steps,
        output_dir=args.output_dir
    )
