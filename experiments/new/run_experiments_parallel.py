"""
Parallel Experiment Runner
===========================

Runs Phase 1 experiments in parallel to save time.
Uses multiprocessing to run multiple experiments simultaneously.
"""

import multiprocessing as mp
import os
import json
import time
from phase1_calibration import run_single_experiment, DATASET_LOADERS
import argparse


# Architecture-Dataset Mapping (using available datasets)
EXPERIMENT_PLAN = [
    # MLPs work with all image datasets
    ('mlp_shallow_2', 'mnist'),
    ('mlp_shallow_2', 'cifar10'),
    ('mlp_medium_5', 'mnist'),
    ('mlp_medium_5', 'cifar10'),
    ('mlp_deep_10', 'mnist'),
    ('mlp_deep_10', 'cifar10'),
    ('mlp_verydeep_15', 'mnist'),
    ('mlp_verydeep_15', 'cifar10'),
    ('mlp_narrow', 'mnist'),
    ('mlp_narrow', 'cifar10'),
    ('mlp_medium', 'mnist'),
    ('mlp_medium', 'cifar10'),
    ('mlp_wide', 'mnist'),
    ('mlp_wide', 'cifar10'),
    ('mlp_verywide', 'mnist'),
    ('mlp_verywide', 'cifar10'),

    # CNNs work with CIFAR-10
    ('cnn_shallow', 'cifar10'),
    ('cnn_medium', 'cifar10'),
    ('cnn_deep', 'cifar10'),

    # ResNet works with CIFAR-10, ImageNet
    ('resnet18', 'cifar10'),
    ('resnet18', 'imagenet'),

    # Transformers work with text and images
    ('transformer_shallow', 'glue_mnli'),
    ('transformer_shallow', 'cifar10'),
    ('transformer_medium', 'glue_mnli'),
    ('transformer_medium', 'cifar10'),
    ('transformer_deep', 'glue_mnli'),
    ('transformer_deep', 'cifar10'),
    ('transformer_narrow', 'glue_mnli'),
    ('transformer_wide', 'glue_mnli'),
]


def run_experiment_worker(args):
    """Worker function to run a single experiment."""
    arch_name, dataset_name, num_steps, output_dir = args

    output_file = os.path.join(output_dir, '{}_{}.json'.format(arch_name, dataset_name))

    # Skip if already exists
    if os.path.exists(output_file):
        return "[SKIP] Skipped {} x {} (already exists)".format(arch_name, dataset_name)

    try:
        start = time.time()
        result = run_single_experiment(
            arch_name=arch_name,
            dataset_name=dataset_name,
            num_steps=num_steps,
            measurement_interval=5
        )

        # Save result
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)

        elapsed = time.time() - start
        return "[OK] Completed {} x {} in {:.1f} min".format(arch_name, dataset_name, elapsed/60)

    except Exception as e:
        return "[FAIL] Failed {} x {}: {}".format(arch_name, dataset_name, str(e)[:100])


def run_parallel_experiments(num_processes=4, num_steps=2000, output_dir='./results/phase1_full'):
    """
    Run experiments in parallel.

    Args:
        num_processes: Number of parallel processes (default: 4)
        num_steps: Steps per experiment
        output_dir: Output directory
    """
    # Create output directory (Python 2/3 compatible)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Prepare experiment arguments
    experiment_args = [
        (arch, dataset, num_steps, output_dir)
        for arch, dataset in EXPERIMENT_PLAN
    ]

    print("=" * 70)
    print("PARALLEL EXPERIMENT RUNNER")
    print("=" * 70)
    print("Total experiments: {}".format(len(experiment_args)))
    print("Parallel processes: {}".format(num_processes))
    print("Steps per experiment: {}".format(num_steps))
    print("Output directory: {}".format(output_dir))
    print("=" * 70)
    print()

    # Run experiments in parallel
    start_time = time.time()

    pool = mp.Pool(processes=num_processes)
    results = []
    for i, result in enumerate(pool.imap_unordered(run_experiment_worker, experiment_args)):
        print("[{}/{}] {}".format(i+1, len(experiment_args), result))
        results.append(result)
    pool.close()
    pool.join()

    elapsed = time.time() - start_time

    # Summary
    completed = sum(1 for r in results if r.startswith("[OK]"))
    skipped = sum(1 for r in results if r.startswith("[SKIP]"))
    failed = sum(1 for r in results if r.startswith("[FAIL]"))

    print()
    print("=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print("Total time: {:.1f} minutes ({:.1f} hours)".format(elapsed/60, elapsed/3600))
    print("Completed: {}".format(completed))
    print("Skipped: {}".format(skipped))
    print("Failed: {}".format(failed))
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Phase 1 experiments in parallel')
    parser.add_argument('--num-processes', type=int, default=4,
                        help='Number of parallel processes (default: 4)')
    parser.add_argument('--num-steps', type=int, default=2000,
                        help='Steps per experiment (default: 2000)')
    parser.add_argument('--output-dir', type=str, default='./results/phase1_full',
                        help='Output directory')

    args = parser.parse_args()

    # Avoid nested parallelism in PyTorch
    import torch
    torch.set_num_threads(1)

    run_parallel_experiments(
        num_processes=args.num_processes,
        num_steps=args.num_steps,
        output_dir=args.output_dir
    )
