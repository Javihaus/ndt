"""
Run 3 Specific Experiments with Checkpoint Saving
==================================================

Re-runs transformer_deep_mnist, cnn_deep_mnist, and mlp_narrow_mnist
with checkpoint saving enabled at steps [100, 1000, 2000].

These checkpoints will be used for modified Phase 2 analysis.
"""

import os
import json
import time
import torch
from pathlib import Path
from phase1_calibration import run_single_experiment
import argparse


# ============================================================================
# CHECKPOINT SAVING CONFIGURATION
# ============================================================================

CHECKPOINT_PLAN = {
    'transformer_deep_mnist': {
        'arch': 'transformer_deep',
        'dataset': 'mnist',
        'checkpoint_steps': [100, 1000, 2000]
    },
    'cnn_deep_mnist': {
        'arch': 'cnn_deep',
        'dataset': 'mnist',
        'checkpoint_steps': [100, 1000, 2000]
    },
    'mlp_narrow_mnist': {
        'arch': 'mlp_narrow',
        'dataset': 'mnist',
        'checkpoint_steps': [100, 1000, 2000]
    }
}


def save_checkpoint(model, optimizer, step, loss, checkpoint_dir, experiment_name):
    """Save model checkpoint."""
    checkpoint_path = Path(checkpoint_dir) / f'checkpoint_step_{step:05d}.pt'

    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'experiment_name': experiment_name
    }, checkpoint_path)

    print(f'  ✓ Saved checkpoint: {checkpoint_path.name}')
    return checkpoint_path


def run_experiment_with_checkpoints(experiment_name, arch_name, dataset_name,
                                    checkpoint_steps, num_steps=2000,
                                    checkpoint_base_dir='../checkpoints'):
    """
    Run experiment with checkpoint saving.

    This is a wrapper that:
    1. Creates checkpoint directory
    2. Runs the experiment
    3. Saves checkpoints at specified steps
    4. Returns results
    """
    # Create checkpoint directory
    checkpoint_dir = Path(checkpoint_base_dir) / experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"RUNNING: {experiment_name}")
    print(f"{'='*70}")
    print(f"Architecture: {arch_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Steps: {num_steps}")
    print(f"Checkpoints: {checkpoint_steps}")
    print(f"Output: {checkpoint_dir}")
    print(f"{'='*70}\n")

    # Import the necessary modules
    from phase1_calibration import (
        get_dataset_loaders, create_architecture,
        train_with_measurement_and_checkpoints
    )

    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load dataset
    train_loader, val_loader, input_dim, num_classes = get_dataset_loaders(
        dataset_name, batch_size=64
    )

    # Create model
    model = create_architecture(arch_name, input_dim, num_classes)
    model = model.to(device)

    print(f"Model created: {sum(p.numel() for p in model.parameters()):,} parameters\n")

    # Train with checkpoints
    start_time = time.time()

    try:
        result = train_with_measurement_and_checkpoints(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_steps=num_steps,
            measurement_interval=5,
            checkpoint_steps=set(checkpoint_steps),
            checkpoint_dir=checkpoint_dir,
            experiment_name=experiment_name
        )
    except AttributeError:
        # If the function doesn't exist, we need to add it to phase1_calibration.py
        print("ERROR: train_with_measurement_and_checkpoints not found")
        print("Using fallback method with modified training loop...")
        result = run_with_custom_checkpoint_saving(
            model, train_loader, val_loader, device,
            num_steps, checkpoint_steps, checkpoint_dir, experiment_name
        )

    elapsed = time.time() - start_time

    result['experiment_name'] = experiment_name
    result['arch_name'] = arch_name
    result['dataset_name'] = dataset_name
    result['elapsed_time'] = elapsed
    result['checkpoint_dir'] = str(checkpoint_dir)
    result['checkpoint_steps'] = checkpoint_steps

    print(f"\n{'='*70}")
    print(f"COMPLETED: {experiment_name}")
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"Final accuracy: {result['final_accuracy']:.4f}")
    print(f"Checkpoints saved: {len(checkpoint_steps)}")
    print(f"{'='*70}\n")

    return result


def run_with_custom_checkpoint_saving(model, train_loader, val_loader, device,
                                      num_steps, checkpoint_steps, checkpoint_dir,
                                      experiment_name):
    """
    Custom training loop with checkpoint saving.
    Fallback if train_with_measurement_and_checkpoints doesn't exist.
    """
    import torch.nn as nn
    import torch.optim as optim
    from tqdm import tqdm

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    checkpoint_steps_set = set(checkpoint_steps)
    measurements = []
    step = 0

    pbar = tqdm(total=num_steps, desc="Training")

    while step < num_steps:
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

            # Compute gradient norm
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.norm().item() ** 2
            grad_norm = grad_norm ** 0.5

            optimizer.step()

            # Save checkpoint if needed
            if step in checkpoint_steps_set:
                save_checkpoint(model, optimizer, step, loss.item(),
                              checkpoint_dir, experiment_name)

            # Measurement (every 5 steps)
            if step % 5 == 0:
                measurements.append({
                    'step': step,
                    'loss': loss.item(),
                    'grad_norm': grad_norm
                })

            step += 1
            pbar.update(1)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    pbar.close()

    # Final validation accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    final_accuracy = correct / total

    return {
        'measurements': measurements,
        'architecture_params': model.get_architecture_params() if hasattr(model, 'get_architecture_params') else {},
        'final_accuracy': final_accuracy
    }


def run_checkpoint_experiments(output_dir='./results/phase1_checkpoints',
                               checkpoint_dir='../checkpoints',
                               num_steps=2000):
    """
    Run all 3 checkpoint experiments sequentially.
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CHECKPOINT EXPERIMENT RUNNER")
    print("=" * 70)
    print(f"Experiments: 3 (transformer, cnn, mlp)")
    print(f"Steps per experiment: {num_steps}")
    print(f"Checkpoints per experiment: 3 ([100, 1000, 2000])")
    print(f"Total checkpoints: 9")
    print(f"Output dir: {output_dir}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print("=" * 70)

    all_results = []
    start_time = time.time()

    for experiment_name, config in CHECKPOINT_PLAN.items():
        arch_name = config['arch']
        dataset_name = config['dataset']
        checkpoint_steps = config['checkpoint_steps']

        result = run_experiment_with_checkpoints(
            experiment_name=experiment_name,
            arch_name=arch_name,
            dataset_name=dataset_name,
            checkpoint_steps=checkpoint_steps,
            num_steps=num_steps,
            checkpoint_base_dir=checkpoint_dir
        )

        all_results.append(result)

        # Save result
        output_file = output_dir / f'{experiment_name}_with_checkpoints.json'
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"✓ Saved result: {output_file}")

    total_time = time.time() - start_time

    # Save summary
    summary = {
        'total_time': total_time,
        'num_experiments': len(CHECKPOINT_PLAN),
        'total_checkpoints': len(CHECKPOINT_PLAN) * 3,
        'experiments': all_results
    }

    summary_file = output_dir / 'checkpoint_experiments_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    print(f"Experiments: {len(CHECKPOINT_PLAN)}")
    print(f"Checkpoints saved: {len(CHECKPOINT_PLAN) * 3}")
    print(f"Summary: {summary_file}")
    print("=" * 70)

    # Verify checkpoints
    print("\nVERIFYING CHECKPOINTS:")
    print("=" * 70)
    for experiment_name, config in CHECKPOINT_PLAN.items():
        checkpoint_exp_dir = Path(checkpoint_dir) / experiment_name
        print(f"\n{experiment_name}:")
        for step in config['checkpoint_steps']:
            checkpoint_path = checkpoint_exp_dir / f'checkpoint_step_{step:05d}.pt'
            if checkpoint_path.exists():
                size_mb = checkpoint_path.stat().st_size / 1e6
                print(f"  ✓ Step {step}: {size_mb:.1f}MB")
            else:
                print(f"  ✗ Step {step}: MISSING")

    print("\n" + "=" * 70)
    print("NEXT STEP: Run modified Phase 2 analysis")
    print("  cd experiments/mechanistic_interpretability")
    print("  python3 modified_phase2_analysis.py")
    print("=" * 70)

    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run 3 experiments with checkpoint saving'
    )
    parser.add_argument('--num-steps', type=int, default=2000,
                        help='Number of training steps (default: 2000)')
    parser.add_argument('--output-dir', type=str,
                        default='./results/phase1_checkpoints',
                        help='Output directory for results')
    parser.add_argument('--checkpoint-dir', type=str,
                        default='../checkpoints',
                        help='Directory to save checkpoints')

    args = parser.parse_args()

    run_checkpoint_experiments(
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        num_steps=args.num_steps
    )
