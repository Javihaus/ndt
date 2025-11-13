"""Reproduce TDS article experiment: High-frequency tracking on MNIST.

This script reproduces the exact experimental setup described in the TDS article:
"I Measured Neural Network Training Every 5 Steps for 10,000 Iterations"

Reference specifications from Figure 1 of the article:
- Architecture: 784-256-128-10 (MLP with 3 hidden layers)
- Dataset: MNIST (60k train/10k test)
- Optimizer: Adam with β1=0.9, β2=0.999
- Learning rate: 0.001
- Batch size: 64
- Training duration: 8000 steps
- Loss function: Cross-entropy
- Measurement frequency: Every 5 steps (high-frequency checkpointing)

Expected results (from TDS article):
- Phase 1: Initial collapse (steps 0-300) - dimensionality drops from ~2500 to ~500
- Phase 2: Expansion (steps 300-5000) - dimensionality climbs to ~1000
- Phase 3: Stabilization (steps 5000-8000) - dimensionality plateaus

The article demonstrates that:
1. High-frequency sampling (every 5 steps) reveals transitions missed by coarse sampling
2. Two-thirds of transitions occur in the first 2000 steps (25% of training)
3. Activation space shows ~85 jumps while weight space shows only 1
4. Dimensionality correlates strongly with loss (ρ = -0.951)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from ndt import HighFrequencyTracker
from ndt import export_to_csv
from ndt import export_to_hdf5
from ndt import plot_metrics_comparison
from ndt import plot_phases


# 1. Define the exact architecture from TDS article: 784-256-128-10
class TDSExperimentMLP(nn.Module):
    """3-layer MLP matching TDS article specifications.

    Architecture: 784 → 256 → 128 → 10
    This is a 3-hidden-layer network (not counting input and output).
    """

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.layer3(x)
        return x


def main():
    """Run the complete TDS experiment reproduction."""

    print("=" * 80)
    print("TDS Article Experiment Reproduction")
    print("=" * 80)
    print("\nConfiguration:")
    print("  Architecture: 784-256-128-10 (3-layer MLP)")
    print("  Dataset: MNIST (60k train / 10k test)")
    print("  Optimizer: Adam (β1=0.9, β2=0.999)")
    print("  Learning rate: 0.001")
    print("  Batch size: 64")
    print("  Training steps: 8000")
    print("  Sampling frequency: Every 5 steps")
    print("  Loss function: Cross-entropy")
    print()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # 2. Initialize model with exact architecture
    model = TDSExperimentMLP().to(device)
    print("Model architecture:")
    print(model)
    print()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print()

    # 3. Load MNIST dataset (60k train, 10k test)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]  # MNIST mean and std
    )

    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)

    # Use exact batch size from TDS article
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2,
        pin_memory=True if device.type == "cuda" else False,
    )

    # 4. Create tracker with high-frequency sampling (every 5 steps)
    tracker = HighFrequencyTracker(
        model,
        layers=[model.layer1, model.layer2, model.layer3],
        layer_names=["Layer1_784-256", "Layer2_256-128", "Layer3_128-10"],
        sampling_frequency=5,  # Every 5 steps as per TDS article
        enable_jump_detection=True,
        device=device,
    )

    # 5. Setup optimizer with exact hyperparameters from TDS article
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))  # β1=0.9, β2=0.999

    # Cross-entropy loss
    criterion = nn.CrossEntropyLoss()

    # 6. Training loop for exactly 8000 steps
    print("Starting training (8000 steps with sampling every 5 steps)...")
    print("Expected ~1600 measurements per layer\n")

    model.train()
    step = 0
    target_steps = 8000

    # Progress indicators
    progress_milestones = [300, 1000, 2000, 3000, 5000, 8000]

    epoch = 0
    while step < target_steps:
        epoch += 1
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # Standard training step
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # Compute gradient norm for correlation analysis
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))

            optimizer.step()

            # Track dimensionality with high frequency (every 5 steps)
            tracker.log(step, loss.item(), grad_norm=grad_norm.item())

            # Progress reporting at key milestones
            if step in progress_milestones:
                print(
                    f"Step {step:5d} | Loss: {loss.item():.6f} | "
                    f"Grad norm: {grad_norm.item():.4f}"
                )

            step += 1

            if step >= target_steps:
                break

    print(f"\nTraining complete! Tracked {step} steps across {epoch} epochs.\n")

    # 7. Analyze results
    print("=" * 80)
    print("Analysis & Results")
    print("=" * 80)

    results = tracker.get_results()

    print("\nTracked layers:")
    for layer_name, df in results.items():
        print(f"  {layer_name}: {len(df)} measurements")
        if len(df) > 0:
            sr_initial = df["stable_rank"].iloc[0]
            sr_final = df["stable_rank"].iloc[-1]
            sr_min = df["stable_rank"].min()
            sr_max = df["stable_rank"].max()
            print(
                f"    Stable rank: initial={sr_initial:.2f}, final={sr_final:.2f}, "
                f"min={sr_min:.2f}, max={sr_max:.2f}"
            )

    # 8. Detect dimensionality jumps (phase transitions)
    print("\n" + "=" * 80)
    print("Jump Detection (Phase Transitions)")
    print("=" * 80)
    print("\nDetecting dimensionality jumps using stable rank metric...")

    jumps_dict = tracker.detect_jumps(metric="stable_rank", threshold_z=2.0)

    total_jumps = 0
    for layer_name, jumps in jumps_dict.items():
        total_jumps += len(jumps)
        print(f"\n{layer_name}: {len(jumps)} jumps detected")

        # Show first 5 jumps
        for i, jump in enumerate(jumps[:5]):
            print(f"  Jump {i+1}: {jump}")

        if len(jumps) > 5:
            print(f"  ... and {len(jumps) - 5} more jumps")

    print(f"\nTotal jumps across all activation layers: {total_jumps}")
    print("\nTDS article reported ~85 jumps in activation space vs 1 in weight space")
    print("Most transitions concentrate in the first 2000 steps (25% of training)")

    # 9. Generate visualizations matching TDS article figures
    print("\n" + "=" * 80)
    print("Generating Visualizations")
    print("=" * 80)

    # Figure 2 equivalent: Activation space analysis
    print("\nCreating activation space dimensionality plot (TDS Figure 2)...")
    fig_activation = plot_phases(
        results, metric="stable_rank", title="Activation Space Dimensionality (Every 5 Steps)"
    )
    fig_activation.savefig("tds_figure2_activation_space.png", dpi=300, bbox_inches="tight")
    print("  Saved: tds_figure2_activation_space.png")

    # Figure 3 equivalent: Dimensionality vs Loss correlation
    print("\nCreating dimensionality vs loss correlation plot (TDS Figure 3)...")
    # Get the middle layer for detailed analysis
    layer2_results = tracker.get_results(layer_name="Layer2_256-128")
    fig_correlation = plot_metrics_comparison(
        layer2_results,
        layer_name="Layer2_256-128",
        title="Dimensionality vs Loss (ρ = -0.951 expected)",
    )
    fig_correlation.savefig("tds_figure3_dimensionality_loss.png", dpi=300, bbox_inches="tight")
    print("  Saved: tds_figure3_dimensionality_loss.png")

    # Figure 4 equivalent: High vs low frequency comparison
    print("\nNote: For Figure 4 (sampling frequency comparison), ")
    print("      run this script twice with different sampling_frequency values")

    # 10. Export results for further analysis
    print("\n" + "=" * 80)
    print("Exporting Results")
    print("=" * 80)

    # CSV export
    print("\nExporting to CSV...")
    export_to_csv(results, "tds_experiment_results.csv")
    print("  Saved: tds_experiment_results.csv")

    # HDF5 export for efficient storage
    print("\nExporting to HDF5...")
    export_to_hdf5(results, "tds_experiment_results.h5")
    print("  Saved: tds_experiment_results.h5")

    # 11. Summary and comparison with TDS article findings
    print("\n" + "=" * 80)
    print("Experiment Summary")
    print("=" * 80)

    print("\nKey findings from TDS article:")
    print("  ✓ Phase 1 (Collapse): Steps 0-300")
    print("    - Dimensionality drops from ~2500 to ~500")
    print("    - Loss landscape restructuring")
    print("  ✓ Phase 2 (Expansion): Steps 300-5000")
    print("    - Dimensionality climbs to ~1000")
    print("    - Capacity expansion and feature formation")
    print("  ✓ Phase 3 (Stabilization): Steps 5000-8000")
    print("    - Dimensionality plateaus")
    print("    - Architectural constraints bind")
    print()
    print("  ✓ High-frequency sampling (5 steps) reveals transitions")
    print("    missed by coarse sampling (50-100 steps)")
    print("  ✓ Dimensionality correlates strongly with loss (ρ = -0.951)")
    print("  ✓ Two-thirds of transitions in first 2000 steps (25% of training)")
    print()

    print("Reproduction complete!")
    print("\nGenerated files:")
    print("  - tds_figure2_activation_space.png")
    print("  - tds_figure3_dimensionality_loss.png")
    print("  - tds_experiment_results.csv")
    print("  - tds_experiment_results.h5")

    # Clean up
    tracker.close()

    print("\n" + "=" * 80)
    print("To reproduce Figure 4 (sampling frequency comparison):")
    print("  1. Run with sampling_frequency=5 (current run)")
    print("  2. Run with sampling_frequency=50 (low frequency)")
    print("  3. Compare the detected jumps between both runs")
    print("=" * 80)


if __name__ == "__main__":
    main()
