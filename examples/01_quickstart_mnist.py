"""Quickstart example: Track dimensionality on MNIST with a simple MLP.

This minimal example demonstrates the core functionality of the Neural
Dimensionality Tracker in just a few lines of code.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from ndt import HighFrequencyTracker
from ndt import export_to_csv
from ndt import plot_phases

# 1. Define a simple MLP
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)

# 2. Set up MNIST data
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST("./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# 3. Create tracker (automatically detects layers to monitor)
tracker = HighFrequencyTracker(
    model, sampling_frequency=10, enable_jump_detection=True  # Record every 10 steps
)

# 4. Standard training loop
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print("Training MNIST with dimensionality tracking...")
model.train()
step = 0

for epoch in range(2):  # Just 2 epochs for demo
    for batch_idx, (data, target) in enumerate(train_loader):
        # Standard training step
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Track dimensionality (single line!)
        tracker.log(step, loss.item())

        step += 1

        if step >= 1000:  # Stop early for demo
            break
    if step >= 1000:
        break

print(f"Training complete. Tracked {step} steps.")

# 5. Analyze results
results = tracker.get_results()

print(f"\nTracked {len(results)} layers:")
for layer_name, df in results.items():
    print(f"  {layer_name}: {len(df)} measurements")

# 6. Detect dimensionality jumps
print("\nDetecting dimensionality jumps...")
jumps_dict = tracker.detect_jumps(metric="stable_rank")

for layer_name, jumps in jumps_dict.items():
    if jumps:
        print(f"  {layer_name}: {len(jumps)} jumps detected")
        for jump in jumps[:3]:  # Show first 3
            print(f"    {jump}")

# 7. Visualize
print("\nCreating visualization...")
fig = plot_phases(results, metric="stable_rank")
fig.savefig("mnist_stable_rank.png", dpi=150, bbox_inches="tight")
print("Saved visualization to mnist_stable_rank.png")

# 8. Export results
print("\nExporting results...")
export_to_csv(results, "mnist_results.csv")
print("Saved results to mnist_results.csv")

# 9. Clean up
tracker.close()

print("\nQuickstart complete! Check mnist_stable_rank.png and mnist_results.csv")
