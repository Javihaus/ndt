"""CNN example: Track dimensionality in a convolutional neural network on CIFAR-10."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from ndt import HighFrequencyTracker
from ndt import export_to_json
from ndt import plot_metrics_comparison


class SimpleCNN(nn.Module):
    """Simple CNN for CIFAR-10."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 128 * 4 * 4)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def main():
    """Train CNN on CIFAR-10 with dimensionality tracking."""

    # Model and data
    model = SimpleCNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    train_data = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2)

    # Create tracker - explicitly specify conv layers and fc layers
    tracker = HighFrequencyTracker(
        model,
        layers=[model.conv1, model.conv2, model.conv3, model.fc1],
        layer_names=["Conv1", "Conv2", "Conv3", "FC1"],
        sampling_frequency=20,  # Sample less frequently for speed
        device=device,
    )

    # Training setup
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    print("Training CNN on CIFAR-10...")
    model.train()
    step = 0

    for epoch in range(3):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # Compute gradient norm
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))

            optimizer.step()

            # Track with gradient norm
            tracker.log(step, loss.item(), grad_norm=grad_norm.item())

            if step % 100 == 0:
                print(f"Step {step}, Loss: {loss.item():.4f}")

            step += 1

            if step >= 2000:
                break
        if step >= 2000:
            break

    print(f"\nTraining complete. Tracked {step} steps.")

    # Analysis
    results = tracker.get_results()

    # Compare all metrics for Conv3 layer
    print("\nGenerating metric comparison plot for Conv3...")
    conv3_results = tracker.get_results(layer_name="Conv3")
    fig = plot_metrics_comparison(conv3_results, layer_name="Conv3")
    fig.savefig("cifar10_conv3_metrics.png", dpi=150, bbox_inches="tight")
    print("Saved to cifar10_conv3_metrics.png")

    # Export
    print("\nExporting results...")
    export_to_json(results, "cifar10_results.json")
    print("Saved to cifar10_results.json")

    # Detect jumps across all layers
    print("\nDetecting dimensionality jumps...")
    jumps_dict = tracker.detect_jumps(metric="stable_rank")

    for layer_name, jumps in jumps_dict.items():
        print(f"{layer_name}: {len(jumps)} jumps")

    tracker.close()
    print("\nExample complete!")


if __name__ == "__main__":
    main()
