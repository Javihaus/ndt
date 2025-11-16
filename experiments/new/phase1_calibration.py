"""
Phase 1: Establish the Phenomenon (Proper Scale)

Goal: Measure α_empirical for each architecture and find α = f(depth, width, connectivity)

Specifications:
- 10+ architectures (MLPs, CNNs, ResNets, Transformers of varying depths/widths)
- 5+ datasets (MNIST, CIFAR-10, Fashion-MNIST, SVHN, subset of ImageNet)
- Measure: α_empirical for each architecture
- High-frequency measurement: every 5-10 steps
- Training duration: 5000-8000 steps per experiment
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import numpy as np
from tqdm import tqdm
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# DIMENSIONALITY MEASUREMENT UTILITIES
# ============================================================================

class DimensionalityEstimator:
    """
    Computes multiple independent measures of effective dimensionality
    for neural network activations.
    """

    @staticmethod
    def stable_rank(matrix: torch.Tensor, eps: float = 1e-10) -> float:
        """
        Stable rank: ||A||_F^2 / ||A||_2^2
        Estimates effective number of linearly independent dimensions.
        """
        if matrix.numel() == 0:
            return 0.0
        frobenius_norm = torch.norm(matrix, p='fro').item()
        spectral_norm = torch.norm(matrix, p=2).item()
        if spectral_norm < eps:
            return 0.0
        return (frobenius_norm ** 2) / (spectral_norm ** 2 + eps)

    @staticmethod
    def participation_ratio(matrix: torch.Tensor, eps: float = 1e-10) -> float:
        """
        Participation ratio: (sum σ_i)^2 / sum(σ_i^2)
        Measures how uniformly singular values are distributed.
        """
        if matrix.numel() == 0:
            return 0.0
        try:
            U, S, V = torch.svd(matrix.float())
            S = S + eps
            return (S.sum().item() ** 2) / (S.pow(2).sum().item() + eps)
        except:
            return 0.0

    @staticmethod
    def effective_rank_90(matrix: torch.Tensor) -> float:
        """
        Number of components needed to capture 90% of variance.
        """
        if matrix.numel() == 0:
            return 0.0
        try:
            U, S, V = torch.svd(matrix.float())
            variance = S.pow(2)
            cumsum = torch.cumsum(variance, dim=0)
            total = variance.sum()
            if total < 1e-10:
                return 0.0
            ratio = cumsum / total
            n_components = (ratio < 0.9).sum().item() + 1
            return float(min(n_components, len(S)))
        except:
            return 0.0

    @staticmethod
    def nuclear_norm_ratio(matrix: torch.Tensor, eps: float = 1e-10) -> float:
        """
        Nuclear norm ratio: ||A||_* / ||A||_2
        Sum of singular values normalized by maximum singular value.
        """
        if matrix.numel() == 0:
            return 0.0
        try:
            U, S, V = torch.svd(matrix.float())
            if S[0].item() < eps:
                return 0.0
            return S.sum().item() / (S[0].item() + eps)
        except:
            return 0.0


class ActivationCapture:
    """Captures activations from specified layers during forward pass."""

    def __init__(self):
        self.activations = {}
        self.hooks = []

    def capture_hook(self, name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                self.activations[name] = output.detach()
        return hook

    def register(self, model: nn.Module, layer_types=(nn.Linear, nn.Conv2d)):
        """Register hooks on specified layer types."""
        for name, module in model.named_modules():
            if isinstance(module, layer_types):
                hook = module.register_forward_hook(self.capture_hook(name))
                self.hooks.append(hook)

    def clear(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}


def measure_network_dimensionality(model: nn.Module,
                                   dataloader: DataLoader,
                                   device: torch.device,
                                   max_batches: int = 10) -> Dict[str, Dict[str, float]]:
    """
    Measure effective dimensionality of network activations across all layers.

    Returns dictionary mapping layer names to their dimensionality metrics.
    """
    model.eval()
    capture = ActivationCapture()
    capture.register(model)

    # Collect activations
    layer_activations = {}

    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            inputs = inputs.to(device)
            _ = model(inputs)

            # Store activations from this batch
            for layer_name, activation in capture.activations.items():
                if layer_name not in layer_activations:
                    layer_activations[layer_name] = []

                # Flatten spatial dimensions if conv layer
                if activation.dim() > 2:
                    act = activation.view(activation.size(0), activation.size(1), -1).mean(dim=2)
                else:
                    act = activation
                layer_activations[layer_name].append(act.cpu())

    capture.clear()

    # Compute metrics for each layer
    estimator = DimensionalityEstimator()
    results = {}

    for layer_name, activations in layer_activations.items():
        activation_matrix = torch.cat(activations, dim=0)
        results[layer_name] = {
            'stable_rank': estimator.stable_rank(activation_matrix),
            'participation_ratio': estimator.participation_ratio(activation_matrix),
            'effective_rank_90': estimator.effective_rank_90(activation_matrix),
            'nuclear_norm_ratio': estimator.nuclear_norm_ratio(activation_matrix)
        }

    return results


# ============================================================================
# ARCHITECTURE DEFINITIONS
# ============================================================================

class SimpleMLP(nn.Module):
    """Multi-layer perceptron with configurable depth and width."""

    def __init__(self, input_dim: int, hidden_dims: List[int],
                 num_classes: int, activation: str = 'relu'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        return self.network(x)

    def get_architecture_params(self) -> Dict:
        """Return architectural parameters for α estimation."""
        total_params = sum(p.numel() for p in self.parameters())
        return {
            'depth': len(self.hidden_dims),
            'width': np.mean(self.hidden_dims) if self.hidden_dims else 0,
            'min_width': min(self.hidden_dims) if self.hidden_dims else 0,
            'max_width': max(self.hidden_dims) if self.hidden_dims else 0,
            'num_params': total_params,
            'connectivity': total_params / (self.input_dim + sum(self.hidden_dims) + self.num_classes)
        }


class SimpleCNN(nn.Module):
    """Simple CNN with configurable conv layers."""

    def __init__(self, in_channels: int, num_classes: int,
                 conv_channels: List[int] = [32, 64, 64]):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.conv_channels = conv_channels

        layers = []
        prev_channels = in_channels

        for channels in conv_channels:
            layers.extend([
                nn.Conv2d(prev_channels, channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ])
            prev_channels = channels

        self.conv_layers = nn.Sequential(*layers)

        # Calculate flattened size (assuming 32x32 input, after maxpools)
        self.flat_size = conv_channels[-1] * (32 // (2 ** len(conv_channels))) ** 2
        self.fc = nn.Linear(self.flat_size, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def get_architecture_params(self) -> Dict:
        total_params = sum(p.numel() for p in self.parameters())
        return {
            'depth': len(self.conv_channels) + 1,
            'width': np.mean(self.conv_channels),
            'min_width': min(self.conv_channels),
            'max_width': max(self.conv_channels),
            'num_params': total_params,
            'connectivity': total_params / (self.in_channels + sum(self.conv_channels) + self.num_classes)
        }


class ResNetWrapper(nn.Module):
    """Wrapper for ResNet with custom classifier."""

    def __init__(self, num_classes: int = 10, depth: str = '18'):
        super().__init__()
        self.depth = depth
        self.num_classes = num_classes

        if depth == '18':
            self.model = resnet18(pretrained=False, num_classes=num_classes)
        else:
            raise ValueError(f"Unsupported depth: {depth}")

        # Modify first conv for smaller images (32x32)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()

    def forward(self, x):
        return self.model(x)

    def get_architecture_params(self) -> Dict:
        total_params = sum(p.numel() for p in self.parameters())
        return {
            'depth': 18 if self.depth == '18' else 34,
            'width': 64,  # Base width for ResNet
            'min_width': 64,
            'max_width': 512,
            'num_params': total_params,
            'connectivity': total_params / (3 + self.num_classes + 512)
        }


class SimpleTransformer(nn.Module):
    """Simple transformer encoder for classification."""

    def __init__(self, input_dim: int, d_model: int, nhead: int,
                 num_layers: int, num_classes: int, seq_len: int = 16):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.seq_len = seq_len

        # Project input to d_model and create sequence
        self.input_proj = nn.Linear(input_dim // seq_len, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.seq_len, -1)
        x = self.input_proj(x)
        x = x + self.pos_encoder.unsqueeze(0)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.fc(x)

    def get_architecture_params(self) -> Dict:
        total_params = sum(p.numel() for p in self.parameters())
        return {
            'depth': self.num_layers,
            'width': self.d_model,
            'min_width': self.d_model,
            'max_width': self.d_model * 4,
            'num_params': total_params,
            'connectivity': total_params / (self.d_model * self.num_layers)
        }


# ============================================================================
# ARCHITECTURE FACTORY
# ============================================================================

def create_architecture(arch_name: str, input_dim: int,
                       num_classes: int, in_channels: int = 3) -> nn.Module:
    """Factory function to create architectures."""

    architectures = {
        # Group 1: Depth variation (MLPs, ~50K params)
        'mlp_shallow_2': lambda: SimpleMLP(input_dim, [128, 64], num_classes),
        'mlp_medium_5': lambda: SimpleMLP(input_dim, [64, 64, 64, 64, 32], num_classes),
        'mlp_deep_10': lambda: SimpleMLP(input_dim, [32]*10, num_classes),
        'mlp_verydeep_15': lambda: SimpleMLP(input_dim, [24]*15, num_classes),

        # Group 2: Width variation (4 layers)
        'mlp_narrow': lambda: SimpleMLP(input_dim, [32, 32, 32, 32], num_classes),
        'mlp_medium': lambda: SimpleMLP(input_dim, [128, 128, 64, 64], num_classes),
        'mlp_wide': lambda: SimpleMLP(input_dim, [256, 256, 128, 128], num_classes),
        'mlp_verywide': lambda: SimpleMLP(input_dim, [512, 512, 256, 256], num_classes),

        # Group 3: CNNs with varying depth
        'cnn_shallow': lambda: SimpleCNN(in_channels, num_classes, [32, 64]),
        'cnn_medium': lambda: SimpleCNN(in_channels, num_classes, [32, 64, 128]),
        'cnn_deep': lambda: SimpleCNN(in_channels, num_classes, [32, 64, 128, 256]),

        # Group 4: ResNet
        'resnet18': lambda: ResNetWrapper(num_classes, depth='18'),

        # Group 5: Transformers with varying depth
        'transformer_shallow': lambda: SimpleTransformer(input_dim, 128, 4, 2, num_classes),
        'transformer_medium': lambda: SimpleTransformer(input_dim, 128, 4, 4, num_classes),
        'transformer_deep': lambda: SimpleTransformer(input_dim, 128, 4, 6, num_classes),

        # Group 6: Width variation for Transformers
        'transformer_narrow': lambda: SimpleTransformer(input_dim, 64, 4, 4, num_classes),
        'transformer_wide': lambda: SimpleTransformer(input_dim, 256, 8, 4, num_classes),
    }

    if arch_name not in architectures:
        raise ValueError(f"Unknown architecture: {arch_name}")

    return architectures[arch_name]()


# ============================================================================
# DATASET LOADERS
# ============================================================================

def get_mnist_loaders(batch_size: int = 64, subset_size: Optional[int] = None):
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    trainset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    if subset_size:
        trainset = Subset(trainset, torch.randperm(len(trainset))[:subset_size])
        testset = Subset(testset, torch.randperm(len(testset))[:subset_size//6])

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader, 784, 10, 1  # input_dim, num_classes, in_channels


def get_fashion_mnist_loaders(batch_size: int = 64, subset_size: Optional[int] = None):
    """Load Fashion-MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    trainset = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform
    )

    if subset_size:
        trainset = Subset(trainset, torch.randperm(len(trainset))[:subset_size])
        testset = Subset(testset, torch.randperm(len(testset))[:subset_size//6])

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader, 784, 10, 1


def get_cifar10_loaders(batch_size: int = 64, subset_size: Optional[int] = None):
    """Load CIFAR-10 dataset."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )

    if subset_size:
        trainset = Subset(trainset, torch.randperm(len(trainset))[:subset_size])
        testset = Subset(testset, torch.randperm(len(testset))[:subset_size//5])

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader, 3072, 10, 3


def get_svhn_loaders(batch_size: int = 64, subset_size: Optional[int] = None):
    """Load SVHN dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
    ])

    trainset = torchvision.datasets.SVHN(
        root='./data', split='train', download=True, transform=transform
    )
    testset = torchvision.datasets.SVHN(
        root='./data', split='test', download=True, transform=transform
    )

    if subset_size:
        trainset = Subset(trainset, torch.randperm(len(trainset))[:subset_size])
        testset = Subset(testset, torch.randperm(len(testset))[:subset_size//7])

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader, 3072, 10, 3


def get_cifar100_loaders(batch_size: int = 64, subset_size: Optional[int] = None):
    """Load CIFAR-100 dataset."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test
    )

    if subset_size:
        trainset = Subset(trainset, torch.randperm(len(trainset))[:subset_size])
        testset = Subset(testset, torch.randperm(len(testset))[:subset_size//5])

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader, 3072, 100, 3


DATASET_LOADERS = {
    'mnist': get_mnist_loaders,
    'fashion_mnist': get_fashion_mnist_loaders,
    'cifar10': get_cifar10_loaders,
    'svhn': get_svhn_loaders,
    'cifar100': get_cifar100_loaders,
}


# ============================================================================
# TRAINING LOOP WITH HIGH-FREQUENCY MEASUREMENT
# ============================================================================

def train_with_measurement(model: nn.Module,
                          train_loader: DataLoader,
                          val_loader: DataLoader,
                          device: torch.device,
                          num_steps: int = 5000,
                          measurement_interval: int = 5,
                          learning_rate: float = 0.001) -> Dict:
    """
    Train model with high-frequency dimensionality measurement.

    Returns dictionary containing:
    - measurements: list of (step, metrics) tuples
    - architecture_params: architectural parameters
    - final_accuracy: validation accuracy
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    measurements = []
    step = 0

    # Get architecture parameters
    arch_params = model.get_architecture_params()

    print(f"Training {arch_params['depth']} layer network, "
          f"{arch_params['num_params']:,} parameters")

    pbar = tqdm(total=num_steps, desc="Training")

    epoch = 0
    while step < num_steps:
        epoch += 1
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

            # Measure dimensionality at specified intervals
            if step % measurement_interval == 0:
                dim_metrics = measure_network_dimensionality(
                    model, val_loader, device, max_batches=5
                )

                measurements.append({
                    'step': step,
                    'epoch': epoch,
                    'loss': loss.item(),
                    'grad_norm': grad_norm,
                    'layer_metrics': dim_metrics
                })

            step += 1
            pbar.update(1)
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'epoch': epoch})

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
        'architecture_params': arch_params,
        'final_accuracy': final_accuracy
    }


# ============================================================================
# EXPERIMENTAL PIPELINE
# ============================================================================

def run_single_experiment(arch_name: str,
                         dataset_name: str,
                         num_steps: int = 5000,
                         measurement_interval: int = 5,
                         device: Optional[torch.device] = None) -> Dict:
    """Run single architecture x dataset experiment."""

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*70}")
    print(f"Experiment: {arch_name} on {dataset_name}")
    print(f"{'='*70}")

    # Load data
    loader_fn = DATASET_LOADERS[dataset_name]
    train_loader, val_loader, input_dim, num_classes, in_channels = loader_fn(
        batch_size=64, subset_size=10000  # Use subset for faster experiments
    )

    # Create model
    model = create_architecture(arch_name, input_dim, num_classes, in_channels)

    # Train with measurement
    start_time = time.time()
    results = train_with_measurement(
        model, train_loader, val_loader, device,
        num_steps=num_steps, measurement_interval=measurement_interval
    )
    elapsed = time.time() - start_time

    results['arch_name'] = arch_name
    results['dataset_name'] = dataset_name
    results['elapsed_time'] = elapsed
    results['num_steps'] = num_steps
    results['measurement_interval'] = measurement_interval

    print(f"\nCompleted in {elapsed/60:.1f} minutes")
    print(f"Final accuracy: {results['final_accuracy']:.4f}")

    return results


def run_phase1_calibration(output_dir: str = './experiments/new/results/phase1',
                           architectures: Optional[List[str]] = None,
                           datasets: Optional[List[str]] = None,
                           num_steps: int = 5000,
                           measurement_interval: int = 5):
    """
    Phase 1: Train all architectures on all datasets.
    This establishes the α = f(architecture) relationship.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if architectures is None:
        architectures = [
            # Depth variation
            'mlp_shallow_2', 'mlp_medium_5', 'mlp_deep_10', 'mlp_verydeep_15',
            # Width variation
            'mlp_narrow', 'mlp_medium', 'mlp_wide', 'mlp_verywide',
            # CNNs
            'cnn_shallow', 'cnn_medium', 'cnn_deep',
            # ResNet
            'resnet18',
            # Transformers
            'transformer_shallow', 'transformer_medium', 'transformer_deep',
            'transformer_narrow', 'transformer_wide',
        ]

    if datasets is None:
        datasets = ['mnist', 'fashion_mnist', 'cifar10', 'svhn', 'cifar100']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"\nPhase 1 Calibration:")
    print(f"  Architectures: {len(architectures)}")
    print(f"  Datasets: {len(datasets)}")
    print(f"  Total experiments: {len(architectures) * len(datasets)}")
    print(f"  Steps per experiment: {num_steps}")
    print(f"  Measurement interval: {measurement_interval}")

    all_results = []

    for dataset_name in datasets:
        for arch_name in architectures:
            try:
                results = run_single_experiment(
                    arch_name, dataset_name,
                    num_steps=num_steps,
                    measurement_interval=measurement_interval,
                    device=device
                )
                all_results.append(results)

                # Save individual result
                output_file = Path(output_dir) / f'{arch_name}_{dataset_name}.json'
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)

                print(f"✓ Saved: {output_file}")

            except Exception as e:
                print(f"✗ Failed {arch_name} on {dataset_name}: {e}")
                continue

    # Save combined results
    combined_file = Path(output_dir) / 'phase1_all_results.json'
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Phase 1 complete. {len(all_results)} experiments saved to {output_dir}")
    print(f"{'='*70}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Phase 1: Calibration Experiments')
    parser.add_argument('--output-dir', type=str, default='./experiments/new/results/phase1',
                       help='Output directory for results')
    parser.add_argument('--num-steps', type=int, default=5000,
                       help='Training steps per experiment')
    parser.add_argument('--measurement-interval', type=int, default=5,
                       help='Measurement frequency (every N steps)')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with 2 archs and 2 datasets')

    args = parser.parse_args()

    if args.quick_test:
        print("Running quick test...")
        archs = ['mlp_shallow_2', 'cnn_shallow']
        datasets = ['mnist', 'cifar10']
        results = run_phase1_calibration(
            args.output_dir, archs, datasets,
            num_steps=500, measurement_interval=10
        )
    else:
        results = run_phase1_calibration(
            args.output_dir,
            num_steps=args.num_steps,
            measurement_interval=args.measurement_interval
        )

    print("\nPhase 1 complete!")
    print("Next: Run phase1_analysis.py to extract α parameters")
