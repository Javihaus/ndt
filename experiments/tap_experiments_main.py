"""
Predicting Training Dynamics from Architectural Parameters
Main Experimental Framework

This code implements systematic measurement of representational dimensionality
during training to test whether TAP-inspired constrained growth models can
predict training dynamics from architectural parameters.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
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


def measure_network_dimensionality(model: nn.Module, 
                                   dataloader: DataLoader,
                                   device: torch.device,
                                   max_batches: int = 10) -> Dict[str, float]:
    """
    Measure effective dimensionality of network activations.
    
    Returns dictionary with stable_rank, participation_ratio, 
    effective_rank_90, and nuclear_norm_ratio.
    """
    model.eval()
    activations = []
    
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            inputs = inputs.to(device)
            
            # Forward pass through network
            x = inputs
            for layer in model.modules():
                if isinstance(layer, (nn.Linear, nn.Conv2d)):
                    if isinstance(layer, nn.Conv2d):
                        x = layer(x)
                        # Flatten spatial dimensions for conv layers
                        x_flat = x.view(x.size(0), x.size(1), -1).mean(dim=2)
                        activations.append(x_flat.cpu())
                    else:
                        x = layer(x)
                        activations.append(x.cpu())
    
    # Concatenate all activations
    if len(activations) == 0:
        return {
            'stable_rank': 0.0,
            'participation_ratio': 0.0,
            'effective_rank_90': 0.0,
            'nuclear_norm_ratio': 0.0
        }
    
    activation_matrix = torch.cat(activations, dim=0)
    
    estimator = DimensionalityEstimator()
    return {
        'stable_rank': estimator.stable_rank(activation_matrix),
        'participation_ratio': estimator.participation_ratio(activation_matrix),
        'effective_rank_90': estimator.effective_rank_90(activation_matrix),
        'nuclear_norm_ratio': estimator.nuclear_norm_ratio(activation_matrix)
    }


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
        return {
            'depth': len(self.hidden_dims),
            'width': np.mean(self.hidden_dims) if self.hidden_dims else 0,
            'num_params': sum(p.numel() for p in self.parameters())
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
        
        # Calculate flattened size (assuming 32x32 input, after 3 maxpools: 4x4)
        self.flat_size = conv_channels[-1] * 4 * 4
        self.fc = nn.Linear(self.flat_size, num_classes)
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    
    def get_architecture_params(self) -> Dict:
        return {
            'depth': len(self.conv_channels) + 1,  # conv layers + fc
            'width': np.mean(self.conv_channels),
            'num_params': sum(p.numel() for p in self.parameters())
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
        return {
            'depth': self.num_layers,
            'width': self.d_model,
            'num_params': sum(p.numel() for p in self.parameters())
        }


def create_architecture(arch_name: str, input_dim: int, 
                       num_classes: int) -> nn.Module:
    """Factory function to create architectures."""
    
    architectures = {
        # Group 1: Depth variation (≈50K params)
        'mlp_2layer': lambda: SimpleMLP(input_dim, [128, 64], num_classes),
        'mlp_5layer': lambda: SimpleMLP(input_dim, [64, 64, 64, 64, 32], num_classes),
        'mlp_10layer': lambda: SimpleMLP(input_dim, [32]*10, num_classes),
        
        # Group 2: Width variation (4 layers)
        'mlp_narrow': lambda: SimpleMLP(input_dim, [32, 32, 32, 32], num_classes),
        'mlp_medium': lambda: SimpleMLP(input_dim, [128, 128, 64, 64], num_classes),
        'mlp_wide': lambda: SimpleMLP(input_dim, [256, 256, 128, 128], num_classes),
        
        # Group 3: Architecture types (≈100K params)
        'cnn_small': lambda: SimpleCNN(3, num_classes, [32, 64, 64]),
        'transformer_small': lambda: SimpleTransformer(input_dim, 128, 4, 2, num_classes),
        'hybrid': lambda: SimpleMLP(input_dim, [256, 128], num_classes)  # Placeholder
    }
    
    if arch_name not in architectures:
        raise ValueError(f"Unknown architecture: {arch_name}")
    
    return architectures[arch_name]()


# ============================================================================
# TRAINING LOOP WITH HIGH-FREQUENCY MEASUREMENT
# ============================================================================

def train_with_measurement(model: nn.Module,
                          train_loader: DataLoader,
                          val_loader: DataLoader,
                          device: torch.device,
                          num_steps: int = 5000,
                          measurement_interval: int = 10,
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
            
            # Measure dimensionality at specified intervals
            if step % measurement_interval == 0:
                dim_metrics = measure_network_dimensionality(
                    model, val_loader, device, max_batches=5
                )
                
                measurements.append({
                    'step': step,
                    'loss': loss.item(),
                    'grad_norm': grad_norm,
                    **dim_metrics
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
        'architecture_params': arch_params,
        'final_accuracy': final_accuracy
    }


# ============================================================================
# DATA LOADING
# ============================================================================

def get_tiny_imagenet_loaders(batch_size: int = 64, 
                              num_classes: int = 10,
                              num_samples: int = 5000):
    """
    Create small subset of ImageNet for fast experimentation.
    Uses CIFAR-10 as proxy since it's similar scale.
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010))
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    
    # Create small subset
    indices = torch.randperm(len(trainset))[:num_samples]
    trainset = Subset(trainset, indices)
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    indices = torch.randperm(len(testset))[:1000]
    testset = Subset(testset, indices)
    
    train_loader = DataLoader(trainset, batch_size=batch_size, 
                             shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=batch_size, 
                            shuffle=False, num_workers=2)
    
    return train_loader, test_loader, 3072, num_classes  # input_dim, num_classes


# ============================================================================
# MAIN EXPERIMENTAL PIPELINE
# ============================================================================

def run_single_experiment(arch_name: str, 
                         task_name: str = 'vision',
                         num_steps: int = 5000,
                         device: Optional[torch.device] = None) -> Dict:
    """Run single architecture x task experiment."""
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*60}")
    print(f"Experiment: {arch_name} on {task_name}")
    print(f"{'='*60}")
    
    # Load data
    train_loader, val_loader, input_dim, num_classes = get_tiny_imagenet_loaders()
    
    # Create model
    model = create_architecture(arch_name, input_dim, num_classes)
    
    # Train with measurement
    start_time = time.time()
    results = train_with_measurement(
        model, train_loader, val_loader, device, 
        num_steps=num_steps, measurement_interval=10
    )
    elapsed = time.time() - start_time
    
    results['arch_name'] = arch_name
    results['task_name'] = task_name
    results['elapsed_time'] = elapsed
    
    print(f"\nCompleted in {elapsed/60:.1f} minutes")
    print(f"Final accuracy: {results['final_accuracy']:.4f}")
    
    return results


def run_phase1_calibration(output_dir: str = './results'):
    """
    Phase 1: Train all 9 architectures on vision task.
    This establishes the α = f(architecture) relationship.
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    architectures = [
        # Group 1: Depth variation
        'mlp_2layer', 'mlp_5layer', 'mlp_10layer',
        # Group 2: Width variation  
        'mlp_narrow', 'mlp_medium', 'mlp_wide',
        # Group 3: Architecture types
        'cnn_small', 'transformer_small', 'hybrid'
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    all_results = []
    
    for arch_name in architectures:
        results = run_single_experiment(arch_name, 'vision', device=device)
        all_results.append(results)
        
        # Save intermediate results
        output_file = Path(output_dir) / f'{arch_name}_vision.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Save combined results
    combined_file = Path(output_dir) / 'phase1_calibration.json'
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Phase 1 complete. Results saved to", output_dir)
    print(f"{'='*60}")
    
    return all_results


if __name__ == "__main__":
    # Run Phase 1 calibration experiments
    results = run_phase1_calibration()
    
    print("\nExperiments complete!")
    print("Next steps:")
    print("1. Analyze results to extract α for each architecture")
    print("2. Establish α = f(depth, width) relationship")
    print("3. Run Phase 2 cross-task validation")
