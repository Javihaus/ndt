"""
Multi-Modal Synthetic Dataset Generator
=========================================

Since real dataset downloads are blocked, this creates realistic synthetic
datasets for different architectures and modalities.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
import numpy as np
from typing import Tuple, List
import torchvision.transforms as transforms


# =============================================================================
# VISION DATASETS
# =============================================================================

class SyntheticImageDataset(Dataset):
    """
    Generate synthetic images with structure (not just noise).
    Supports multiple image sizes for different architectures.
    """
    def __init__(self, num_samples=10000, img_size=224, num_classes=10,
                 channels=3, pattern='structured'):
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_classes = num_classes
        self.channels = channels
        self.pattern = pattern

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Seed for reproducibility
        rng = np.random.RandomState(idx)

        # Generate label
        label = idx % self.num_classes

        # Generate image with structure based on label
        if self.pattern == 'structured':
            # Create images with different patterns per class
            img = self._generate_structured_image(label, rng)
        else:
            # Simple colored noise
            img = rng.randn(self.channels, self.img_size, self.img_size) * 0.3 + 0.5

        img = torch.FloatTensor(img).clamp(0, 1)
        return img, label

    def _generate_structured_image(self, label, rng):
        """Generate image with geometric patterns that vary by class."""
        img = np.zeros((self.channels, self.img_size, self.img_size))

        # Background color varies by class
        bg_color = label / self.num_classes
        img[:] = bg_color

        # Add geometric shapes based on label
        center_x, center_y = self.img_size // 2, self.img_size // 2
        radius = self.img_size // 4

        y, x = np.ogrid[:self.img_size, :self.img_size]

        if label % 4 == 0:  # Circle
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            img[:, mask] = 1 - bg_color
        elif label % 4 == 1:  # Square
            mask = (np.abs(x - center_x) <= radius) & (np.abs(y - center_y) <= radius)
            img[:, mask] = 1 - bg_color
        elif label % 4 == 2:  # Stripes
            mask = (x % 20) < 10
            img[:, mask] = 1 - bg_color
        else:  # Grid
            mask = ((x % 30) < 15) ^ ((y % 30) < 15)
            img[:, mask] = 1 - bg_color

        # Add noise
        img += rng.randn(*img.shape) * 0.1

        return img


# =============================================================================
# TEXT/SEQUENCE DATASETS
# =============================================================================

class SyntheticTextDataset(Dataset):
    """
    Generate synthetic text sequences for NLP transformers.
    Creates sequences with patterns that correlate with labels.
    """
    def __init__(self, num_samples=10000, seq_len=128, vocab_size=5000,
                 num_classes=10):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        rng = np.random.RandomState(idx)

        # Generate label
        label = idx % self.num_classes

        # Create sequence with bias toward certain tokens based on label
        # This creates learnable patterns
        sequence = []
        for i in range(self.seq_len):
            # Bias distribution based on label
            if i < self.seq_len // 2:
                # First half: strong class correlation
                token = (label * 100 + rng.randint(0, 100)) % self.vocab_size
            else:
                # Second half: weaker correlation
                if rng.rand() < 0.7:
                    token = (label * 50 + rng.randint(0, 200)) % self.vocab_size
                else:
                    token = rng.randint(0, self.vocab_size)
            sequence.append(token)

        return torch.LongTensor(sequence), label


class SyntheticSequenceDataset(Dataset):
    """
    Generate continuous-valued sequences for RNNs/LSTMs.
    """
    def __init__(self, num_samples=10000, seq_len=100, input_dim=10,
                 num_classes=10):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        rng = np.random.RandomState(idx)
        label = idx % self.num_classes

        # Generate sequence with sinusoidal patterns based on label
        t = np.linspace(0, 10, self.seq_len)
        sequence = np.zeros((self.seq_len, self.input_dim))

        for dim in range(self.input_dim):
            # Frequency varies by label
            freq = 0.5 + label * 0.1 + dim * 0.05
            phase = rng.rand() * 2 * np.pi
            sequence[:, dim] = np.sin(freq * t + phase) + rng.randn(self.seq_len) * 0.1

        return torch.FloatTensor(sequence), label


# =============================================================================
# REINFORCEMENT LEARNING DATASETS
# =============================================================================

class SyntheticRLDataset(Dataset):
    """
    Generate synthetic state-action trajectories for RL policy learning.
    """
    def __init__(self, num_samples=10000, state_dim=4, num_actions=4):
        self.num_samples = num_samples
        self.state_dim = state_dim
        self.num_actions = num_actions

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        rng = np.random.RandomState(idx)

        # Generate state
        state = rng.randn(self.state_dim)

        # Action correlated with state (e.g., largest state dimension)
        action = np.argmax(np.abs(state)) % self.num_actions

        return torch.FloatTensor(state), action


# =============================================================================
# DATASET FACTORY
# =============================================================================

def create_multimodal_dataset(dataset_name: str, train: bool = True,
                               transform=None, **kwargs):
    """
    Create synthetic datasets for different modalities.

    Args:
        dataset_name: Name of dataset type
        train: Whether this is training set
        transform: Optional transform (for compatibility)
        **kwargs: Additional arguments

    Returns:
        Dataset instance
    """
    num_samples = 10000 if train else 2000

    if dataset_name == 'imagenet_small':
        # Small ImageNet-like: 224x224 RGB for Vision Transformers
        return SyntheticImageDataset(
            num_samples=num_samples,
            img_size=224,
            num_classes=100,
            channels=3,
            pattern='structured'
        )

    elif dataset_name == 'imagenet_large':
        # Larger ImageNet-like
        return SyntheticImageDataset(
            num_samples=num_samples,
            img_size=384,
            num_classes=1000,
            channels=3,
            pattern='structured'
        )

    elif dataset_name == 'cifar_synthetic':
        # CIFAR-like: 32x32 RGB
        return SyntheticImageDataset(
            num_samples=num_samples,
            img_size=32,
            num_classes=10,
            channels=3,
            pattern='structured'
        )

    elif dataset_name == 'mnist_synthetic':
        # MNIST-like: 28x28 grayscale
        return SyntheticImageDataset(
            num_samples=num_samples,
            img_size=28,
            num_classes=10,
            channels=1,
            pattern='structured'
        )

    elif dataset_name == 'text_short':
        # Short text sequences (for BERT-like models)
        return SyntheticTextDataset(
            num_samples=num_samples,
            seq_len=128,
            vocab_size=30000,
            num_classes=4
        )

    elif dataset_name == 'text_long':
        # Long text sequences (for GPT-like models)
        return SyntheticTextDataset(
            num_samples=num_samples,
            seq_len=512,
            vocab_size=50000,
            num_classes=10
        )

    elif dataset_name == 'sequence':
        # Continuous sequences (for RNN/LSTM)
        return SyntheticSequenceDataset(
            num_samples=num_samples,
            seq_len=100,
            input_dim=10,
            num_classes=10
        )

    elif dataset_name == 'rl_states':
        # RL state-action pairs
        return SyntheticRLDataset(
            num_samples=num_samples,
            state_dim=4,
            num_actions=4
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_dataloader(dataset_name: str, batch_size: int = 64,
                   train: bool = True, num_workers: int = 0):
    """Create dataloader for a multimodal dataset."""
    dataset = create_multimodal_dataset(dataset_name, train=train)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers
    )


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing Multi-Modal Synthetic Datasets\n")

    datasets_to_test = [
        ('imagenet_small', 'Vision Transformers (224x224)'),
        ('imagenet_large', 'Large ViT (384x384)'),
        ('cifar_synthetic', 'CNNs/ResNets (32x32)'),
        ('mnist_synthetic', 'MLPs (28x28)'),
        ('text_short', 'BERT/Text Classification'),
        ('text_long', 'GPT/Language Modeling'),
        ('sequence', 'RNN/LSTM'),
        ('rl_states', 'RL Policy Networks'),
    ]

    for name, description in datasets_to_test:
        dataset = create_multimodal_dataset(name, train=True)
        sample, label = dataset[0]
        print(f"✓ {name:20s} - {description}")
        print(f"  Shape: {sample.shape}, Label: {label}, Size: {len(dataset)}")
