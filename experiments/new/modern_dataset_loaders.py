"""
Modern Dataset Loaders for TAP Validation
==========================================

Loads real datasets from experiments/new/data/
Supports: CIFAR-10, ImageNet, GLUE, Conceptual Captions
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path
import pickle
import pandas as pd
from PIL import Image
import json

# Base data directory
DATA_DIR = Path(__file__).parent / 'data'


# =============================================================================
# CIFAR-10 LOADER
# =============================================================================

class CIFAR10Dataset(Dataset):
    """Load CIFAR-10 from pickle files."""

    def __init__(self, root=DATA_DIR / 'cifar-10-batches-py', train=True, transform=None):
        self.root = Path(root)
        self.train = train
        self.transform = transform

        # Load data
        if train:
            self.data = []
            self.labels = []
            for i in range(1, 6):
                batch_file = self.root / f'data_batch_{i}'
                with open(batch_file, 'rb') as f:
                    batch = pickle.load(f, encoding='bytes')
                    self.data.append(batch[b'data'])
                    self.labels.extend(batch[b'labels'])
            self.data = np.concatenate(self.data, axis=0)
        else:
            batch_file = self.root / 'test_batch'
            with open(batch_file, 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
                self.data = batch[b'data']
                self.labels = batch[b'labels']

        # Reshape to images (N, 32, 32, 3)
        self.data = self.data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img, label = self.data[idx], self.labels[idx]
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        return img, label


# =============================================================================
# IMAGENET PARQUET LOADER
# =============================================================================

class ImageNetParquetDataset(Dataset):
    """Load ImageNet from parquet file."""

    def __init__(self, parquet_path=DATA_DIR / 'imagenet' / 'train-00001-of-00021.parquet',
                 transform=None, max_samples=None):
        self.parquet_path = Path(parquet_path)
        self.transform = transform

        # Load parquet
        if self.parquet_path.exists():
            self.df = pd.read_parquet(self.parquet_path)
            if max_samples:
                self.df = self.df.head(max_samples)
            print(f"✓ Loaded ImageNet: {len(self.df)} images from {parquet_path.name}")
        else:
            print(f"✗ ImageNet not found at {parquet_path}")
            # Create empty dataset
            self.df = pd.DataFrame({'image': [], 'label': []})

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Handle image (might be bytes or PIL Image depending on parquet format)
        if isinstance(row['image'], bytes):
            img = Image.open(io.BytesIO(row['image'])).convert('RGB')
        else:
            img = row['image']
            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.array(img))

        label = int(row['label']) if 'label' in row else 0

        if self.transform:
            img = self.transform(img)
        else:
            # Default: resize to 224x224
            img = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])(img)

        return img, label


# =============================================================================
# GLUE (MNLI) TEXT LOADER
# =============================================================================

class GLUEDataset(Dataset):
    """Load GLUE/MNLI text classification dataset."""

    def __init__(self, parquet_path=DATA_DIR / 'glue' / 'mnli' / 'train-00000-of-00001.parquet',
                 max_length=128, vocab_size=30000, max_samples=None):
        self.parquet_path = Path(parquet_path)
        self.max_length = max_length
        self.vocab_size = vocab_size

        # Load parquet
        if self.parquet_path.exists():
            self.df = pd.read_parquet(self.parquet_path)
            if max_samples:
                self.df = self.df.head(max_samples)
            print(f"✓ Loaded GLUE/MNLI: {len(self.df)} samples")
        else:
            print(f"✗ GLUE not found at {parquet_path}")
            self.df = pd.DataFrame({'premise': [], 'hypothesis': [], 'label': []})

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Simple tokenization (hash-based)
        text = f"{row.get('premise', '')} {row.get('hypothesis', '')}"
        tokens = self._simple_tokenize(text, self.max_length, self.vocab_size)

        label = int(row.get('label', 0))

        return torch.LongTensor(tokens), label

    @staticmethod
    def _simple_tokenize(text, max_length, vocab_size):
        """Simple hash-based tokenization."""
        words = text.lower().split()[:max_length]
        tokens = [hash(word) % vocab_size for word in words]
        # Pad to max_length
        tokens = tokens + [0] * (max_length - len(tokens))
        return tokens[:max_length]


# =============================================================================
# CONCEPTUAL CAPTIONS (Vision-Language) LOADER
# =============================================================================

class ConceptualCaptionsDataset(Dataset):
    """Load Conceptual Captions image-text pairs."""

    def __init__(self, root=DATA_DIR / 'conceptual_captions',
                 transform=None, max_samples=None):
        self.root = Path(root)
        self.transform = transform

        # Try to load from various possible formats
        self.data = []

        # Option 1: JSON/JSONL format
        json_files = list(self.root.glob('*.json*'))
        if json_files:
            for json_file in json_files[:1]:  # Use first file
                with open(json_file) as f:
                    if json_file.suffix == '.jsonl':
                        self.data = [json.loads(line) for line in f]
                    else:
                        self.data = json.load(f)
                print(f"✓ Loaded Conceptual Captions: {len(self.data)} pairs")
                break

        # Option 2: TSV format (URL, caption)
        tsv_files = list(self.root.glob('*.tsv'))
        if tsv_files and not self.data:
            df = pd.read_csv(tsv_files[0], sep='\t', header=None, names=['url', 'caption'])
            self.data = df.to_dict('records')
            print(f"✓ Loaded Conceptual Captions: {len(self.data)} pairs")

        if not self.data:
            print(f"✗ Conceptual Captions not found in {root}")

        if max_samples and self.data:
            self.data = self.data[:max_samples]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # For now, return dummy image and tokenized caption
        # In real implementation, would download image from URL
        img = torch.randn(3, 224, 224)  # Placeholder

        # Simple caption tokenization
        caption = item.get('caption', '')
        tokens = [hash(word) % 30000 for word in caption.lower().split()[:128]]
        tokens = tokens + [0] * (128 - len(tokens))

        return img, torch.LongTensor(tokens[:128])


# =============================================================================
# DATASET FACTORY
# =============================================================================

def get_dataset(dataset_name, train=True, transform=None, **kwargs):
    """
    Get dataset by name.

    Args:
        dataset_name: Name of dataset
        train: Whether to load training set
        transform: Optional transform
        **kwargs: Additional arguments

    Returns:
        Dataset instance
    """

    if dataset_name == 'cifar10':
        return CIFAR10Dataset(train=train, transform=transform)

    elif dataset_name == 'imagenet':
        # For training, use parquet file
        # For validation, could use a different file
        return ImageNetParquetDataset(transform=transform, **kwargs)

    elif dataset_name == 'glue_mnli':
        return GLUEDataset(**kwargs)

    elif dataset_name == 'conceptual_captions':
        return ConceptualCaptionsDataset(transform=transform, **kwargs)

    elif dataset_name == 'mnist':
        # Fallback to torchvision MNIST
        import torchvision.datasets as datasets
        return datasets.MNIST(root=DATA_DIR, train=train, download=True, transform=transform)

    elif dataset_name == 'fashion_mnist':
        import torchvision.datasets as datasets
        return datasets.FashionMNIST(root=DATA_DIR, train=train, download=True, transform=transform)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_dataloader(dataset_name, batch_size=64, train=True, num_workers=0, **kwargs):
    """Create dataloader for a dataset."""

    # Set default transforms based on dataset
    if 'transform' not in kwargs:
        if dataset_name in ['cifar10']:
            kwargs['transform'] = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        elif dataset_name in ['imagenet']:
            kwargs['transform'] = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif dataset_name in ['mnist', 'fashion_mnist']:
            kwargs['transform'] = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

    dataset = get_dataset(dataset_name, train=train, **kwargs)

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
    print("Testing Modern Dataset Loaders\n")

    # Test CIFAR-10
    print("1. Testing CIFAR-10:")
    try:
        loader = get_dataloader('cifar10', batch_size=32, train=True)
        img, label = next(iter(loader))
        print(f"   ✓ CIFAR-10: batch shape {img.shape}, labels {label.shape}")
    except Exception as e:
        print(f"   ✗ CIFAR-10: {e}")

    # Test ImageNet
    print("\n2. Testing ImageNet:")
    try:
        loader = get_dataloader('imagenet', batch_size=16, train=True, max_samples=100)
        img, label = next(iter(loader))
        print(f"   ✓ ImageNet: batch shape {img.shape}, labels {label.shape}")
    except Exception as e:
        print(f"   ✗ ImageNet: {e}")

    # Test GLUE
    print("\n3. Testing GLUE/MNLI:")
    try:
        loader = get_dataloader('glue_mnli', batch_size=32, max_samples=100)
        tokens, label = next(iter(loader))
        print(f"   ✓ GLUE: batch shape {tokens.shape}, labels {label.shape}")
    except Exception as e:
        print(f"   ✗ GLUE: {e}")

    # Test Conceptual Captions
    print("\n4. Testing Conceptual Captions:")
    try:
        dataset = get_dataset('conceptual_captions', max_samples=100)
        print(f"   ✓ Conceptual Captions: {len(dataset)} samples")
    except Exception as e:
        print(f"   ✗ Conceptual Captions: {e}")
