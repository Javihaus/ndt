"""
Dataset Downloader with Fallback Options
==========================================

Handles dataset downloads with multiple fallback mirrors and
creates synthetic datasets when downloads fail.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path
import urllib.request
import gzip
import shutil

def download_with_fallback(url_list, save_path):
    """Try multiple URLs until one works."""
    for url in url_list:
        try:
            print(f"Trying {url}...")
            urllib.request.urlretrieve(url, save_path)
            return True
        except Exception as e:
            print(f"Failed: {e}")
            continue
    return False


def create_synthetic_mnist(num_samples=10000, train=True):
    """
    Create synthetic MNIST-like dataset.

    Args:
        num_samples: Number of samples to generate
        train: Whether this is training set

    Returns:
        TensorDataset with images and labels
    """
    # Generate random grayscale images 28x28
    images = torch.randn(num_samples, 1, 28, 28) * 0.3 + 0.5
    images = torch.clamp(images, 0, 1)

    # Generate random labels (10 classes)
    labels = torch.randint(0, 10, (num_samples,))

    return TensorDataset(images, labels)


def create_synthetic_cifar10(num_samples=10000, train=True):
    """
    Create synthetic CIFAR10-like dataset.

    Args:
        num_samples: Number of samples to generate
        train: Whether this is training set

    Returns:
        TensorDataset with images and labels
    """
    # Generate random RGB images 32x32
    images = torch.randn(num_samples, 3, 32, 32) * 0.3 + 0.5
    images = torch.clamp(images, 0, 1)

    # Generate random labels (10 classes)
    labels = torch.randint(0, 10, (num_samples,))

    return TensorDataset(images, labels)


def create_synthetic_fashion_mnist(num_samples=10000, train=True):
    """Create synthetic Fashion-MNIST-like dataset."""
    return create_synthetic_mnist(num_samples, train)


def create_synthetic_svhn(num_samples=10000, split='train'):
    """Create synthetic SVHN-like dataset."""
    return create_synthetic_cifar10(num_samples, train=(split == 'train'))


def create_synthetic_cifar100(num_samples=10000, train=True):
    """Create synthetic CIFAR100-like dataset."""
    images = torch.randn(num_samples, 3, 32, 32) * 0.3 + 0.5
    images = torch.clamp(images, 0, 1)
    labels = torch.randint(0, 100, (num_samples,))
    return TensorDataset(images, labels)


def get_dataset(name, root='./data', train=True, download=True, transform=None, use_synthetic=False):
    """
    Get dataset with fallback to synthetic data.

    Args:
        name: Dataset name ('mnist', 'cifar10', 'fashion_mnist', 'svhn', 'cifar100')
        root: Root directory for data
        train: Whether to load training set
        download: Whether to try downloading
        transform: Transform to apply
        use_synthetic: Force use of synthetic data

    Returns:
        Dataset object
    """
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)

    if use_synthetic:
        print(f"Using synthetic {name} dataset")
        if name == 'mnist':
            return create_synthetic_mnist(10000 if train else 2000, train)
        elif name == 'fashion_mnist':
            return create_synthetic_fashion_mnist(10000 if train else 2000, train)
        elif name == 'cifar10':
            return create_synthetic_cifar10(10000 if train else 2000, train)
        elif name == 'svhn':
            return create_synthetic_svhn(10000 if train else 2000, 'train' if train else 'test')
        elif name == 'cifar100':
            return create_synthetic_cifar100(10000 if train else 2000, train)

    # Try downloading real dataset
    if download:
        try:
            if name == 'mnist':
                dataset = torchvision.datasets.MNIST(root=root, train=train, download=True, transform=transform)
                print(f"✓ Downloaded {name}")
                return dataset
            elif name == 'fashion_mnist':
                dataset = torchvision.datasets.FashionMNIST(root=root, train=train, download=True, transform=transform)
                print(f"✓ Downloaded {name}")
                return dataset
            elif name == 'cifar10':
                dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
                print(f"✓ Downloaded {name}")
                return dataset
            elif name == 'svhn':
                split = 'train' if train else 'test'
                dataset = torchvision.datasets.SVHN(root=root, split=split, download=True, transform=transform)
                print(f"✓ Downloaded {name}")
                return dataset
            elif name == 'cifar100':
                dataset = torchvision.datasets.CIFAR100(root=root, train=train, download=True, transform=transform)
                print(f"✓ Downloaded {name}")
                return dataset
        except Exception as e:
            print(f"✗ Failed to download {name}: {e}")
            print(f"Falling back to synthetic {name}")
            return get_dataset(name, root, train, False, transform, use_synthetic=True)

    # Check if data exists locally
    try:
        if name == 'mnist':
            return torchvision.datasets.MNIST(root=root, train=train, download=False, transform=transform)
        elif name == 'fashion_mnist':
            return torchvision.datasets.FashionMNIST(root=root, train=train, download=False, transform=transform)
        elif name == 'cifar10':
            return torchvision.datasets.CIFAR10(root=root, train=train, download=False, transform=transform)
        elif name == 'svhn':
            split = 'train' if train else 'test'
            return torchvision.datasets.SVHN(root=root, split=split, download=False, transform=transform)
        elif name == 'cifar100':
            return torchvision.datasets.CIFAR100(root=root, train=train, download=False, transform=transform)
    except Exception:
        print(f"No local {name} data found, using synthetic")
        return get_dataset(name, root, train, False, transform, use_synthetic=True)


if __name__ == '__main__':
    print("Testing dataset downloader...")

    for name in ['mnist', 'cifar10']:
        print(f"\nTesting {name}:")
        dataset = get_dataset(name, train=True, download=True)
        print(f"  Dataset size: {len(dataset)}")
        print(f"  Sample shape: {dataset[0][0].shape if hasattr(dataset[0][0], 'shape') else 'N/A'}")
