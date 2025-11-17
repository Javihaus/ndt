"""
Dataset Downloader - REAL DATA ONLY
====================================

NO SYNTHETIC DATA. If download fails, raise error.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path


def get_dataset(name, root='./data', train=True, download=True, transform=None, use_synthetic=False):
    """
    Get dataset - REAL DATA ONLY, NO FALLBACK.

    Args:
        name: Dataset name ('mnist', 'fashion_mnist', 'qmnist')
        root: Root directory for data
        train: Whether to load training set
        download: Whether to try downloading
        transform: Transform to apply

    Returns:
        Dataset object

    Raises:
        RuntimeError: If dataset cannot be downloaded
    """
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)

    # NO SYNTHETIC DATA - if use_synthetic is requested, fail immediately
    if use_synthetic:
        raise RuntimeError("SYNTHETIC DATA NOT ALLOWED - use real datasets only")

    # Try downloading real dataset
    if download:
        try:
            if name == 'mnist':
                dataset = torchvision.datasets.MNIST(root=root, train=train, download=True, transform=transform)
                print("Downloaded MNIST successfully")
                return dataset
            elif name == 'fashion_mnist':
                dataset = torchvision.datasets.FashionMNIST(root=root, train=train, download=True, transform=transform)
                print("Downloaded Fashion-MNIST successfully")
                return dataset
            elif name == 'qmnist':
                dataset = torchvision.datasets.QMNIST(root=root, train='train' if train else 'test', download=True, transform=transform)
                print("Downloaded QMNIST successfully")
                return dataset
            else:
                raise RuntimeError("Dataset '{}' not supported. Use: mnist, fashion_mnist, qmnist".format(name))
        except Exception as e:
            raise RuntimeError("Failed to download {}: {}".format(name, str(e)))

    # Check if data exists locally
    try:
        if name == 'mnist':
            return torchvision.datasets.MNIST(root=root, train=train, download=False, transform=transform)
        elif name == 'fashion_mnist':
            return torchvision.datasets.FashionMNIST(root=root, train=train, download=False, transform=transform)
        elif name == 'qmnist':
            return torchvision.datasets.QMNIST(root=root, train='train' if train else 'test', download=False, transform=transform)
        else:
            raise RuntimeError("Dataset '{}' not supported".format(name))
    except Exception as e:
        raise RuntimeError("No local {} data found and download failed: {}".format(name, str(e)))
