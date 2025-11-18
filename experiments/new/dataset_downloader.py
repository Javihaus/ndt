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

# Fix MNIST download URLs - use Google Cloud Storage mirror
# The default URLs (yann.lecun.com and ossci-datasets.s3.amazonaws.com) often fail with 403
torchvision.datasets.MNIST.resources = [
    ("https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
    ("https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
    ("https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
    ("https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
]

# Fix Fashion-MNIST URLs similarly
torchvision.datasets.FashionMNIST.resources = [
    ("https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte.gz", "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
    ("https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-labels-idx1-ubyte.gz", "25c81989df183df01b3e8a0aad5dffbe"),
    ("https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-images-idx3-ubyte.gz", "bef4ecab320f06d8554ea6380940ec79"),
    ("https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-labels-idx1-ubyte.gz", "bb300cfdad3c16e7a12a480ee83cd310"),
]


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

    # First try loading from existing data (no download)
    try:
        if name == 'mnist':
            dataset = torchvision.datasets.MNIST(root=root, train=train, download=False, transform=transform)
            return dataset
        elif name == 'fashion_mnist':
            dataset = torchvision.datasets.FashionMNIST(root=root, train=train, download=False, transform=transform)
            return dataset
        elif name == 'qmnist':
            dataset = torchvision.datasets.QMNIST(root=root, train='train' if train else 'test', download=False, transform=transform)
            return dataset
        else:
            raise RuntimeError("Dataset '{}' not supported. Use: mnist, fashion_mnist, qmnist".format(name))
    except Exception as e:
        if not download:
            raise RuntimeError("No local {} data found: {}".format(name, str(e)))
        # Fall through to download attempt

    # Try downloading real dataset
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
    except Exception as e:
        raise RuntimeError("Failed to download {}: {}".format(name, str(e)))
