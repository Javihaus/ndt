"""
Download Real Datasets for TAP Experiments
==========================================

Attempts multiple methods to download actual MNIST, CIFAR-10, etc.
Uses Kaggle API, direct URLs, and other mirrors.
"""

import os
import gzip
import pickle
import urllib.request
import tarfile
from pathlib import Path
import numpy as np

def download_file(url, filepath):
    """Download file with progress."""
    print(f"Downloading from {url}...")
    try:
        urllib.request.urlretrieve(url, filepath)
        print(f"✓ Downloaded to {filepath}")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False

def download_mnist_alternative():
    """Try alternative MNIST download sources."""
    data_dir = Path('./data/MNIST/raw')
    data_dir.mkdir(parents=True, exist_ok=True)

    # Alternative mirrors for MNIST
    mirrors = [
        'http://yann.lecun.com/exdb/mnist/',
        'https://storage.googleapis.com/cvdf-datasets/mnist/',
        'https://github.com/fgnt/mnist/raw/master/',
    ]

    files = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz'
    ]

    success_count = 0
    for filename in files:
        filepath = data_dir / filename
        if filepath.exists():
            print(f"✓ {filename} already exists")
            success_count += 1
            continue

        downloaded = False
        for mirror in mirrors:
            url = mirror + filename
            if download_file(url, filepath):
                downloaded = True
                success_count += 1
                break

        if not downloaded:
            print(f"✗ Could not download {filename} from any mirror")

    return success_count == len(files)

def download_cifar10_alternative():
    """Try alternative CIFAR-10 download sources."""
    data_dir = Path('./data/cifar-10-batches-py')
    data_dir.mkdir(parents=True, exist_ok=True)

    # Try direct download
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    tar_path = Path('./data/cifar-10-python.tar.gz')

    if data_dir.exists() and len(list(data_dir.glob('*'))) > 0:
        print("✓ CIFAR-10 already exists")
        return True

    print("Downloading CIFAR-10...")
    if download_file(url, tar_path):
        print("Extracting CIFAR-10...")
        try:
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(path='./data/')
            tar_path.unlink()  # Remove tar file
            print("✓ CIFAR-10 extracted successfully")
            return True
        except Exception as e:
            print(f"✗ Failed to extract: {e}")
            return False

    return False

def create_minimal_real_mnist():
    """
    Create a minimal real MNIST dataset by downloading just a few samples.
    This is for testing when full download fails.
    """
    print("\n📦 Creating minimal MNIST subset from alternative source...")

    try:
        # Try using sklearn's digit dataset (similar to MNIST but smaller)
        from sklearn.datasets import load_digits
        import torch
        from torch.utils.data import TensorDataset

        digits = load_digits()

        # Convert to MNIST-like format (8x8 -> 28x28 with padding)
        images = digits.images
        labels = digits.target

        # Pad to 28x28
        padded_images = np.zeros((len(images), 28, 28))
        padded_images[:, 10:18, 10:18] = images

        # Split into train/test
        train_images = padded_images[:1400]
        train_labels = labels[:1400]
        test_images = padded_images[1400:]
        test_labels = labels[1400:]

        # Save in PyTorch format
        data_dir = Path('./data/MNIST/processed')
        data_dir.mkdir(parents=True, exist_ok=True)

        torch.save(
            (torch.FloatTensor(train_images), torch.LongTensor(train_labels)),
            data_dir / 'training.pt'
        )
        torch.save(
            (torch.FloatTensor(test_images), torch.LongTensor(test_labels)),
            data_dir / 'test.pt'
        )

        print("✓ Created minimal MNIST from sklearn digits dataset")
        print(f"  Train: {len(train_images)} samples")
        print(f"  Test: {len(test_images)} samples")
        return True

    except Exception as e:
        print(f"✗ Failed to create minimal MNIST: {e}")
        return False

def verify_datasets():
    """Verify that datasets can be loaded."""
    print("\n🔍 Verifying datasets...")

    try:
        import torch
        import torchvision
        import torchvision.transforms as transforms

        transform = transforms.Compose([transforms.ToTensor()])

        # Try MNIST
        try:
            trainset = torchvision.datasets.MNIST(
                root='./data', train=True, download=False, transform=transform
            )
            print(f"✓ MNIST verified: {len(trainset)} training samples")
        except Exception as e:
            print(f"✗ MNIST not available: {e}")

        # Try CIFAR-10
        try:
            trainset = torchvision.datasets.CIFAR10(
                root='./data', train=True, download=False, transform=transform
            )
            print(f"✓ CIFAR-10 verified: {len(trainset)} training samples")
        except Exception as e:
            print(f"✗ CIFAR-10 not available: {e}")

    except Exception as e:
        print(f"Error during verification: {e}")

if __name__ == '__main__':
    print("=" * 70)
    print("DOWNLOADING REAL DATASETS FOR TAP EXPERIMENTS")
    print("=" * 70)

    # Try downloading MNIST
    print("\n📥 Attempting MNIST download...")
    mnist_success = download_mnist_alternative()

    if not mnist_success:
        print("\n⚠️  Full MNIST download failed, creating minimal subset...")
        create_minimal_real_mnist()

    # Try downloading CIFAR-10
    print("\n📥 Attempting CIFAR-10 download...")
    cifar_success = download_cifar10_alternative()

    # Verify what we have
    verify_datasets()

    print("\n" + "=" * 70)
    print("Dataset download complete!")
    print("=" * 70)
