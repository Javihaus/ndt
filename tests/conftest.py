"""Pytest fixtures for testing neural dimensionality tracker."""

import pytest
import torch
import torch.nn as nn


@pytest.fixture
def small_mlp():
    """Create a small MLP for testing."""
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 15),
        nn.ReLU(),
        nn.Linear(15, 5),
    )


@pytest.fixture
def simple_cnn():
    """Create a simple CNN for testing."""
    return nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 8 * 8, 10),
    )


@pytest.fixture
def sample_batch():
    """Create a sample batch of data."""
    return torch.randn(32, 10)


@pytest.fixture
def sample_image_batch():
    """Create a sample batch of images."""
    return torch.randn(8, 3, 32, 32)


@pytest.fixture
def random_matrix():
    """Create a random 2D matrix."""
    return torch.randn(100, 50)


@pytest.fixture
def low_rank_matrix():
    """Create a low-rank matrix for testing."""
    # Create rank-5 matrix
    U = torch.randn(100, 5)
    V = torch.randn(5, 50)
    return U @ V


@pytest.fixture
def identity_matrix():
    """Create an identity-like matrix."""
    return torch.eye(50, 50)


@pytest.fixture
def near_singular_matrix():
    """Create a near-singular matrix."""
    matrix = torch.randn(50, 50)
    # Make it near-singular by setting small singular values to near-zero
    U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
    S[40:] = 1e-8  # Make last 10 singular values tiny
    return U @ torch.diag(S) @ Vh
