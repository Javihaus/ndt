"""Dimensionality estimators for neural network representations.

This module provides four key dimensionality metrics:
1. Stable Rank: Ratio of squared Frobenius norm to squared spectral norm
2. Participation Ratio: Inverse of sum of squared normalized singular values
3. Cumulative Energy 90: Number of components needed for 90% variance
4. Nuclear Norm Ratio: Nuclear norm normalized by spectral norm
"""

from typing import Tuple

import torch


def stable_rank(matrix: torch.Tensor, eps: float = 1e-10) -> float:
    """Compute stable rank of a matrix.

    Stable rank is the ratio of squared Frobenius norm to squared spectral norm.
    It estimates the effective number of linearly independent dimensions.

    Args:
        matrix: Input matrix of shape (m, n)
        eps: Small constant for numerical stability (default: 1e-10)

    Returns:
        Stable rank value (float)

    Raises:
        ValueError: If matrix is not 2D or contains NaN/Inf values

    Example:
        >>> import torch
        >>> A = torch.randn(100, 50)
        >>> sr = stable_rank(A)
        >>> print(f"Stable rank: {sr:.2f}")

    References:
        Vershynin, R. (2010). Introduction to the non-asymptotic analysis
        of random matrices. arXiv preprint arXiv:1011.3027.
    """
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got {matrix.ndim}D")

    if torch.isnan(matrix).any() or torch.isinf(matrix).any():
        raise ValueError("Matrix contains NaN or Inf values")

    with torch.no_grad():
        frobenius_norm = torch.norm(matrix, p="fro")
        # Spectral norm is the largest singular value
        _, singular_values, _ = torch.linalg.svd(matrix, full_matrices=False)
        spectral_norm = singular_values[0]

        result = (frobenius_norm**2) / (spectral_norm**2 + eps)
        return result.item()


def participation_ratio(matrix: torch.Tensor, eps: float = 1e-10) -> float:
    """Compute participation ratio from singular values.

    Participation ratio is the inverse of the sum of squared normalized
    singular values. It measures the effective dimensionality of the
    representation space.

    Args:
        matrix: Input matrix of shape (m, n)
        eps: Small constant for numerical stability (default: 1e-10)

    Returns:
        Participation ratio value (float)

    Raises:
        ValueError: If matrix is not 2D or SVD computation fails

    Example:
        >>> import torch
        >>> A = torch.randn(100, 50)
        >>> pr = participation_ratio(A)
        >>> print(f"Participation ratio: {pr:.2f}")

    References:
        Grassberger, P., & Procaccia, I. (1983). Measuring the strangeness
        of strange attractors. Physica D: Nonlinear Phenomena, 9(1-2), 189-208.
    """
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got {matrix.ndim}D")

    with torch.no_grad():
        try:
            _, singular_values, _ = torch.linalg.svd(matrix, full_matrices=False)
        except RuntimeError as e:
            raise ValueError(f"SVD computation failed: {e}")

        # Normalize singular values
        sv_normalized = singular_values / (torch.sum(singular_values) + eps)

        # Compute participation ratio
        pr = 1.0 / (torch.sum(sv_normalized**2) + eps)
        return pr.item()


def cumulative_energy_90(matrix: torch.Tensor, threshold: float = 0.90) -> int:
    """Compute number of components needed for 90% cumulative energy.

    This metric counts how many principal components are needed to capture
    a given percentage (default 90%) of the total variance.

    Args:
        matrix: Input matrix of shape (m, n)
        threshold: Energy threshold (default: 0.90 for 90%)

    Returns:
        Number of components needed (int)

    Raises:
        ValueError: If matrix is not 2D, threshold invalid, or SVD fails

    Example:
        >>> import torch
        >>> A = torch.randn(100, 50)
        >>> n_components = cumulative_energy_90(A)
        >>> print(f"Components for 90% energy: {n_components}")

    References:
        Jolliffe, I. T. (2002). Principal component analysis (2nd ed.).
        Springer.
    """
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got {matrix.ndim}D")

    if not 0 < threshold <= 1:
        raise ValueError(f"Threshold must be in (0, 1], got {threshold}")

    with torch.no_grad():
        try:
            _, singular_values, _ = torch.linalg.svd(matrix, full_matrices=False)
        except RuntimeError as e:
            raise ValueError(f"SVD computation failed: {e}")

        # Compute explained variance
        explained_variance = singular_values**2
        total_variance = torch.sum(explained_variance)

        # Compute cumulative explained variance ratio
        cumsum = torch.cumsum(explained_variance, dim=0)
        cumsum_ratio = cumsum / total_variance

        # Find number of components for threshold
        n_components = torch.searchsorted(cumsum_ratio, threshold).item() + 1
        return min(n_components, len(singular_values))


def nuclear_norm_ratio(matrix: torch.Tensor, eps: float = 1e-10) -> float:
    """Compute nuclear norm ratio (nuclear norm / spectral norm).

    This metric normalizes the nuclear norm (sum of singular values) by
    the spectral norm (largest singular value), providing a normalized
    measure of rank.

    Args:
        matrix: Input matrix of shape (m, n)
        eps: Small constant for numerical stability (default: 1e-10)

    Returns:
        Nuclear norm ratio value (float)

    Raises:
        ValueError: If matrix is not 2D or SVD computation fails

    Example:
        >>> import torch
        >>> A = torch.randn(100, 50)
        >>> nnr = nuclear_norm_ratio(A)
        >>> print(f"Nuclear norm ratio: {nnr:.2f}")

    References:
        Candès, E. J., & Recht, B. (2009). Exact matrix completion via
        convex optimization. Foundations of Computational mathematics, 9(6), 717-772.
    """
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got {matrix.ndim}D")

    with torch.no_grad():
        try:
            _, singular_values, _ = torch.linalg.svd(matrix, full_matrices=False)
        except RuntimeError as e:
            raise ValueError(f"SVD computation failed: {e}")

        nuclear_norm = torch.sum(singular_values)
        spectral_norm = singular_values[0]  # Largest singular value

        ratio = nuclear_norm / (spectral_norm + eps)
        return ratio.item()


def compute_all_metrics(
    matrix: torch.Tensor, eps: float = 1e-10
) -> Tuple[float, float, int, float]:
    """Compute all four dimensionality metrics efficiently.

    This function computes SVD once and reuses it for all metrics,
    improving efficiency when all metrics are needed.

    Args:
        matrix: Input matrix of shape (m, n)
        eps: Small constant for numerical stability (default: 1e-10)

    Returns:
        Tuple of (stable_rank, participation_ratio, cumulative_90, nuclear_norm_ratio)

    Raises:
        ValueError: If matrix is not 2D or computations fail

    Example:
        >>> import torch
        >>> A = torch.randn(100, 50)
        >>> sr, pr, ce90, nnr = compute_all_metrics(A)
        >>> print(f"Metrics: SR={sr:.2f}, PR={pr:.2f}, CE90={ce90}, NNR={nnr:.2f}")
    """
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got {matrix.ndim}D")

    with torch.no_grad():
        # Compute SVD once for all metrics
        try:
            _, singular_values, _ = torch.linalg.svd(matrix, full_matrices=False)
        except RuntimeError as e:
            raise ValueError(f"SVD computation failed: {e}")

        # Compute norms for stable rank
        frobenius_norm = torch.norm(matrix, p="fro")
        spectral_norm = singular_values[0]  # Largest singular value
        sr = ((frobenius_norm**2) / (spectral_norm**2 + eps)).item()

        # Participation ratio
        sv_normalized = singular_values / (torch.sum(singular_values) + eps)
        pr = (1.0 / (torch.sum(sv_normalized**2) + eps)).item()

        # Cumulative energy 90%
        explained_variance = singular_values**2
        total_variance = torch.sum(explained_variance)
        cumsum = torch.cumsum(explained_variance, dim=0)
        cumsum_ratio = cumsum / total_variance
        ce90 = torch.searchsorted(cumsum_ratio, 0.90).item() + 1
        ce90 = min(ce90, len(singular_values))

        # Nuclear norm ratio
        nuclear_norm = torch.sum(singular_values)
        nnr = (nuclear_norm / (singular_values[0] + eps)).item()

        return sr, pr, ce90, nnr
