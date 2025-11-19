"""Activation analysis tools for mechanistic interpretability.

This module provides tools for analyzing neural network activations:
- PCA for understanding principal directions
- Clustering for identifying activation patterns
- Manifold visualization (t-SNE, UMAP)
- Geometry analysis (effective dimensionality, singular value distributions)
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


class ActivationAnalyzer:
    """Analyzes neural network activations for mechanistic interpretability.

    This class provides tools to understand what representations are learned
    at different layers and training stages, particularly focusing on
    critical moments identified by dimensionality jumps.

    Attributes:
        results: Dictionary storing analysis results

    Example:
        >>> analyzer = ActivationAnalyzer()
        >>> # Capture activations from model
        >>> activations = capture.get_all_activations()
        >>> # Flatten for analysis
        >>> flat_acts = activations['layer_0'].view(batch_size, -1).cpu().numpy()
        >>> # Run PCA analysis
        >>> pca_results = analyzer.pca_analysis(flat_acts, n_components=10)
        >>> # Visualize
        >>> analyzer.plot_pca_variance(pca_results)
    """

    def __init__(self) -> None:
        """Initialize the activation analyzer."""
        self.results: Dict[str, Any] = {}

    def flatten_activation(self, activation: torch.Tensor) -> np.ndarray:
        """Flatten activation tensor to 2D array for analysis.

        Args:
            activation: Activation tensor of shape (batch, ...)

        Returns:
            Flattened array of shape (batch, features)
        """
        if isinstance(activation, torch.Tensor):
            activation = activation.detach().cpu().numpy()

        batch_size = activation.shape[0]
        return activation.reshape(batch_size, -1)

    def pca_analysis(
        self,
        activations: np.ndarray,
        n_components: Optional[int] = None,
        return_transformed: bool = True
    ) -> Dict[str, Any]:
        """Perform PCA analysis on activations.

        Analyzes the principal components of the activation space to understand
        the main directions of variation in representations.

        Args:
            activations: Array of shape (n_samples, n_features)
            n_components: Number of components to compute (default: min(n_samples, n_features))
            return_transformed: Whether to include transformed data

        Returns:
            Dictionary containing:
                - 'explained_variance_ratio': Variance explained by each component
                - 'cumulative_variance': Cumulative variance explained
                - 'components': Principal component vectors
                - 'singular_values': Singular values
                - 'n_components_90': Components needed for 90% variance
                - 'n_components_95': Components needed for 95% variance
                - 'transformed': Transformed data (if return_transformed=True)
        """
        if n_components is None:
            n_components = min(activations.shape)

        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(activations)

        cumulative_var = np.cumsum(pca.explained_variance_ratio_)
        n_90 = np.searchsorted(cumulative_var, 0.90) + 1
        n_95 = np.searchsorted(cumulative_var, 0.95) + 1

        results = {
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': cumulative_var,
            'components': pca.components_,
            'singular_values': pca.singular_values_,
            'n_components_90': min(n_90, n_components),
            'n_components_95': min(n_95, n_components),
            'mean': pca.mean_,
            'pca_object': pca
        }

        if return_transformed:
            results['transformed'] = transformed

        return results

    def cluster_analysis(
        self,
        activations: np.ndarray,
        method: str = 'kmeans',
        n_clusters: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """Perform clustering analysis on activations.

        Identifies groups of similar activation patterns, useful for
        understanding how different inputs are represented.

        Args:
            activations: Array of shape (n_samples, n_features)
            method: Clustering method ('kmeans' or 'dbscan')
            n_clusters: Number of clusters (for kmeans)
            **kwargs: Additional arguments for clustering algorithm

        Returns:
            Dictionary containing:
                - 'labels': Cluster assignments
                - 'n_clusters': Number of clusters found
                - 'silhouette_score': Silhouette score if applicable
                - 'cluster_centers': Cluster centroids (kmeans only)
                - 'inertia': Within-cluster sum of squares (kmeans only)
        """
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, **kwargs)
            labels = clusterer.fit_predict(activations)

            results = {
                'labels': labels,
                'n_clusters': n_clusters,
                'cluster_centers': clusterer.cluster_centers_,
                'inertia': clusterer.inertia_
            }

        elif method == 'dbscan':
            eps = kwargs.get('eps', 0.5)
            min_samples = kwargs.get('min_samples', 5)
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
            labels = clusterer.fit_predict(activations)

            n_found = len(set(labels)) - (1 if -1 in labels else 0)
            results = {
                'labels': labels,
                'n_clusters': n_found,
                'n_noise': np.sum(labels == -1)
            }
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        # Compute silhouette score if more than 1 cluster
        unique_labels = set(labels)
        if len(unique_labels) > 1 and -1 not in unique_labels:
            results['silhouette_score'] = silhouette_score(activations, labels)
        elif len(unique_labels) > 2:  # Has noise but also clusters
            mask = labels != -1
            if mask.sum() > 1:
                results['silhouette_score'] = silhouette_score(
                    activations[mask], labels[mask]
                )

        return results

    def manifold_embedding(
        self,
        activations: np.ndarray,
        method: str = 'tsne',
        n_components: int = 2,
        **kwargs
    ) -> Dict[str, Any]:
        """Compute low-dimensional manifold embedding.

        Reduces high-dimensional activations to 2D/3D for visualization,
        preserving local structure.

        Args:
            activations: Array of shape (n_samples, n_features)
            method: Embedding method ('tsne' or 'umap')
            n_components: Output dimensions (2 or 3)
            **kwargs: Additional arguments for embedding algorithm

        Returns:
            Dictionary containing:
                - 'embedding': Low-dimensional coordinates
                - 'method': Method used
                - 'params': Parameters used
        """
        if method == 'tsne':
            perplexity = kwargs.get('perplexity', min(30, activations.shape[0] - 1))
            tsne = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                random_state=42,
                **{k: v for k, v in kwargs.items() if k != 'perplexity'}
            )
            embedding = tsne.fit_transform(activations)

            results = {
                'embedding': embedding,
                'method': 'tsne',
                'params': {'perplexity': perplexity, 'n_components': n_components}
            }

        elif method == 'umap':
            if not HAS_UMAP:
                raise ImportError("UMAP not installed. Install with: pip install umap-learn")

            n_neighbors = kwargs.get('n_neighbors', min(15, activations.shape[0] - 1))
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                random_state=42,
                **{k: v for k, v in kwargs.items() if k != 'n_neighbors'}
            )
            embedding = reducer.fit_transform(activations)

            results = {
                'embedding': embedding,
                'method': 'umap',
                'params': {'n_neighbors': n_neighbors, 'n_components': n_components}
            }
        else:
            raise ValueError(f"Unknown embedding method: {method}")

        return results

    def singular_value_analysis(
        self,
        activations: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze singular value distribution of activations.

        Computes detailed statistics about the singular value spectrum,
        useful for understanding representation geometry.

        Args:
            activations: Array of shape (n_samples, n_features)

        Returns:
            Dictionary containing:
                - 'singular_values': All singular values
                - 'normalized_sv': Normalized singular values
                - 'stable_rank': Stable rank
                - 'participation_ratio': Participation ratio
                - 'entropy': Spectral entropy
                - 'condition_number': Condition number
        """
        U, S, Vt = np.linalg.svd(activations, full_matrices=False)

        # Normalized singular values
        S_norm = S / S.sum()

        # Stable rank
        stable_rank = (S ** 2).sum() / (S[0] ** 2) if S[0] > 0 else 0

        # Participation ratio
        participation_ratio = 1.0 / (S_norm ** 2).sum() if S_norm.sum() > 0 else 0

        # Spectral entropy
        S_norm_nonzero = S_norm[S_norm > 0]
        entropy = -np.sum(S_norm_nonzero * np.log(S_norm_nonzero))

        # Condition number
        condition_number = S[0] / S[-1] if S[-1] > 0 else np.inf

        return {
            'singular_values': S,
            'normalized_sv': S_norm,
            'stable_rank': stable_rank,
            'participation_ratio': participation_ratio,
            'spectral_entropy': entropy,
            'condition_number': condition_number
        }

    def compare_activations(
        self,
        activations_before: np.ndarray,
        activations_after: np.ndarray
    ) -> Dict[str, Any]:
        """Compare activations before and after a critical moment (e.g., jump).

        Analyzes how representations change during dimensionality transitions.

        Args:
            activations_before: Activations before critical moment
            activations_after: Activations after critical moment

        Returns:
            Dictionary containing comparison metrics:
                - 'pca_before': PCA results before
                - 'pca_after': PCA results after
                - 'sv_before': Singular value analysis before
                - 'sv_after': Singular value analysis after
                - 'dim_change': Change in effective dimensionality
                - 'subspace_overlap': Overlap between principal subspaces
        """
        # PCA analysis
        pca_before = self.pca_analysis(activations_before, return_transformed=False)
        pca_after = self.pca_analysis(activations_after, return_transformed=False)

        # Singular value analysis
        sv_before = self.singular_value_analysis(activations_before)
        sv_after = self.singular_value_analysis(activations_after)

        # Dimensionality change
        dim_change = {
            'stable_rank': sv_after['stable_rank'] - sv_before['stable_rank'],
            'participation_ratio': sv_after['participation_ratio'] - sv_before['participation_ratio'],
            'n_components_90': pca_after['n_components_90'] - pca_before['n_components_90']
        }

        # Subspace overlap (using principal angles)
        n_components = min(10, min(pca_before['components'].shape[0], pca_after['components'].shape[0]))
        V1 = pca_before['components'][:n_components].T
        V2 = pca_after['components'][:n_components].T

        # Compute principal angles via SVD
        M = V1.T @ V2
        _, s, _ = np.linalg.svd(M)
        principal_angles = np.arccos(np.clip(s, -1, 1))
        subspace_overlap = np.cos(principal_angles).mean()

        return {
            'pca_before': pca_before,
            'pca_after': pca_after,
            'sv_before': sv_before,
            'sv_after': sv_after,
            'dim_change': dim_change,
            'subspace_overlap': subspace_overlap,
            'principal_angles': principal_angles
        }

    def neuron_importance(
        self,
        activations: np.ndarray,
        method: str = 'variance'
    ) -> Dict[str, Any]:
        """Compute importance scores for individual neurons.

        Identifies which neurons contribute most to the representation.

        Args:
            activations: Array of shape (n_samples, n_neurons)
            method: Scoring method ('variance', 'mean_abs', 'sparsity')

        Returns:
            Dictionary containing:
                - 'scores': Importance score for each neuron
                - 'ranking': Indices sorted by importance
                - 'dead_neurons': Indices of neurons with zero activation
                - 'top_10_indices': Top 10 most important neurons
        """
        if method == 'variance':
            scores = np.var(activations, axis=0)
        elif method == 'mean_abs':
            scores = np.mean(np.abs(activations), axis=0)
        elif method == 'sparsity':
            # Lower sparsity = higher importance
            scores = 1.0 - np.mean(activations == 0, axis=0)
        else:
            raise ValueError(f"Unknown importance method: {method}")

        ranking = np.argsort(scores)[::-1]
        dead_neurons = np.where(np.all(activations == 0, axis=0))[0]

        return {
            'scores': scores,
            'ranking': ranking,
            'dead_neurons': dead_neurons,
            'n_dead': len(dead_neurons),
            'top_10_indices': ranking[:10],
            'method': method
        }

    def activation_statistics(
        self,
        activations: np.ndarray
    ) -> Dict[str, float]:
        """Compute summary statistics for activations.

        Args:
            activations: Array of shape (n_samples, n_features)

        Returns:
            Dictionary of statistics
        """
        return {
            'mean': float(np.mean(activations)),
            'std': float(np.std(activations)),
            'min': float(np.min(activations)),
            'max': float(np.max(activations)),
            'sparsity': float(np.mean(activations == 0)),
            'n_samples': activations.shape[0],
            'n_features': activations.shape[1]
        }
