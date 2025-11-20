"""Analysis modules for Phase 2 mechanistic interpretability.

This package provides tools for:
- Activation analysis (PCA, clustering, manifold visualization)
- Feature visualization (CAM, saliency, attention maps)
- Neuron importance scoring
"""

from .activation_analysis import ActivationAnalyzer
from .feature_visualization import FeatureVisualizer

__all__ = ["ActivationAnalyzer", "FeatureVisualizer"]
