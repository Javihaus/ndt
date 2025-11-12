"""Architecture-specific handlers for dimensionality tracking."""

from ndt.architectures.base import ArchitectureHandler
from ndt.architectures.mlp import MLPHandler
from ndt.architectures.cnn import CNNHandler
from ndt.architectures.transformer import TransformerHandler
from ndt.architectures.vit import ViTHandler
from ndt.architectures.registry import detect_architecture, get_handler

__all__ = [
    "ArchitectureHandler",
    "MLPHandler",
    "CNNHandler",
    "TransformerHandler",
    "ViTHandler",
    "detect_architecture",
    "get_handler",
]
