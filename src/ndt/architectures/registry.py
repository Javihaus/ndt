"""Registry for auto-detecting and selecting architecture handlers."""

from typing import List
from typing import Optional
from typing import Tuple

import torch.nn as nn

from ndt.architectures.base import ArchitectureHandler
from ndt.architectures.cnn import CNNHandler
from ndt.architectures.mlp import MLPHandler
from ndt.architectures.transformer import TransformerHandler
from ndt.architectures.vit import ViTHandler

# Default handler priority order (more specific first)
DEFAULT_HANDLERS = [
    ViTHandler(),
    TransformerHandler(),
    CNNHandler(),
    MLPHandler(),
]


def detect_architecture(model: nn.Module) -> Optional[ArchitectureHandler]:
    """Automatically detect the architecture type of a model.

    Tries each handler in priority order and returns the first match.

    Args:
        model: The neural network model

    Returns:
        ArchitectureHandler instance if detected, None otherwise

    Example:
        >>> import torch.nn as nn
        >>> model = nn.Sequential(nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 10))
        >>> handler = detect_architecture(model)
        >>> print(handler.get_name())
        'MLP'
    """
    for handler in DEFAULT_HANDLERS:
        if handler.validate_model(model):
            return handler
    return None


def get_handler(model: nn.Module, architecture: Optional[str] = None) -> ArchitectureHandler:
    """Get an architecture handler for a model.

    Args:
        model: The neural network model
        architecture: Optional architecture name ("mlp", "cnn", "transformer", "vit").
                     If None, auto-detects.

    Returns:
        ArchitectureHandler instance

    Raises:
        ValueError: If architecture cannot be detected or invalid name provided

    Example:
        >>> handler = get_handler(model, architecture="mlp")
        >>> layers = handler.get_activation_layers(model)
    """
    if architecture is not None:
        # Explicit architecture specified
        architecture = architecture.lower()
        handler_map = {
            "mlp": MLPHandler(),
            "cnn": CNNHandler(),
            "transformer": TransformerHandler(),
            "vit": ViTHandler(),
        }

        if architecture not in handler_map:
            raise ValueError(
                f"Unknown architecture: {architecture}. "
                f"Must be one of {list(handler_map.keys())}"
            )

        handler = handler_map[architecture]

        # Validate that the model actually matches
        if not handler.validate_model(model):
            raise ValueError(
                f"Model does not appear to be a {architecture.upper()}. "
                "Try auto-detection by not specifying architecture."
            )

        return handler

    # Auto-detect architecture
    handler = detect_architecture(model)
    if handler is None:
        raise ValueError(
            "Could not auto-detect model architecture. "
            "Please specify architecture explicitly: 'mlp', 'cnn', 'transformer', or 'vit'"
        )

    return handler


def get_layers_for_model(
    model: nn.Module, architecture: Optional[str] = None
) -> Tuple[List[nn.Module], List[str]]:
    """Convenience function to get layers and names for a model.

    Args:
        model: The neural network model
        architecture: Optional architecture name for explicit selection

    Returns:
        Tuple of (layers, layer_names)

    Example:
        >>> layers, names = get_layers_for_model(model)
        >>> print(f"Monitoring {len(layers)} layers: {names}")
    """
    handler = get_handler(model, architecture)
    layers = handler.get_activation_layers(model)
    names = handler.get_layer_names(model)
    return layers, names
