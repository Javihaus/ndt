"""Forward hooks for capturing neural network activations."""

from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

import torch
import torch.nn as nn


class ActivationCapture:
    """Captures activations from specified layers using forward hooks.

    This class manages PyTorch forward hooks to capture activations during
    the forward pass without modifying the model code.

    Attributes:
        activations: Dictionary mapping layer names to captured activations
        hooks: List of registered hook handles

    Example:
        >>> import torch.nn as nn
        >>> model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 10))
        >>> capture = ActivationCapture()
        >>> capture.register_hooks(model, [model[0], model[2]])
        >>> output = model(torch.randn(32, 10))
        >>> print(capture.activations.keys())
        >>> capture.remove_hooks()
    """

    def __init__(self) -> None:
        """Initialize the activation capture."""
        self.activations: Dict[str, torch.Tensor] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._layer_names: Dict[nn.Module, str] = {}

    def _create_hook(self, name: str) -> Callable:
        """Create a forward hook function for a specific layer.

        Args:
            name: Name identifier for the layer

        Returns:
            Hook function that captures activations
        """

        def hook(module: nn.Module, input: tuple, output: torch.Tensor) -> None:
            """Forward hook that stores the output activation.

            Args:
                module: The layer module
                input: Input tuple to the layer
                output: Output tensor from the layer
            """
            # Detach to avoid keeping computation graph
            self.activations[name] = output.detach()

        return hook

    def register_hooks(
        self, model: nn.Module, layers: List[nn.Module], layer_names: Optional[List[str]] = None
    ) -> None:
        """Register forward hooks on specified layers.

        Args:
            model: The neural network model
            layers: List of layer modules to monitor
            layer_names: Optional custom names for layers. If None, uses module class names.

        Raises:
            ValueError: If number of layer names doesn't match number of layers
        """
        if layer_names is not None and len(layer_names) != len(layers):
            raise ValueError(
                f"Number of layer names ({len(layer_names)}) must match "
                f"number of layers ({len(layers)})"
            )

        # Generate default names if not provided
        if layer_names is None:
            layer_names = [f"{layer.__class__.__name__}_{i}" for i, layer in enumerate(layers)]

        # Register hooks
        for layer, name in zip(layers, layer_names):
            hook_handle = layer.register_forward_hook(self._create_hook(name))
            self.hooks.append(hook_handle)
            self._layer_names[layer] = name

    def remove_hooks(self) -> None:
        """Remove all registered hooks and clear stored activations."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.activations.clear()
        self._layer_names.clear()

    def clear_activations(self) -> None:
        """Clear stored activations without removing hooks."""
        self.activations.clear()

    def get_activation(self, layer_name: str) -> Optional[torch.Tensor]:
        """Get activation for a specific layer.

        Args:
            layer_name: Name of the layer

        Returns:
            Activation tensor if available, None otherwise
        """
        return self.activations.get(layer_name)

    def get_all_activations(self) -> Dict[str, torch.Tensor]:
        """Get all captured activations.

        Returns:
            Dictionary mapping layer names to activation tensors
        """
        return self.activations.copy()

    def __enter__(self) -> "ActivationCapture":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - removes hooks."""
        self.remove_hooks()

    def __del__(self) -> None:
        """Cleanup hooks on deletion."""
        self.remove_hooks()
