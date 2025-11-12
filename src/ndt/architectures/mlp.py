"""Handler for Multi-Layer Perceptron (MLP) architectures."""

from typing import List

import torch.nn as nn

from ndt.architectures.base import ArchitectureHandler


class MLPHandler(ArchitectureHandler):
    """Handler for fully-connected (MLP) networks.

    Monitors all Linear layers in the network.
    """

    def validate_model(self, model: nn.Module) -> bool:
        """Check if model is an MLP (contains Linear layers, no Conv/Attention).

        Args:
            model: The neural network model

        Returns:
            True if model appears to be an MLP
        """
        has_linear = False
        has_conv = False
        has_attention = False

        for module in model.modules():
            if isinstance(module, nn.Linear):
                has_linear = True
            elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                has_conv = True
            elif isinstance(module, nn.MultiheadAttention):
                has_attention = True

        # MLP should have Linear layers but not Conv or Attention
        return has_linear and not has_conv and not has_attention

    def get_activation_layers(self, model: nn.Module) -> List[nn.Module]:
        """Get all Linear layers for monitoring.

        Args:
            model: The neural network model

        Returns:
            List of Linear layer modules
        """
        layers = []
        for module in model.modules():
            if isinstance(module, nn.Linear):
                layers.append(module)
        return layers

    def get_layer_names(self, model: nn.Module) -> List[str]:
        """Generate names for Linear layers.

        Args:
            model: The neural network model

        Returns:
            List of layer names like "Linear_0", "Linear_1", etc.
        """
        layers = self.get_activation_layers(model)
        names = []

        for i, layer in enumerate(layers):
            in_features = layer.in_features
            out_features = layer.out_features
            names.append(f"Linear_{i}_{in_features}x{out_features}")

        return names
