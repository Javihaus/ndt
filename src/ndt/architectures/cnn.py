"""Handler for Convolutional Neural Network (CNN) architectures."""

from typing import List

import torch.nn as nn

from ndt.architectures.base import ArchitectureHandler


class CNNHandler(ArchitectureHandler):
    """Handler for Convolutional Neural Networks (CNNs).

    Monitors Conv2d layers and the first Linear layer (pre-classifier).
    Supports both simple CNNs and ResNet-style architectures.
    """

    def validate_model(self, model: nn.Module) -> bool:
        """Check if model is a CNN (contains Conv layers).

        Args:
            model: The neural network model

        Returns:
            True if model appears to be a CNN
        """
        has_conv = False
        for module in model.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                has_conv = True
                break
        return has_conv

    def get_activation_layers(self, model: nn.Module) -> List[nn.Module]:
        """Get Conv2d and key Linear layers for monitoring.

        For ResNets, monitors after each residual block.
        For simple CNNs, monitors each conv layer.

        Args:
            model: The neural network model

        Returns:
            List of layer modules to track
        """
        layers = []

        # Check if it's a ResNet-style architecture
        if self._is_resnet(model):
            layers = self._get_resnet_layers(model)
        else:
            # Simple CNN: monitor all conv layers
            for module in model.modules():
                if isinstance(module, nn.Conv2d):
                    layers.append(module)

            # Also monitor first Linear layer (before classifier)
            linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
            if linear_layers:
                layers.append(linear_layers[0])

        return layers

    def _is_resnet(self, model: nn.Module) -> bool:
        """Check if model is ResNet-style architecture.

        Args:
            model: The neural network model

        Returns:
            True if model appears to be a ResNet
        """
        # Check for common ResNet layer names
        for name, module in model.named_modules():
            if "layer1" in name or "layer2" in name or "layer3" in name or "layer4" in name:
                return True
        return False

    def _get_resnet_layers(self, model: nn.Module) -> List[nn.Module]:
        """Get layers to monitor in ResNet architecture.

        Monitors the last conv in each residual block.

        Args:
            model: ResNet model

        Returns:
            List of key layers to monitor
        """
        layers = []

        # Get the main residual blocks (layer1, layer2, layer3, layer4)
        for name, module in model.named_children():
            if name.startswith("layer"):
                # Get last conv in this layer group
                conv_layers = [m for m in module.modules() if isinstance(m, nn.Conv2d)]
                if conv_layers:
                    layers.append(conv_layers[-1])

        # Add final Linear layer if present
        linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
        if linear_layers:
            layers.append(linear_layers[-1])

        return layers

    def get_layer_names(self, model: nn.Module) -> List[str]:
        """Generate descriptive names for CNN layers.

        Args:
            model: The neural network model

        Returns:
            List of layer names
        """
        layers = self.get_activation_layers(model)
        names = []

        conv_count = 0
        linear_count = 0

        for layer in layers:
            if isinstance(layer, nn.Conv2d):
                names.append(f"Conv_{conv_count}_{layer.out_channels}ch")
                conv_count += 1
            elif isinstance(layer, nn.Linear):
                names.append(f"FC_{linear_count}_{layer.out_features}")
                linear_count += 1

        return names
