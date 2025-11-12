"""Abstract base class for architecture-specific handlers."""

from abc import ABC
from abc import abstractmethod
from typing import List

import torch.nn as nn


class ArchitectureHandler(ABC):
    """Abstract base class for architecture-specific dimensionality tracking.

    Each architecture handler implements methods to identify relevant layers
    and extract activations appropriately for that architecture type.
    """

    @abstractmethod
    def validate_model(self, model: nn.Module) -> bool:
        """Check if model matches this handler's architecture.

        Args:
            model: The neural network model

        Returns:
            True if model is compatible with this handler
        """
        pass

    @abstractmethod
    def get_activation_layers(self, model: nn.Module) -> List[nn.Module]:
        """Return list of layers to monitor for dimensionality.

        Args:
            model: The neural network model

        Returns:
            List of layer modules to track
        """
        pass

    @abstractmethod
    def get_layer_names(self, model: nn.Module) -> List[str]:
        """Generate descriptive names for the layers to monitor.

        Args:
            model: The neural network model

        Returns:
            List of layer names corresponding to activation layers
        """
        pass

    def get_name(self) -> str:
        """Get the name of this architecture handler.

        Returns:
            Architecture name (e.g., "MLP", "CNN", "Transformer")
        """
        return self.__class__.__name__.replace("Handler", "")
