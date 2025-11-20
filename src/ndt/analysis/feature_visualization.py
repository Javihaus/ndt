"""Feature visualization tools for mechanistic interpretability.

This module provides tools for visualizing what neural networks learn:
- Class Activation Mapping (CAM) for CNNs
- Gradient-based saliency maps
- Attention visualization for Transformers
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureVisualizer:
    """Visualizes learned features in neural networks.

    Provides methods for understanding what features the network
    has learned at different layers and training stages.

    Example:
        >>> visualizer = FeatureVisualizer(model)
        >>> # Generate CAM for an image
        >>> cam = visualizer.grad_cam(input_image, target_class, 'conv5')
        >>> # Visualize saliency
        >>> saliency = visualizer.saliency_map(input_image, target_class)
    """

    def __init__(self, model: nn.Module) -> None:
        """Initialize the feature visualizer.

        Args:
            model: The neural network model to visualize
        """
        self.model = model
        self.gradients: Dict[str, torch.Tensor] = {}
        self.activations: Dict[str, torch.Tensor] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

    def _save_gradient(self, name: str):
        """Create hook to save gradients."""

        def hook(grad):
            self.gradients[name] = grad.detach()

        return hook

    def _save_activation(self, name: str):
        """Create forward hook to save activations."""

        def hook(module, input, output):
            self.activations[name] = output.detach()
            if output.requires_grad:
                output.register_hook(self._save_gradient(name))

        return hook

    def register_hooks(self, layer_names: Dict[str, nn.Module]) -> None:
        """Register hooks on specified layers.

        Args:
            layer_names: Dictionary mapping names to layer modules
        """
        for name, layer in layer_names.items():
            hook = layer.register_forward_hook(self._save_activation(name))
            self.hooks.append(hook)

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.gradients.clear()
        self.activations.clear()

    def grad_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        target_layer: nn.Module,
        layer_name: str = "target",
    ) -> np.ndarray:
        """Generate Grad-CAM visualization.

        Grad-CAM uses gradients flowing into the target layer to produce
        a coarse localization map highlighting important regions.

        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class index
            target_layer: Layer to compute CAM for
            layer_name: Name for the layer (for storage)

        Returns:
            CAM heatmap as numpy array

        References:
            Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
            via Gradient-based Localization", ICCV 2017
        """
        self.model.eval()

        # Register hooks
        activations = []
        gradients = []

        def forward_hook(module, input, output):
            activations.append(output.detach())

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0].detach())

        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)

        try:
            # Forward pass
            output = self.model(input_tensor)

            # Get prediction score for target class
            if output.dim() == 1:
                score = output[target_class]
            else:
                score = output[0, target_class]

            # Backward pass
            self.model.zero_grad()
            score.backward()

            # Get the activations and gradients
            activation = activations[0]  # (1, C, H, W)
            gradient = gradients[0]  # (1, C, H, W)

            # Global average pooling of gradients
            weights = torch.mean(gradient, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

            # Weighted combination of activation maps
            cam = torch.sum(weights * activation, dim=1, keepdim=True)  # (1, 1, H, W)

            # ReLU to keep only positive contributions
            cam = F.relu(cam)

            # Normalize
            cam = cam - cam.min()
            if cam.max() > 0:
                cam = cam / cam.max()

            return cam.squeeze().cpu().numpy()

        finally:
            forward_handle.remove()
            backward_handle.remove()

    def saliency_map(
        self, input_tensor: torch.Tensor, target_class: int, absolute: bool = True
    ) -> np.ndarray:
        """Generate vanilla gradient saliency map.

        Computes the gradient of the target class score with respect to
        the input pixels.

        Args:
            input_tensor: Input tensor (1, C, H, W) or (1, features)
            target_class: Target class index
            absolute: Whether to take absolute value of gradients

        Returns:
            Saliency map as numpy array
        """
        self.model.eval()

        input_tensor = input_tensor.clone().requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)

        # Get score for target class
        if output.dim() == 1:
            score = output[target_class]
        else:
            score = output[0, target_class]

        # Backward pass
        self.model.zero_grad()
        score.backward()

        # Get gradients
        saliency = input_tensor.grad.detach()

        if absolute:
            saliency = torch.abs(saliency)

        # For multi-channel inputs, take max across channels
        if saliency.dim() == 4 and saliency.shape[1] > 1:
            saliency = saliency.max(dim=1, keepdim=True)[0]

        return saliency.squeeze().cpu().numpy()

    def integrated_gradients(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        baseline: Optional[torch.Tensor] = None,
        steps: int = 50,
    ) -> np.ndarray:
        """Compute Integrated Gradients attribution.

        Accumulates gradients along the path from baseline to input,
        providing more stable attributions than vanilla gradients.

        Args:
            input_tensor: Input tensor
            target_class: Target class index
            baseline: Baseline tensor (default: zeros)
            steps: Number of interpolation steps

        Returns:
            Attribution map as numpy array

        References:
            Sundararajan et al., "Axiomatic Attribution for Deep Networks", ICML 2017
        """
        self.model.eval()

        if baseline is None:
            baseline = torch.zeros_like(input_tensor)

        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, steps).to(input_tensor.device)
        scaled_inputs = torch.stack(
            [baseline + alpha * (input_tensor - baseline) for alpha in alphas]
        )

        # Compute gradients for all interpolated inputs
        scaled_inputs.requires_grad_(True)

        # Forward pass (batch all steps together)
        outputs = self.model(scaled_inputs.view(-1, *input_tensor.shape[1:]))

        # Get scores
        if outputs.dim() == 1:
            scores = outputs.view(steps, -1)[:, target_class]
        else:
            scores = outputs[:, target_class]

        # Backward pass
        self.model.zero_grad()
        scores.sum().backward()

        # Average gradients
        avg_gradients = scaled_inputs.grad.mean(dim=0)

        # Integrated gradients
        ig = (input_tensor - baseline) * avg_gradients

        return ig.squeeze().detach().cpu().numpy()

    def attention_visualization(
        self, attention_weights: torch.Tensor, head_idx: Optional[int] = None
    ) -> np.ndarray:
        """Visualize attention weights from transformer models.

        Args:
            attention_weights: Attention tensor (batch, heads, seq, seq)
            head_idx: Specific head to visualize (None for average)

        Returns:
            Attention matrix as numpy array
        """
        if head_idx is not None:
            attn = attention_weights[:, head_idx]
        else:
            # Average across heads
            attn = attention_weights.mean(dim=1)

        return attn.squeeze().detach().cpu().numpy()

    def feature_maps(self, activation: torch.Tensor, top_k: int = 16) -> List[np.ndarray]:
        """Extract top-k feature maps from a convolutional layer.

        Args:
            activation: Activation tensor (batch, channels, H, W)
            top_k: Number of feature maps to return

        Returns:
            List of feature map arrays
        """
        # Compute activation magnitudes
        magnitudes = activation.abs().mean(dim=(2, 3))  # (batch, channels)

        # Get top-k channels
        _, top_indices = magnitudes[0].topk(min(top_k, activation.shape[1]))

        feature_maps = []
        for idx in top_indices:
            fmap = activation[0, idx].detach().cpu().numpy()
            # Normalize
            fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
            feature_maps.append(fmap)

        return feature_maps

    def neuron_activation_maximization(
        self,
        target_layer: nn.Module,
        neuron_idx: int,
        input_shape: Tuple[int, ...],
        steps: int = 100,
        lr: float = 0.1,
    ) -> torch.Tensor:
        """Generate input that maximizes a specific neuron's activation.

        Uses gradient ascent to find an input that maximally activates
        a target neuron.

        Args:
            target_layer: Layer containing the neuron
            neuron_idx: Index of the neuron to maximize
            input_shape: Shape of input tensor
            steps: Number of optimization steps
            lr: Learning rate

        Returns:
            Optimized input tensor
        """
        # Initialize with random noise
        input_tensor = torch.randn(1, *input_shape, requires_grad=True)

        activation_value = []

        def hook(module, input, output):
            if output.dim() == 2:
                activation_value.append(output[0, neuron_idx])
            elif output.dim() == 4:
                # For conv layers, take spatial mean
                activation_value.append(output[0, neuron_idx].mean())

        handle = target_layer.register_forward_hook(hook)

        try:
            for _ in range(steps):
                activation_value.clear()

                # Forward pass
                _ = self.model(input_tensor)

                # Get activation
                act = activation_value[0]

                # Backward pass
                self.model.zero_grad()
                act.backward()

                # Gradient ascent
                with torch.no_grad():
                    input_tensor += lr * input_tensor.grad
                    input_tensor.grad.zero_()

            return input_tensor.detach()

        finally:
            handle.remove()

    def layer_conductance(
        self, input_tensor: torch.Tensor, target_class: int, target_layer: nn.Module
    ) -> np.ndarray:
        """Compute layer conductance attribution.

        Measures how much each neuron in a layer contributes to the
        final prediction.

        Args:
            input_tensor: Input tensor
            target_class: Target class index
            target_layer: Layer to compute conductance for

        Returns:
            Conductance values for each neuron
        """
        # Use integrated gradients through the layer
        activation_grads = []

        def hook(module, input, output):
            output.retain_grad()
            activation_grads.append(output)

        handle = target_layer.register_forward_hook(hook)

        try:
            self.model.eval()

            # Forward pass
            output = self.model(input_tensor)

            # Get score
            if output.dim() == 1:
                score = output[target_class]
            else:
                score = output[0, target_class]

            # Backward pass
            self.model.zero_grad()
            score.backward()

            # Get activation and gradient
            activation = activation_grads[0]
            gradient = activation.grad

            # Conductance = activation * gradient
            conductance = (activation * gradient).sum(dim=0)

            return conductance.detach().cpu().numpy()

        finally:
            handle.remove()

    def compare_features(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        target_layer: nn.Module,
        checkpoint_before: str,
        checkpoint_after: str,
    ) -> Dict[str, Any]:
        """Compare feature visualizations before and after a critical moment.

        Loads model weights from checkpoints and compares Grad-CAM visualizations.

        Args:
            input_tensor: Input to visualize
            target_class: Target class
            target_layer: Layer for visualization
            checkpoint_before: Path to checkpoint before critical moment
            checkpoint_after: Path to checkpoint after critical moment

        Returns:
            Dictionary with CAMs before and after
        """
        # Load before checkpoint
        state_before = torch.load(checkpoint_before, map_location="cpu")
        self.model.load_state_dict(state_before["model_state_dict"])
        cam_before = self.grad_cam(input_tensor, target_class, target_layer)

        # Load after checkpoint
        state_after = torch.load(checkpoint_after, map_location="cpu")
        self.model.load_state_dict(state_after["model_state_dict"])
        cam_after = self.grad_cam(input_tensor, target_class, target_layer)

        # Compute difference
        cam_diff = cam_after - cam_before

        return {
            "cam_before": cam_before,
            "cam_after": cam_after,
            "cam_diff": cam_diff,
            "mean_change": np.abs(cam_diff).mean(),
        }
