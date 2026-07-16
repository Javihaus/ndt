"""High-frequency tracking of neural network dimensionality during training."""

import logging
from dataclasses import asdict
from dataclasses import dataclass
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import pandas as pd
import torch
import torch.nn as nn

from ndt.core.estimators import compute_all_metrics
from ndt.core.hooks import ActivationCapture
from ndt.core.jump_detector import JumpDetector


@dataclass
class DimensionalityMetrics:
    """Container for dimensionality measurements at a single training step.

    Attributes:
        step: Training step number
        stable_rank: Stable rank estimate
        participation_ratio: Participation ratio estimate
        cumulative_90: Number of components for 90% energy
        nuclear_norm_ratio: Nuclear norm ratio estimate
        loss: Training loss at this step
        grad_norm: Gradient norm at this step (optional)
    """

    step: int
    stable_rank: float
    participation_ratio: float
    cumulative_90: int
    nuclear_norm_ratio: float
    loss: float
    grad_norm: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class HighFrequencyTracker:
    """High-frequency tracker for neural network representational dimensionality.

    This class provides minimal-intrusion tracking of dimensionality metrics
    during neural network training. It automatically captures activations and
    computes multiple dimensionality estimates at specified intervals.

    Attributes:
        model: The neural network model to track
        layers: List of layers to monitor
        sampling_frequency: Record metrics every N steps
        jump_detector: Optional jump detector for identifying phase transitions

    Example:
        >>> import torch.nn as nn
        >>> model = nn.Sequential(
        ...     nn.Linear(784, 512), nn.ReLU(),
        ...     nn.Linear(512, 256), nn.ReLU(),
        ...     nn.Linear(256, 10)
        ... )
        >>> tracker = HighFrequencyTracker(model, layers=[model[0], model[2], model[4]])
        >>> # In training loop:
        >>> for step, (x, y) in enumerate(dataloader):
        ...     loss = train_step(model, x, y)
        ...     tracker.log(step, loss)
        >>> results = tracker.get_results()
        >>> tracker.close()
    """

    def __init__(
        self,
        model: nn.Module,
        layers: Optional[List[nn.Module]] = None,
        layer_names: Optional[List[str]] = None,
        sampling_frequency: int = 1,
        enable_jump_detection: bool = True,
        jump_window_size: int = 50,
        jump_z_threshold: float = 3.0,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize the high-frequency tracker.

        Args:
            model: Neural network model to track
            layers: List of layers to monitor. If None, auto-detects based on architecture
            layer_names: Optional custom names for layers
            sampling_frequency: Record metrics every N steps (default: 1)
            enable_jump_detection: Whether to detect dimensionality jumps (default: True)
            jump_window_size: Window size for jump detection (default: 50)
            jump_z_threshold: Z-score threshold for jump detection (default: 3.0)
            device: Device for computations (default: model's device)

        Raises:
            ValueError: If model or parameters are invalid
        """
        if not isinstance(model, nn.Module):
            raise ValueError("model must be a torch.nn.Module")

        self.model = model
        self.sampling_frequency = sampling_frequency
        self.device = device or next(model.parameters()).device

        # Auto-detect layers if not provided
        if layers is None:
            layers = self._auto_detect_layers()

        if not layers:
            raise ValueError(
                "No layers to track. Provide layers explicitly or use supported architecture."
            )

        # Set up activation capture
        self.activation_capture = ActivationCapture()
        self.activation_capture.register_hooks(model, layers, layer_names)
        self.layer_names = layer_names or [
            f"{layer.__class__.__name__}_{i}" for i, layer in enumerate(layers)
        ]

        # Initialize storage
        self.metrics_history: Dict[str, List[DimensionalityMetrics]] = {
            name: [] for name in self.layer_names
        }
        self.step_counter = 0

        # Jump detection
        self.enable_jump_detection = enable_jump_detection
        if enable_jump_detection:
            self.jump_detector = JumpDetector(
                window_size=jump_window_size, z_threshold=jump_z_threshold
            )
        else:
            self.jump_detector = None

        # Logging
        self.logger = logging.getLogger(__name__)

    def _auto_detect_layers(self) -> List[nn.Module]:
        """Auto-detect layers to monitor based on model architecture.

        Returns:
            List of layers to monitor
        """
        layers = []

        for module in self.model.modules():
            # Monitor Linear, Conv2d, and MultiheadAttention layers
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                layers.append(module)
            # Skip if too many layers (>20), sample evenly
            if len(layers) > 20:
                # Keep every other layer
                layers = layers[::2]

        return layers

    def log(
        self, step: int, loss: float, grad_norm: Optional[float] = None, force: bool = False
    ) -> None:
        """Log dimensionality metrics for the current step.

        This is the main method called during training. It captures activations
        and computes dimensionality metrics according to the sampling frequency.

        Args:
            step: Current training step
            loss: Training loss value
            grad_norm: Optional gradient norm
            force: Force logging even if not at sampling interval

        Example:
            >>> for step, (x, y) in enumerate(dataloader):
            ...     optimizer.zero_grad()
            ...     output = model(x)
            ...     loss = criterion(output, y)
            ...     loss.backward()
            ...     optimizer.step()
            ...     tracker.log(step, loss.item())
        """
        # Check if we should record this step
        if not force and step % self.sampling_frequency != 0:
            return

        self.step_counter = step

        # Activations are already captured by hooks during forward pass
        # Now compute metrics for each layer
        with torch.no_grad():
            for layer_name in self.layer_names:
                activation = self.activation_capture.get_activation(layer_name)

                if activation is None:
                    self.logger.warning(
                        f"No activation captured for layer {layer_name} at step {step}"
                    )
                    continue

                try:
                    # Reshape activation to 2D matrix (batch_size, features)
                    matrix = self._prepare_activation_matrix(activation)

                    # Compute all dimensionality metrics
                    sr, pr, ce90, nnr = compute_all_metrics(matrix)

                    # Create metrics object
                    metrics = DimensionalityMetrics(
                        step=step,
                        stable_rank=sr,
                        participation_ratio=pr,
                        cumulative_90=ce90,
                        nuclear_norm_ratio=nnr,
                        loss=loss,
                        grad_norm=grad_norm,
                    )

                    # Store metrics
                    self.metrics_history[layer_name].append(metrics)

                except Exception as e:
                    self.logger.error(
                        f"Error computing metrics for {layer_name} at step {step}: {e}"
                    )

        # Clear activations for next step
        self.activation_capture.clear_activations()

    def _prepare_activation_matrix(self, activation: torch.Tensor) -> torch.Tensor:
        """Prepare activation tensor as 2D matrix for dimensionality computation.

        Args:
            activation: Activation tensor of arbitrary shape

        Returns:
            2D matrix of shape (batch_size, features)
        """
        # Handle different tensor shapes
        if activation.ndim == 2:
            # Already 2D (batch_size, features)
            return activation
        elif activation.ndim == 4:
            # Conv2d output: (batch, channels, height, width)
            # Flatten spatial dimensions: (batch, channels * height * width)
            batch_size = activation.size(0)
            return activation.view(batch_size, -1)
        elif activation.ndim == 3:
            # Transformer output: (batch, seq_len, hidden_dim)
            # Flatten sequence: (batch * seq_len, hidden_dim)
            return activation.view(-1, activation.size(-1))
        else:
            # General case: flatten all but first dimension
            batch_size = activation.size(0)
            return activation.view(batch_size, -1)

    def get_results(
        self, layer_name: Optional[str] = None
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Get tracked metrics as pandas DataFrame(s).

        Args:
            layer_name: If specified, return results for that layer only.
                       If None, return dict of DataFrames for all layers.

        Returns:
            DataFrame or dict of DataFrames containing metrics history

        Example:
            >>> results = tracker.get_results()
            >>> for layer_name, df in results.items():
            ...     print(f"{layer_name}: {len(df)} measurements")
        """
        if layer_name is not None:
            if layer_name not in self.metrics_history:
                raise ValueError(f"Unknown layer name: {layer_name}")
            return self._metrics_to_dataframe(self.metrics_history[layer_name])

        return {
            name: self._metrics_to_dataframe(metrics)
            for name, metrics in self.metrics_history.items()
        }

    def _metrics_to_dataframe(self, metrics_list: List[DimensionalityMetrics]) -> pd.DataFrame:
        """Convert list of metrics to DataFrame.

        Args:
            metrics_list: List of DimensionalityMetrics objects

        Returns:
            DataFrame with metrics
        """
        if not metrics_list:
            return pd.DataFrame()

        data = [m.to_dict() for m in metrics_list]
        return pd.DataFrame(data)

    def detect_jumps(
        self, layer_name: Optional[str] = None, metric: str = "stable_rank"
    ) -> Dict[str, List]:
        """Detect dimensionality jumps in tracked metrics.

        Args:
            layer_name: If specified, detect jumps for that layer only.
                       If None, detect for all layers.
            metric: Which metric to analyze ("stable_rank", "participation_ratio",
                   "cumulative_90", or "nuclear_norm_ratio")

        Returns:
            Dictionary mapping layer names to lists of Jump objects

        Raises:
            ValueError: If jump detection is disabled or metric name is invalid
        """
        if not self.enable_jump_detection:
            raise ValueError("Jump detection is disabled. Enable it during initialization.")

        valid_metrics = [
            "stable_rank",
            "participation_ratio",
            "cumulative_90",
            "nuclear_norm_ratio",
        ]
        if metric not in valid_metrics:
            raise ValueError(f"metric must be one of {valid_metrics}, got {metric}")

        layers_to_analyze = [layer_name] if layer_name else self.layer_names

        jumps_dict = {}
        for name in layers_to_analyze:
            if name not in self.metrics_history:
                continue

            # Extract metric values
            values = [getattr(m, metric) for m in self.metrics_history[name]]

            # Detect jumps
            jumps = self.jump_detector.detect_jumps(values, metric_name=f"{name}_{metric}")
            jumps_dict[name] = jumps

        return jumps_dict

    def close(self) -> None:
        """Clean up hooks and resources.

        Should be called when done tracking to free resources.
        """
        self.activation_capture.remove_hooks()
        self.logger.info(f"Tracker closed. Recorded {self.step_counter} steps.")

    def __enter__(self) -> "HighFrequencyTracker":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def __del__(self) -> None:
        """Cleanup on deletion."""
        if hasattr(self, "activation_capture"):
            self.activation_capture.remove_hooks()
