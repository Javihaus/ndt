"""Statistical detection of dimensionality jumps during training."""

from dataclasses import dataclass
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np


@dataclass
class Jump:
    """Represents a detected jump in dimensionality.

    Attributes:
        step: Training step where jump occurred
        z_score: Z-score magnitude of the jump
        value_before: Dimensionality value before jump
        value_after: Dimensionality value after jump
        metric_name: Name of the metric that jumped
    """

    step: int
    z_score: float
    value_before: float
    value_after: float
    metric_name: str

    def __repr__(self) -> str:
        """String representation of the jump."""
        return (
            f"Jump(step={self.step}, metric={self.metric_name}, "
            f"z_score={self.z_score:.2f}, "
            f"Δ={self.value_after - self.value_before:.2f})"
        )


class JumpDetector:
    """Detects significant jumps in dimensionality metrics using Z-score analysis.

    This class uses a rolling window to compute mean and standard deviation,
    then identifies jumps that exceed a specified Z-score threshold.

    Attributes:
        window_size: Size of the rolling window for statistics
        z_threshold: Z-score threshold for jump detection
        min_samples: Minimum number of samples before detecting jumps

    Example:
        >>> detector = JumpDetector(window_size=50, z_threshold=3.0)
        >>> values = [10.0] * 100 + [20.0] * 100  # Simulated jump
        >>> jumps = detector.detect_jumps(values, metric_name="stable_rank")
        >>> for jump in jumps:
        ...     print(jump)
    """

    def __init__(
        self, window_size: int = 50, z_threshold: float = 3.0, min_samples: int = 20
    ) -> None:
        """Initialize the jump detector.

        Args:
            window_size: Size of rolling window for computing statistics (default: 50)
            z_threshold: Z-score threshold for detecting jumps (default: 3.0)
            min_samples: Minimum samples before starting detection (default: 20)

        Raises:
            ValueError: If parameters are invalid
        """
        if window_size < 2:
            raise ValueError(f"window_size must be >= 2, got {window_size}")
        if z_threshold <= 0:
            raise ValueError(f"z_threshold must be > 0, got {z_threshold}")
        if min_samples < 1:
            raise ValueError(f"min_samples must be >= 1, got {min_samples}")

        self.window_size = window_size
        self.z_threshold = z_threshold
        self.min_samples = min_samples

    def detect_jumps(
        self, values: List[float], metric_name: str = "dimensionality", step_offset: int = 0
    ) -> List[Jump]:
        """Detect jumps in a sequence of dimensionality values.

        Args:
            values: List of dimensionality measurements
            metric_name: Name of the metric being analyzed
            step_offset: Offset to add to step numbers (for aligning with training steps)

        Returns:
            List of detected Jump objects

        Raises:
            ValueError: If values list is too short
        """
        if len(values) < self.min_samples:
            return []

        values_array = np.array(values)
        jumps = []

        # Start detection after minimum samples
        for i in range(self.min_samples, len(values_array)):
            # Define window for statistics
            window_start = max(0, i - self.window_size)
            window = values_array[window_start:i]

            # Compute statistics
            mean = np.mean(window)
            std = np.std(window)

            # Avoid division by zero
            if std < 1e-10:
                continue

            # Compute z-score for current value
            current_value = values_array[i]
            z_score = abs((current_value - mean) / std)

            # Detect jump
            if z_score > self.z_threshold:
                jump = Jump(
                    step=i + step_offset,
                    z_score=z_score,
                    value_before=values_array[i - 1],
                    value_after=current_value,
                    metric_name=metric_name,
                )
                jumps.append(jump)

        return jumps

    def detect_jumps_with_direction(
        self,
        values: List[float],
        metric_name: str = "dimensionality",
        step_offset: int = 0,
        direction: Optional[str] = None,
    ) -> List[Jump]:
        """Detect jumps with optional direction filtering.

        Args:
            values: List of dimensionality measurements
            metric_name: Name of the metric being analyzed
            step_offset: Offset to add to step numbers
            direction: Filter by direction - "up", "down", or None for both

        Returns:
            List of detected Jump objects filtered by direction

        Raises:
            ValueError: If direction is not valid
        """
        if direction not in [None, "up", "down"]:
            raise ValueError(f"direction must be None, 'up', or 'down', got {direction}")

        all_jumps = self.detect_jumps(values, metric_name, step_offset)

        if direction is None:
            return all_jumps

        # Filter by direction
        filtered_jumps = []
        for jump in all_jumps:
            change = jump.value_after - jump.value_before
            if direction == "up" and change > 0:
                filtered_jumps.append(jump)
            elif direction == "down" and change < 0:
                filtered_jumps.append(jump)

        return filtered_jumps

    def compute_rolling_statistics(
        self, values: List[float]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute rolling mean, std, and z-scores for visualization.

        Args:
            values: List of dimensionality measurements

        Returns:
            Tuple of (means, stds, z_scores) as numpy arrays

        Raises:
            ValueError: If values list is empty
        """
        if len(values) == 0:
            raise ValueError("values list cannot be empty")

        values_array = np.array(values)
        n = len(values_array)

        means = np.zeros(n)
        stds = np.zeros(n)
        z_scores = np.zeros(n)

        for i in range(self.min_samples, n):
            window_start = max(0, i - self.window_size)
            window = values_array[window_start:i]

            means[i] = np.mean(window)
            stds[i] = np.std(window)

            if stds[i] > 1e-10:
                z_scores[i] = abs((values_array[i] - means[i]) / stds[i])

        return means, stds, z_scores
