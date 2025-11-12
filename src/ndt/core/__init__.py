"""Core functionality for neural dimensionality tracking."""

from ndt.core.estimators import (
    stable_rank,
    participation_ratio,
    cumulative_energy_90,
    nuclear_norm_ratio,
    compute_all_metrics,
)
from ndt.core.hooks import ActivationCapture
from ndt.core.tracker import HighFrequencyTracker, DimensionalityMetrics
from ndt.core.jump_detector import JumpDetector

__all__ = [
    "stable_rank",
    "participation_ratio",
    "cumulative_energy_90",
    "nuclear_norm_ratio",
    "compute_all_metrics",
    "ActivationCapture",
    "HighFrequencyTracker",
    "DimensionalityMetrics",
    "JumpDetector",
]
