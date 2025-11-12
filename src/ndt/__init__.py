"""Neural Dimensionality Tracker (NDT)

High-frequency monitoring of neural network representational dimensionality during training.
"""

from ndt.__version__ import __version__

# Architecture handlers
from ndt.architectures import (
    CNNHandler,
    MLPHandler,
    TransformerHandler,
    ViTHandler,
    detect_architecture,
    get_handler,
)
from ndt.core.estimators import (
    compute_all_metrics,
    cumulative_energy_90,
    nuclear_norm_ratio,
    participation_ratio,
    stable_rank,
)
from ndt.core.hooks import ActivationCapture
from ndt.core.jump_detector import Jump, JumpDetector

# Core functionality
from ndt.core.tracker import DimensionalityMetrics, HighFrequencyTracker

# Export
from ndt.export import (
    export_to_csv,
    export_to_hdf5,
    export_to_json,
)

# Utilities
from ndt.utils import load_config, save_config, setup_logger

# Visualization
from ndt.visualization import (
    create_interactive_plot,
    create_multi_layer_plot,
    plot_jumps,
    plot_metrics_comparison,
    plot_phases,
    plot_single_metric,
)

__all__ = [
    # Version
    "__version__",
    # Core
    "HighFrequencyTracker",
    "DimensionalityMetrics",
    "stable_rank",
    "participation_ratio",
    "cumulative_energy_90",
    "nuclear_norm_ratio",
    "compute_all_metrics",
    "JumpDetector",
    "Jump",
    "ActivationCapture",
    # Architectures
    "detect_architecture",
    "get_handler",
    "MLPHandler",
    "CNNHandler",
    "TransformerHandler",
    "ViTHandler",
    # Visualization
    "plot_phases",
    "plot_jumps",
    "plot_metrics_comparison",
    "plot_single_metric",
    "create_interactive_plot",
    "create_multi_layer_plot",
    # Export
    "export_to_csv",
    "export_to_json",
    "export_to_hdf5",
    # Utils
    "setup_logger",
    "load_config",
    "save_config",
]
