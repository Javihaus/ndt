"""Neural Dimensionality Tracker (NDT)

High-frequency monitoring of neural network representational dimensionality during training.
"""

from ndt.__version__ import __version__

# Core functionality
from ndt.core.tracker import HighFrequencyTracker, DimensionalityMetrics
from ndt.core.estimators import (
    stable_rank,
    participation_ratio,
    cumulative_energy_90,
    nuclear_norm_ratio,
    compute_all_metrics,
)
from ndt.core.jump_detector import JumpDetector, Jump
from ndt.core.hooks import ActivationCapture

# Architecture handlers
from ndt.architectures import (
    detect_architecture,
    get_handler,
    MLPHandler,
    CNNHandler,
    TransformerHandler,
    ViTHandler,
)

# Visualization
from ndt.visualization import (
    plot_phases,
    plot_jumps,
    plot_metrics_comparison,
    plot_single_metric,
    create_interactive_plot,
    create_multi_layer_plot,
)

# Export
from ndt.export import (
    export_to_csv,
    export_to_json,
    export_to_hdf5,
)

# Utilities
from ndt.utils import setup_logger, load_config, save_config

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
