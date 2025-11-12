"""Neural Dimensionality Tracker (NDT)

High-frequency monitoring of neural network representational dimensionality during training.
"""

from ndt.__version__ import __version__
from ndt.architectures import CNNHandler
from ndt.architectures import MLPHandler
from ndt.architectures import TransformerHandler
from ndt.architectures import ViTHandler
from ndt.architectures import detect_architecture
from ndt.architectures import get_handler
from ndt.core.estimators import compute_all_metrics
from ndt.core.estimators import cumulative_energy_90
from ndt.core.estimators import nuclear_norm_ratio
from ndt.core.estimators import participation_ratio
from ndt.core.estimators import stable_rank
from ndt.core.hooks import ActivationCapture
from ndt.core.jump_detector import Jump
from ndt.core.jump_detector import JumpDetector
from ndt.core.tracker import DimensionalityMetrics
from ndt.core.tracker import HighFrequencyTracker
from ndt.export import export_to_csv
from ndt.export import export_to_hdf5
from ndt.export import export_to_json
from ndt.utils import load_config
from ndt.utils import save_config
from ndt.utils import setup_logger
from ndt.visualization import create_interactive_plot
from ndt.visualization import create_multi_layer_plot
from ndt.visualization import plot_jumps
from ndt.visualization import plot_metrics_comparison
from ndt.visualization import plot_phases
from ndt.visualization import plot_single_metric

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
