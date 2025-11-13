"""Example scripts for NDT (Neural Dimensionality Tracker).

This package contains example scripts demonstrating various use cases.
The numbered example files are aliased for easier importing in tests.
"""

# Import from numbered example files for easier access in tests
import importlib.util
import sys
from pathlib import Path

# Load reproduce_tds_experiment from 03_reproduce_tds_experiment.py
_examples_dir = Path(__file__).parent
_tds_path = _examples_dir / "03_reproduce_tds_experiment.py"

reproduce_tds_experiment = None
if _tds_path.exists():
    try:
        _spec = importlib.util.spec_from_file_location("reproduce_tds_experiment", _tds_path)
        reproduce_tds_experiment = importlib.util.module_from_spec(_spec)
        sys.modules["examples.reproduce_tds_experiment"] = reproduce_tds_experiment
        _spec.loader.exec_module(reproduce_tds_experiment)
    except ImportError:
        # torchvision might not be available in test environment
        reproduce_tds_experiment = None
        if "examples.reproduce_tds_experiment" in sys.modules:
            del sys.modules["examples.reproduce_tds_experiment"]

__all__ = ["reproduce_tds_experiment"]
