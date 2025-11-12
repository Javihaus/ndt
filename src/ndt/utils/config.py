"""Configuration management utilities."""

from pathlib import Path
from typing import Any
from typing import Dict

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary

    Example:
        >>> config = load_config("tracker_config.yaml")
        >>> tracker = HighFrequencyTracker(**config)

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML is invalid
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        output_path: Output file path

    Example:
        >>> config = {
        ...     "sampling_frequency": 10,
        ...     "enable_jump_detection": True,
        ...     "jump_z_threshold": 3.0
        ... }
        >>> save_config(config, "tracker_config.yaml")
    """
    output_path = Path(output_path)

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Saved configuration to {output_path}")


def get_default_config() -> Dict[str, Any]:
    """Get default configuration for the tracker.

    Returns:
        Default configuration dictionary

    Example:
        >>> config = get_default_config()
        >>> config['sampling_frequency'] = 5  # Customize
        >>> tracker = HighFrequencyTracker(model, **config)
    """
    return {
        "sampling_frequency": 1,
        "enable_jump_detection": True,
        "jump_window_size": 50,
        "jump_z_threshold": 3.0,
    }
