"""JSON export functionality for tracking results."""

import json
from pathlib import Path
from typing import Any
from typing import Dict

import pandas as pd


def export_to_json(results: Dict[str, pd.DataFrame], output_path: str, pretty: bool = True) -> None:
    """Export tracking results to JSON format.

    Args:
        output_path: Output file path
        results: Dictionary mapping layer names to DataFrames
        pretty: If True, format JSON with indentation

    Example:
        >>> results = tracker.get_results()
        >>> export_to_json(results, "results.json")

    Raises:
        ValueError: If output_path is invalid
    """
    output_path = Path(output_path)

    # Convert DataFrames to dictionaries
    json_data = {}
    for layer_name, df in results.items():
        json_data[layer_name] = df.to_dict(orient="records")

    # Write to file
    with open(output_path, "w") as f:
        if pretty:
            json.dump(json_data, f, indent=2)
        else:
            json.dump(json_data, f)

    print(f"Exported results to {output_path}")


def load_from_json(input_path: str) -> Dict[str, pd.DataFrame]:
    """Load tracking results from JSON format.

    Args:
        input_path: Input file path

    Returns:
        Dictionary mapping layer names to DataFrames

    Example:
        >>> results = load_from_json("results.json")

    Raises:
        FileNotFoundError: If input_path doesn't exist
        ValueError: If JSON format is invalid
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    with open(input_path, "r") as f:
        json_data = json.load(f)

    if not isinstance(json_data, dict):
        raise ValueError("JSON must be a dictionary mapping layer names to data")

    # Convert back to DataFrames
    results = {}
    for layer_name, data in json_data.items():
        results[layer_name] = pd.DataFrame(data)

    return results


def export_metadata(metadata: Dict[str, Any], output_path: str, pretty: bool = True) -> None:
    """Export tracker metadata to JSON.

    Args:
        metadata: Dictionary containing tracker configuration and metadata
        output_path: Output file path
        pretty: If True, format JSON with indentation

    Example:
        >>> metadata = {
        ...     "model_name": "ResNet18",
        ...     "dataset": "CIFAR-10",
        ...     "sampling_frequency": 10,
        ...     "total_steps": 5000
        ... }
        >>> export_metadata(metadata, "metadata.json")
    """
    output_path = Path(output_path)

    with open(output_path, "w") as f:
        if pretty:
            json.dump(metadata, f, indent=2)
        else:
            json.dump(metadata, f)

    print(f"Exported metadata to {output_path}")
