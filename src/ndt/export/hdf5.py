"""HDF5 export functionality for large-scale tracking results."""

from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import h5py
import numpy as np
import pandas as pd


def export_to_hdf5(
    results: Dict[str, pd.DataFrame],
    output_path: str,
    compression: str = "gzip",
    compression_opts: int = 4,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Export tracking results to HDF5 format (efficient for large data).

    Args:
        results: Dictionary mapping layer names to DataFrames
        output_path: Output HDF5 file path
        compression: Compression algorithm ("gzip", "lzf", or None)
        compression_opts: Compression level (0-9 for gzip)
        metadata: Optional metadata to store as attributes

    Example:
        >>> results = tracker.get_results()
        >>> metadata = {"model": "ResNet18", "dataset": "CIFAR-10"}
        >>> export_to_hdf5(results, "results.h5", metadata=metadata)

    Raises:
        ValueError: If output_path is invalid
    """
    output_path = Path(output_path)

    with h5py.File(output_path, "w") as f:
        # Store metadata as root attributes
        if metadata:
            for key, value in metadata.items():
                f.attrs[key] = value

        # Store each layer's data as a group
        for layer_name, df in results.items():
            # Create group for this layer
            group = f.create_group(layer_name)

            # Store each column as a dataset
            for col in df.columns:
                data = df[col].values

                # Handle different data types
                if data.dtype == object:
                    # Convert to string for object columns
                    data = data.astype(str)

                group.create_dataset(
                    col,
                    data=data,
                    compression=compression,
                    compression_opts=compression_opts if compression == "gzip" else None,
                )

            # Store column names as attribute
            group.attrs["columns"] = list(df.columns)

    print(f"Exported results to {output_path}")


def load_from_hdf5(input_path: str) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """Load tracking results from HDF5 format.

    Args:
        input_path: Input HDF5 file path

    Returns:
        Tuple of (results_dict, metadata_dict)

    Example:
        >>> results, metadata = load_from_hdf5("results.h5")
        >>> print(f"Model: {metadata['model']}")

    Raises:
        FileNotFoundError: If input_path doesn't exist
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    results = {}
    metadata = {}

    with h5py.File(input_path, "r") as f:
        # Load root metadata
        for key in f.attrs.keys():
            metadata[key] = f.attrs[key]

        # Load each layer's data
        for layer_name in f.keys():
            group = f[layer_name]

            # Reconstruct DataFrame
            data = {}
            for col in group.attrs["columns"]:
                data[col] = group[col][:]

            results[layer_name] = pd.DataFrame(data)

    return results, metadata


def append_to_hdf5(
    results: Dict[str, pd.DataFrame], output_path: str, compression: str = "gzip"
) -> None:
    """Append new tracking results to existing HDF5 file.

    Useful for checkpointing during long training runs.

    Args:
        results: Dictionary mapping layer names to DataFrames with new data
        output_path: Output HDF5 file path
        compression: Compression algorithm

    Example:
        >>> # After training 1000 more steps
        >>> new_results = tracker.get_results()
        >>> append_to_hdf5(new_results, "results.h5")

    Raises:
        FileNotFoundError: If output_path doesn't exist
        ValueError: If layer structure doesn't match
    """
    output_path = Path(output_path)

    if not output_path.exists():
        raise FileNotFoundError(
            f"File not found: {output_path}. Use export_to_hdf5() to create new file."
        )

    with h5py.File(output_path, "a") as f:
        for layer_name, new_df in results.items():
            if layer_name not in f:
                raise ValueError(f"Layer {layer_name} not found in existing file")

            group = f[layer_name]

            # For each column, append data
            for col in new_df.columns:
                if col not in group:
                    raise ValueError(f"Column {col} not found in layer {layer_name}")

                # Load existing data
                existing_data = group[col][:]
                new_data = new_df[col].values

                # Concatenate
                combined_data = np.concatenate([existing_data, new_data])

                # Delete old dataset and create new one
                del group[col]
                group.create_dataset(col, data=combined_data, compression=compression)

    print(f"Appended results to {output_path}")
