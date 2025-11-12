"""CSV export functionality for tracking results."""

from pathlib import Path
from typing import Dict

import pandas as pd


def export_to_csv(
    results: Dict[str, pd.DataFrame], output_path: str, separate_files: bool = False
) -> None:
    """Export tracking results to CSV format.

    Args:
        results: Dictionary mapping layer names to DataFrames
        output_path: Output file path or directory path
        separate_files: If True, create separate CSV for each layer.
                       If False, combine all layers into one CSV with layer column.

    Example:
        >>> results = tracker.get_results()
        >>> export_to_csv(results, "results.csv")
        >>> # Or separate files:
        >>> export_to_csv(results, "results_dir/", separate_files=True)

    Raises:
        ValueError: If output_path is invalid
    """
    output_path = Path(output_path)

    if separate_files:
        # Create directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)

        # Export each layer to separate file
        for layer_name, df in results.items():
            filename = output_path / f"{layer_name}.csv"
            df.to_csv(filename, index=False)
            print(f"Exported {layer_name} to {filename}")

    else:
        # Combine all layers into one CSV
        combined_data = []
        for layer_name, df in results.items():
            df_copy = df.copy()
            df_copy["layer"] = layer_name
            combined_data.append(df_copy)

        if combined_data:
            combined_df = pd.concat(combined_data, ignore_index=True)

            # Reorder columns to put layer first
            cols = ["layer"] + [col for col in combined_df.columns if col != "layer"]
            combined_df = combined_df[cols]

            combined_df.to_csv(output_path, index=False)
            print(f"Exported all layers to {output_path}")
        else:
            print("No data to export")


def load_from_csv(input_path: str, separate_files: bool = False) -> Dict[str, pd.DataFrame]:
    """Load tracking results from CSV format.

    Args:
        input_path: Input file path or directory path
        separate_files: If True, load separate CSVs from directory.
                       If False, load combined CSV and split by layer.

    Returns:
        Dictionary mapping layer names to DataFrames

    Example:
        >>> results = load_from_csv("results.csv")
        >>> # Or from directory:
        >>> results = load_from_csv("results_dir/", separate_files=True)

    Raises:
        FileNotFoundError: If input_path doesn't exist
        ValueError: If CSV format is invalid
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Path not found: {input_path}")

    if separate_files:
        # Load from directory
        if not input_path.is_dir():
            raise ValueError(f"Expected directory for separate_files=True, got: {input_path}")

        results = {}
        for csv_file in input_path.glob("*.csv"):
            layer_name = csv_file.stem
            df = pd.read_csv(csv_file)
            results[layer_name] = df

        return results

    else:
        # Load combined CSV
        df = pd.read_csv(input_path)

        if "layer" not in df.columns:
            raise ValueError("CSV must have 'layer' column for combined format")

        results = {}
        for layer_name in df["layer"].unique():
            layer_df = df[df["layer"] == layer_name].copy()
            layer_df = layer_df.drop(columns=["layer"])
            results[layer_name] = layer_df

        return results
