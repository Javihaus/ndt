# API Reference

This page provides the API reference for **ndt** (Neural Dimensionality Tracker). All public classes and functions are documented with their signatures, parameters, return types, and examples.

For auto-generated documentation from source docstrings, ensure `mkdocstrings` is configured in your MkDocs build.

## Core

### HighFrequencyTracker

::: ndt.core.tracker.HighFrequencyTracker

### DimensionalityMetrics

::: ndt.core.tracker.DimensionalityMetrics

### ActivationCapture

::: ndt.core.hooks.ActivationCapture

### JumpDetector

::: ndt.core.jump_detector.JumpDetector

### Jump

::: ndt.core.jump_detector.Jump

## Dimensionality Estimators

### stable_rank

::: ndt.core.estimators.stable_rank

### participation_ratio

::: ndt.core.estimators.participation_ratio

### cumulative_energy_90

::: ndt.core.estimators.cumulative_energy_90

### nuclear_norm_ratio

::: ndt.core.estimators.nuclear_norm_ratio

### compute_all_metrics

::: ndt.core.estimators.compute_all_metrics

## Architectures

### detect_architecture

::: ndt.architectures.detect_architecture

### get_handler

::: ndt.architectures.get_handler

### MLPHandler

::: ndt.architectures.MLPHandler

### CNNHandler

::: ndt.architectures.CNNHandler

### TransformerHandler

::: ndt.architectures.TransformerHandler

### ViTHandler

::: ndt.architectures.ViTHandler

## Visualization

### plot_phases

::: ndt.visualization.plot_phases

### plot_jumps

::: ndt.visualization.plot_jumps

### plot_metrics_comparison

::: ndt.visualization.plot_metrics_comparison

### plot_single_metric

::: ndt.visualization.plot_single_metric

### create_interactive_plot

::: ndt.visualization.create_interactive_plot

### create_multi_layer_plot

::: ndt.visualization.create_multi_layer_plot

## Export

### export_to_csv

::: ndt.export.export_to_csv

### export_to_json

::: ndt.export.export_to_json

### export_to_hdf5

::: ndt.export.export_to_hdf5

## Utilities

### setup_logger

::: ndt.utils.setup_logger

### load_config

::: ndt.utils.load_config

### save_config

::: ndt.utils.save_config
