"""Matplotlib-based plotting utilities for dimensionality tracking."""

from typing import Dict
from typing import List
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ndt.core.jump_detector import Jump

# Set nice default style
sns.set_style("whitegrid")
sns.set_palette("husl")


def plot_single_metric(
    df: pd.DataFrame,
    metric: str = "stable_rank",
    layer_name: str = "Layer",
    ax: Optional[plt.Axes] = None,
    show_loss: bool = True,
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """Plot a single dimensionality metric over training.

    Args:
        df: DataFrame with metrics (must have 'step' and metric columns)
        metric: Name of metric to plot
        layer_name: Name of the layer for title
        ax: Optional matplotlib axis to plot on
        show_loss: Whether to show loss on secondary y-axis
        figsize: Figure size if creating new figure

    Returns:
        Matplotlib figure object

    Example:
        >>> fig = plot_single_metric(results_df, metric="stable_rank", layer_name="Linear_0")
        >>> plt.savefig("stable_rank.png")
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Plot main metric
    ax.plot(df["step"], df[metric], linewidth=2, label=metric, color="steelblue")
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12, color="steelblue")
    ax.tick_params(axis="y", labelcolor="steelblue")
    ax.grid(True, alpha=0.3)

    # Plot loss on secondary axis if requested
    if show_loss and "loss" in df.columns:
        ax2 = ax.twinx()
        ax2.plot(
            df["step"],
            df["loss"],
            linewidth=1,
            alpha=0.6,
            label="Loss",
            color="coral",
            linestyle="--",
        )
        ax2.set_ylabel("Loss", fontsize=12, color="coral")
        ax2.tick_params(axis="y", labelcolor="coral")

    ax.set_title(
        f'{metric.replace("_", " ").title()} - {layer_name}', fontsize=14, fontweight="bold"
    )
    fig.tight_layout()

    return fig


def plot_metrics_comparison(
    df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    layer_name: str = "Layer",
    figsize: tuple = (15, 10),
) -> plt.Figure:
    """Plot multiple dimensionality metrics for comparison.

    Args:
        df: DataFrame with metrics
        metrics: List of metric names to plot (default: all 4 main metrics)
        layer_name: Name of the layer for title
        figsize: Figure size

    Returns:
        Matplotlib figure object

    Example:
        >>> fig = plot_metrics_comparison(results_df, layer_name="Linear_0")
        >>> plt.savefig("all_metrics.png")
    """
    if metrics is None:
        metrics = ["stable_rank", "participation_ratio", "cumulative_90", "nuclear_norm_ratio"]

    # Filter to available metrics
    metrics = [m for m in metrics if m in df.columns]

    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=figsize, sharex=True)

    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        ax.plot(df["step"], df[metric], linewidth=2, color="steelblue")
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=11)
        ax.grid(True, alpha=0.3)

        # Add loss as background on last plot
        if ax == axes[-1] and "loss" in df.columns:
            ax2 = ax.twinx()
            ax2.plot(df["step"], df["loss"], linewidth=1, alpha=0.4, color="coral", linestyle="--")
            ax2.set_ylabel("Loss", fontsize=10, color="coral")
            ax2.tick_params(axis="y", labelcolor="coral")

    axes[-1].set_xlabel("Training Step", fontsize=12)
    fig.suptitle(f"Dimensionality Metrics - {layer_name}", fontsize=14, fontweight="bold")
    fig.tight_layout()

    return fig


def plot_phases(
    results_dict: Dict[str, pd.DataFrame], metric: str = "stable_rank", figsize: tuple = (15, 8)
) -> plt.Figure:
    """Plot metric across multiple layers to visualize phase transitions.

    Args:
        results_dict: Dictionary mapping layer names to DataFrames
        metric: Metric to plot
        figsize: Figure size

    Returns:
        Matplotlib figure object

    Example:
        >>> results = tracker.get_results()
        >>> fig = plot_phases(results, metric="stable_rank")
        >>> plt.savefig("phases.png")
    """
    fig, ax = plt.subplots(figsize=figsize)

    for layer_name, df in results_dict.items():
        if metric in df.columns:
            ax.plot(df["step"], df[metric], linewidth=2, label=layer_name, alpha=0.8)

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(
        f'{metric.replace("_", " ").title()} Across Layers', fontsize=14, fontweight="bold"
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig


def plot_jumps(
    df: pd.DataFrame,
    jumps: List[Jump],
    metric: str = "stable_rank",
    layer_name: str = "Layer",
    figsize: tuple = (14, 6),
) -> plt.Figure:
    """Plot metric with detected jumps highlighted.

    Args:
        df: DataFrame with metrics
        jumps: List of detected Jump objects
        metric: Metric name that was analyzed for jumps
        layer_name: Name of the layer
        figsize: Figure size

    Returns:
        Matplotlib figure object

    Example:
        >>> jumps = tracker.detect_jumps(layer_name="Linear_0", metric="stable_rank")
        >>> fig = plot_jumps(results_df, jumps["Linear_0"], metric="stable_rank")
        >>> plt.savefig("jumps.png")
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot metric
    ax.plot(df["step"], df[metric], linewidth=2, color="steelblue", label=metric)

    # Highlight jumps
    if jumps:
        jump_steps = [j.step for j in jumps]
        jump_values = [
            (
                df[df["step"] == j.step][metric].values[0]
                if j.step in df["step"].values
                else j.value_after
            )
            for j in jumps
        ]

        ax.scatter(
            jump_steps,
            jump_values,
            color="red",
            s=100,
            zorder=5,
            label=f"Jumps (n={len(jumps)})",
            marker="o",
            edgecolors="darkred",
            linewidths=2,
        )

        # Add vertical lines at jumps
        for step in jump_steps:
            ax.axvline(x=step, color="red", linestyle="--", alpha=0.3, linewidth=1)

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(f"Dimensionality Jumps - {layer_name}", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame, metrics: Optional[List[str]] = None, figsize: tuple = (10, 8)
) -> plt.Figure:
    """Plot correlation heatmap between metrics.

    Args:
        df: DataFrame with metrics
        metrics: List of metrics to include (default: all numeric columns)
        figsize: Figure size

    Returns:
        Matplotlib figure object

    Example:
        >>> fig = plot_correlation_heatmap(results_df)
        >>> plt.savefig("correlations.png")
    """
    if metrics is None:
        # Use all numeric columns except 'step'
        metrics = [col for col in df.select_dtypes(include=[np.number]).columns if col != "step"]

    correlation_matrix = df[metrics].corr()

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )

    ax.set_title("Metric Correlations", fontsize=14, fontweight="bold")
    fig.tight_layout()

    return fig
