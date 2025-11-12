"""Interactive Plotly-based visualizations for dimensionality tracking."""

from typing import Dict
from typing import List
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_interactive_plot(
    df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    layer_name: str = "Layer",
    show_loss: bool = True,
) -> go.Figure:
    """Create interactive plot of dimensionality metrics.

    Args:
        df: DataFrame with metrics
        metrics: List of metric names to plot (default: all 4 main metrics)
        layer_name: Name of the layer for title
        show_loss: Whether to include loss curve

    Returns:
        Plotly Figure object

    Example:
        >>> fig = create_interactive_plot(results_df, layer_name="Linear_0")
        >>> fig.show()
        >>> fig.write_html("interactive_plot.html")
    """
    if metrics is None:
        metrics = ["stable_rank", "participation_ratio", "cumulative_90", "nuclear_norm_ratio"]

    # Filter to available metrics
    metrics = [m for m in metrics if m in df.columns]

    # Create subplots
    n_plots = len(metrics) + (1 if show_loss and "loss" in df.columns else 0)
    fig = make_subplots(
        rows=n_plots,
        cols=1,
        subplot_titles=[m.replace("_", " ").title() for m in metrics]
        + (["Loss"] if show_loss and "loss" in df.columns else []),
        vertical_spacing=0.05,
    )

    # Add metric traces
    for i, metric in enumerate(metrics, 1):
        fig.add_trace(
            go.Scatter(
                x=df["step"],
                y=df[metric],
                mode="lines",
                name=metric.replace("_", " ").title(),
                line=dict(width=2),
                hovertemplate="Step: %{x}<br>" + metric + ": %{y:.2f}<extra></extra>",
            ),
            row=i,
            col=1,
        )

    # Add loss if requested
    if show_loss and "loss" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["step"],
                y=df["loss"],
                mode="lines",
                name="Loss",
                line=dict(width=2, color="coral"),
                hovertemplate="Step: %{x}<br>Loss: %{y:.3f}<extra></extra>",
            ),
            row=n_plots,
            col=1,
        )

    # Update layout
    fig.update_layout(
        title_text=f"Dimensionality Metrics - {layer_name}",
        title_font_size=18,
        height=300 * n_plots,
        showlegend=False,
        hovermode="x unified",
    )

    # Update x-axes
    fig.update_xaxes(title_text="Training Step", row=n_plots, col=1)

    return fig


def create_multi_layer_plot(
    results_dict: Dict[str, pd.DataFrame], metric: str = "stable_rank"
) -> go.Figure:
    """Create interactive plot comparing a metric across multiple layers.

    Args:
        results_dict: Dictionary mapping layer names to DataFrames
        metric: Metric to plot

    Returns:
        Plotly Figure object

    Example:
        >>> results = tracker.get_results()
        >>> fig = create_multi_layer_plot(results, metric="stable_rank")
        >>> fig.show()
    """
    fig = go.Figure()

    for layer_name, df in results_dict.items():
        if metric in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["step"],
                    y=df[metric],
                    mode="lines",
                    name=layer_name,
                    line=dict(width=2),
                    hovertemplate=(
                        f"{layer_name}<br>Step: %{{x}}<br>{metric}: %{{y:.2f}}<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        title=f'{metric.replace("_", " ").title()} Across Layers',
        title_font_size=18,
        xaxis_title="Training Step",
        yaxis_title=metric.replace("_", " ").title(),
        hovermode="x unified",
        height=600,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    return fig


def create_dashboard(
    results_dict: Dict[str, pd.DataFrame], layer_name: Optional[str] = None
) -> go.Figure:
    """Create comprehensive dashboard with all metrics.

    Args:
        results_dict: Dictionary mapping layer names to DataFrames
        layer_name: If specified, show detailed view for one layer.
                   If None, show comparison across layers.

    Returns:
        Plotly Figure object

    Example:
        >>> results = tracker.get_results()
        >>> fig = create_dashboard(results, layer_name="Linear_0")
        >>> fig.write_html("dashboard.html")
    """
    if layer_name is not None:
        # Single-layer detailed dashboard
        df = results_dict[layer_name]

        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=(
                "Stable Rank",
                "Participation Ratio",
                "Cumulative 90%",
                "Nuclear Norm Ratio",
                "Training Loss",
                "Gradient Norm",
            ),
            vertical_spacing=0.1,
            horizontal_spacing=0.1,
        )

        metrics = [
            ("stable_rank", 1, 1),
            ("participation_ratio", 1, 2),
            ("cumulative_90", 2, 1),
            ("nuclear_norm_ratio", 2, 2),
            ("loss", 3, 1),
            ("grad_norm", 3, 2),
        ]

        for metric, row, col in metrics:
            if metric in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df["step"],
                        y=df[metric],
                        mode="lines",
                        name=metric.replace("_", " ").title(),
                        line=dict(width=2),
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )

        fig.update_layout(
            title_text=f"Comprehensive Dashboard - {layer_name}", title_font_size=20, height=900
        )

    else:
        # Multi-layer comparison dashboard
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Stable Rank",
                "Participation Ratio",
                "Cumulative 90%",
                "Nuclear Norm Ratio",
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.1,
        )

        metrics = [
            ("stable_rank", 1, 1),
            ("participation_ratio", 1, 2),
            ("cumulative_90", 2, 1),
            ("nuclear_norm_ratio", 2, 2),
        ]

        for metric, row, col in metrics:
            for layer_name, df in results_dict.items():
                if metric in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df["step"],
                            y=df[metric],
                            mode="lines",
                            name=layer_name,
                            line=dict(width=2),
                            showlegend=(row == 1 and col == 1),  # Only show legend once
                        ),
                        row=row,
                        col=col,
                    )

        fig.update_layout(
            title_text="Multi-Layer Comparison Dashboard", title_font_size=20, height=800
        )

    fig.update_xaxes(title_text="Training Step")

    return fig
