"""Tests for matplotlib and plotly visualizations."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from ndt.core.jump_detector import Jump
from ndt.visualization import create_interactive_plot
from ndt.visualization import create_multi_layer_plot
from ndt.visualization import plot_jumps
from ndt.visualization import plot_metrics_comparison
from ndt.visualization import plot_phases
from ndt.visualization import plot_single_metric
from ndt.visualization.interactive import create_dashboard
from ndt.visualization.plots import plot_correlation_heatmap

METRICS = ["stable_rank", "participation_ratio", "cumulative_90", "nuclear_norm_ratio"]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def make_df(offset=0.0, with_loss=True, with_grad=False):
    steps = np.arange(0, 50, 5)
    data = {
        "step": steps,
        "stable_rank": 1.0 + 0.1 * steps + offset,
        "participation_ratio": 2.0 + 0.05 * steps + offset,
        "cumulative_90": (3 + steps // 10).astype(float),
        "nuclear_norm_ratio": 0.5 + 0.01 * steps,
    }
    if with_loss:
        data["loss"] = 1.0 / (1.0 + steps)
    if with_grad:
        data["grad_norm"] = 0.2 + 0.0 * steps
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Matplotlib plots
# ---------------------------------------------------------------------------


class TestPlotSingleMetric:
    def test_with_loss(self):
        df = make_df()
        fig = plot_single_metric(df, metric="stable_rank", layer_name="L0")
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2  # main + twin loss axis
        main, twin = fig.axes
        np.testing.assert_allclose(main.lines[0].get_xdata(), df["step"])
        np.testing.assert_allclose(main.lines[0].get_ydata(), df["stable_rank"])
        np.testing.assert_allclose(twin.lines[0].get_ydata(), df["loss"])
        assert main.get_title() == "Stable Rank - L0"
        assert main.get_ylabel() == "Stable Rank"
        assert main.get_xlabel() == "Training Step"
        assert twin.get_ylabel() == "Loss"

    def test_without_loss(self):
        fig = plot_single_metric(make_df(), metric="cumulative_90", show_loss=False)
        assert len(fig.axes) == 1
        assert fig.axes[0].get_title() == "Cumulative 90 - Layer"

    def test_loss_column_missing(self):
        fig = plot_single_metric(make_df(with_loss=False))
        assert len(fig.axes) == 1

    def test_uses_given_axis(self):
        fig, ax = plt.subplots()
        returned = plot_single_metric(make_df(), ax=ax, show_loss=False)
        assert returned is fig
        assert len(ax.lines) == 1


class TestPlotMetricsComparison:
    def test_default_metrics(self):
        df = make_df()
        fig = plot_metrics_comparison(df, layer_name="L1")
        # 4 metric axes + 1 twin axis for loss on the last subplot.
        assert len(fig.axes) == 5
        for ax, metric in zip(fig.axes[:4], METRICS):
            np.testing.assert_allclose(ax.lines[0].get_ydata(), df[metric])
            assert ax.get_ylabel() == metric.replace("_", " ").title()
        np.testing.assert_allclose(fig.axes[4].lines[0].get_ydata(), df["loss"])
        assert fig._suptitle.get_text() == "Dimensionality Metrics - L1"
        assert fig.axes[3].get_xlabel() == "Training Step"

    def test_single_metric_and_missing_filtered(self):
        df = make_df(with_loss=False)
        fig = plot_metrics_comparison(df, metrics=["stable_rank", "does_not_exist"])
        assert len(fig.axes) == 1
        np.testing.assert_allclose(fig.axes[0].lines[0].get_ydata(), df["stable_rank"])


class TestPlotPhases:
    def test_one_line_per_layer_with_metric(self):
        results = {
            "a": make_df(0.0),
            "b": make_df(1.0),
            "no_metric": pd.DataFrame({"step": [0, 1], "loss": [1.0, 0.5]}),
        }
        fig = plot_phases(results, metric="participation_ratio")
        ax = fig.axes[0]
        assert [line.get_label() for line in ax.lines] == ["a", "b"]
        np.testing.assert_allclose(ax.lines[1].get_ydata(), results["b"]["participation_ratio"])
        assert ax.get_title() == "Participation Ratio Across Layers"
        assert [t.get_text() for t in ax.get_legend().get_texts()] == ["a", "b"]


class TestPlotJumps:
    def test_jumps_highlighted(self):
        df = make_df()
        jumps = [
            Jump(step=10, z_score=4.0, value_before=1.5, value_after=2.0, metric_name="sr"),
            Jump(step=12, z_score=5.0, value_before=2.0, value_after=9.9, metric_name="sr"),
        ]
        fig = plot_jumps(df, jumps, metric="stable_rank", layer_name="L2")
        ax = fig.axes[0]
        offsets = ax.collections[0].get_offsets()
        # Step 10 is in the frame -> value read from df; step 12 is not -> value_after.
        np.testing.assert_allclose(offsets, [[10, 2.0], [12, 9.9]])
        # One metric line + one vertical line per jump.
        assert len(ax.lines) == 3
        assert [line.get_xdata()[0] for line in ax.lines[1:]] == [10, 12]
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert labels == ["stable_rank", "Jumps (n=2)"]
        assert ax.get_title() == "Dimensionality Jumps - L2"

    def test_no_jumps(self):
        fig = plot_jumps(make_df(), [], metric="stable_rank")
        ax = fig.axes[0]
        assert len(ax.collections) == 0
        assert len(ax.lines) == 1


class TestCorrelationHeatmap:
    def test_default_metrics_exclude_step(self):
        df = make_df()
        fig = plot_correlation_heatmap(df)
        ax = fig.axes[0]
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert labels == METRICS + ["loss"]
        assert "step" not in labels
        annotations = [float(t.get_text()) for t in ax.texts]
        expected = df[METRICS + ["loss"]].corr().to_numpy().ravel()
        np.testing.assert_allclose(annotations, np.round(expected, 2), atol=1e-9)
        assert ax.get_title() == "Metric Correlations"

    def test_explicit_metrics(self):
        df = make_df()
        fig = plot_correlation_heatmap(df, metrics=["stable_rank", "loss"])
        ax = fig.axes[0]
        assert len(ax.texts) == 4
        assert float(ax.texts[0].get_text()) == 1.0


# ---------------------------------------------------------------------------
# Plotly
# ---------------------------------------------------------------------------


class TestInteractivePlot:
    def test_default_metrics_with_loss(self, tmp_path):
        df = make_df()
        fig = create_interactive_plot(df, layer_name="L0")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 5
        assert [t.name for t in fig.data] == [
            "Stable Rank",
            "Participation Ratio",
            "Cumulative 90",
            "Nuclear Norm Ratio",
            "Loss",
        ]
        np.testing.assert_allclose(fig.data[0].y, df["stable_rank"])
        np.testing.assert_allclose(fig.data[4].y, df["loss"])
        assert fig.data[4].line.color == "coral"
        assert fig.layout.title.text == "Dimensionality Metrics - L0"
        assert fig.layout.height == 300 * 5
        titles = [a.text for a in fig.layout.annotations]
        assert titles[-1] == "Loss"

        out = tmp_path / "plot.html"
        fig.write_html(str(out), include_plotlyjs=False, auto_open=False)
        assert "Dimensionality Metrics - L0" in out.read_text()

    def test_without_loss_and_filtered_metrics(self):
        df = make_df()
        fig = create_interactive_plot(df, metrics=["stable_rank", "bogus"], show_loss=False)
        assert len(fig.data) == 1
        assert fig.layout.height == 300
        assert [a.text for a in fig.layout.annotations] == ["Stable Rank"]


class TestMultiLayerPlot:
    def test_traces_per_layer(self):
        results = {"a": make_df(0.0), "b": make_df(2.0), "c": pd.DataFrame({"step": [0]})}
        fig = create_multi_layer_plot(results, metric="stable_rank")
        assert [t.name for t in fig.data] == ["a", "b"]
        np.testing.assert_allclose(fig.data[1].y, results["b"]["stable_rank"])
        assert fig.layout.title.text == "Stable Rank Across Layers"
        assert fig.layout.yaxis.title.text == "Stable Rank"
        assert fig.layout.xaxis.title.text == "Training Step"


class TestDashboard:
    def test_single_layer(self):
        df = make_df(with_grad=True)
        fig = create_dashboard({"L0": df, "L1": make_df()}, layer_name="L0")
        assert len(fig.data) == 6
        assert fig.layout.title.text == "Comprehensive Dashboard - L0"
        assert fig.layout.height == 900
        np.testing.assert_allclose(fig.data[5].y, df["grad_norm"])
        assert all(t.showlegend is False for t in fig.data)

    def test_single_layer_missing_columns(self):
        fig = create_dashboard({"L0": make_df(with_loss=False)}, layer_name="L0")
        assert len(fig.data) == 4

    def test_multi_layer(self):
        results = {"a": make_df(0.0), "b": make_df(1.0)}
        fig = create_dashboard(results)
        assert len(fig.data) == 4 * 2
        assert fig.layout.title.text == "Multi-Layer Comparison Dashboard"
        assert fig.layout.height == 800
        # Legend only shown for the first subplot's traces.
        assert [t.showlegend for t in fig.data] == [True, True] + [False] * 6
        assert [t.name for t in fig.data[:2]] == ["a", "b"]

    def test_unknown_layer_raises(self):
        with pytest.raises(KeyError):
            create_dashboard({"a": make_df()}, layer_name="zzz")
