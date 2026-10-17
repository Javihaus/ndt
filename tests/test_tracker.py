"""Tests for HighFrequencyTracker."""

import logging

import pytest
import torch
import torch.nn as nn

from ndt.core.tracker import DimensionalityMetrics
from ndt.core.tracker import HighFrequencyTracker


class TestDimensionalityMetrics:
    """Tests for DimensionalityMetrics dataclass."""

    def test_creation(self):
        """Should create metrics object successfully."""
        metrics = DimensionalityMetrics(
            step=100,
            stable_rank=25.5,
            participation_ratio=20.3,
            cumulative_90=18,
            nuclear_norm_ratio=15.2,
            loss=0.5,
            grad_norm=1.2,
        )

        assert metrics.step == 100
        assert metrics.stable_rank == 25.5
        assert metrics.loss == 0.5

    def test_to_dict(self):
        """Should convert to dictionary."""
        metrics = DimensionalityMetrics(
            step=100,
            stable_rank=25.5,
            participation_ratio=20.3,
            cumulative_90=18,
            nuclear_norm_ratio=15.2,
            loss=0.5,
        )

        d = metrics.to_dict()
        assert d["step"] == 100
        assert d["stable_rank"] == 25.5
        assert "grad_norm" in d


class TestHighFrequencyTracker:
    """Tests for HighFrequencyTracker class."""

    def test_initialization_with_mlp(self, small_mlp):
        """Should initialize successfully with MLP."""
        tracker = HighFrequencyTracker(small_mlp, layers=[small_mlp[0], small_mlp[2], small_mlp[4]])

        assert tracker.model is small_mlp
        assert len(tracker.layer_names) == 3
        tracker.close()

    def test_initialization_auto_detect(self, small_mlp):
        """Should auto-detect layers if not provided."""
        tracker = HighFrequencyTracker(small_mlp)

        assert len(tracker.layer_names) > 0
        tracker.close()

    def test_single_log_call(self, small_mlp, sample_batch):
        """Should log metrics after forward pass."""
        tracker = HighFrequencyTracker(small_mlp, layers=[small_mlp[0]])

        # Forward pass (triggers hooks)
        _ = small_mlp(sample_batch)

        # Log metrics
        tracker.log(step=0, loss=1.0)

        # Check that metrics were recorded
        results = tracker.get_results()
        assert len(results) == 1

        layer_df = list(results.values())[0]
        assert len(layer_df) == 1
        assert layer_df["step"].iloc[0] == 0
        assert "stable_rank" in layer_df.columns

        tracker.close()

    def test_multiple_log_calls(self, small_mlp, sample_batch):
        """Should accumulate metrics over multiple calls."""
        tracker = HighFrequencyTracker(small_mlp, layers=[small_mlp[0]])

        for step in range(10):
            _ = small_mlp(sample_batch)
            tracker.log(step=step, loss=1.0 / (step + 1))

        results = tracker.get_results()
        layer_df = list(results.values())[0]

        assert len(layer_df) == 10
        assert layer_df["step"].tolist() == list(range(10))

        tracker.close()

    def test_sampling_frequency(self, small_mlp, sample_batch):
        """Should respect sampling frequency."""
        tracker = HighFrequencyTracker(small_mlp, layers=[small_mlp[0]], sampling_frequency=5)

        for step in range(20):
            _ = small_mlp(sample_batch)
            tracker.log(step=step, loss=1.0)

        results = tracker.get_results()
        layer_df = list(results.values())[0]

        # Should only log at steps 0, 5, 10, 15
        assert len(layer_df) == 4

        tracker.close()

    def test_force_logging(self, small_mlp, sample_batch):
        """Should log when forced even if not at sampling interval."""
        tracker = HighFrequencyTracker(small_mlp, layers=[small_mlp[0]], sampling_frequency=10)

        # Log at step 3 with force=True
        _ = small_mlp(sample_batch)
        tracker.log(step=3, loss=1.0, force=True)

        results = tracker.get_results()
        layer_df = list(results.values())[0]

        assert len(layer_df) == 1
        assert layer_df["step"].iloc[0] == 3

        tracker.close()

    def test_context_manager(self, small_mlp, sample_batch):
        """Should work as context manager."""
        with HighFrequencyTracker(small_mlp, layers=[small_mlp[0]]) as tracker:
            _ = small_mlp(sample_batch)
            tracker.log(step=0, loss=1.0)

            results = tracker.get_results()
            assert len(results) > 0

        # Hooks should be removed after context exit
        assert len(tracker.activation_capture.hooks) == 0

    def test_cnn_support(self, simple_cnn, sample_image_batch):
        """Should work with CNN architecture."""
        tracker = HighFrequencyTracker(simple_cnn, layers=[simple_cnn[0], simple_cnn[3]])

        _ = simple_cnn(sample_image_batch)
        tracker.log(step=0, loss=1.0)

        results = tracker.get_results()
        assert len(results) == 2

        tracker.close()

    def test_get_results_single_layer(self, small_mlp, sample_batch):
        """Should get results for specific layer."""
        tracker = HighFrequencyTracker(
            small_mlp, layers=[small_mlp[0], small_mlp[2]], layer_names=["layer0", "layer1"]
        )

        _ = small_mlp(sample_batch)
        tracker.log(step=0, loss=1.0)

        # Get all results
        all_results = tracker.get_results()
        assert len(all_results) == 2

        # Get specific layer
        layer0_results = tracker.get_results(layer_name="layer0")
        assert len(layer0_results) == 1

        tracker.close()

    def test_invalid_layer_name(self, small_mlp, sample_batch):
        """Should raise error for invalid layer name."""
        tracker = HighFrequencyTracker(small_mlp, layers=[small_mlp[0]])

        _ = small_mlp(sample_batch)
        tracker.log(step=0, loss=1.0)

        with pytest.raises(ValueError, match="Unknown layer name"):
            tracker.get_results(layer_name="nonexistent")

        tracker.close()


class TestJumpDetectionIntegration:
    """Tests for jump detection integration in tracker."""

    def test_jump_detection_enabled(self, small_mlp):
        """Should create jump detector when enabled."""
        tracker = HighFrequencyTracker(small_mlp, enable_jump_detection=True)

        assert tracker.jump_detector is not None
        tracker.close()

    def test_jump_detection_disabled(self, small_mlp):
        """Should not create jump detector when disabled."""
        tracker = HighFrequencyTracker(small_mlp, enable_jump_detection=False)

        assert tracker.jump_detector is None
        tracker.close()

    def test_detect_jumps_raises_when_disabled(self, small_mlp, sample_batch):
        """Should raise error if jump detection disabled."""
        tracker = HighFrequencyTracker(small_mlp, enable_jump_detection=False)

        _ = small_mlp(sample_batch)
        tracker.log(step=0, loss=1.0)

        with pytest.raises(ValueError, match="Jump detection is disabled"):
            tracker.detect_jumps()

        tracker.close()


class TestTrackerEdgeCases:
    """Validation, auto-detection and activation-shape handling."""

    def test_rejects_non_module(self):
        with pytest.raises(ValueError, match="must be a torch.nn.Module"):
            HighFrequencyTracker("not a model")

    def test_no_trackable_layers(self):
        with pytest.raises(ValueError, match="No layers to track"):
            HighFrequencyTracker(nn.Sequential(nn.LayerNorm(4)))

    def test_auto_detect_includes_linear_and_conv(self, simple_cnn):
        tracker = HighFrequencyTracker(simple_cnn)
        assert tracker.layer_names == ["Conv2d_0", "Conv2d_1", "Linear_2"]
        tracker.close()

    def test_auto_detect_subsamples_more_than_20_layers(self):
        model = nn.Sequential(*[nn.Linear(3, 3) for _ in range(21)])
        tracker = HighFrequencyTracker(model)
        expected = [model[i] for i in range(0, 21, 2)]
        assert list(tracker.activation_capture._layer_names) == expected
        assert len(tracker.layer_names) == 11
        tracker.close()

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "src bug: tracker.py _auto_detect_layers halves the list inside the loop, "
            "so with 30 layers it keeps 20 (early layers thinned, late layers all kept) "
            "instead of every other layer"
        ),
    )
    def test_auto_detect_samples_evenly_for_30_layers(self):
        model = nn.Sequential(*[nn.Linear(3, 3) for _ in range(30)])
        tracker = HighFrequencyTracker(model)
        try:
            assert len(tracker.layer_names) == 15
        finally:
            tracker.close()

    def test_log_without_forward_warns_and_records_nothing(self, small_mlp, caplog):
        tracker = HighFrequencyTracker(small_mlp, layers=[small_mlp[0]], layer_names=["fc"])
        with caplog.at_level(logging.WARNING, logger="ndt.core.tracker"):
            tracker.log(step=3, loss=1.0)
        assert "No activation captured for layer fc at step 3" in caplog.text
        assert tracker.metrics_history["fc"] == []
        assert tracker.get_results("fc").empty
        tracker.close()

    def test_metric_errors_are_logged_not_raised(self, small_mlp, caplog):
        tracker = HighFrequencyTracker(small_mlp, layers=[small_mlp[0]], layer_names=["fc"])
        x = torch.randn(8, 10)
        x[0, 0] = float("nan")
        small_mlp(x)
        with caplog.at_level(logging.ERROR, logger="ndt.core.tracker"):
            tracker.log(step=0, loss=1.0)
        assert "Error computing metrics for fc at step 0" in caplog.text
        assert tracker.metrics_history["fc"] == []
        # Activations are still cleared for the next step.
        assert tracker.activation_capture.activations == {}
        tracker.close()

    def test_three_dimensional_activations(self):
        lin = nn.Linear(6, 4)
        tracker = HighFrequencyTracker(lin, layers=[lin], layer_names=["proj"])
        lin(torch.randn(2, 5, 6))  # (batch, seq, hidden)
        tracker.log(step=0, loss=0.5, grad_norm=0.1)
        df = tracker.get_results("proj")
        assert len(df) == 1
        # Flattened to (batch * seq, hidden) = (10, 4): rank is at most 4.
        assert 1.0 <= df["stable_rank"].iloc[0] <= 4.0
        assert df["grad_norm"].iloc[0] == pytest.approx(0.1)
        tracker.close()

    @pytest.mark.parametrize(
        "shape,expected",
        [
            ((3, 7), (3, 7)),
            ((2, 3, 4), (6, 4)),
            ((2, 3, 4, 5), (2, 60)),
            ((2, 1, 2, 2, 2), (2, 8)),
        ],
    )
    def test_prepare_activation_matrix(self, small_mlp, shape, expected):
        tracker = HighFrequencyTracker(small_mlp, layers=[small_mlp[0]])
        act = torch.arange(float(torch.Size(shape).numel())).view(shape)
        matrix = tracker._prepare_activation_matrix(act)
        assert tuple(matrix.shape) == expected
        torch.testing.assert_close(matrix.flatten(), act.flatten())
        tracker.close()

    def test_detect_jumps_invalid_metric(self, small_mlp):
        tracker = HighFrequencyTracker(small_mlp, layers=[small_mlp[0]])
        with pytest.raises(ValueError, match="metric must be one of"):
            tracker.detect_jumps(metric="loss")
        tracker.close()

    def test_detect_jumps_unknown_layer_is_skipped(self, small_mlp):
        tracker = HighFrequencyTracker(small_mlp, layers=[small_mlp[0]])
        assert tracker.detect_jumps(layer_name="nonexistent") == {}
        tracker.close()

    def test_detect_jumps_finds_rank_collapse(self):
        lin = nn.Linear(6, 6, bias=False)
        with torch.no_grad():
            lin.weight.copy_(torch.eye(6))
        tracker = HighFrequencyTracker(
            lin, layers=[lin], layer_names=["id"], jump_window_size=10, jump_z_threshold=3.0
        )
        gen = torch.Generator().manual_seed(0)
        for step in range(40):
            if step < 30:
                x = torch.randn(32, 6, generator=gen)
            else:
                # Rank-1 inputs: stable rank collapses to 1.
                x = torch.randn(32, 1, generator=gen) @ torch.ones(1, 6)
            lin(x)
            tracker.log(step=step, loss=1.0)

        jumps = tracker.detect_jumps(layer_name="id", metric="stable_rank")["id"]
        assert 30 in [j.step for j in jumps]
        first = next(j for j in jumps if j.step == 30)
        assert first.value_after == pytest.approx(1.0, abs=1e-3)
        assert first.value_before > 1.5
        assert first.metric_name == "id_stable_rank"
        tracker.close()
