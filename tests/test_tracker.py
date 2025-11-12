"""Tests for HighFrequencyTracker."""

import pytest

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
