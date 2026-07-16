"""Tests for jump detection."""

import numpy as np
import pytest

from ndt.core.jump_detector import Jump
from ndt.core.jump_detector import JumpDetector


class TestJump:
    """Tests for Jump dataclass."""

    def test_creation(self):
        """Should create jump object successfully."""
        jump = Jump(
            step=100, z_score=5.2, value_before=10.0, value_after=20.0, metric_name="stable_rank"
        )

        assert jump.step == 100
        assert jump.z_score == 5.2
        assert jump.metric_name == "stable_rank"

    def test_repr(self):
        """Should have readable string representation."""
        jump = Jump(
            step=100, z_score=5.2, value_before=10.0, value_after=20.0, metric_name="stable_rank"
        )

        repr_str = repr(jump)
        assert "step=100" in repr_str
        assert "stable_rank" in repr_str


class TestJumpDetector:
    """Tests for JumpDetector class."""

    def test_initialization(self):
        """Should initialize with valid parameters."""
        detector = JumpDetector(window_size=50, z_threshold=3.0)

        assert detector.window_size == 50
        assert detector.z_threshold == 3.0

    def test_invalid_initialization(self):
        """Should raise error for invalid parameters."""
        with pytest.raises(ValueError, match="window_size"):
            JumpDetector(window_size=0)

        with pytest.raises(ValueError, match="z_threshold"):
            JumpDetector(z_threshold=-1)

    def test_no_jumps_in_constant_signal(self):
        """Should detect no jumps in constant signal."""
        detector = JumpDetector(window_size=10, z_threshold=3.0)
        values = [10.0] * 100

        jumps = detector.detect_jumps(values)

        assert len(jumps) == 0

    def test_detect_single_jump(self):
        """Should detect a clear jump."""
        detector = JumpDetector(window_size=20, z_threshold=3.0, min_samples=10)

        # Create signal with jump at step 50
        values = [10.0] * 50 + [20.0] * 50

        jumps = detector.detect_jumps(values, metric_name="test_metric")

        assert len(jumps) >= 1
        # First jump should be near step 50
        assert 48 <= jumps[0].step <= 52
        assert jumps[0].metric_name == "test_metric"

    def test_detect_multiple_jumps(self):
        """Should detect multiple jumps."""
        detector = JumpDetector(window_size=15, z_threshold=2.5, min_samples=10)

        # Create signal with multiple jumps
        values = [5.0] * 30 + [15.0] * 30 + [25.0] * 30

        jumps = detector.detect_jumps(values)

        assert len(jumps) >= 2

    def test_minimum_samples_requirement(self):
        """Should not detect jumps before minimum samples."""
        detector = JumpDetector(window_size=10, z_threshold=3.0, min_samples=50)

        values = [10.0] * 30 + [20.0] * 30  # Only 60 samples

        jumps = detector.detect_jumps(values)

        # Should only detect jumps after step 50
        for jump in jumps:
            assert jump.step >= 50

    def test_z_score_threshold(self):
        """Should respect z-score threshold."""
        # Low threshold - should detect more jumps
        detector_low = JumpDetector(window_size=20, z_threshold=1.5, min_samples=10)
        # High threshold - should detect fewer jumps
        detector_high = JumpDetector(window_size=20, z_threshold=5.0, min_samples=10)

        # Signal with moderate variation
        np.random.seed(42)
        values = list(10 + np.random.randn(50)) + list(15 + np.random.randn(50))

        jumps_low = detector_low.detect_jumps(values)
        jumps_high = detector_high.detect_jumps(values)

        assert len(jumps_low) >= len(jumps_high)

    def test_step_offset(self):
        """Should apply step offset correctly."""
        detector = JumpDetector(window_size=10, z_threshold=3.0, min_samples=5)

        values = [5.0] * 20 + [15.0] * 20

        jumps = detector.detect_jumps(values, step_offset=1000)

        # Jump steps should be offset
        for jump in jumps:
            assert jump.step >= 1000

    def test_direction_filtering(self):
        """Should filter jumps by direction."""
        detector = JumpDetector(window_size=10, z_threshold=2.0, min_samples=5)

        # Signal with up and down jumps
        values = [10.0] * 20 + [20.0] * 20 + [5.0] * 20

        jumps_all = detector.detect_jumps_with_direction(values, direction=None)
        jumps_up = detector.detect_jumps_with_direction(values, direction="up")
        jumps_down = detector.detect_jumps_with_direction(values, direction="down")

        assert len(jumps_up) < len(jumps_all)
        assert len(jumps_down) < len(jumps_all)

        # Check direction is correct
        for jump in jumps_up:
            assert jump.value_after > jump.value_before

        for jump in jumps_down:
            assert jump.value_after < jump.value_before

    def test_invalid_direction(self):
        """Should raise error for invalid direction."""
        detector = JumpDetector()

        with pytest.raises(ValueError, match="direction must be"):
            detector.detect_jumps_with_direction([1, 2, 3], direction="sideways")

    def test_rolling_statistics(self):
        """Should compute rolling statistics correctly."""
        detector = JumpDetector(window_size=10, z_threshold=3.0, min_samples=5)

        values = list(range(50))  # Increasing signal

        means, stds, z_scores = detector.compute_rolling_statistics(values)

        assert len(means) == 50
        assert len(stds) == 50
        assert len(z_scores) == 50

        # Early values should be zero (before min_samples)
        assert means[0] == 0
        assert stds[0] == 0

        # Later values should be computed
        assert means[10] > 0
        assert stds[10] > 0

    def test_empty_values(self):
        """Should handle empty values list."""
        detector = JumpDetector()

        jumps = detector.detect_jumps([])
        assert len(jumps) == 0

        with pytest.raises(ValueError, match="cannot be empty"):
            detector.compute_rolling_statistics([])

    def test_noisy_signal(self):
        """Should handle noisy signals reasonably."""
        detector = JumpDetector(window_size=20, z_threshold=3.0, min_samples=10)

        np.random.seed(42)
        # Noisy signal with one clear jump
        noise_level = 0.5
        values = list(10 + noise_level * np.random.randn(50)) + list(
            20 + noise_level * np.random.randn(50)
        )

        jumps = detector.detect_jumps(values)

        # Should detect at least one jump near step 50
        assert len(jumps) >= 1
        # But not too many false positives
        assert len(jumps) < 10
