"""Tests for dimensionality estimators."""

import pytest
import torch

from ndt.core.estimators import compute_all_metrics
from ndt.core.estimators import cumulative_energy_90
from ndt.core.estimators import nuclear_norm_ratio
from ndt.core.estimators import participation_ratio
from ndt.core.estimators import stable_rank


class TestStableRank:
    """Tests for stable rank estimator."""

    def test_identity_matrix(self, identity_matrix):
        """Identity matrix should have stable rank equal to its dimension."""
        sr = stable_rank(identity_matrix)
        assert sr == pytest.approx(50.0, rel=0.01)

    def test_low_rank_matrix(self, low_rank_matrix):
        """Low rank matrix should have small stable rank."""
        sr = stable_rank(low_rank_matrix)
        assert sr < 10  # Should be close to 5 (true rank)

    def test_random_matrix(self, random_matrix):
        """Random matrix should have positive stable rank."""
        sr = stable_rank(random_matrix)
        # For random matrices, stable rank depends on singular value distribution
        # Expect it to be between 10 and the min dimension (50)
        assert 10 < sr <= 50

    def test_invalid_input(self):
        """Should raise error for non-2D input."""
        with pytest.raises(ValueError, match="Expected 2D matrix"):
            stable_rank(torch.randn(10, 20, 30))

    def test_nan_input(self):
        """Should raise error for NaN values."""
        matrix = torch.randn(10, 10)
        matrix[0, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN or Inf"):
            stable_rank(matrix)


class TestParticipationRatio:
    """Tests for participation ratio estimator."""

    def test_identity_matrix(self, identity_matrix):
        """Identity matrix should have PR equal to dimension."""
        pr = participation_ratio(identity_matrix)
        assert pr == pytest.approx(50.0, rel=0.01)

    def test_low_rank_matrix(self, low_rank_matrix):
        """Low rank matrix should have small PR."""
        pr = participation_ratio(low_rank_matrix)
        assert pr < 10

    def test_random_matrix(self, random_matrix):
        """Random matrix should have positive PR."""
        pr = participation_ratio(random_matrix)
        # For random matrices, PR depends on singular value distribution
        # Expect it to be between 10 and the min dimension (50)
        assert 10 < pr <= 50

    def test_invalid_input(self):
        """Should raise error for non-2D input."""
        with pytest.raises(ValueError, match="Expected 2D matrix"):
            participation_ratio(torch.randn(10))


class TestCumulativeEnergy90:
    """Tests for cumulative energy 90% estimator."""

    def test_identity_matrix(self, identity_matrix):
        """Identity matrix should need all components for 90%."""
        ce90 = cumulative_energy_90(identity_matrix)
        assert ce90 == pytest.approx(45, abs=5)  # Around 90% of 50

    def test_low_rank_matrix(self, low_rank_matrix):
        """Low rank matrix should need few components."""
        ce90 = cumulative_energy_90(low_rank_matrix)
        assert ce90 <= 10  # Should be close to rank 5

    def test_custom_threshold(self, random_matrix):
        """Should work with custom thresholds."""
        ce50 = cumulative_energy_90(random_matrix, threshold=0.50)
        ce90 = cumulative_energy_90(random_matrix, threshold=0.90)
        ce99 = cumulative_energy_90(random_matrix, threshold=0.99)
        assert ce50 < ce90 < ce99

    def test_invalid_threshold(self, random_matrix):
        """Should raise error for invalid threshold."""
        with pytest.raises(ValueError, match="Threshold must be in"):
            cumulative_energy_90(random_matrix, threshold=1.5)


class TestNuclearNormRatio:
    """Tests for nuclear norm ratio estimator."""

    def test_identity_matrix(self, identity_matrix):
        """Identity matrix should have NNR equal to dimension."""
        nnr = nuclear_norm_ratio(identity_matrix)
        assert nnr == pytest.approx(50.0, rel=0.01)

    def test_low_rank_matrix(self, low_rank_matrix):
        """Low rank matrix should have small NNR."""
        nnr = nuclear_norm_ratio(low_rank_matrix)
        assert nnr < 10

    def test_near_singular(self, near_singular_matrix):
        """Near-singular matrix should have moderate NNR."""
        nnr = nuclear_norm_ratio(near_singular_matrix)
        assert 10 < nnr < 50


class TestComputeAllMetrics:
    """Tests for computing all metrics at once."""

    def test_all_metrics_computed(self, random_matrix):
        """Should return all four metrics."""
        sr, pr, ce90, nnr = compute_all_metrics(random_matrix)

        assert isinstance(sr, float)
        assert isinstance(pr, float)
        assert isinstance(ce90, int)
        assert isinstance(nnr, float)

        # All should be positive
        assert sr > 0
        assert pr > 0
        assert ce90 > 0
        assert nnr > 0

    def test_consistency_with_individual(self, random_matrix):
        """Should match individual function results."""
        sr_all, pr_all, ce90_all, nnr_all = compute_all_metrics(random_matrix)

        sr_indiv = stable_rank(random_matrix)
        pr_indiv = participation_ratio(random_matrix)
        ce90_indiv = cumulative_energy_90(random_matrix)
        nnr_indiv = nuclear_norm_ratio(random_matrix)

        assert sr_all == pytest.approx(sr_indiv)
        # Note: PR and NNR might differ slightly due to SVD randomness
        # but should be close
        assert abs(pr_all - pr_indiv) < 0.1
        assert ce90_all == ce90_indiv
        assert abs(nnr_all - nnr_indiv) < 0.1

    def test_invalid_input(self):
        """Should raise error for invalid input."""
        with pytest.raises(ValueError, match="Expected 2D matrix"):
            compute_all_metrics(torch.randn(10, 20, 30))
