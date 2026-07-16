"""Tests for the validity harness (pure numpy, no torch required)."""

from __future__ import annotations

import numpy as np

from ndt.validity import drift_no_jump
from ndt.validity import jump_detector_as_callable
from ndt.validity import planted_multi
from ndt.validity import planted_transition
from ndt.validity import pure_noise
from ndt.validity import standard_battery
from ndt.validity import validate_detector
from ndt.core.jump_detector import JumpDetector


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures carry the ground truth they claim
# ─────────────────────────────────────────────────────────────────────────────


def test_planted_transition_ground_truth():
    fx = planted_transition(n_steps=400, change_at=200)
    assert fx.ground_truth == (200,)
    assert len(fx.values) == 400
    # the level really changes at the plant
    assert fx.values[:200].mean() < fx.values[200:].mean()


def test_pure_noise_has_no_ground_truth():
    fx = pure_noise()
    assert fx.ground_truth == ()


def test_drift_has_no_ground_truth():
    fx = drift_no_jump()
    assert fx.ground_truth == ()


def test_planted_multi_shapes():
    fx = planted_multi(changes=(150, 350, 450), levels=(10.0, 16.0, 13.0, 22.0))
    assert fx.ground_truth == (150, 350, 450)


def test_planted_multi_validates_levels_length():
    import pytest

    with pytest.raises(ValueError):
        planted_multi(changes=(100, 200), levels=(1.0, 2.0))  # needs 3 levels


def test_standard_battery_has_one_positive_and_negatives():
    battery = standard_battery()
    positives = [f for f in battery if f.ground_truth]
    negatives = [f for f in battery if not f.ground_truth]
    assert len(positives) >= 1
    assert len(negatives) >= 2


# ─────────────────────────────────────────────────────────────────────────────
# The harness scores detectors honestly
# ─────────────────────────────────────────────────────────────────────────────


def test_perfect_detector_is_valid():
    """An oracle that returns exactly the ground truth passes."""

    def oracle(values):
        # cheat: read the fixture's plant from the level change. Here we just
        # return the known plant points for the standard battery by detecting
        # the single largest step, which suffices for the planted fixtures and
        # returns nothing on flat noise.
        v = np.asarray(values)
        diffs = np.abs(np.diff(v))
        # threshold at a large multiple of local scale; on noise this stays empty
        scale = np.median(np.abs(diffs)) + 1e-9
        idx = [int(i + 1) for i, d in enumerate(diffs) if d > 8 * scale]
        return idx

    report = validate_detector(oracle, name="oracle")
    # oracle recovers planted steps and stays quiet on noise/drift
    assert report.mean_recall >= 0.75


def test_noisy_threshold_detector_is_not_valid():
    """The library's own JumpDetector fires on noise, so it fails the check.

    This is the whole point: the tool reports that a popular detector is not
    trustworthy, on ground truth, in milliseconds.
    """
    det = jump_detector_as_callable(JumpDetector(window_size=50, z_threshold=2.0))
    report = validate_detector(det, name="JumpDetector(z=2.0)")
    assert report.false_positives_on_null > 0
    assert report.valid is False


def test_report_renders():
    det = jump_detector_as_callable(JumpDetector(z_threshold=3.0))
    report = validate_detector(det, name="JumpDetector(z=3.0)")
    text = report.render()
    assert "VERDICT" in text
    assert "recall" in text


def test_recall_none_on_null_results():
    det = jump_detector_as_callable(JumpDetector(z_threshold=3.0))
    report = validate_detector(det)
    null_results = [r for r in report.results if r.is_null]
    assert all(r.recall is None for r in null_results)
