"""Validity harness for transition detectors.

Before you claim a phase transition, check that your detector is not reading
noise. This module runs any detector against ground-truth fixtures (known
transitions and known nulls) and returns a verdict on whether its detections
can be trusted.

Quick start:

    >>> from ndt import JumpDetector
    >>> from ndt.validity import validate_detector, jump_detector_as_callable
    >>> det = jump_detector_as_callable(JumpDetector(z_threshold=3.0))
    >>> report = validate_detector(det, name="JumpDetector(z=3)")
    >>> print(report.render())
"""

from ndt.validity.fixtures import Fixture
from ndt.validity.fixtures import drift_no_jump
from ndt.validity.fixtures import planted_multi
from ndt.validity.fixtures import planted_transition
from ndt.validity.fixtures import pure_noise
from ndt.validity.fixtures import standard_battery
from ndt.validity.harness import Detector
from ndt.validity.harness import FixtureResult
from ndt.validity.harness import ValidityReport
from ndt.validity.harness import jump_detector_as_callable
from ndt.validity.harness import validate_detector

__all__ = [
    "Fixture",
    "planted_transition",
    "planted_multi",
    "pure_noise",
    "drift_no_jump",
    "standard_battery",
    "Detector",
    "FixtureResult",
    "ValidityReport",
    "validate_detector",
    "jump_detector_as_callable",
]
