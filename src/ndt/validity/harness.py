"""Validity harness: does a transition detector measure transitions, or noise?

The harness runs a detector against ground-truth fixtures and reports two
numbers that decide whether its detections can be trusted:

- **recall** on the positive controls: did it find the transitions that are
  really there?
- **false positives** on the negative controls: did it fire where there is
  nothing to find?

A detector with high recall and near-zero false positives is valid on these
fixtures. A detector that fires on pure noise is reading noise, whatever it
reports on real data. This is a construct-validity check, the same idea the
evaluation-science literature applies to any measurement instrument.

Any detector works, as long as it is a callable that takes a sequence of values
and returns the step indices where it detects a transition. The built-in
``JumpDetector`` is adapted for you by :func:`jump_detector_as_callable`.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ndt.validity.fixtures import Fixture
from ndt.validity.fixtures import standard_battery

__all__ = [
    "Detector",
    "FixtureResult",
    "ValidityReport",
    "validate_detector",
    "jump_detector_as_callable",
]

# A detector is any callable: values -> the step indices it flags as transitions.
Detector = Callable[[Sequence[float]], "list[int]"]


def jump_detector_as_callable(detector, metric_name: str = "metric") -> Detector:
    """Adapt an ``ndt.JumpDetector`` to the harness detector protocol."""

    def _call(values: Sequence[float]) -> list[int]:
        jumps = detector.detect_jumps(list(values), metric_name=metric_name)
        return [j.step for j in jumps]

    return _call


def _match(
    detected: Sequence[int], truth: Sequence[int], tol: int
) -> tuple[int, int, int]:
    """Greedy one-to-one matching within ``tol`` steps.

    Returns (true_positives, false_positives, false_negatives).
    """
    truth_left = list(truth)
    tp = 0
    fp = 0
    for d in sorted(detected):
        hit = None
        for t in truth_left:
            if abs(d - t) <= tol:
                hit = t
                break
        if hit is None:
            fp += 1
        else:
            tp += 1
            truth_left.remove(hit)
    fn = len(truth_left)
    return tp, fp, fn


@dataclass(frozen=True)
class FixtureResult:
    """A detector's outcome on one fixture."""

    fixture: str
    n_steps: int
    n_ground_truth: int
    detected: tuple[int, ...]
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def is_null(self) -> bool:
        """True when the fixture has no transition (a negative control)."""
        return self.n_ground_truth == 0

    @property
    def recall(self) -> float | None:
        """Fraction of ground-truth transitions recovered, or None for a null."""
        if self.is_null:
            return None
        return self.true_positives / self.n_ground_truth

    @property
    def false_positive_rate_per_1000(self) -> float:
        """False detections per 1000 steps. The honest noise number."""
        return 1000.0 * self.false_positives / max(self.n_steps, 1)


@dataclass(frozen=True)
class ValidityReport:
    """A verdict on whether a detector's transitions can be trusted.

    ``valid`` is True only when the detector recovers most planted transitions
    and stays near-silent on the null controls. The thresholds are deliberately
    lenient; a detector that fails even these is not measuring transitions.
    """

    detector_name: str
    results: tuple[FixtureResult, ...]
    mean_recall: float
    false_positives_on_null: int
    valid: bool
    message: str

    def render(self) -> str:
        """Full multi-line report."""
        lines = [f"Validity report for: {self.detector_name}", ""]
        for r in self.results:
            if r.is_null:
                lines.append(
                    f"  {r.fixture:<20} null control   "
                    f"false positives: {r.false_positives:>3} "
                    f"({r.false_positive_rate_per_1000:.1f}/1000 steps)"
                )
            else:
                lines.append(
                    f"  {r.fixture:<20} recall: {r.recall:.2f}   "
                    f"false positives: {r.false_positives:>3}"
                )
        lines.append("")
        lines.append(f"  mean recall on planted transitions : {self.mean_recall:.2f}")
        lines.append(
            f"  false positives on null controls   : {self.false_positives_on_null}"
        )
        lines.append("")
        lines.append(
            f"  VERDICT: {'VALID' if self.valid else 'NOT VALID'} on these fixtures"
        )
        lines.append(f"  {self.message}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()


def validate_detector(
    detector: Detector,
    fixtures: Sequence[Fixture] | None = None,
    *,
    name: str = "detector",
    min_recall: float = 0.75,
    max_false_positives_per_1000: float = 2.0,
) -> ValidityReport:
    """Run ``detector`` against ground-truth fixtures and return a verdict.

    Args:
        detector: A callable mapping a value sequence to detected step indices.
            Wrap an ``ndt.JumpDetector`` with :func:`jump_detector_as_callable`.
        fixtures: The ground-truth battery. Defaults to
            :func:`ndt.validity.fixtures.standard_battery`.
        name: Label for the report.
        min_recall: Minimum mean recall on planted transitions to pass.
        max_false_positives_per_1000: Maximum false-positive rate on any null
            control to pass.

    Returns:
        A :class:`ValidityReport`. ``valid`` is True only when the detector
        clears both bars.
    """
    fixtures = list(fixtures) if fixtures is not None else standard_battery()
    results: list[FixtureResult] = []
    recalls: list[float] = []
    fp_on_null = 0
    worst_null_rate = 0.0

    for fx in fixtures:
        detected = [int(d) for d in detector(np.asarray(fx.values))]
        tp, fp, fn = _match(detected, fx.ground_truth, fx.tolerance)
        res = FixtureResult(
            fixture=fx.name,
            n_steps=len(fx.values),
            n_ground_truth=len(fx.ground_truth),
            detected=tuple(sorted(detected)),
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
        )
        results.append(res)
        if res.is_null:
            fp_on_null += res.false_positives
            worst_null_rate = max(worst_null_rate, res.false_positive_rate_per_1000)
        else:
            recalls.append(res.recall)

    mean_recall = float(np.mean(recalls)) if recalls else 0.0
    passes_recall = mean_recall >= min_recall
    passes_fp = worst_null_rate <= max_false_positives_per_1000
    valid = passes_recall and passes_fp

    if valid:
        message = (
            "Recovers planted transitions and stays quiet on noise and drift. "
            "Its detections are trustworthy on data like these fixtures."
        )
    elif not passes_fp and not passes_recall:
        message = (
            "Misses planted transitions AND fires on the null controls. "
            "These detections reflect the method, not the data."
        )
    elif not passes_fp:
        message = (
            "Fires on pure noise and continuous drift. It manufactures "
            "transitions where there are none, so a detection on real data "
            "cannot be trusted without this check."
        )
    else:
        message = (
            "Stays quiet on the null controls but misses planted transitions. "
            "It under-reports; real transitions can pass unnoticed."
        )

    return ValidityReport(
        detector_name=name,
        results=tuple(results),
        mean_recall=mean_recall,
        false_positives_on_null=fp_on_null,
        valid=valid,
        message=message,
    )
