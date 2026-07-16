"""Ground-truth fixtures for validating transition detectors.

A transition detector claims to find discrete changes in a training-dynamics
trajectory. You cannot know whether such a claim is real unless you test the
detector on a trajectory whose transitions you already know. These fixtures
provide that known ground truth.

Two controls, mirrored on the scientific method:

- A *planted transition* is a positive control: a trajectory with exactly the
  transitions you put in it. A detector that misses them produces a false
  negative you can measure.
- *Pure noise* is a negative control: a trajectory with no transition at all.
  A detector that fires on it produces a false positive you can measure.

These fixtures are synthetic and cost nothing to generate: no model, no
training, no GPU. They let you check a detector's validity in milliseconds.
Real-model fixtures (for example a grokking run, where a transition is visible
in test accuracy) plug into the same interface; see ``Fixture``.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

import numpy as np

__all__ = [
    "Fixture",
    "planted_transition",
    "planted_multi",
    "pure_noise",
    "drift_no_jump",
]


@dataclass(frozen=True)
class Fixture:
    """A trajectory paired with its known transition steps.

    Attributes:
        values: The metric trajectory, one value per training step.
        ground_truth: The step indices where a transition truly occurs. Empty
            for a null fixture (no transition).
        name: A short label for reports.
        tolerance: How many steps away a detection may fall and still count as
            recovering a ground-truth transition. Detection is a noisy business;
            an exact-step match is too strict.
    """

    values: np.ndarray
    ground_truth: tuple[int, ...]
    name: str
    tolerance: int = 5
    meta: dict = field(default_factory=dict)


def planted_transition(
    n_steps: int = 400,
    change_at: int = 200,
    before: float = 10.0,
    after: float = 20.0,
    noise: float = 0.3,
    seed: int = 0,
) -> Fixture:
    """A positive control: one known step change at ``change_at``.

    The trajectory sits near ``before``, then steps to ``after`` at
    ``change_at``, with Gaussian measurement noise. Ground truth is the single
    step ``change_at``. A valid detector recovers it; the absence of any other
    detection is the precision test.
    """
    rng = np.random.default_rng(seed)
    values = np.empty(n_steps, dtype=float)
    values[:change_at] = before
    values[change_at:] = after
    values += rng.normal(0.0, noise, size=n_steps)
    return Fixture(values=values, ground_truth=(change_at,), name="planted_transition")


def planted_multi(
    n_steps: int = 600,
    changes: tuple[int, ...] = (150, 350, 450),
    levels: tuple[float, ...] = (10.0, 16.0, 13.0, 22.0),
    noise: float = 0.3,
    seed: int = 0,
) -> Fixture:
    """A positive control with several known transitions.

    ``levels`` gives the value in each segment; ``changes`` gives the steps
    between segments. ``len(levels)`` must be ``len(changes) + 1``.
    """
    if len(levels) != len(changes) + 1:
        raise ValueError("levels must have exactly one more entry than changes")
    rng = np.random.default_rng(seed)
    values = np.empty(n_steps, dtype=float)
    bounds = (0, *changes, n_steps)
    for level, lo, hi in zip(levels, bounds[:-1], bounds[1:]):
        values[lo:hi] = level
    values += rng.normal(0.0, noise, size=n_steps)
    return Fixture(values=values, ground_truth=tuple(changes), name="planted_multi")


def pure_noise(n_steps: int = 400, noise: float = 1.0, seed: int = 0) -> Fixture:
    """A negative control: no transition, only noise.

    A detector should return no transitions on this fixture. Anything it fires
    is a false positive, and the rate of firing on pure noise is the single most
    honest number you can put on a transition detector.
    """
    rng = np.random.default_rng(seed)
    values = rng.normal(0.0, noise, size=n_steps)
    return Fixture(values=values, ground_truth=(), name="pure_noise")


def drift_no_jump(
    n_steps: int = 400,
    start: float = 10.0,
    end: float = 20.0,
    noise: float = 0.3,
    seed: int = 0,
) -> Fixture:
    """A negative control with a slow, continuous drift and no discrete jump.

    This is the case the paper argues most training dynamics actually are:
    smooth, continuous change with no step. A detector that reports discrete
    transitions here is mistaking a ramp for a jump. Ground truth is empty.
    """
    rng = np.random.default_rng(seed)
    values = np.linspace(start, end, n_steps) + rng.normal(0.0, noise, size=n_steps)
    return Fixture(values=values, ground_truth=(), name="drift_no_jump")


def standard_battery() -> list[Fixture]:
    """The default validity battery: one positive control and three negatives.

    Returns a planted single transition, a planted multi transition, pure noise,
    and a continuous drift. A detector worth trusting recovers the planted
    transitions and stays silent on the noise and the drift.
    """
    return [
        planted_transition(),
        planted_multi(),
        pure_noise(),
        drift_no_jump(),
    ]
