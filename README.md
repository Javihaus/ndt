<div align="center">

<img src="docs/assets/ndt-logo.svg" alt="ndt" width="140">

# ndt

### Know whether a detected phase transition is real, or an artifact of your method.

[![Python](https://img.shields.io/badge/python-3.8%20|%203.9%20|%203.10%20|%203.11-469BA7?style=flat-square)](https://github.com/Javihaus/ndt)
[![CI](https://img.shields.io/github/actions/workflow/status/Javihaus/ndt/tests.yml?branch=main&label=CI&style=flat-square&color=469BA7)](https://github.com/Javihaus/ndt/actions)
[![Docs](https://img.shields.io/badge/docs-javihaus.github.io%2Fndt-005765?style=flat-square)](https://javihaus.github.io/ndt)
[![PyPI](https://img.shields.io/pypi/v/ndtracker?style=flat-square&color=005765&label=pypi)](https://pypi.org/project/ndtracker/)
[![License](https://img.shields.io/badge/License-Apache_2.0-012731.svg?style=flat-square)](LICENSE)
[![OpenSSF Scorecard](https://img.shields.io/ossf-scorecard/github.com/Javihaus/ndt?style=flat-square&label=OpenSSF%20Scorecard)](https://scorecard.dev/viewer/?uri=github.com/Javihaus/ndt)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/13530/badge)](https://www.bestpractices.dev/projects/13530)

[Documentation](https://javihaus.github.io/ndt) · [Tutorials](#tutorials) · [The finding](#the-finding) · [Examples](examples/) · [Contributing](CONTRIBUTING.md)

</div>

---

**ndt** tracks how a neural network's representational dimensionality evolves during training, across MLPs, CNNs, Transformers and Vision Transformers. It also does one thing no other tracker does: it tells you whether a detected phase transition is real, or an artifact of your detection method.

That check matters because the mistake is easy to make. In a companion study across 55 experiments and 30,147 measurement points, transition detectors built on these same metrics disagreed with each other almost completely (a z-score threshold detector and the threshold-free PELT algorithm correlated at -0.029), and no transition appeared consistently across metrics. A detector can report a clean "phase transition" that is nothing but its own response to noise. So before you build a theory on a detected transition, run the validity harness on your detector and watch what it does on ground truth you control.

## Why ndt?

| The situation | What ndt gives you |
| --- | --- |
| A jump detector reports a clean transition in your training curve. | ndt runs that same detector on trajectories whose transitions you control, and reports its recall and its false-positive rate on pure noise. |
| Two detectors on the same run disagree almost completely. | One verdict on ground truth, in milliseconds, before you commit to a claim. No model, no GPU, no training. |
| You want to see representation geometry, not just loss. | Four dimensionality metrics per layer, logged on any PyTorch model in three lines. |
| A reviewer asks how you know the transition is not an artifact. | A rendered report with a positive control, three null controls, and a plain PASS or FAIL. |

## I want to...

| Goal | Start here |
| --- | --- |
| Check a detector before I trust it | [Tutorial 1](#tutorial-1-check-your-detector-before-you-trust-it) |
| Track dimensionality during training | [Tutorial 2](#tutorial-2-track-dimensionality-during-training) |
| Validate my own detector or my own data | [Tutorial 3](#tutorial-3-plug-in-your-own-detector-and-fixtures) |
| Understand the result behind the tool | [The finding](#the-finding) |
| Read the full API | [Documentation](https://javihaus.github.io/ndt) |

## Installation

```bash
pip install ndtracker
```

This pulls PyTorch and NumPy, everything the tracker and the harness need. The validity harness itself is pure NumPy, so its tests run without a GPU.

## Tutorials

### Tutorial 1: check your detector before you trust it

This is the reason ndt exists. Take any transition detector, run it against a battery of ground-truth fixtures, and read a verdict. The built-in `JumpDetector` is included as one detector to be validated, not as a truth oracle.

```python
from ndt import JumpDetector
from ndt.validity import validate_detector, jump_detector_as_callable

detector = jump_detector_as_callable(JumpDetector(z_threshold=3.0))
report = validate_detector(detector, name="JumpDetector(z=3.0)")
print(report.render())
```

```
Validity report for: JumpDetector(z=3.0)

  planted_transition   recall: 1.00   false positives:   5
  planted_multi        recall: 1.00   false positives:  14
  pure_noise           null control   false positives:   2 (5.0/1000 steps)
  drift_no_jump        null control   false positives:   4 (10.0/1000 steps)

  mean recall on planted transitions : 1.00
  false positives on null controls   : 6

  VERDICT: NOT VALID on these fixtures
  Fires on pure noise and continuous drift. It manufactures transitions where
  there are none, so a detection on real data cannot be trusted without this check.
```

Read this carefully: the detector recovers every planted transition (recall 1.00) *and* fires on pure noise and on a smooth drift. A bare detection count says nothing until you know that false-positive rate. The harness gives you the number, and the verdict is a function of it:

```python
report.valid                 # False
report.mean_recall           # 1.0
report.false_positives_on_null  # 6
```

### Tutorial 2: track dimensionality during training

Add representational-dimensionality tracking to any PyTorch model in three lines. The tracker attaches hooks, samples activations at a fixed frequency, and records four metrics per layer.

```python
import torch.nn as nn
from ndt import HighFrequencyTracker, plot_phases

model = nn.Sequential(
    nn.Linear(784, 512), nn.ReLU(),
    nn.Linear(512, 256), nn.ReLU(),
    nn.Linear(256, 10),
)

tracker = HighFrequencyTracker(model, sampling_frequency=10)

for step, (x, y) in enumerate(dataloader):
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    tracker.log(step, loss.item())   # the one added line

results = tracker.get_results()       # dict: layer name -> DataFrame
fig = plot_phases(results, metric="stable_rank")
```

`results` is a dictionary keyed by layer name; each value is a DataFrame with one row per sampled step and one column per metric. The four metrics:

| Metric | What it measures |
| --- | --- |
| `participation_ratio` | effective number of active dimensions in the representation |
| `stable_rank` | Frobenius-to-spectral norm ratio, a noise-robust rank |
| `cumulative_energy_90` | dimensions needed to hold 90% of the spectral energy |
| `nuclear_norm_ratio` | nuclear norm normalized by dimension |

### Tutorial 3: plug in your own detector and fixtures

The harness accepts any detector that is a callable from a value sequence to the step indices it flags. It also accepts any fixtures you build, including ones drawn from a real training run whose transition you can see in test accuracy (for example a grokking run). This is how you move from synthetic controls to your own data without changing the interface.

```python
import numpy as np
from ndt.validity import validate_detector, Fixture, planted_transition, pure_noise

# 1. Your own detector: values -> detected step indices.
def slope_detector(values, k=20, threshold=3.0):
    v = np.asarray(values)
    steps = []
    for i in range(k, len(v) - k):
        before = v[i - k:i].mean()
        after = v[i:i + k].mean()
        if abs(after - before) > threshold:
            steps.append(i)
    return steps

# 2. Your own fixtures: a positive control, a null, and your real run.
my_run = Fixture(
    values=load_my_stable_rank_trajectory(),   # your metric over steps
    ground_truth=(4200,),                        # the step you trust from test accuracy
    name="grokking_run",
    tolerance=25,
)

battery = [planted_transition(), pure_noise(), my_run]
report = validate_detector(slope_detector, battery, name="slope_detector")
print(report.render())
```

A detector that passes here recovers your known transition and stays quiet on the null. A detector that fails is measuring its own method, whatever it reports on data you cannot check.

## The finding

ndt is the tool built out of a result, so the result is worth stating plainly.

Across 55 training runs and 30,147 high-frequency measurements, transition detection turned out to be a property of the detector as much as of the data. Change the detector, or change the metric, and the "phase transitions" move or vanish. A z-score threshold detector and the threshold-free PELT algorithm, run on the same trajectories, correlated at -0.029. Most of what looks like a discrete jump in a training curve is a smooth, continuous change that a detector rounds into a step.

The conclusion is not that transitions never happen. It is that a detection is a claim, and a claim about a transition is only as good as the detector's behavior on ground truth. ndt exists to make that behavior measurable in milliseconds, so the claim can carry weight. This is the same construct-validity idea that the evaluation-science literature applies to any measurement instrument, brought to training dynamics.

Full method and results are in the companion paper, *Phase Transitions or Continuous Evolution? Methodological Sensitivity in Neural Network Training Dynamics*.

## Architecture

```
                 your model (MLP / CNN / Transformer / ViT)
                                  │
                    HighFrequencyTracker.log(step, loss)
                                  │
              ┌───────────────────┴───────────────────┐
              │                                         │
        activation hooks                         four dimensionality
       (per-layer capture)                        metrics per layer
              │                                         │
              └───────────────────┬───────────────────┘
                                  │
                         results  →  export (CSV / JSON / HDF5)
                                  │
                             plot_phases
                                  │
                       ── validity harness ──
        any detector  →  ground-truth fixtures  →  PASS / FAIL verdict
```

## Examples

Complete, runnable scripts live in [examples/](examples/):

- [`01_quickstart_mnist.py`](examples/01_quickstart_mnist.py): basic MLP on MNIST
- [`02_cnn_cifar10.py`](examples/02_cnn_cifar10.py): CNN on CIFAR-10
- [`03_reproduce_tds_experiment.py`](examples/03_reproduce_tds_experiment.py): reproduces the 8000-step, 784-256-128-10 MNIST experiment from the Towards Data Science article, measurements every 5 steps

## Citation

```bibtex
@software{marin2026ndt,
  author    = {Marín, Javier},
  title     = {ndt: Validity-Checked Tracking of Neural Network Training Dynamics},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/Javihaus/ndt},
  version   = {2.0.0}
}
```

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, the ruff-based lint and format checks, and the test layout. The validity tests run on pure NumPy and need no GPU.

## About

Built by [Javier Marín](https://jmarin.info). ndt is part of a body of work on trustworthy AI: respect the system, do not just fit the data.

## License

Apache 2.0. See [LICENSE](LICENSE).
