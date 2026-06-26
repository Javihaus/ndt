# How It Works

**ndt** instruments a PyTorch model with forward hooks, captures layer activations at a configurable frequency during training, and reduces each activation matrix to a small set of complementary *dimensionality* metrics. Tracking these metrics step-by-step reveals how a network's effective representational capacity evolves — and exposes discrete phase transitions ("jumps") that coarse checkpointing misses.

## Why track dimensionality?

Recent research (Ansuini et al. 2019, Yang et al. 2024) shows that neural networks **expand their effective representational capacity during training**, despite having a fixed parameter count. A few insights motivate high-frequency tracking:

1. **Capacity expands during training** — networks don't just explore a fixed space, they build new representational structure.
2. **Transitions are discrete** — learning happens in distinct phases with rapid jumps.
3. **Critical periods exist** — early training (often the first ~25%) is when most structure forms.
4. **High-frequency sampling reveals hidden dynamics** — coarse checkpointing averages over the transitions that matter.

## The measurement pipeline

1. **Hook** — `ActivationCapture` registers forward hooks on the tracked layers (auto-detected `Linear`/`Conv` layers by default, or a layer list you provide).
2. **Sample** — every `sampling_frequency` steps, `HighFrequencyTracker.log(step, loss)` captures the current activations.
3. **Estimate** — each activation matrix is reduced to the dimensionality metrics below via its singular-value spectrum.
4. **Detect** — `JumpDetector` flags discrete transitions in the metric time series.
5. **Export / visualize** — results stream to pandas DataFrames, CSV/JSON/HDF5, and Matplotlib / Plotly figures.

## Dimensionality metrics

ndt computes four complementary measures from the singular values of each layer's activation matrix. Using several metrics together guards against any single measure's blind spots.

| Metric | Function | Intuition |
|---|---|---|
| **Stable rank** | `stable_rank` | Squared Frobenius norm over squared spectral norm — a soft, noise-robust rank estimate. |
| **Participation ratio** | `participation_ratio` | How many singular directions carry meaningful energy. |
| **Cumulative energy (90%)** | `cumulative_energy_90` | Number of components needed to explain 90% of the variance. |
| **Nuclear-norm ratio** | `nuclear_norm_ratio` | Nuclear norm normalized by spectral norm — sensitivity to the spectrum's tail. |

Compute them all at once with `compute_all_metrics`.

## Jump detection

`JumpDetector` maintains rolling statistics over each metric's time series and flags a **jump** when a step's value deviates beyond `jump_z_threshold` standard deviations from the rolling mean (window `jump_window_size`). Each detected transition is returned as a `Jump` record, making it straightforward to align phase boundaries with the loss curve.

## Architecture support

ndt auto-detects the model family and applies the matching activation-extraction handler:

- `MLPHandler` — multilayer perceptrons
- `CNNHandler` — convolutional networks
- `TransformerHandler` — transformer encoders/decoders
- `ViTHandler` — vision transformers

Use `detect_architecture` / `get_handler` directly, or let `HighFrequencyTracker` select the handler automatically. See [Architecture Support](../guides/architecture_support.md) for details and per-architecture notes.
