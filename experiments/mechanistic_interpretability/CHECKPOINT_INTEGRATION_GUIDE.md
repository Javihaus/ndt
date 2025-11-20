# Checkpoint Integration Guide

## Quick Start: Add Checkpoints to Your Training Script

This guide shows exactly how to modify your training code to save the 9 checkpoints needed for modified Phase 2 analysis.

---

## Step 1: Load Checkpoint Plan

Add this to your training script:

```python
import json
from pathlib import Path
import torch

# Load checkpoint plan
checkpoint_plan_file = Path('experiments/mechanistic_interpretability/checkpoint_plan_modified.json')
with open(checkpoint_plan_file, 'r') as f:
    checkpoint_plan = json.load(f)

# Get checkpoint steps for your experiment
experiment_name = 'transformer_deep_mnist'  # or 'cnn_deep_mnist', 'mlp_narrow_mnist'
checkpoint_steps = set(checkpoint_plan['experiments'][experiment_name]['checkpoint_steps'])

print(f"Will save checkpoints at steps: {sorted(checkpoint_steps)}")
# Output: Will save checkpoints at steps: [100, 1000, 2000]
```

---

## Step 2: Modify Training Loop

Add checkpoint saving logic:

```python
# Setup
checkpoint_dir = Path(f'experiments/checkpoints/{experiment_name}')
checkpoint_dir.mkdir(parents=True, exist_ok=True)

# Training loop
for step in range(num_training_steps):

    # ... your existing training code ...
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    # Save checkpoint at specified steps
    if step in checkpoint_steps:
        checkpoint_path = checkpoint_dir / f'checkpoint_step_{step:05d}.pt'

        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
            'epoch': epoch,
            'model_config': {
                'architecture': experiment_name,
                'num_layers': len(model.layers) if hasattr(model, 'layers') else None,
                'hidden_dim': model.hidden_dim if hasattr(model, 'hidden_dim') else None,
            }
        }, checkpoint_path)

        print(f'✓ Saved checkpoint: {checkpoint_path.name}')
```

---

## Step 3: Run Experiments

Run your 3 experiments with checkpoint saving enabled:

```bash
# Transformer experiment
python train.py --architecture transformer --experiment transformer_deep_mnist --steps 2000

# CNN experiment
python train.py --architecture cnn --experiment cnn_deep_mnist --steps 2000

# MLP experiment
python train.py --architecture mlp --experiment mlp_narrow_mnist --steps 2000
```

Expected output:
```
Step 100/2000: loss=0.234
✓ Saved checkpoint: checkpoint_step_00100.pt

Step 1000/2000: loss=0.089
✓ Saved checkpoint: checkpoint_step_01000.pt

Step 2000/2000: loss=0.034
✓ Saved checkpoint: checkpoint_step_02000.pt
```

---

## Step 4: Verify Checkpoints

Check that all 9 checkpoints exist:

```bash
cd experiments/mechanistic_interpretability
python3 -c "
import json
from pathlib import Path

# Load plan
with open('checkpoint_plan_modified.json', 'r') as f:
    plan = json.load(f)

# Check each experiment
checkpoint_base = Path('../checkpoints')
for exp_name, exp_config in plan['experiments'].items():
    print(f'\\n{exp_name}:')
    for step in exp_config['checkpoint_steps']:
        checkpoint_path = checkpoint_base / exp_name / f'checkpoint_step_{step:05d}.pt'
        exists = '✓' if checkpoint_path.exists() else '✗'
        size = f'{checkpoint_path.stat().st_size / 1e6:.1f}MB' if checkpoint_path.exists() else 'missing'
        print(f'  {exists} Step {step}: {size}')
"
```

Expected output:
```
transformer_deep_mnist:
  ✓ Step 100: 12.3MB
  ✓ Step 1000: 12.3MB
  ✓ Step 2000: 12.3MB

cnn_deep_mnist:
  ✓ Step 100: 8.7MB
  ✓ Step 1000: 8.7MB
  ✓ Step 2000: 8.7MB

mlp_narrow_mnist:
  ✓ Step 100: 4.2MB
  ✓ Step 1000: 4.2MB
  ✓ Step 2000: 4.2MB
```

---

## Step 5: Run Modified Phase 2 Analysis

Once all 9 checkpoints are available:

```bash
cd experiments/mechanistic_interpretability
python3 modified_phase2_analysis.py
```

The script will:
1. Load checkpoints (early/mid/late for each experiment)
2. Extract architecture-specific features:
   - **Transformers**: Attention patterns across heads/layers
   - **CNNs**: Convolutional filter weights
   - **MLPs**: Hidden layer activation patterns
3. Compare feature sets:
   - Cosine similarity between early/mid/late
   - Feature diversity measures (clustering)
   - Visualizations
4. Test hypothesis: Are early features qualitatively different from late features?

---

## Expected Results Structure

After analysis completes, you'll have:

```
results/modified_phase2/
├── transformer_deep_mnist_analysis.json       # Similarity & diversity metrics
├── transformer_deep_mnist_features.png        # Feature visualizations
├── cnn_deep_mnist_analysis.json
├── cnn_deep_mnist_filters.png                 # Filter evolution visualization
├── mlp_narrow_mnist_analysis.json
├── mlp_narrow_mnist_activations.png           # Activation pattern visualization
└── summary_report.md                          # Integrated findings
```

---

## Timeline

- **Checkpoint generation**: 2-3 hours (running 3 experiments)
- **Analysis execution**: 10-15 minutes (loading + feature extraction + comparison)
- **Total time**: ~3 hours (vs 8 weeks in original plan)

---

## What the Analysis Will Test

### Hypothesis
Early training (steps 0-100, first 5%) forms foundational representations that differ qualitatively from late training (steps 1000-2000, final 50%).

### Test Metric
If `cosine_similarity(early, late) < cosine_similarity(mid, late)`, this supports the hypothesis that:
- Early phase: **Foundation** (qualitatively different features)
- Mid→Late phase: **Refinement** (features become more similar over time)

### Expected Outcomes

**If hypothesis supported:**
- Early features show distinct patterns (low diversity, broad/general)
- Late features show specialized patterns (high diversity, specific/refined)
- Temporal boundary aligns across architectures
- **Contribution**: "Dimensionality tracking identifies when representations shift from foundation to refinement"

**If hypothesis not supported:**
- Features evolve continuously, no clear early/late distinction
- Temporal patterns in dimensionality don't map to feature-level changes
- **Contribution**: "Dimensionality measurements reflect optimization dynamics, not representational structure"

**Either way:**
- Establishes relationship between dimensionality tracking and learned features
- Provides empirical characterization of temporal training dynamics
- Identifies appropriate checkpointing strategy for representation analysis

---

## Honest Framing

### What We're Testing
- ✅ Whether temporal patterns (83.3% early) correspond to feature differences
- ✅ When to checkpoint for representation analysis (early vs late)
- ✅ Architecture-specific patterns in feature formation timing

### What We're NOT Claiming
- ❌ Discrete phase transitions (magnitudes too small)
- ❌ Mechanistic understanding (features ≠ mechanisms)
- ❌ Causal relationships (correlation only)
- ❌ Individual "jump" significance (numerical artifacts)

### Contribution Type
**Measurement Infrastructure**, not mechanistic discovery.

Appropriate venues:
- ICML/NeurIPS workshops on training dynamics
- Distill.pub as measurement methodology
- Position: "Tools for identifying critical training periods"

---

## Troubleshooting

### Problem: Checkpoint files too large
**Solution**: Only save what's needed for feature extraction:
```python
torch.save({
    'step': step,
    'model_state_dict': model.state_dict(),  # Only this is essential
    # Optionally omit optimizer state to save space
}, checkpoint_path)
```

### Problem: Different model architectures between checkpoints
**Solution**: Save model configuration with each checkpoint:
```python
torch.save({
    'step': step,
    'model_state_dict': model.state_dict(),
    'model_config': {
        'class_name': model.__class__.__name__,
        'init_args': model_init_args,
    }
}, checkpoint_path)
```

### Problem: Analysis takes too long
**Solution**: Reduce sample size in feature extraction functions:
```python
# In extract_transformer_features(), extract_mlp_features()
if batch_idx >= 20:  # Use fewer batches
    break
```

---

## Summary

**Total checkpoints needed**: 9 (3 per experiment)
**Storage required**: ~75MB (all checkpoints)
**Time to generate**: ~3 hours (running experiments)
**Time to analyze**: ~15 minutes (feature extraction + comparison)

**Next step**: Modify your training script following Step 2 above, run 3 experiments, and execute the analysis.
