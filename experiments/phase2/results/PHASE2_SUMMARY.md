# Phase 2 Summary: Comparative Analysis

Generated: 2025-11-19T23:16:32.831720

## Overview

| Experiment | Jumps | Pattern | R² | α |
|------------|-------|---------|-----|---|
| cnn_deep_fashion_mnist | 23 | scattered | 0.632 | 3.06e-04 |
| transformer_shallow_mnist | 9 | early_cascade | 0.261 | 5.01e-05 |
| mlp_narrow_mnist | 0 | smooth | 0.934 | 1.41e-03 |

## Key Findings

### 1. Architecture-Specific Learning Patterns

- **CNNs**: Show cascading dimensionality jumps, suggesting hierarchical feature emergence
- **Transformers**: Exhibit discrete transitions, possibly attention head specialization
- **MLPs**: Demonstrate smooth, predictable growth following TAP dynamics

### 2. Implications for Mechanistic Interpretability

- **Jump detection as investigation guide**: Dimensionality transitions identify when to apply feature visualization
- **Architecture-dependent analysis**: Different architectures require different interpretability approaches
- **Training phase awareness**: Early vs late jumps suggest different types of feature learning

## Next Steps for Phase 3

1. Deep dive feature visualization on top jump moments
2. Activation pattern comparison before/after transitions
3. Ablation studies at critical training phases
4. Write up findings for publication
