# Phase 2 Mechanistic Analysis: cnn_deep/fashion_mnist

Generated: 2025-11-19T23:16:32.826563

## Executive Summary

**Pattern**: Scattered with 23 dimensionality transitions

**Growth Rate**: slow (α=0.000306)

**Max Dimensionality**: 6.33

## Dimensionality Transitions (Jumps)

- **Total jumps**: 23
- **Pattern**: Irregular distribution of jumps
- **Early training jumps**: 14
- **Late training jumps**: 9
- **Average magnitude**: 0.0274
- **Maximum magnitude**: 0.0391
- **Average z-score**: 2.93

### Top 5 Jumps by Magnitude

| Rank | Index | Magnitude | Z-Score |
|------|-------|-----------|---------|
| 1 | 7 | 0.0391 | 4.36 |
| 2 | 6 | 0.0380 | 4.23 |
| 3 | 8 | 0.0364 | 4.03 |
| 4 | 9 | 0.0323 | 3.53 |
| 5 | 5 | 0.0320 | 3.49 |

## Dimensionality Growth Dynamics

- **Growth rate (α)**: 0.000306 - Gradual dimensionality increase
- **Maximum dimensionality (D_max)**: 6.33
- **TAP fit quality (R²)**: 0.6317 - Some deviation from TAP predictions
- **Onset time (t0)**: 1.15e-11

## Mechanistic Interpretation

### Key Findings

- The high number of jumps suggests hierarchical feature learning where different levels of abstraction emerge at distinct training phases.

### Hypotheses for Investigation

- Each jump may correspond to the emergence of a new feature type (edges → textures → parts → objects)

### Suggested Next Steps

1. Feature visualization at jump indices: [7, 6, 8, 9, 5]
1. Compare activation patterns before/after largest jumps
1. Analyze which neurons become active at each transition

## Architecture Context

- **Depth**: 5 layers
- **Width**: 120.0
- **Parameters**: 390,410
- **Connectivity**: 795.13
- **Final Accuracy**: 0.8480
