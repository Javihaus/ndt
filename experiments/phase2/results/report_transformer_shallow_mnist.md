# Phase 2 Mechanistic Analysis: transformer_shallow/mnist

Generated: 2025-11-19T23:16:32.828718

## Executive Summary

**Pattern**: Early Cascade with 9 dimensionality transitions

**Growth Rate**: very_slow (α=0.000050)

**Max Dimensionality**: 13.18

## Dimensionality Transitions (Jumps)

- **Total jumps**: 9
- **Pattern**: Concentrated burst of jumps in early training
- **Early training jumps**: 7
- **Late training jumps**: 2
- **Average magnitude**: 0.1010
- **Maximum magnitude**: 0.1639
- **Average z-score**: 4.42

### Top 5 Jumps by Magnitude

| Rank | Index | Magnitude | Z-Score |
|------|-------|-----------|---------|
| 1 | 0 | 0.1639 | 7.26 |
| 2 | 1 | 0.1564 | 6.92 |
| 3 | 2 | 0.1420 | 6.27 |
| 4 | 3 | 0.1223 | 5.38 |
| 5 | 4 | 0.0992 | 4.34 |

## Dimensionality Growth Dynamics

- **Growth rate (α)**: 0.000050 - Very slow expansion, possible bottleneck
- **Maximum dimensionality (D_max)**: 13.18
- **TAP fit quality (R²)**: 0.2610 - Significant deviation from TAP model - likely jumps/transitions
- **Onset time (t0)**: 7.29e-10

## Mechanistic Interpretation

### Key Findings

- Transformers show discrete attention pattern transitions rather than gradual feature refinement.

### Hypotheses for Investigation

- Jumps may correspond to attention heads specializing for different aspects of the input.

### Suggested Next Steps

1. Feature visualization at jump indices: [0, 1, 2, 3, 4]
1. Compare activation patterns before/after largest jumps
1. Analyze which neurons become active at each transition

## Architecture Context

- **Depth**: 2 layers
- **Width**: 128
- **Parameters**: 406,282
- **Connectivity**: 1587.04
- **Final Accuracy**: 0.9358
