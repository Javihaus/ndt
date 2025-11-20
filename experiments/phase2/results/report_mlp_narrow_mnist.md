# Phase 2 Mechanistic Analysis: mlp_narrow/mnist

Generated: 2025-11-19T23:16:32.830799

## Executive Summary

**Pattern**: Smooth learning with excellent TAP fit (R²=0.934)

**Growth Rate**: fast (α=0.001415)

**Max Dimensionality**: 8.54

## Dimensionality Transitions (Jumps)

No dimensionality transitions detected. Learning follows smooth TAP dynamics.

## Dimensionality Growth Dynamics

- **Growth rate (α)**: 0.001415 - Rapid dimensionality expansion
- **Maximum dimensionality (D_max)**: 8.54
- **TAP fit quality (R²)**: 0.9336 - TAP model fits extremely well - smooth, predictable growth
- **Onset time (t0)**: 3.35e-11

## Mechanistic Interpretation

### Key Findings

- Smooth, predictable growth suggests MLPs learn features gradually without sharp phase transitions.

### Hypotheses for Investigation

- MLPs may rely on distributed representations that incrementally refine rather than discrete feature emergence.

### Suggested Next Steps

1. Feature visualization at jump indices: []
1. Compare activation patterns before/after largest jumps
1. Analyze which neurons become active at each transition

## Architecture Context

- **Depth**: 4 layers
- **Width**: 32.0
- **Parameters**: 28,618
- **Connectivity**: 31.04
- **Final Accuracy**: 0.9268
