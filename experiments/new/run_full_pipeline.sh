#!/bin/bash
#
# Full TAP Experimental Pipeline
# Runs all 4 phases sequentially with real data
#
# Usage: ./run_full_pipeline.sh
#

set -e  # Exit on error

echo "========================================================================"
echo "TAP DYNAMICS - FULL EXPERIMENTAL PIPELINE"
echo "========================================================================"
echo ""
echo "Running complete validation as specified in CLAUDE.md:"
echo "  Phase 1: Calibration (17 arch × 5 datasets × 2000 steps)"
echo "  Phase 2: Prediction validation (R² > 0.8 criterion)"
echo "  Phase 3: Real-time monitoring tool"
echo "  Phase 4: Capability emergence correlation"
echo ""
echo "========================================================================"

# Phase 1: Calibration
echo ""
echo "PHASE 1: CALIBRATION"
echo "========================================================================"
if [ -d "results/phase1_full" ] && [ "$(ls -A results/phase1_full/*.json 2>/dev/null | wc -l)" -ge "85" ]; then
    echo "✓ Phase 1 already complete (85 experiments found)"
else
    echo "Running Phase 1 calibration..."
    python phase1_calibration.py --num-steps 2000 --output-dir results/phase1_full
    echo "✓ Phase 1 complete"
fi

# Phase 1 Analysis
echo ""
echo "Analyzing Phase 1 results..."
python phase1_analysis.py \
    --results-dir results/phase1_full \
    --output-dir results/phase1_analysis_full \
    --metric stable_rank

echo "✓ Phase 1 analysis complete"

# Phase 2: Prediction
echo ""
echo "PHASE 2: PREDICTION VALIDATION"
echo "========================================================================"
python phase2_prediction.py \
    --alpha-summary results/phase1_analysis_full/alpha_summary.csv \
    --output-dir results/phase2_full \
    --num-steps 2000 \
    --success-threshold 0.8

echo "✓ Phase 2 complete"

# Phase 3: Real-time Monitoring
echo ""
echo "PHASE 3: REAL-TIME MONITORING"
echo "========================================================================"
python phase3_realtime_monitor.py \
    --alpha-models results/phase1_analysis_full/alpha_models.json \
    --output-dir results/phase3_full

echo "✓ Phase 3 complete"

# Phase 4: Capability Emergence
echo ""
echo "PHASE 4: CAPABILITY EMERGENCE"
echo "========================================================================"
python phase4_capability_emergence.py \
    --output-dir results/phase4_full \
    --num-steps 5000

echo "✓ Phase 4 complete"

# Final Summary
echo ""
echo "========================================================================"
echo "ALL PHASES COMPLETE"
echo "========================================================================"
echo ""
echo "Results saved in:"
echo "  - results/phase1_full/         (85 calibration experiments)"
echo "  - results/phase1_analysis_full/ (α parameter extraction)"
echo "  - results/phase2_full/         (prediction validation)"
echo "  - results/phase3_full/         (monitoring tool)"
echo "  - results/phase4_full/         (capability emergence)"
echo ""
echo "Next steps:"
echo "  1. Review results/*/phase*_report.md for detailed findings"
echo "  2. Check if success criteria met:"
echo "     - Phase 1: α = f(architecture) with R² > 0.7"
echo "     - Phase 2: Prediction R² > 0.8"
echo "     - Phase 4: Dimensionality-capability correlation p < 0.05"
echo "  3. Refine TAP model based on findings"
echo ""
echo "========================================================================"
