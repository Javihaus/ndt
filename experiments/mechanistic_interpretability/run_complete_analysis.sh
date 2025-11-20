#!/bin/bash

################################################################################
# Complete Mechanistic Interpretability Analysis Pipeline
################################################################################
#
# This script runs the entire analysis framework in the correct order:
#   1. Phase 1: Phenomenological analysis (convergence, jumps, critical periods)
#   2. Phase 2: Mechanistic investigations (transformer, CNN/MLP, early/late)
#   3. Integration: Comprehensive report generation
#
# Usage:
#   bash run_complete_analysis.sh [phase]
#
# Options:
#   phase1      - Run only Phase 1 analyses
#   phase2      - Run only Phase 2 analyses
#   integration - Run only integration report
#   all         - Run complete pipeline (default)
#
# Requirements:
#   - Python 3.8+
#   - numpy, pandas, matplotlib, seaborn, scikit-learn
#   - Phase 1 data in ../new/results/phase1_full/
#
# Output:
#   - results/ directory with all analysis outputs
#   - Visualizations (PNG), data files (CSV/JSON), reports (MD)
#
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Parse arguments
PHASE="${1:-all}"

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

run_script() {
    local script_name="$1"
    local description="$2"

    print_info "Running: $description"
    echo "Script: $script_name"

    if python3 "$script_name"; then
        print_success "Completed: $description"
    else
        print_error "Failed: $description"
        exit 1
    fi
    echo ""
}

################################################################################
# Phase 1: Phenomenological Analysis
################################################################################

run_phase1() {
    print_header "PHASE 1: PHENOMENOLOGICAL ANALYSIS"

    print_info "Analyzing 55 experiments across 3 architectures..."
    echo ""

    # Step 1.1: Convergence Analysis
    run_script "step1_1_convergence_analysis.py" \
               "Step 1.1: Layer-wise Convergence Analysis"

    # Step 1.2: Jump Characterization
    run_script "step1_2_jump_characterization.py" \
               "Step 1.2: Dimensionality Jump Characterization"

    # Step 1.3: Critical Periods
    run_script "step1_3_critical_periods.py" \
               "Step 1.3: Critical Period Identification"

    print_success "Phase 1 Complete!"
    print_info "Generated:"
    echo "  - 55 convergence heatmaps"
    echo "  - 2,991 jumps characterized"
    echo "  - 49 critical periods identified"
    echo ""
}

################################################################################
# Phase 2: Mechanistic Investigation
################################################################################

run_phase2() {
    print_header "PHASE 2: MECHANISTIC INVESTIGATION"

    print_info "Investigating 3 key hypotheses..."
    echo ""

    # Week 3-4: Transformer Analysis
    run_script "phase2_week3_4_transformer_analysis.py" \
               "Week 3-4: Transformer Deep Dive"

    # Week 5-6: CNN vs MLP Comparison
    run_script "phase2_week5_6_cnn_mlp_comparison.py" \
               "Week 5-6: CNN vs MLP Comparison"

    # Week 7-8: Early vs Late Jumps
    run_script "phase2_week7_8_early_late_jumps.py" \
               "Week 7-8: Early vs Late Jump Analysis"

    print_success "Phase 2 Complete!"
    print_info "Generated:"
    echo "  - Transformer analysis (5 representative jumps)"
    echo "  - CNN vs MLP comparison (architecture differences)"
    echo "  - Early vs late analysis (temporal patterns)"
    echo ""
}

################################################################################
# Integration: Comprehensive Report
################################################################################

run_integration() {
    print_header "INTEGRATION: COMPREHENSIVE REPORT"

    print_info "Synthesizing all findings..."
    echo ""

    # Generate integrated report
    run_script "generate_integrated_report.py" \
               "Integration: Complete Analysis Synthesis"

    print_success "Integration Complete!"
    print_info "Generated:"
    echo "  - Executive summary dashboard (10 panels)"
    echo "  - Architecture deep dive (6 panels)"
    echo "  - Integrated JSON report"
    echo ""
}

################################################################################
# Main Execution
################################################################################

print_header "MECHANISTIC INTERPRETABILITY ANALYSIS PIPELINE"

echo "Starting analysis with option: $PHASE"
echo "Working directory: $SCRIPT_DIR"
echo ""

# Check prerequisites
print_info "Checking prerequisites..."

# Check Python
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 not found. Please install Python 3.8+."
    exit 1
fi
print_success "Python 3 found: $(python3 --version)"

# Check required packages
python3 -c "import numpy, pandas, matplotlib, seaborn, sklearn" 2>/dev/null
if [ $? -eq 0 ]; then
    print_success "Required packages installed"
else
    print_error "Missing required packages. Please install:"
    echo "  pip install numpy pandas matplotlib seaborn scikit-learn"
    exit 1
fi

# Check Phase 1 data
if [ ! -f "../new/results/phase1_full/transformer_deep_mnist.json" ]; then
    print_error "Phase 1 data not found. Expected at: ../new/results/phase1_full/"
    exit 1
fi
print_success "Phase 1 data found"

echo ""

# Run requested analyses
case "$PHASE" in
    phase1)
        run_phase1
        ;;
    phase2)
        run_phase2
        ;;
    integration)
        run_integration
        ;;
    all)
        run_phase1
        run_phase2
        run_integration
        ;;
    *)
        print_error "Unknown option: $PHASE"
        echo "Usage: bash run_complete_analysis.sh [phase1|phase2|integration|all]"
        exit 1
        ;;
esac

################################################################################
# Summary
################################################################################

print_header "ANALYSIS COMPLETE"

echo "Results saved to: $SCRIPT_DIR/results/"
echo ""
echo "Key outputs:"
echo "  - Phase 1: results/step1_1/, step1_2/, step1_3/"
echo "  - Phase 2: results/phase2_week3_4/, week5_6/, week7_8/"
echo "  - Integration: results/integrated_report/"
echo ""
echo "View main visualization:"
echo "  open results/integrated_report/executive_summary_dashboard.png"
echo ""
echo "View complete report:"
echo "  cat results/integrated_report/integrated_report.json"
echo ""

print_success "All analyses completed successfully!"

echo ""
print_info "Next steps:"
echo "  1. Review visualizations in results/ directories"
echo "  2. Check PHASE1_SUMMARY.md and PHASE2_SUMMARY.md for detailed reports"
echo "  3. See README.md for checkpoint integration to complete mechanistic validation"
echo ""

exit 0
