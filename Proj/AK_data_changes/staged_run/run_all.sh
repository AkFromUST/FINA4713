#!/usr/bin/env bash
# =============================================================================
# Run all pipeline stages in order.
# Execute from Proj/:
#   bash AK_data_changes/staged_run/run_all.sh
# =============================================================================

set -e  # stop on first error

PYTHON=../venv/bin/python
STAGED=AK_data_changes/staged_run

echo "=========================================="
echo "STAGE 1: Preprocessing + IC + PCA"
echo "=========================================="
$PYTHON -u $STAGED/stage1_preprocess.py

echo ""
echo "=========================================="
echo "STAGE 2: Ridge + OLS"
echo "=========================================="
$PYTHON -u $STAGED/stage2_ridge.py

echo ""
echo "=========================================="
echo "STAGE 3: PTK-SDF"
echo "=========================================="
$PYTHON -u $STAGED/stage3_ptk.py

echo ""
echo "=========================================="
echo "STAGE 4: GKX NN3"
echo "=========================================="
$PYTHON -u $STAGED/stage4_gkx.py

echo ""
echo "=========================================="
echo "STAGE 5: Plots + Comparison Table"
echo "=========================================="
$PYTHON -u $STAGED/stage5_plot.py

echo ""
echo "All stages complete. Plots in AK_data_changes/files/results_staged/"
