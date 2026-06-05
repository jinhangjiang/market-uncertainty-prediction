#!/bin/bash
# ============================================================
# run_pipeline.sh — EUI Prediction Pipeline (JAIS Revision)
# ============================================================
# Usage:
#   ./run_pipeline.sh [dataset] [mode]
#
#   dataset : reddit | stackexchange | bogleheads | all  (default: all)
#   mode    : smoke | full                               (default: smoke)
#
# Examples:
#   ./run_pipeline.sh all smoke       # local validation, CPU, tiny data (~5 min)
#   ./run_pipeline.sh all full        # cloud GPU, full datasets
#   ./run_pipeline.sh reddit full     # run only Reddit dataset, full mode
#
# Smoke mode is the DEFAULT to prevent accidental local GPU runs.
# ============================================================

set -e

DATASET="${1:-all}"
MODE="${2:-smoke}"
CONFIG="configs/pipeline_config.yaml"
LOG_DIR="logs/$(date +%Y%m%d_%H%M%S)_${MODE}_${DATASET}"

mkdir -p "$LOG_DIR"

echo "=============================================="
echo " EUI Prediction Pipeline"
echo " Dataset : $DATASET"
echo " Mode    : $MODE"
echo " Config  : $CONFIG"
echo " Log dir : $LOG_DIR"
echo "=============================================="

# Activate conda environment if available
if command -v conda &>/dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate eui-revision 2>/dev/null || echo "[WARN] Could not activate eui-revision env; using current Python"
fi

PYTHON="${PYTHON:-python}"

run_step() {
    local step_num="$1"
    local script="$2"
    local step_name
    step_name=$(basename "$script" .py)
    local log_file="$LOG_DIR/${step_num}_${step_name}.log"

    echo ""
    echo "----------------------------------------------"
    echo "[$(date '+%H:%M:%S')] Step $step_num: $script"
    echo "----------------------------------------------"

    $PYTHON "$script" \
        --dataset "$DATASET" \
        --mode "$MODE" \
        --config "$CONFIG" \
        2>&1 | tee "$log_file"

    local exit_code=${PIPESTATUS[0]}
    if [ $exit_code -ne 0 ]; then
        echo ""
        echo "ERROR: Step $step_num failed (exit code $exit_code). See $log_file"
        exit $exit_code
    fi
    echo "[$(date '+%H:%M:%S')] Step $step_num DONE"
}

run_step "00" "src/00_download_data.py"
run_step "01" "src/01_preprocess.py"
run_step "02" "src/02_gdcm_train.py"
run_step "03" "src/03_feature_engineering.py"
run_step "04" "src/04_feature_selection.py"
run_step "05" "src/05_forecast.py"
run_step "06" "src/06_shap_interpret.py"
run_step "07" "src/07_report.py"

echo ""
echo "=============================================="
echo " Pipeline COMPLETE [$MODE]"
echo " Dataset  : $DATASET"
echo " Reports  : outputs/final_report/"
echo " Logs     : $LOG_DIR/"
echo "=============================================="
