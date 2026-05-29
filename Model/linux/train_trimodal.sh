#!/bin/bash
# DepressoSpeech - Trimodal Training Pipeline (Linux)
# Trains Audio + Video + Text trimodal fusion model
# Supports both regression (PHQ-8 score) and binary classification (depressed/not)
#
# Usage:
#   Regression (default):    bash linux/train_trimodal.sh
#   Classification mode:     CLASSIFICATION=1 bash linux/train_trimodal.sh
#   With NPZ cache:          NPZ_CACHE=data/npz bash linux/train_trimodal.sh
set -e

echo "=========================================="
echo "  DepressoSpeech - Trimodal Training"
echo "=========================================="

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
source .venv/bin/activate

CONFIG=${CONFIG:-"configs/trimodal_v2_config.yaml"}
PATIENTS_DIR=${PATIENTS_DIR:-"data/raw"}
LABELS_CSV=${LABELS_CSV:-"data/labels.csv"}
CLASSIFICATION=${CLASSIFICATION:-"0"}
NPZ_CACHE=${NPZ_CACHE:-"data/npz"}

echo "Configuration:"
echo "  Config:          $CONFIG"
echo "  Patients Dir:    $PATIENTS_DIR"
echo "  Labels CSV:      $LABELS_CSV"
echo "  Classification:  $CLASSIFICATION"
if [ -n "$NPZ_CACHE" ]; then
    echo "  NPZ Cache:       $NPZ_CACHE"
else
    echo "  NPZ Cache:       (disabled, loads raw CSVs)"
fi
echo ""

# Check for GPU
python3 -c "
import torch
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU Device: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('Device: CPU')
"

# Check patient data exists
if [ ! -d "$PATIENTS_DIR" ] || [ -z "$(ls -A "$PATIENTS_DIR" 2>/dev/null)" ]; then
    echo "ERROR: Patient data not found at $PATIENTS_DIR"
    echo "Expected structure: data/raw/<PID>/<PID>_OpenSMILE2.3.0_egemaps.csv, etc."
    exit 1
fi

# Ensure checkpoint dirs exist
mkdir -p checkpoints

# Auto-build NPZ cache if requested but not present
if [ -n "$NPZ_CACHE" ] && [ ! -d "$NPZ_CACHE" ]; then
    echo ""
    echo "[NPZ] Cache directory not found. Building from raw CSVs..."
    echo "      This may take a few minutes on first run."
    echo ""
    python3 scripts/verify_npz.py --build \
        --raw-dir "$PATIENTS_DIR" \
        --output-dir "$NPZ_CACHE" \
        --labels "$LABELS_CSV"
    echo ""
    echo "[NPZ] Cache built."
fi

# Assemble optional npz_cache flag
NPZ_FLAG=""
if [ -n "$NPZ_CACHE" ]; then
    NPZ_FLAG="--npz_cache $NPZ_CACHE"
fi

echo ""
if [ "$CLASSIFICATION" = "1" ]; then
    echo "Starting trimodal CLASSIFIER training..."
    echo "------------------------------------------"
    python3 scripts/train_trimodal_classifier.py \
        --config "$CONFIG" \
        --data_dir "$PATIENTS_DIR" \
        --labels "$LABELS_CSV" \
        $NPZ_FLAG
    echo "------------------------------------------"
    echo "  Classification training complete!"
    echo "  Artifacts saved:"
    echo "    Best Model:  checkpoints/best_classifier.pt"
    echo "    Latest:      checkpoints/latest_classifier.pt"
else
    echo "Starting trimodal REGRESSION training..."
    echo "------------------------------------------"
    python3 scripts/train_trimodal.py \
        --config "$CONFIG" \
        --patients-dir "$PATIENTS_DIR" \
        --labels "$LABELS_CSV" \
        $NPZ_FLAG
    echo "------------------------------------------"
    echo "  Regression training complete!"
    echo "  Artifacts saved:"
    echo "    Best Model:  checkpoints/best_trimodal.pt"
    echo "    Stage 1:     checkpoints/best_trimodal_text.pt"
    echo "    Stage 2:     checkpoints/best_trimodal_audio_text.pt"
    echo "    Logs:        logs/trimodal_training_*.log"
fi
echo "=========================================="
