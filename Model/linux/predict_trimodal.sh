#!/bin/bash
# DepressoSpeech - Trimodal Inference Pipeline (Linux)
# Predicts PHQ-8 from pre-extracted trimodal features (audio + video + text)
#
# Supports three modes:
#   1. Single participant folder
#   2. Batch inference on all participants in a directory
#   3. Individual feature files
#
# Usage:
#   Single participant:
#     bash linux/predict_trimodal.sh --participant-dir data/raw/300
#
#   Batch inference:
#     bash linux/predict_trimodal.sh --batch-dir data/raw --output results.csv
#
#   Individual files:
#     bash linux/predict_trimodal.sh \
#       --mfcc data/raw/300/300_OpenSMILE2.3.0_mfcc.csv \
#       --egemaps data/raw/300/300_OpenSMILE2.3.0_egemaps.csv \
#       --openface data/raw/300/300_BoVW_openFace_2.1.0_Pose_Gaze_AUs.csv \
#       --cnn data/raw/300/300_CNN_ResNet.mat \
#       --transcript data/raw/300/300_Transcript.csv
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
source .venv/bin/activate

CHECKPOINT=${CHECKPOINT:-"checkpoints/best_trimodal.pt"}

# Check checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    echo "Please train the model first:"
    echo "  bash linux/train_trimodal.sh"
    exit 1
fi

echo "=========================================="
echo "  DepressoSpeech - Trimodal Inference"
echo "=========================================="
echo "  Checkpoint: $CHECKPOINT"
echo ""

# Pass all arguments to Python script
python3 scripts/predict_trimodal.py \
    --checkpoint "$CHECKPOINT" \
    "$@"

echo "=========================================="
