#!/bin/bash
# DepressoSpeech - Full Trimodal Pipeline (Linux)
# Runs: organize data → train trimodal model
# Supports regression (PHQ-8) and binary classification (depressed/not)
#
# Expected files per participant in data/raw/<PID>/:
#   <PID>_OpenSMILE2.3.0_egemaps.csv      (88-dim eGeMAPS)
#   <PID>_OpenSMILE2.3.0_mfcc.csv         (39-dim or 120-dim MFCC)
#   <PID>_BoVW_openFace_2.1.0_Pose_Gaze_AUs.csv  (49-dim OpenFace)
#   <PID>_CNN_ResNet.csv                  (512-dim CNN visual embeddings)
#   <PID>_Transcript.csv                  (text for SBERT embeddings)
#
# Usage:
#   Regression (default):    bash linux/run_trimodal_pipeline.sh
#   Classification mode:     CLASSIFICATION=1 bash linux/run_trimodal_pipeline.sh
set -e

echo "=========================================="
echo "  DepressoSpeech - Full Trimodal Pipeline"
echo "=========================================="
echo ""

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# Step 1: Organize Data
if [ -n "$SOURCE_DIR" ]; then
    echo "[Step 1/2] Organizing patient data from $SOURCE_DIR..."
    bash linux/organize_data.sh
else
    echo "[Step 1/2] Skipping data organization (SOURCE_DIR not set)"
    echo "  To organize: SOURCE_DIR=/path/to/parent bash linux/run_trimodal_pipeline.sh"
fi

# Step 2: Trimodal Training
echo ""
if [ "$CLASSIFICATION" = "1" ]; then
    echo "[Step 2/2] Training trimodal CLASSIFIER..."
else
    echo "[Step 2/2] Training trimodal REGRESSION model..."
fi
bash linux/train_trimodal.sh

echo ""
echo "=========================================="
echo "  Full trimodal pipeline complete!"
echo ""
echo "  Next steps:"
echo "    Evaluate:    python scripts/evaluate_classifier.py --checkpoint checkpoints/best_classifier.pt"
echo "    Serve API:   bash linux/serve.sh"
echo "=========================================="
