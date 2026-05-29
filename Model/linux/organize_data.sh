#!/bin/bash
# DepressoSpeech - Patient Data Organization (Linux)
# Organizes audio, transcripts, and pre-extracted features into per-patient folders
set -e

echo "=========================================="
echo "  DepressoSpeech - Data Organization"
echo "=========================================="

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
source .venv/bin/activate

SOURCE_DIR=${SOURCE_DIR:-""}
OUTPUT_DIR=${OUTPUT_DIR:-"data/patients"}

if [ -z "$SOURCE_DIR" ]; then
    echo "Usage: SOURCE_DIR=/path/to/parent bash linux/organize_data.sh"
    echo ""
    echo "Expected parent folder structure:"
    echo "  /path/to/parent/"
    echo "    audio/         - WAV/MP3 files"
    echo "    transcripts/   - CSV/TXT transcript files"
    echo "    features/      - Pre-extracted feature CSVs/MATs"
    exit 1
fi

if [ ! -d "$SOURCE_DIR" ]; then
    echo "ERROR: Source directory not found: $SOURCE_DIR"
    exit 1
fi

echo "Configuration:"
echo "  Source: $SOURCE_DIR"
echo "  Output: $OUTPUT_DIR"
echo ""

python3 scripts/organize_patient_data.py \
    --source-dir "$SOURCE_DIR" \
    --output-dir "$OUTPUT_DIR"

echo "=========================================="
echo "  Data organization complete!"
echo "  Output: $OUTPUT_DIR/"
echo "=========================================="
