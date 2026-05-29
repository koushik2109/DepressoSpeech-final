#!/bin/bash
# DepressoSpeech - FastAPI Server (Linux)
# Runs the REST API for inference (keeps running on host)
# Endpoints: POST /predict, POST /predict/batch, GET /health
set -e

echo "=========================================="
echo "  DepressoSpeech - API Server"
echo "=========================================="

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
source .venv/bin/activate

CONFIG=${CONFIG:-"configs/inference_config.yaml"}
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-8000}
LOG_LEVEL=${LOG_LEVEL:-"info"}
RELOAD=${RELOAD:-""}

# Check that model artifacts exist
if [ ! -f "checkpoints/best_model.pt" ]; then
    echo "ERROR: Model checkpoint not found at checkpoints/best_model.pt"
    echo "Please train the model first: bash linux/train.sh"
    exit 1
fi

echo "Server Configuration:"
echo "  Config:    $CONFIG"
echo "  Host:      $HOST"
echo "  Port:      $PORT"
echo "  Log Level: $LOG_LEVEL"
echo ""
echo "Endpoints:"
echo "  POST http://$HOST:$PORT/predict              - Single file prediction (audio-only)"
echo "  POST http://$HOST:$PORT/predict/batch        - Batch prediction"
echo "  POST http://$HOST:$PORT/predict/multimodal   - Multimodal prediction (audio+video+text)"
echo "  GET  http://$HOST:$PORT/health               - Health check"
echo ""
echo "Classification mode: num_classes=2 in config enables binary depression detection"
echo "  Output: depression_probability, predicted_label (0/1), phq8_score (scaled)"
echo ""
echo "Starting server (Ctrl+C to stop)..."
echo "------------------------------------------"

CMD="python3 scripts/serve.py --host $HOST --port $PORT --config $CONFIG --log-level $LOG_LEVEL"
[ -n "$RELOAD" ] && CMD="$CMD --reload"

$CMD
