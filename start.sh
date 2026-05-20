#!/bin/bash
set -e

echo "Starting Hallo4 RunPod Worker..."
MODEL_DIR="${MODEL_DIR:-/runpod-volume/models}"
SENTINEL="${MODEL_DIR}/.download_complete_hallo4"

if [ ! -f "${SENTINEL}" ]; then
    echo "First boot for Hallo4 at ${MODEL_DIR}"
    mkdir -p "${MODEL_DIR}"
    python /app/download_models.py || true
    touch "${SENTINEL}"
fi

export PYTHONPATH="/app/hallo4:${PYTHONPATH}"
echo "Starting Hallo4 serverless handler..."
python -u /app/handler.py
