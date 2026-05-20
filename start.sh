#!/bin/bash
set -e

echo "Starting Hallo4 RunPod Worker..."
MODEL_DIR="${MODEL_DIR:-/runpod-volume/models}"
SENTINEL="${MODEL_DIR}/.download_complete_hallo4"

if [ ! -f "${SENTINEL}" ]; then
    echo "First boot for Hallo4 at ${MODEL_DIR}"
    mkdir -p "${MODEL_DIR}"
    if [ -n "${HALLO4_HF_REPO:-}" ]; then
        python /app/download_models.py && touch "${SENTINEL}"
    elif [ -d "${MODEL_DIR}/hallo4" ] || [ -d "${MODEL_DIR}/pretrained_models/hallo4" ]; then
        echo "Using mounted Hallo4 models in ${MODEL_DIR}"
        touch "${SENTINEL}"
    else
        echo "HALLO4_HF_REPO is not set and no mounted Hallo4 model folder was found."
        echo "Skipping automatic model download; handler will report missing model paths if a job starts."
    fi
fi

export PYTHONPATH="/app/hallo4:${PYTHONPATH}"
echo "Starting Hallo4 serverless handler..."
python -u /app/handler.py
