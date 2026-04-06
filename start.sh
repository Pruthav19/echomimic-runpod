#!/bin/bash
set -e

echo "Starting EchoMimicV3-Flash RunPod Worker..."
MODEL_DIR="${MODEL_DIR:-/runpod-volume/models}"

# ── Download models if not present (first boot on a fresh network volume) ──
SENTINEL="${MODEL_DIR}/.download_complete_v3"
if [ ! -f "${SENTINEL}" ]; then
    echo "Models not found at ${MODEL_DIR} — downloading now (first boot)..."
    mkdir -p "${MODEL_DIR}"
    python /app/download_models.py
    touch "${SENTINEL}"
    echo "Models downloaded and ready."
else
    echo "Models already present at ${MODEL_DIR} — skipping download."
fi

export PYTHONPATH="/app/echomimic_v3:${PYTHONPATH}"

echo "Starting serverless handler..."
python -u /app/handler.py
