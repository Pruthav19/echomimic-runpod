#!/bin/bash
set -e

echo "Starting EchoMimicV3-Flash + LatentSync RunPod Worker..."
MODEL_DIR="${MODEL_DIR:-/runpod-volume/models}"

# ── Download models if not present (first boot on a fresh network volume) ──
# Sentinel bumped to v3_latentsync — forces re-download to pick up LatentSync
# checkpoints (latentsync_unet.pt + whisper/tiny.pt) on existing volumes.
SENTINEL="${MODEL_DIR}/.download_complete_v3_latentsync"
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

echo "Starting EchoMimicV3-Flash + LatentSync serverless handler..."
python -u /app/handler.py
