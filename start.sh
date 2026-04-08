#!/bin/bash
set -e

echo "Starting EchoMimicV3-Preview RunPod Worker..."
MODEL_DIR="${MODEL_DIR:-/runpod-volume/models}"

# ── Download models if not present (first boot on a fresh network volume) ──
# New sentinel since the Preview variant lives under a different directory
# (echomimicv3-preview/) and downloads different transformer weights than the
# old Flash build. The rename forces a clean re-download on the network volume.
SENTINEL="${MODEL_DIR}/.download_complete_v3_preview"
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
