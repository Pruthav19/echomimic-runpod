#!/bin/bash
set -e

echo "🚀 Starting EchoMimic RunPod Worker..."
MODEL_DIR="${MODEL_DIR:-/runpod-volume/echomimic_models}"

# Download models on first boot
if [ ! -f "${MODEL_DIR}/.download_complete" ]; then
    python /app/download_models.py
    touch "${MODEL_DIR}/.download_complete"
fi

# Symlink models to where EchoMimic's code expects them
rm -rf /app/EchoMimic/pretrained_weights
ln -sfn "${MODEL_DIR}" /app/EchoMimic/pretrained_weights
echo "🔗 Symlinked models to /app/EchoMimic/pretrained_weights"

export PYTHONPATH="/app/EchoMimic:${PYTHONPATH}"

echo "🎬 Starting serverless handler..."
python -u /app/handler.py