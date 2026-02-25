#!/bin/bash
set -e

echo "🚀 Starting EchoMimic RunPod Worker..."
MODEL_DIR="${MODEL_DIR:-/runpod-volume/echomimic_models}"

# Download models if missing or incomplete.
# We check for sd-image-variations-diffusers (added in v2) rather than just
# a generic flag, so previously-deployed workers with the old structure
# will automatically re-download the corrected layout.
if [ ! -d "${MODEL_DIR}/sd-image-variations-diffusers" ]; then
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