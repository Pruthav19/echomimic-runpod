#!/bin/bash
set -e

echo "🚀 Starting EchoMimic RunPod Worker..."
MODEL_DIR="${MODEL_DIR:-/runpod-volume/echomimic_models}"

# Download models if missing or incomplete.
# Check for both sd-image-variations-diffusers and the VAE model file.
# If either is missing the full download runs (it's idempotent).
VAE_BIN="${MODEL_DIR}/sd-vae-ft-mse/diffusion_pytorch_model.bin"
VAE_SAFE="${MODEL_DIR}/sd-vae-ft-mse/diffusion_pytorch_model.safetensors"
if [ ! -d "${MODEL_DIR}/sd-image-variations-diffusers" ] || { [ ! -f "$VAE_BIN" ] && [ ! -f "$VAE_SAFE" ]; }; then
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