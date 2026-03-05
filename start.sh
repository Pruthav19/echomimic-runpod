#!/bin/bash
set -e

echo "🚀 Starting EchoMimic RunPod Worker..."
MODEL_DIR="${MODEL_DIR:-/runpod-volume/echomimic_models}"

has_vae_weights() {
    [ -f "${MODEL_DIR}/sd-vae-ft-mse/diffusion_pytorch_model.bin" ] || \
    [ -f "${MODEL_DIR}/sd-vae-ft-mse/diffusion_pytorch_model.safetensors" ] || \
    [ -f "${MODEL_DIR}/sd-vae-ft-mse/diffusion_pytorch_model.fp16.safetensors" ]
}

models_ready() {
    [ -f "${MODEL_DIR}/denoising_unet.pth" ] && \
    [ -f "${MODEL_DIR}/reference_unet.pth" ] && \
    [ -f "${MODEL_DIR}/face_locator.pth" ] && \
    [ -f "${MODEL_DIR}/motion_module.pth" ] && \
    [ -f "${MODEL_DIR}/audio_processor/whisper_tiny.pt" ] && \
    [ -d "${MODEL_DIR}/sd-image-variations-diffusers" ] && \
    has_vae_weights
}

if ! models_ready; then
    echo "📦 Model directory incomplete. Downloading/repairing model files..."
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