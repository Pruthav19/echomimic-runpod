#!/bin/bash
set -e

echo "Starting Hallo4 RunPod Worker..."
MODEL_DIR="${MODEL_DIR:-/runpod-volume/models}"
SENTINEL="${MODEL_DIR}/.download_complete_hallo4"

models_present() {
    { [ -f "${MODEL_DIR}/hallo4/model_weight.ckpt" ] || [ -f "${MODEL_DIR}/hallo4/model_weight.pth" ] || [ -f "${MODEL_DIR}/pretrained_models/hallo4/model_weight.ckpt" ] || [ -f "${MODEL_DIR}/pretrained_models/hallo4/model_weight.pth" ]; } &&
    { [ -d "${MODEL_DIR}/Wan2.1_Encoders" ] || [ -d "${MODEL_DIR}/pretrained_models/Wan2.1_Encoders" ]; } &&
    { [ -f "${MODEL_DIR}/audio_separator/Kim_Vocal_2.onnx" ] || [ -f "${MODEL_DIR}/pretrained_models/audio_separator/Kim_Vocal_2.onnx" ]; } &&
    { [ -d "${MODEL_DIR}/wav2vec2-base-960h" ] || [ -d "${MODEL_DIR}/wav2vec/wav2vec2-base-960h" ] || [ -d "${MODEL_DIR}/pretrained_models/wav2vec2-base-960h" ] || [ -d "${MODEL_DIR}/pretrained_models/wav2vec/wav2vec2-base-960h" ]; }
}

if [ -f "${SENTINEL}" ] && ! models_present; then
    echo "Found stale Hallo4 download sentinel, but model assets are missing. Refreshing model cache."
    rm -f "${SENTINEL}"
fi

if [ ! -f "${SENTINEL}" ]; then
    echo "First boot for Hallo4 at ${MODEL_DIR}"
    mkdir -p "${MODEL_DIR}"
    if models_present; then
        echo "Using mounted Hallo4 models in ${MODEL_DIR}"
        touch "${SENTINEL}"
    else
        echo "Hallo4 models not found in ${MODEL_DIR}; downloading from ${HALLO4_HF_REPO:-fudan-generative-ai/hallo4}."
        echo "For the official gated repo, accept the Hugging Face terms and set HF_TOKEN."
        python /app/download_models.py && touch "${SENTINEL}"
    fi
fi

export PYTHONPATH="/app/hallo4:${PYTHONPATH}"
echo "Starting Hallo4 serverless handler..."
python -u /app/handler.py
