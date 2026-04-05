#!/bin/bash
set -e

echo "Starting EchoMimic RunPod Worker..."
MODEL_DIR="${MODEL_DIR:-/app/models}"

# Symlink baked-in models to where EchoMimic's code expects them
rm -rf /app/EchoMimic/pretrained_weights
ln -sfn "${MODEL_DIR}" /app/EchoMimic/pretrained_weights
echo "Symlinked ${MODEL_DIR} → /app/EchoMimic/pretrained_weights"

export PYTHONPATH="/app/EchoMimic:${PYTHONPATH}"

echo "Starting serverless handler..."
python -u /app/handler.py
