# ══════════════════════════════════════════════════════════════════
# EchoMimic Serverless Image for RunPod
# ══════════════════════════════════════════════════════════════════
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=0

RUN pip install --upgrade pip

# ── 1. System Dependencies (Crucial for OpenCV & MediaPipe) ──────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1-mesa-glx libglib2.0-0 git wget curl pkg-config \
    libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
    libswscale-dev libswresample-dev libavfilter-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── 2. Clone EchoMimic ───────────────────────────────────────────
RUN git clone https://github.com/BadToBest/EchoMimic.git /app/EchoMimic

# ── 3. Install EchoMimic Dependencies ────────────────────────────
WORKDIR /app/EchoMimic
RUN pip install -r requirements.txt

# 🚨 THE MAGIC FIXES: Downgrade protobuf and force correct mediapipe
RUN pip install "protobuf<4"
RUN pip uninstall -y mediapipe && pip install mediapipe==0.10.15
RUN pip install onnxruntime-gpu "numpy<2.0"

# ── 4. Install Serverless Handler Dependencies ───────────────────
WORKDIR /app
COPY requirements.txt /app/requirements.txt
# Install gfpgan first so basicsr/facexlib are resolved before the hf_hub pin
RUN pip install gfpgan
RUN pip install -r requirements.txt

# ── 5. Lock entire stack to mutually compatible versions ─────────
# torch/torchvision/torchaudio: gfpgan can pull in a mismatched torchvision
#   (built against a different torch ABI) causing "torchvision::nms does not exist".
#   Force the exact CUDA 12.1 wheels matching the base image's torch 2.2.0.
# huggingface_hub==0.21.3: has both cached_download and is_offline_mode.
# transformers<4.40: compatible with hf_hub 0.21.x and torch 2.2.0.
# numpy<2.0: gfpgan/basicsr silently pull in numpy 2.x which breaks torch here.
# accelerate: prevents low_cpu_mem_usage fallback / slower model loads.
RUN pip install --force-reinstall \
    --index-url https://download.pytorch.org/whl/cu121 \
    "torch==2.2.0" \
    "torchvision==0.17.0" \
    "torchaudio==2.2.0" \
  && pip install --force-reinstall \
    "huggingface_hub==0.21.3" \
    "transformers>=4.35.0,<4.40.0" \
    "numpy<2.0" \
    "accelerate"

# ── 6. Copy Scripts ──────────────────────────────────────────────
COPY download_models.py /app/download_models.py
COPY handler.py /app/handler.py
COPY preprocess.py /app/preprocess.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# ── Environment Variables ────────────────────────────────────────
ENV PYTHONPATH="/app/EchoMimic"
ENV MODEL_DIR="/runpod-volume/echomimic_models"

CMD ["/app/start.sh"]