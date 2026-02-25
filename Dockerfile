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
RUN pip install -r requirements.txt

# ── 5. Copy Scripts ──────────────────────────────────────────────
COPY download_models.py /app/download_models.py
COPY handler.py /app/handler.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# ── Environment Variables ────────────────────────────────────────
ENV PYTHONPATH="/app/EchoMimic"
ENV MODEL_DIR="/runpod-volume/echomimic_models"

CMD ["/app/start.sh"]