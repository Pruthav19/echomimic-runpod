# ══════════════════════════════════════════════════════════════════
# EchoMimic Serverless Image for RunPod
# Models are baked in at build time — no network volume required.
# ══════════════════════════════════════════════════════════════════
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=0

RUN pip install --upgrade pip

# ── 1. System Dependencies ──────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1-mesa-glx libglib2.0-0 git wget curl pkg-config \
    libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
    libswscale-dev libswresample-dev libavfilter-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── 2. Clone EchoMimic ──────────────────────────────────────────
RUN git clone https://github.com/BadToBest/EchoMimic.git /app/EchoMimic

# ── 2b. Patch: shift face mask to nose level ────────────────────
COPY patch_echomimic.py /app/patch_echomimic.py
RUN python3 /app/patch_echomimic.py

# ── 3. Install EchoMimic Dependencies ───────────────────────────
WORKDIR /app/EchoMimic
RUN pip install -r requirements.txt
RUN pip install "protobuf<4"
RUN pip uninstall -y mediapipe && pip install mediapipe==0.10.15
RUN pip install onnxruntime-gpu "numpy<2.0" \
    "opencv-python<4.10.0" \
    "opencv-contrib-python<4.10.0" \
    "opencv-python-headless<4.10.0"

# ── 4. Install Serverless Handler Dependencies ───────────────────
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install gfpgan --no-deps
RUN pip install basicsr facexlib realesrgan
RUN pip install -r requirements.txt

# ── 5. Lock HuggingFace + NumPy + Torch ─────────────────────────
# Must happen before model download so the correct hf_hub version is used.
RUN pip install --force-reinstall \
    "huggingface_hub==0.21.3" \
    "transformers>=4.35.0,<4.40.0" \
    "numpy<2.0" \
    "accelerate" \
    "torch==2.2.0" \
    "torchvision==0.17.0" \
    --extra-index-url https://download.pytorch.org/whl/cu121

# ── 6. Copy Application Scripts ──────────────────────────────────
COPY download_models.py /app/download_models.py
COPY handler.py         /app/handler.py
COPY preprocess.py      /app/preprocess.py
COPY start.sh           /app/start.sh
RUN chmod +x /app/start.sh

# ── 7. Bake All Models Into The Image ───────────────────────────
# Downloaded once at build time → instant cold starts, no volume needed.
ENV MODEL_DIR="/app/models"
RUN python /app/download_models.py

# ── Environment ──────────────────────────────────────────────────
ENV PYTHONPATH="/app/EchoMimic"

CMD ["/app/start.sh"]
