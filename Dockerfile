# ══════════════════════════════════════════════════════════════════
# EchoMimicV3-Flash Serverless Image for RunPod — H100 / CUDA 12.1
# ══════════════════════════════════════════════════════════════════
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ── 1. System packages ───────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git wget curl \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# ── 2. Upgrade pip/setuptools ────────────────────────────────────
RUN pip install --upgrade pip setuptools wheel

# ── 3. Clone EchoMimicV3 ────────────────────────────────────────
RUN git clone https://github.com/antgroup/echomimic_v3.git /app/echomimic_v3

# ── 4. PyTorch stack — cu121, installed ONCE, never overwritten ──
RUN pip install \
    torch==2.2.2 \
    torchvision==0.17.2 \
    torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cu121

# ── 5. EchoMimicV3 dependencies ─────────────────────────────────
# Install in deliberate order to avoid resolution conflicts.

# 5a. Core ML — pinned to last versions compatible with torch 2.2.2
#     diffusers >= 0.31 references torch.xpu (added in torch 2.4)
#     transformers >= 4.47 disables PyTorch if < 2.4
RUN pip install \
    "diffusers==0.30.3" \
    "transformers==4.46.3" \
    "accelerate==0.34.2" \
    "omegaconf" \
    "safetensors" \
    "einops"

# 5b. Audio processing
RUN pip install \
    "librosa" \
    "pyloudnorm" \
    "SentencePiece"

# 5c. Video / image processing
RUN pip install \
    "decord" \
    "moviepy==2.2.1" \
    "pillow" \
    "opencv-python-headless" \
    "imageio[ffmpeg]" \
    "imageio[pyav]"

# 5d. Utilities
RUN pip install \
    "onnxruntime" \
    "timm" \
    "tomesd" \
    "torchdiffeq" \
    "torchsde" \
    "albumentations" \
    "beautifulsoup4" \
    "ftfy" \
    "func_timeout" \
    "scikit-image" \
    "tensorboard"

# 5e. Final numpy lock — EchoMimicV3 deps may upgrade to 2.x
#     torch 2.2.2 was compiled against numpy 1.x ABI
RUN pip install "numpy==1.26.4" \
    && pip uninstall -y bitsandbytes 2>/dev/null || true

# ── 6. Handler dependencies ──────────────────────────────────────
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# ── 7. Application scripts ───────────────────────────────────────
COPY download_models.py /app/download_models.py
COPY handler.py         /app/handler.py
COPY preprocess.py      /app/preprocess.py
COPY start.sh           /app/start.sh
RUN chmod +x /app/start.sh

# ── 8. Models live on a RunPod Network Volume ────────────────────
ENV MODEL_DIR="/runpod-volume/models"

# ── Runtime environment ───────────────────────────────────────────
ENV PYTHONPATH="/app/echomimic_v3"
CMD ["/app/start.sh"]
