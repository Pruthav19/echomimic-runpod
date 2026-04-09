# ══════════════════════════════════════════════════════════════════
# EchoMimicV3-Flash + LatentSync Serverless Image for RunPod — H100 / CUDA 12.1
# Two-stage pipeline: EchoMimicV3-Flash for video generation,
# LatentSync 1.6 for production-grade lip sync refinement.
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

# ── 4. Clone LatentSync ──────────────────────────────────────────
RUN git clone https://github.com/bytedance/LatentSync.git /app/latentsync

# ── 5. PyTorch stack (EchoMimicV3 env) — cu121, torch 2.2.2 ─────
# NOTE: diffusers must stay ≤0.30.x here because 0.31+ calls torch.xpu
# which does not exist in torch 2.2.x (added in 2.4).
RUN pip install \
    torch==2.2.2 \
    torchvision==0.17.2 \
    torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cu121

# ── 6. EchoMimicV3 dependencies ─────────────────────────────────

# 6a. Core ML — pinned for torch 2.2.2 compatibility
RUN pip install \
    "diffusers==0.30.3" \
    "transformers==4.46.3" \
    "accelerate==0.34.2" \
    "omegaconf" \
    "safetensors" \
    "einops"

# 6b. Audio processing
RUN pip install \
    "librosa" \
    "pyloudnorm" \
    "SentencePiece"

# 6c. Video / image processing
RUN pip install \
    "decord" \
    "moviepy==2.2.1" \
    "pillow" \
    "opencv-python-headless" \
    "imageio[ffmpeg]" \
    "imageio[pyav]"

# 6d. Utilities
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

# 6e. Final numpy lock — EchoMimicV3 deps may upgrade to 2.x
#     torch 2.2.2 was compiled against numpy 1.x ABI
RUN pip install "numpy==1.26.4" \
    && pip uninstall -y bitsandbytes 2>/dev/null || true

# ── 7. LatentSync — isolated venv (torch 2.5.1 + diffusers 0.32.2) ──
# LatentSync requires torch 2.5.1 and diffusers 0.32.2 which are incompatible
# with EchoMimicV3's torch 2.2.2 / diffusers 0.30.3. We isolate them in a
# separate Python venv so both environments coexist without conflict.
# Handler calls LatentSync via subprocess using /app/latentsync_env/bin/python.
RUN python -m venv /app/latentsync_env

# Install torch 2.5.1 + CUDA in the venv first (from CUDA wheel index)
RUN /app/latentsync_env/bin/pip install --upgrade pip && \
    /app/latentsync_env/bin/pip install \
    torch==2.5.1 \
    torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Install LatentSync's remaining deps (exclude torch/torchvision lines already installed)
RUN grep -Ev "^torch" /app/latentsync/requirements.txt | \
    /app/latentsync_env/bin/pip install -r /dev/stdin

# ── 8. Handler dependencies ──────────────────────────────────────
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# ── 9. Application scripts ───────────────────────────────────────
COPY download_models.py /app/download_models.py
COPY handler.py         /app/handler.py
COPY preprocess.py      /app/preprocess.py
COPY start.sh           /app/start.sh
RUN chmod +x /app/start.sh

# ── 10. Models live on a RunPod Network Volume ───────────────────
ENV MODEL_DIR="/runpod-volume/models"

# ── Runtime environment ───────────────────────────────────────────
ENV PYTHONPATH="/app/echomimic_v3"
CMD ["/app/start.sh"]
