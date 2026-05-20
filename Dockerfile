FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git wget curl \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

# Hallo4 repository
RUN git clone https://github.com/fudan-generative-vision/hallo4.git /app/hallo4

WORKDIR /app/hallo4
# Sanitize upstream requirements to avoid broken local wheel paths.
# Drops only local wheel file lines such as:
#   /cpfs01/.../*.whl, ./x.whl, ../x.whl, ~/x.whl
RUN if [ -f requirements.txt ]; then \
      cat > /tmp/sanitize_hallo4_reqs.awk <<'AWK' \
/^[[:space:]]*($|#)/ { next }
{
  line=$0
  s=$0
  sub(/^[[:space:]]+/, "", s)
  split(s, tok, /[[:space:]]+/)
  t=tolower(tok[1])
  if ((index(t, "/")==1 || index(t, "./")==1 || index(t, "../")==1 || index(t, "~/")==1) && t ~ /\.whl$/) next
  print line
}
AWK
      awk -f /tmp/sanitize_hallo4_reqs.awk requirements.txt > /tmp/hallo4.requirements.clean.txt && \
      pip install -r /tmp/hallo4.requirements.clean.txt; \
    fi

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY download_models.py /app/download_models.py
COPY handler.py /app/handler.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

ENV MODEL_DIR="/runpod-volume/models"
ENV PYTHONPATH="/app/hallo4"
CMD ["/app/start.sh"]
