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
RUN if [ -f requirements.txt ]; then \
      cat > /tmp/sanitize_hallo4_requirements.py <<'PY'
from pathlib import Path
import shlex

src = Path("requirements.txt")
out = Path("/tmp/hallo4.requirements.clean.txt")

keep = []
for raw in src.read_text().splitlines():
    s = raw.strip()
    if not s or s.startswith("#"):
        continue
    if s.startswith(("-r ", "--requirement ", "-c ", "--constraint ", "--find-links ", "-f ")):
        keep.append(raw)
        continue

    token = (shlex.split(s, comments=True)[:1] or [""])[0]
    low = token.lower()
    is_local_wheel = (
        low.endswith(".whl")
        and (token.startswith("/") or token.startswith("./") or token.startswith("../") or token.startswith("~/"))
    )
    if not is_local_wheel:
        keep.append(raw)

out.write_text("\n".join(keep) + ("\n" if keep else ""))
print(f"Sanitized requirements: kept {len(keep)} lines")
PY
      python /tmp/sanitize_hallo4_requirements.py && \
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
