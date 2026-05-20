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
# Install Hallo4 deps from a sanitized requirements file.
# Upstream may contain absolute/local wheel paths (e.g. /cpfs01/...flash_attn...whl)
# that do not exist in container builds and cause OSError during pip install.
RUN if [ -f requirements.txt ]; then \
      python - <<'PY' \
from pathlib import Path \
import shlex \
 \
src = Path('requirements.txt') \
out = Path('/tmp/hallo4.requirements.clean.txt') \
 \
def should_skip(raw_line: str) -> bool: \
    line = raw_line.strip() \
    if not line or line.startswith('#'): \
        return True \
    if line.startswith(('-r ', '--requirement ', '-c ', '--constraint ', '--find-links ', '-f ')): \
        return False \
    token = shlex.split(line, comments=True)[0] if line else '' \
    lower = token.lower() \
    if lower.endswith('.whl') and (token.startswith('/') or token.startswith('./') or token.startswith('../') or token.startswith('~/')): \
        return True \
    return False \
 \
cleaned = [] \
for raw in src.read_text().splitlines(): \
    if should_skip(raw): \
        continue \
    cleaned.append(raw) \
out.write_text('\n'.join(cleaned) + ('\n' if cleaned else '')) \
print(f'Wrote sanitized requirements to {out} with {len(cleaned)} entries') \
PY
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
