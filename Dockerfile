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
# Install Hallo4 deps from sanitized requirements:
# - drops blank/comment lines
# - drops local wheel paths/references like /cpfs01/.../*.whl, file:///cpfs01/.../*.whl,
#   ./x.whl, ../x.whl, ~/x.whl
RUN if [ -f requirements.txt ]; then \
      python -c $'from pathlib import Path\nfrom urllib.parse import unquote, urlparse\nimport shlex\nsrc = Path("requirements.txt")\nout = Path("/tmp/hallo4.requirements.clean.txt")\nkeep = []\nskipped = []\nlocal_prefixes = ("/", "./", "../", "~/")\n\ndef is_local_wheel_ref(value):\n    value = value.strip()\n    low = value.lower()\n    if low.startswith("file://"):\n        path = unquote(urlparse(value).path)\n        return path.lower().endswith(".whl") and path.startswith(local_prefixes)\n    return low.endswith(".whl") and value.startswith(local_prefixes)\n\nfor raw in src.read_text().splitlines():\n    s = raw.strip()\n    if not s or s.startswith("#"):\n        continue\n    parts = shlex.split(s, comments=True)\n    if any(is_local_wheel_ref(part) for part in parts):\n        skipped.append(raw)\n        continue\n    keep.append(raw)\nout.write_text("\\n".join(keep) + ("\\n" if keep else ""))\nprint(f"Sanitized requirements: kept {len(keep)} lines, skipped {len(skipped)} local wheel lines")' && \
      pip install -r /tmp/hallo4.requirements.clean.txt; \
    fi

# Hallo4's requirements reference a machine-local Wan wheel, which the sanitizer
# drops. Install the public package without deps because Hallo4 pins the shared
# runtime deps above, and Wan can fall back when flash-attn is unavailable.
RUN pip install --no-deps "wan@git+https://github.com/Wan-Video/Wan2.1"
RUN python -c "import importlib.util; assert importlib.util.find_spec('wan.text2video'), 'wan.text2video not installed'; print('Wan package OK')"

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
