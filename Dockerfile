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
RUN python -c $'from pathlib import Path\npath = Path("vace/models/wan/wan_vace.py")\ntext = path.read_text()\nold = """            self.model = VaceWanModel.from_ckpt(\\n                model_path,\\n                audio_model_path,\\n                additional_kwargs={\\n                    \\"enable_skeleton_cross_attn\\": enable_skeleton_cross_attn,\\n                    \\"enable_audio_cross_attn\\": enable_audio_cross_attn,\\n                    \\"use_gradient_checkpointing\\": use_gradient_checkpointing,\\n                },\\n            )\\n"""\nnew = """            self.model = VaceWanModel.from_ckpt(\\n                model_path,\\n                audio_model_path,\\n                additional_kwargs={\\n                    \\"enable_skeleton_cross_attn\\": enable_skeleton_cross_attn,\\n                    \\"enable_audio_cross_attn\\": enable_audio_cross_attn,\\n                    \\"use_gradient_checkpointing\\": use_gradient_checkpointing,\\n                },\\n                config_path=os.path.join(checkpoint_dir, \\"config.json\\"),\\n            )\\n"""\npatched = text.replace(old, new, 1)\nassert patched != text, "Could not patch Hallo4 Wan config_path call site"\npath.write_text(patched)\nprint("Patched Hallo4 Wan config_path call site")'
RUN python -c "from pathlib import Path; text = Path('vace/models/wan/wan_vace.py').read_text(); assert 'config_path=os.path.join(checkpoint_dir, \"config.json\")' in text, 'Hallo4 Wan config_path patch missing'; print('Verified Hallo4 Wan config_path patch')"
RUN python -c $'from pathlib import Path\npath = Path("vace/models/wan/modules/attention.py")\ntext = path.read_text()\nold = """    else:\\n        assert FLASH_ATTN_2_AVAILABLE\\n        x = flash_attn.flash_attn_varlen_func(\\n            q=q,\\n            k=k,\\n            v=v,\\n            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(\\n                0, dtype=torch.int32).to(q.device, non_blocking=True),\\n            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(\\n                0, dtype=torch.int32).to(q.device, non_blocking=True),\\n            max_seqlen_q=lq,\\n            max_seqlen_k=lk,\\n            dropout_p=dropout_p,\\n            softmax_scale=softmax_scale,\\n            causal=causal,\\n            window_size=window_size,\\n            deterministic=deterministic).unflatten(0, (b, lq))\\n\\n    # output\\n    return x.type(out_dtype)\\n"""\nnew = """    elif FLASH_ATTN_2_AVAILABLE:\\n        x = flash_attn.flash_attn_varlen_func(\\n            q=q,\\n            k=k,\\n            v=v,\\n            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(\\n                0, dtype=torch.int32).to(q.device, non_blocking=True),\\n            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(\\n                0, dtype=torch.int32).to(q.device, non_blocking=True),\\n            max_seqlen_q=lq,\\n            max_seqlen_k=lk,\\n            dropout_p=dropout_p,\\n            softmax_scale=softmax_scale,\\n            causal=causal,\\n            window_size=window_size,\\n            deterministic=deterministic).unflatten(0, (b, lq))\\n    else:\\n        warnings.warn(\\n            \\"flash-attn is unavailable; using PyTorch scaled_dot_product_attention fallback.\\"\\n        )\\n        q = q.unflatten(0, (b, lq)).transpose(1, 2)\\n        k = k.unflatten(0, (b, lk)).transpose(1, 2)\\n        v = v.unflatten(0, (b, lk)).transpose(1, 2)\\n        x = torch.nn.functional.scaled_dot_product_attention(\\n            q, k, v, is_causal=causal, dropout_p=dropout_p, scale=softmax_scale\\n        )\\n        x = x.transpose(1, 2).contiguous()\\n\\n    # output\\n    return x.type(out_dtype)\\n"""\npatched = text.replace(old, new, 1)\nassert patched != text, "Could not patch Hallo4 flash attention fallback"\npath.write_text(patched)\nprint("Patched Hallo4 flash attention fallback")'
RUN python -c "from pathlib import Path; text = Path('vace/models/wan/modules/attention.py').read_text(); assert 'assert FLASH_ATTN_2_AVAILABLE' not in text and 'scaled_dot_product_attention fallback' in text, 'Hallo4 attention fallback patch missing'; print('Verified Hallo4 attention fallback patch')"
RUN python -c $'from pathlib import Path\npath = Path("vace/models/wan/modules/attention.py")\ntext = path.read_text()\nold = """    def half(x):\\n        return x if x.dtype in half_dtypes else x.to(dtype)\\n\\n    # preprocess query\\n"""\nnew = """    def half(x):\\n        return x if x.dtype in half_dtypes else x.to(dtype)\\n\\n    if not FLASH_ATTN_2_AVAILABLE and not FLASH_ATTN_3_AVAILABLE:\\n        if q_lens is not None or k_lens is not None:\\n            warnings.warn(\\n                \\"Padding mask is disabled when using scaled_dot_product_attention fallback.\\"\\n            )\\n        if q_scale is not None:\\n            q = q * q_scale\\n        q = q.transpose(1, 2).to(dtype)\\n        k = k.transpose(1, 2).to(dtype)\\n        v = v.transpose(1, 2).to(dtype)\\n        x = torch.nn.functional.scaled_dot_product_attention(\\n            q, k, v, is_causal=causal, dropout_p=dropout_p, scale=softmax_scale\\n        )\\n        return x.transpose(1, 2).contiguous().type(out_dtype)\\n\\n    # preprocess query\\n"""\npatched = text.replace(old, new, 1)\nassert patched != text, "Could not patch early Hallo4 attention fallback"\npath.write_text(patched)\nprint("Patched early Hallo4 attention fallback")'
RUN python -c "from pathlib import Path; text = Path('vace/models/wan/modules/attention.py').read_text(); assert 'if not FLASH_ATTN_2_AVAILABLE and not FLASH_ATTN_3_AVAILABLE' in text, 'early attention fallback patch missing'; print('Verified early Hallo4 attention fallback patch')"
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
RUN python -c "from importlib.metadata import files; assert any(str(p) == 'wan/text2video.py' for p in files('wan')), 'wan.text2video not installed'; print('Wan package OK')"

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
