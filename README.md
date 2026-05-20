# EchoMimic RunPod - Parameters Reference

This document explains all parameters currently used by the API handler, including:
- Dynamic input parameters (passed per request)
- Environment variables (set at deploy/runtime)
- Hardcoded defaults/fallbacks
- Common LLM parameters you can add later (not currently used)

---

## 1) Request Input Parameters (`event.input`)

These are read in `handler.py` and passed into preprocessing/inference.

### Required

| Parameter | Type | Required | Default | What it does |
|---|---|---:|---|---|
| `avatar_image_url` | string (URL) | Yes | — | Source avatar image to animate. |
| `audio_url` | string (URL) | Yes | — | Source speech audio used to drive lip sync. |

### Optional Processing Flags

| Parameter | Type | Required | Default | What it does |
|---|---|---:|---|---|
| `skip_preprocess` | bool | No | `false` | If `true`, skips portrait crop, face enhancement, and color correction on the input image. |
| `skip_enhance` | bool | No | `false` | If `true`, skips post video enhancement (Real-ESRGAN upscale + ffmpeg encode tuning). |
| `background_lock` | float | No | `0.0` | Reduces drifting/changing backgrounds by blending non-face areas back toward the original avatar image. Suggested range: `0.6`-`0.9`. |

### Optional Inference Parameters (Dynamic)

| Parameter | Type | Required | Default | Passed to | What it does |
|---|---|---:|---|---|---|
| `target_size` | int | No | `512` | `-W`, `-H` | Output resolution (square). Larger = more detail, slower inference. |
| `inference_steps` | int | No | `40` | `--steps` | Number of denoising steps. Higher can improve quality but increases latency. |
| `cfg_scale` | float | No | `2.5` | `--cfg` | Guidance strength. Higher can look more rigid/stiff; lower is more natural but less constrained. |
| `fps` | int | No | `24` | `--fps` | Output video frame rate. |
| `seed` | int | No | `42` | `--seed` | Random seed for reproducibility. |
| `context_frames` | int | No | `16` | `--context_frames` | Number of temporal context frames for generation windows. Safe supported max is `32`; recommended range is `16`-`24`. |
| `context_overlap` | int | No | `6` | `--context_overlap` | Overlap between context windows to smooth transitions. Must be smaller than `context_frames`; recommended range is `4`-`8`. |
| `face_expand_ratio` | float | No | `0.5` | `--facecrop_dilation_ratio` | Expands face crop region around detected face. |
| `face_mask_dilation` | float | No | `0.0` | `--facemusk_dilation_ratio` | Extra mask expansion around animated lower-face region. |

### Optional YAML Fields (Only for alternate script path)

These are written into `job_config.yaml`, but current runtime uses `infer_audio2vid.py` CLI args for visual controls.

| Parameter | Type | Required | Default | What it does |
|---|---|---:|---|---|
| `pose_weight` | float | No | model/config default | Pose contribution weight (only active if using `infer_audio2vid_pose.py`). |
| `face_weight` | float | No | model/config default | Face expression contribution weight (pose variant flow). |
| `lip_weight` | float | No | model/config default | Lip-sync contribution weight (pose variant flow). |

---

## 2) Environment Variables

These are process-level settings (not per request).

| Variable | Default | What it does |
|---|---|---|
| `S3_BUCKET` | `your-bucket-name` | Destination S3 bucket for generated videos. |
| `S3_REGION` | `us-east-1` | AWS region for S3 client. |
| `S3_ACCESS_KEY` | empty | AWS access key used by `boto3` client. |
| `S3_SECRET_KEY` | empty | AWS secret key used by `boto3` client. |
| `MODEL_DIR` | `/runpod-volume/echomimic_models` | Model cache/weights directory (used across scripts). |

---

## 3) Hardcoded Values (Current Code)

These are currently fixed in code unless edited:

| Hardcoded value | Location | Purpose |
|---|---|---|
| `WORKSPACE = "/tmp/workspace"` | `handler.py` | Per-job temporary work folder. |
| `ECHOMIMIC_DIR = "/app/EchoMimic"` | `handler.py` | Inference repo root used as process cwd. |
| `ExpiresIn=3600` | `upload_to_s3()` | Presigned URL valid for 1 hour. |
| `ContentType="video/mp4"` | `upload_to_s3()` | Upload metadata/content type. |
| Inference script `infer_audio2vid.py` | `run_echomimic()` | Fixed script entrypoint. |

---

## 4) LLM Parameters (Not Currently Used, But Can Be Added)

There is no LLM inference call in this repository right now. If you add one, these are the common tunables:

| LLM Parameter | Typical Type | What it controls |
|---|---|---|
| `model` | string | Which model family/version to run. |
| `temperature` | float | Randomness/creativity of outputs (higher = more varied). |
| `top_p` | float | Nucleus sampling cutoff (alternative/companion to temperature). |
| `max_tokens` | int | Maximum length of generated output. |
| `seed` | int | Deterministic generation when supported. |
| `frequency_penalty` | float | Reduces repeated tokens/phrases. |
| `presence_penalty` | float | Encourages introducing new topics/tokens. |
| `stop` | string or list | Stop sequences to end generation early. |
| `n` | int | Number of completions to generate per request. |
| `response_format` | object/string | Enforces JSON or structured outputs when supported. |

### Suggested API shape if you add LLM controls

Add a nested object in request payload:

```json
{
  "input": {
    "avatar_image_url": "https://...",
    "audio_url": "https://...",
    "target_size": 512,
    "inference_steps": 40,
    "cfg_scale": 2.5,
    "llm": {
      "model": "gpt-4.1-mini",
      "temperature": 0.3,
      "top_p": 1.0,
      "max_tokens": 300
    }
  }
}
```

> Note: the `llm` block above is documentation-only until you implement an actual LLM call path in code.

---

## 5) Example Current Request (Supported Today)

```json
{
  "input": {
    "avatar_image_url": "https://example.com/avatar.png",
    "audio_url": "https://example.com/audio.wav",
    "target_size": 512,
    "inference_steps": 40,
    "cfg_scale": 2.5,
    "fps": 24,
    "seed": 42,
    "context_frames": 16,
    "context_overlap": 6,
    "face_expand_ratio": 0.5,
    "face_mask_dilation": 0.0,
    "background_lock": 0.75,
    "skip_preprocess": false,
    "skip_enhance": false
  }
}
```

---

## 6) Hallo4 RunPod Setup (Branch Guide)

This branch adds a migration/setup plan so you can test **Hallo4** on RunPod while keeping request inputs as close as possible to the current EchoMimic API.

### Branch

- Branch name: `feat/hallo4-runpod-setup`
- Base repo: this `echomimic-runpod` repo

### Recommended RunPod GPU for Hallo4

If you want the least friction and fastest iteration:

1. **H100 80GB (SXM/PCIe)** — best default for Hallo4 testing (largest VRAM headroom, fastest inference/training experimentation).
2. **A100 80GB** — strong fallback if H100 availability/cost is not ideal.
3. **L40S 48GB** — budget-conscious option for inference-only tests with lower concurrency.

### Why H100 first

- Hallo4 pipelines commonly combine multiple heavy components (audio encoder + diffusion/video modules + face processing).
- H100 gives margin for:
  - larger resolution,
  - longer clips,
  - safer batching,
  - fewer OOM retries.

### Keep Input Parameters Mostly the Same

Use the same public API contract in `event.input` and map fields into Hallo4 internals.

**Keep unchanged**
- `avatar_image_url`
- `audio_url`
- `target_size`
- `inference_steps`
- `cfg_scale`
- `fps`
- `seed`
- `context_frames`
- `context_overlap`
- `face_expand_ratio`
- `face_mask_dilation`
- `skip_preprocess`
- `skip_enhance`
- `background_lock`

### Suggested Hallo4 Mapping Layer

| Existing input | Hallo4 mapped knob (example) | Notes |
|---|---|---|
| `target_size` | output resolution | keep square default for parity |
| `inference_steps` | sampler steps | direct map |
| `cfg_scale` | CFG scale | direct map |
| `fps` | render fps | direct map |
| `seed` | generator seed | direct map |
| `context_frames` | temporal window/chunk | map to Hallo4 sequence window |
| `context_overlap` | overlap/crossfade window | direct map where available |
| `face_expand_ratio` | face crop dilation | direct map |
| `face_mask_dilation` | face mask dilation | direct map |
| `background_lock` | bg stabilization alpha | optional post-process blend |

### Minimal RunPod Bring-up Checklist for Hallo4

1. Replace model checkout in Dockerfile from EchoMimic source to Hallo4 source (`https://github.com/fudan-generative-vision/hallo4`).
2. Keep handler request schema stable; only swap backend execution function.
3. Keep S3 upload output format and response JSON unchanged so clients do not break.
4. Start with H100 + network volume for model cache.
5. Validate with current sample payload from Section 5 and compare latency/quality.

### Suggested first test payload

Use the same payload shape already documented in Section 5 so your client can switch backends without changing request construction.

