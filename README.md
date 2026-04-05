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
