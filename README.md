# Hallo4 RunPod Serverless

This repo now targets a **single Hallo4 pipeline** for RunPod serverless.
EchoMimic and LatentSync runtime paths were removed.

## Input API (`event.input`)

Required:
- `avatar_image_url`
- `audio_url`

Optional (kept for client compatibility):
- `target_size` (default `512`)
- `inference_steps` (default `30`)
- `cfg_scale` (default `3.0`)
- `fps` (default `25`)
- `seed` (default `42`)

## Environment variables

- `MODEL_DIR` (default `/runpod-volume/models`)
- `S3_BUCKET`, `S3_REGION`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`
- `HALLO4_COMMAND_TEMPLATE` (override Hallo4 command if your checkout uses different CLI)
- `HALLO4_HF_REPO` (optional HuggingFace repo for model auto-download)
- `HF_TOKEN` (required for automatic download from the gated official repo)
- `HALLO4_MODEL_PATH`, `HALLO4_CKPT_DIR`, `HALLO4_AUDIO_SEPARATOR_MODEL_PATH`, `HALLO4_WAV2VEC_MODEL_PATH` (optional explicit model paths)

The official Hallo4 model repo is `fudan-generative-ai/hallo4`, but it is gated on Hugging Face. Accept the model terms first, then either mount the snapshot into `MODEL_DIR` or set `HF_TOKEN` so the container can download it on first boot. `HALLO4_HF_REPO` defaults to `fudan-generative-ai/hallo4`.

Expected `MODEL_DIR` layout after download/mount:

```text
/runpod-volume/models/
  hallo4/model_weight.ckpt
  Wan2.1_Encoders/
  audio_separator/Kim_Vocal_2.onnx
  wav2vec2-base-960h/
```

## Default Hallo4 command template

```bash
python -m vace.vace_wan_inference \
  --prompt {prompt} \
  --src_video {conditioning_video} \
  --src_ref_images {avatar} \
  --src_audio {audio} \
  --save_dir {output_dir} \
  --model_path {model_path} \
  --ckpt_dir {ckpt_dir} \
  --audio_separator_model_path {audio_separator_model_path} \
  --wav2vec_model_path {wav2vec_model_path} \
  --sample_steps {steps} \
  --sample_guide_scale {cfg} \
  --base_seed {seed} \
  --size {size}
```

Supported placeholders:
- `{avatar}` `{audio}` `{conditioning_video}` `{output}` `{output_dir}`
- `{steps}` `{cfg}` `{fps}` `{seed}` `{size}` `{prompt}`
- `{model_dir}` `{model_path}` `{ckpt_dir}` `{audio_separator_model_path}` `{wav2vec_model_path}`

The handler creates `{conditioning_video}` from the avatar image because upstream Hallo4 expects `--src_video` plus `--src_ref_images`. Hallo4 writes `*_out_video.mp4` inside `{output_dir}`; the handler copies the latest generated output to `{output}` before S3 upload.

Hallo4 supports `hallo_size` values `480*832` and `832*480`. If you send the older numeric `target_size` value, the handler keeps it in `params_used` for compatibility but runs Hallo4 with the default `480*832`.

If Hallo4 exposes a different entrypoint/flags, set `HALLO4_COMMAND_TEMPLATE` accordingly.

## Recommended RunPod GPUs

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

