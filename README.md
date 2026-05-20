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

## Default Hallo4 command template

```bash
python inference.py \
  --image {avatar} \
  --audio {audio} \
  --output {output} \
  --steps {steps} \
  --cfg {cfg} \
  --fps {fps} \
  --seed {seed} \
  --size {size}
```

Supported placeholders:
- `{avatar}` `{audio}` `{output}` `{steps}` `{cfg}` `{fps}` `{seed}` `{size}`

If Hallo4 exposes a different entrypoint/flags, set `HALLO4_COMMAND_TEMPLATE` accordingly.

## Recommended RunPod GPUs

1. H100 80GB (best headroom)
2. A100 80GB (strong fallback)
3. L40S 48GB (budget inference option)
