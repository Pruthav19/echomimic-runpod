"""RunPod serverless handler for Hallo4-only pipeline.

This removes EchoMimic/LatentSync runtime paths and acts as a thin API adapter:
- keeps request fields similar to previous API
- downloads input assets
- invokes Hallo4 inference command
- uploads resulting mp4 to S3
"""
import os
import uuid
import shutil
import logging
import subprocess
import shlex
from pathlib import Path

import runpod
import boto3
import requests

from download_models import download_models

WORKSPACE = Path("/tmp/workspace")
HALLO4_DIR = Path("/app/hallo4")
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/runpod-volume/models"))

S3_BUCKET = os.environ.get("S3_BUCKET", "your-bucket-name")
S3_REGION = os.environ.get("S3_REGION", "us-east-1")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY", "")

# Default command follows upstream Hallo4's inf.sh entrypoint:
# python -m vace.vace_wan_inference ...
#
# Hallo4 writes outputs into --save_dir as <src_video_stem>_out_video.mp4, so
# run_hallo4 copies that generated file to the stable output path expected by
# the RunPod/S3 wrapper.
#
# The default uses absolute model paths under MODEL_DIR. Mount/download the
# Hugging Face snapshot layout there, or override the HALLO4_* path env vars.
#
# Tokens available for templating:
# {avatar} {audio} {conditioning_video} {output} {output_dir}
# {steps} {cfg} {fps} {seed} {size} {prompt}
# {frame_num} {n_motion_frame} {sample_solver} {wav2vec_features} {extra_args}
# {model_dir} {model_path} {ckpt_dir}
# {audio_separator_model_path} {wav2vec_model_path}
HALLO4_COMMAND_TEMPLATE = os.environ.get(
    "HALLO4_COMMAND_TEMPLATE",
    "python -m vace.vace_wan_inference "
    "--prompt {prompt} "
    "--src_video {conditioning_video} "
    "--src_ref_images {avatar} "
    "--src_audio {audio} "
    "--save_dir {output_dir} "
    "--model_path {model_path} "
    "--ckpt_dir {ckpt_dir} "
    "--audio_separator_model_path {audio_separator_model_path} "
    "--wav2vec_model_path {wav2vec_model_path} "
    "--sample_steps {steps} "
    "--sample_guide_scale {cfg} "
    "--base_seed {seed} "
    "--size {size} "
    "--frame_num {frame_num} "
    "--n_motion_frame {n_motion_frame} "
    "--sample_solver {sample_solver} "
    "--wav2vec_features {wav2vec_features} "
    "{extra_args}",
)

PRESETS = {
    "quality": {
        "inference_steps": 30,
        "cfg_scale": 3.0,
        "frame_num": 81,
        "n_motion_frame": 1,
        "sample_solver": "unipc",
        "wav2vec_features": "all",
    },
    "balanced": {
        "inference_steps": 20,
        "cfg_scale": 3.0,
        "frame_num": 81,
        "n_motion_frame": 1,
        "sample_solver": "unipc",
        "wav2vec_features": "all",
    },
    "fast": {
        "inference_steps": 12,
        "cfg_scale": 2.5,
        "frame_num": 81,
        "n_motion_frame": 1,
        "sample_solver": "unipc",
        "wav2vec_features": "all",
    },
    "preview": {
        "inference_steps": 8,
        "cfg_scale": 2.5,
        "frame_num": 49,
        "n_motion_frame": 1,
        "sample_solver": "unipc",
        "wav2vec_features": "all",
        "max_round": 1,
    },
}

DEFAULTS = {
    "preset": os.environ.get("HALLO4_DEFAULT_PRESET", "quality"),
    "target_size": 512,
    "hallo_size": "480*832",
    "inference_steps": 30,
    "cfg_scale": 3.0,
    "fps": 25,
    "seed": 42,
    "prompt": "a person is talking",
    "frame_num": 81,
    "n_motion_frame": 1,
    "sample_solver": "unipc",
    "wav2vec_features": "all",
}

SUPPORTED_HALLO_SIZES = {"480*832", "832*480"}
SUPPORTED_PRESETS = set(PRESETS)
SUPPORTED_SOLVERS = {"unipc", "dpm++"}
SUPPORTED_WAV2VEC_FEATURES = {"all", "last"}
AUTO_DOWNLOAD_MODELS = os.environ.get("HALLO4_AUTO_DOWNLOAD", "1").lower() not in {
    "0",
    "false",
    "no",
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

s3_client = boto3.client(
    "s3",
    region_name=S3_REGION,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
)


def download_file(url: str, out_path: Path):
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    out_path.write_bytes(r.content)


def upload_to_s3(file_path: Path, key: str):
    s3_client.upload_file(str(file_path), S3_BUCKET, key, ExtraArgs={"ContentType": "video/mp4"})
    return s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": key},
        ExpiresIn=3600,
    )


def _first_existing(candidates):
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    return Path(candidates[0])


def resolve_model_paths():
    model_path = os.environ.get("HALLO4_MODEL_PATH")
    ckpt_dir = os.environ.get("HALLO4_CKPT_DIR")
    audio_separator_model_path = os.environ.get("HALLO4_AUDIO_SEPARATOR_MODEL_PATH")
    wav2vec_model_path = os.environ.get("HALLO4_WAV2VEC_MODEL_PATH")

    if not model_path:
        model_path = _first_existing(
            [
                MODEL_DIR / "hallo4" / "model_weight.ckpt",
                MODEL_DIR / "hallo4" / "model_weight.pth",
                MODEL_DIR / "pretrained_models" / "hallo4" / "model_weight.ckpt",
                MODEL_DIR / "pretrained_models" / "hallo4" / "model_weight.pth",
                HALLO4_DIR / "pretrained_models" / "hallo4" / "model_weight.pth",
                HALLO4_DIR / "pretrained_models" / "hallo4" / "model_weight.ckpt",
            ]
        )
    if not ckpt_dir:
        ckpt_dir = _first_existing(
            [
                MODEL_DIR / "Wan2.1_Encoders",
                MODEL_DIR / "pretrained_models" / "Wan2.1_Encoders",
                HALLO4_DIR / "pretrained_models" / "Wan2.1_Encoders",
            ]
        )
    if not audio_separator_model_path:
        audio_separator_model_path = _first_existing(
            [
                MODEL_DIR / "audio_separator" / "Kim_Vocal_2.onnx",
                MODEL_DIR / "pretrained_models" / "audio_separator" / "Kim_Vocal_2.onnx",
                HALLO4_DIR / "pretrained_models" / "audio_separator" / "Kim_Vocal_2.onnx",
            ]
        )
    if not wav2vec_model_path:
        wav2vec_model_path = _first_existing(
            [
                MODEL_DIR / "wav2vec2-base-960h",
                MODEL_DIR / "wav2vec" / "wav2vec2-base-960h",
                MODEL_DIR / "pretrained_models" / "wav2vec" / "wav2vec2-base-960h",
                MODEL_DIR / "pretrained_models" / "wav2vec2-base-960h",
                HALLO4_DIR / "pretrained_models" / "wav2vec" / "wav2vec2-base-960h",
                HALLO4_DIR / "pretrained_models" / "wav2vec2-base-960h",
            ]
        )

    return {
        "model_path": Path(model_path),
        "ckpt_dir": Path(ckpt_dir),
        "audio_separator_model_path": Path(audio_separator_model_path),
        "wav2vec_model_path": Path(wav2vec_model_path),
    }


def validate_model_paths(paths):
    missing = []
    for name, path in paths.items():
        if not path.exists():
            missing.append(f"{name}={path}")
    if missing:
        raise FileNotFoundError(
            "Hallo4 model assets are missing: "
            + ", ".join(missing)
            + ". The official repo is gated. Accept access at "
            "https://huggingface.co/fudan-generative-ai/hallo4, set HF_TOKEN, "
            "and let start.sh download into MODEL_DIR. Alternatively, mount the "
            "accepted Hugging Face snapshot at MODEL_DIR, or set "
            "HALLO4_MODEL_PATH, HALLO4_CKPT_DIR, HALLO4_AUDIO_SEPARATOR_MODEL_PATH, "
            "and HALLO4_WAV2VEC_MODEL_PATH."
        )


def ensure_model_assets():
    model_paths = resolve_model_paths()
    try:
        validate_model_paths(model_paths)
        return resolve_model_paths()
    except FileNotFoundError:
        if not AUTO_DOWNLOAD_MODELS:
            raise
        logger.warning("Hallo4 model assets are missing; attempting on-demand download.")
        download_models()
        model_paths = resolve_model_paths()
        validate_model_paths(model_paths)
        return model_paths


def media_duration_seconds(path: Path) -> float:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {proc.stderr}")
    return max(float(proc.stdout.strip() or "0"), 1.0)


def create_conditioning_video(avatar_path: Path, audio_path: Path, video_path: Path, fps: int):
    duration = media_duration_seconds(audio_path)
    proc = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-loop",
            "1",
            "-i",
            str(avatar_path),
            "-t",
            f"{duration:.3f}",
            "-vf",
            f"fps={fps},scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video_path),
        ],
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to create Hallo4 conditioning video: {proc.stderr}")


def prepare_source_video(input_path: Path, audio_path: Path, video_path: Path, fps: int):
    duration = media_duration_seconds(audio_path)
    proc = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-stream_loop",
            "-1",
            "-i",
            str(input_path),
            "-t",
            f"{duration:.3f}",
            "-vf",
            f"fps={fps},scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video_path),
        ],
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to prepare Hallo4 source video: {proc.stderr}")


def prepare_audio(input_path: Path, output_path: Path):
    proc = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ],
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to prepare audio for Hallo4: {proc.stderr}")


def find_hallo4_output(output_dir: Path) -> Path:
    candidates = sorted(
        output_dir.glob("*_out_video.mp4"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        candidates = sorted(
            output_dir.glob("*.mp4"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    if not candidates:
        raise FileNotFoundError(f"Hallo4 finished but no mp4 output was found in {output_dir}")
    return candidates[0]


def optional_cli_args(params: dict) -> str:
    args = []
    if params.get("max_round") is not None:
        args.extend(["--max_round", str(params["max_round"])])
    if params.get("sample_shift") is not None:
        args.extend(["--sample_shift", str(params["sample_shift"])])
    if params.get("start_inf_frame"):
        args.extend(["--start_inf_frame", str(params["start_inf_frame"])])
    if params.get("offload_model"):
        args.append("--offload_model")
    if params.get("t5_cpu"):
        args.append("--t5_cpu")
    return " ".join(shlex.quote(arg) for arg in args)


def run_hallo4(
    avatar_path: Path,
    audio_path: Path,
    output_path: Path,
    params: dict,
    source_video_path: Path | None = None,
):
    output_dir = output_path.parent / "hallo4_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    conditioning_video_path = output_path.parent / "conditioning.mp4"

    model_paths = ensure_model_assets()
    if source_video_path:
        logger.info("Using provided source video for Hallo4 conditioning: %s", source_video_path)
        prepare_source_video(source_video_path, audio_path, conditioning_video_path, params["fps"])
        conditioning_mode = "source_video"
    else:
        logger.warning(
            "No source_video_url provided; using still-image conditioning. "
            "For results closer to official Hallo4 examples, pass source_video_url."
        )
        create_conditioning_video(
            avatar_path,
            audio_path,
            conditioning_video_path,
            params["fps"],
        )
        conditioning_mode = "still_avatar_loop"
    params["conditioning_mode"] = conditioning_mode

    values = {
        "avatar": str(avatar_path),
        "audio": str(audio_path),
        "conditioning_video": str(conditioning_video_path),
        "output": str(output_path),
        "output_dir": str(output_dir),
        "steps": str(params["inference_steps"]),
        "cfg": str(params["cfg_scale"]),
        "fps": str(params["fps"]),
        "seed": str(params["seed"]),
        "size": params["hallo_size"],
        "prompt": params["prompt"],
        "frame_num": str(params["frame_num"]),
        "n_motion_frame": str(params["n_motion_frame"]),
        "sample_solver": params["sample_solver"],
        "wav2vec_features": params["wav2vec_features"],
        "extra_args": optional_cli_args(params),
        "model_dir": str(MODEL_DIR),
        "model_path": str(model_paths["model_path"]),
        "ckpt_dir": str(model_paths["ckpt_dir"]),
        "audio_separator_model_path": str(model_paths["audio_separator_model_path"]),
        "wav2vec_model_path": str(model_paths["wav2vec_model_path"]),
    }
    quoted_values = {key: shlex.quote(value) for key, value in values.items()}
    quoted_values["extra_args"] = values["extra_args"]
    cmd = HALLO4_COMMAND_TEMPLATE.format(**quoted_values)
    logger.info("Running Hallo4 command: %s", cmd)
    proc = subprocess.run(
        cmd,
        shell=True,
        cwd=str(HALLO4_DIR),
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Hallo4 failed: {proc.stderr}\nSTDOUT:\n{proc.stdout}")
    generated_output = find_hallo4_output(output_dir)
    if generated_output != output_path:
        shutil.copyfile(generated_output, output_path)
    if not output_path.exists():
        raise FileNotFoundError(f"Hallo4 finished but output not found: {output_path}")


def resolve_hallo_size(data):
    raw_size = data.get("hallo_size", data.get("size"))
    if raw_size is None:
        raw_size = data.get("target_size")
    raw_size = str(raw_size) if raw_size is not None else DEFAULTS["hallo_size"]
    if raw_size in SUPPORTED_HALLO_SIZES:
        return raw_size
    return DEFAULTS["hallo_size"]


def resolve_preset(data):
    preset = str(data.get("hallo_preset", data.get("preset", DEFAULTS["preset"]))).lower()
    if preset in SUPPORTED_PRESETS:
        return preset
    return "quality"


def frame_num_value(value):
    frame_num = int(value)
    if frame_num < 5:
        return 5
    remainder = (frame_num - 1) % 4
    if remainder:
        frame_num += 4 - remainder
    return frame_num


def optional_int(data, key):
    value = data.get(key)
    if value in (None, ""):
        return None
    return int(value)


def optional_float(data, key):
    value = data.get(key)
    if value in (None, ""):
        return None
    return float(value)


def bool_value(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "on"}


def handler(event):
    data = event.get("input", {})
    avatar_url = data.get("avatar_image_url")
    audio_url = data.get("audio_url")
    source_video_url = (
        data.get("source_video_url")
        or data.get("driving_video_url")
        or data.get("conditioning_video_url")
    )

    if not avatar_url or not audio_url:
        return {"error": "avatar_image_url and audio_url are required"}

    preset_name = resolve_preset(data)
    preset = PRESETS[preset_name]
    sample_solver = str(data.get("sample_solver", preset["sample_solver"]))
    if sample_solver not in SUPPORTED_SOLVERS:
        sample_solver = preset["sample_solver"]
    wav2vec_features = str(data.get("wav2vec_features", preset["wav2vec_features"]))
    if wav2vec_features not in SUPPORTED_WAV2VEC_FEATURES:
        wav2vec_features = preset["wav2vec_features"]

    params = {
        "preset": preset_name,
        "target_size": data.get("target_size", DEFAULTS["target_size"]),
        "hallo_size": resolve_hallo_size(data),
        "inference_steps": int(data.get("inference_steps", preset["inference_steps"])),
        "cfg_scale": float(data.get("cfg_scale", preset["cfg_scale"])),
        "fps": int(data.get("fps", DEFAULTS["fps"])),
        "seed": int(data.get("seed", DEFAULTS["seed"])),
        "prompt": str(data.get("prompt", DEFAULTS["prompt"])),
        "frame_num": frame_num_value(data.get("frame_num", preset["frame_num"])),
        "n_motion_frame": int(data.get("n_motion_frame", preset["n_motion_frame"])),
        "sample_solver": sample_solver,
        "wav2vec_features": wav2vec_features,
        "max_round": optional_int(data, "max_round")
        if "max_round" in data
        else preset.get("max_round"),
        "sample_shift": optional_float(data, "sample_shift"),
        "start_inf_frame": int(data.get("start_inf_frame", 0)),
        "offload_model": bool_value(data.get("offload_model"), False),
        "t5_cpu": bool_value(data.get("t5_cpu"), False),
        "source_video_provided": bool(source_video_url),
    }

    job_id = str(uuid.uuid4())
    job_dir = WORKSPACE / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    avatar_path = job_dir / "avatar.png"
    input_audio_path = job_dir / "input_audio"
    audio_path = job_dir / "audio.wav"
    source_video_path = job_dir / "source_video_input.mp4"
    output_path = job_dir / "output.mp4"

    try:
        download_file(avatar_url, avatar_path)
        download_file(audio_url, input_audio_path)
        if source_video_url:
            download_file(source_video_url, source_video_path)
        prepare_audio(input_audio_path, audio_path)
        run_hallo4(
            avatar_path,
            audio_path,
            output_path,
            params,
            source_video_path if source_video_url else None,
        )

        key = f"hallo4/{job_id}.mp4"
        url = upload_to_s3(output_path, key)
        return {
            "status": "success",
            "video_url": url,
            "s3_key": key,
            "engine": "hallo4",
            "params_used": params,
        }
    except Exception as e:
        logger.exception("Job failed")
        return {"status": "error", "error": str(e)}
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


if __name__ == "__main__":
    logger.info("Starting Hallo4 RunPod serverless handler...")
    logger.info("MODEL_DIR=%s", MODEL_DIR)
    logger.info("HALLO4_COMMAND_TEMPLATE=%s", HALLO4_COMMAND_TEMPLATE)
    runpod.serverless.start({"handler": handler})
