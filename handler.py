"""RunPod serverless handler for Hallo4-only pipeline.

This removes EchoMimic/LatentSync runtime paths and acts as a thin API adapter:
- keeps request fields similar to previous API
- downloads input assets
- invokes Hallo4 inference command
- uploads resulting mp4 to S3
"""
import os
import uuid
import json
import shutil
import logging
import subprocess
from pathlib import Path

import runpod
import boto3
import requests

WORKSPACE = Path("/tmp/workspace")
HALLO4_DIR = Path("/app/hallo4")
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/runpod-volume/models"))

S3_BUCKET = os.environ.get("S3_BUCKET", "your-bucket-name")
S3_REGION = os.environ.get("S3_REGION", "us-east-1")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY", "")

# Default command is intentionally explicit and easy to replace if Hallo4 API differs.
# Tokens available for templating:
# {avatar} {audio} {output} {steps} {cfg} {fps} {seed} {size}
HALLO4_COMMAND_TEMPLATE = os.environ.get(
    "HALLO4_COMMAND_TEMPLATE",
    "python inference.py --image {avatar} --audio {audio} --output {output} --steps {steps} --cfg {cfg} --fps {fps} --seed {seed} --size {size}",
)

DEFAULTS = {
    "target_size": 512,
    "inference_steps": 30,
    "cfg_scale": 3.0,
    "fps": 25,
    "seed": 42,
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


def run_hallo4(avatar_path: Path, audio_path: Path, output_path: Path, params: dict):
    cmd = HALLO4_COMMAND_TEMPLATE.format(
        avatar=str(avatar_path),
        audio=str(audio_path),
        output=str(output_path),
        steps=params["inference_steps"],
        cfg=params["cfg_scale"],
        fps=params["fps"],
        seed=params["seed"],
        size=params["target_size"],
    )
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
    if not output_path.exists():
        raise FileNotFoundError(f"Hallo4 finished but output not found: {output_path}")


def handler(event):
    data = event.get("input", {})
    avatar_url = data.get("avatar_image_url")
    audio_url = data.get("audio_url")

    if not avatar_url or not audio_url:
        return {"error": "avatar_image_url and audio_url are required"}

    params = {
        "target_size": int(data.get("target_size", DEFAULTS["target_size"])),
        "inference_steps": int(data.get("inference_steps", DEFAULTS["inference_steps"])),
        "cfg_scale": float(data.get("cfg_scale", DEFAULTS["cfg_scale"])),
        "fps": int(data.get("fps", DEFAULTS["fps"])),
        "seed": int(data.get("seed", DEFAULTS["seed"])),
    }

    job_id = str(uuid.uuid4())
    job_dir = WORKSPACE / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    avatar_path = job_dir / "avatar.png"
    audio_path = job_dir / "audio.wav"
    output_path = job_dir / "output.mp4"

    try:
        download_file(avatar_url, avatar_path)
        download_file(audio_url, audio_path)
        run_hallo4(avatar_path, audio_path, output_path, params)

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
