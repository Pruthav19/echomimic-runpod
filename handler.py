import os
import uuid
import subprocess
import runpod
import boto3
import requests
import yaml
import logging
import glob

WORKSPACE = "/tmp/workspace"
ECHOMIMIC_DIR = "/app/EchoMimic"
S3_BUCKET = os.environ.get("S3_BUCKET", "your-bucket-name")
S3_REGION = os.environ.get("S3_REGION", "us-east-1")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY", "")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        region_name=S3_REGION,
    )

def download_file(url, dest_path):
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return dest_path

def upload_to_s3(local_path, s3_key):
    s3 = get_s3_client()
    s3.upload_file(local_path, S3_BUCKET, s3_key, ExtraArgs={"ContentType": "video/mp4"})
    url = s3.generate_presigned_url("get_object", Params={"Bucket": S3_BUCKET, "Key": s3_key}, ExpiresIn=3600)
    return url

def run_echomimic(image_path, audio_path, output_dir):
    """Dynamically generates a YAML config and runs EchoMimic."""
    
    # 1. Create a dynamic config for this specific job
    job_config_path = os.path.join(output_dir, "job_config.yaml")
    config_data = {
        "test_cases": {
            image_path: [audio_path]
        }
    }
    
    with open(job_config_path, "w") as f:
        yaml.dump(config_data, f)

    # 2. Run EchoMimic Inference
    cmd = [
        "python", "-u", "infer_audio2vid.py",
        "--config", job_config_path
    ]

    logger.info(f"Running EchoMimic: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=ECHOMIMIC_DIR)

    if result.returncode != 0:
        logger.error(f"EchoMimic STDERR: {result.stderr}")
        raise RuntimeError(f"EchoMimic inference failed: {result.stderr}")

    # 3. Locate the generated output video
    # EchoMimic usually saves to ./output/ based on the image name
    mp4_files = glob.glob(os.path.join(ECHOMIMIC_DIR, "output", "**", "*.mp4"), recursive=True)
    if not mp4_files:
        raise RuntimeError("EchoMimic finished but no mp4 was found!")
    
    # Find the most recently created mp4
    generated_video = max(mp4_files, key=os.path.getctime)
    return generated_video

def handler(event):
    try:
        input_data = event["input"]
        job_id = str(uuid.uuid4())[:8]
        job_dir = os.path.join(WORKSPACE, job_id)
        os.makedirs(job_dir, exist_ok=True)

        if not input_data.get("avatar_image_url") or not input_data.get("audio_url"):
            return {"error": "avatar_image_url and audio_url are required"}

        # 1. Download Inputs
        image_path = os.path.join(job_dir, "avatar.png")
        audio_path = os.path.join(job_dir, "audio.wav")
        download_file(input_data["avatar_image_url"], image_path)
        download_file(input_data["audio_url"], audio_path)

        # 2. Run Generation
        final_video_path = run_echomimic(image_path, audio_path, job_dir)

        # 3. Upload to S3
        s3_key = f"echomimic_videos/{job_id}.mp4"
        video_url = upload_to_s3(final_video_path, s3_key)

        # Cleanup
        subprocess.run(["rm", "-rf", job_dir], capture_output=True)

        return {
            "video_url": video_url,
            "job_id": job_id,
            "status": "success",
        }

    except Exception as e:
        logger.exception("Handler error")
        return {"error": str(e), "status": "failed"}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})