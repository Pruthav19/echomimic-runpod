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

def run_echomimic(image_path, audio_path, output_dir, user_params):
    """Dynamically generates a YAML config overriding defaults with API input."""
    
    # 1. Load the default animation config so we keep all the correct internal model paths
    default_config_path = os.path.join(ECHOMIMIC_DIR, "configs", "prompts", "animation.yaml")
    with open(default_config_path, "r") as f:
        config_data = yaml.safe_load(f)

    # 2. Inject the specific job image and audio
    config_data["test_cases"] = {
        image_path: [audio_path]
    }
    
    # 3. Apply user tweaks dynamically
    # Maps your custom API keys to EchoMimic's internal config keys
    if "inference_steps" in user_params:
        config_data["steps"] = int(user_params["inference_steps"])
    if "cfg_scale" in user_params:
        config_data["cfg"] = float(user_params["cfg_scale"])
    if "face_expand_ratio" in user_params:
        config_data["facemask_ratio"] = float(user_params["face_expand_ratio"])
    if "target_size" in user_params:
        config_data["W"] = int(user_params["target_size"])
        config_data["H"] = int(user_params["target_size"])
    if "fps" in user_params:
        config_data["fps"] = int(user_params["fps"])
    if "seed" in user_params:
        config_data["seed"] = int(user_params["seed"])
        
    # Inject weights into the config (only active if you switch to infer_audio2vid_pose.py)
    if "pose_weight" in user_params:
        config_data["pose_weight"] = float(user_params["pose_weight"])
    if "face_weight" in user_params:
        config_data["face_weight"] = float(user_params["face_weight"])
    if "lip_weight" in user_params:
        config_data["lip_weight"] = float(user_params["lip_weight"])

    # 4. Save the modified config for this specific job
    job_config_path = os.path.join(output_dir, "job_config.yaml")
    with open(job_config_path, "w") as f:
        yaml.dump(config_data, f)

    # 5. Run EchoMimic Inference
    cmd = [
        "python", "-u", "infer_audio2vid.py",
        "--config", job_config_path
    ]

    logger.info(f"Running EchoMimic with config: {config_data}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=ECHOMIMIC_DIR)

    if result.returncode != 0:
        logger.error(f"EchoMimic STDERR: {result.stderr}")
        raise RuntimeError(f"EchoMimic inference failed: {result.stderr}")

    # 6. Locate the generated output video
    mp4_files = glob.glob(os.path.join(ECHOMIMIC_DIR, "output", "**", "*.mp4"), recursive=True)
    if not mp4_files:
        raise RuntimeError("EchoMimic finished but no mp4 was found!")
    
    generated_video = max(mp4_files, key=os.path.getctime)
    return generated_video

def handler(event):
    try:
        input_data = event.get("input", {})
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

        # 2. Run Generation (now passing the entire input payload to extract params)
        final_video_path = run_echomimic(image_path, audio_path, job_dir, input_data)

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