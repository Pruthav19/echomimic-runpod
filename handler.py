import os
import uuid
import subprocess
import runpod
import boto3
import requests
import yaml
import logging
import glob

from preprocess import preprocess_image, prepare_background_reference, stabilize_background, composite_face_video

WORKSPACE = "/tmp/workspace"
ECHOMIMIC_DIR = "/app/EchoMimic"
S3_BUCKET = os.environ.get("S3_BUCKET", "your-bucket-name")
S3_REGION = os.environ.get("S3_REGION", "us-east-1")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY", "")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_CONTEXT_FRAMES = 32
DEFAULT_CONTEXT_FRAMES = 16   # EchoMimic temporal attention is trained on 12-16; larger causes jitter
DEFAULT_CONTEXT_OVERLAP = 8   # 50% of context_frames — critical for smooth seams between windows

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


def _resolve_temporal_window(user_params):
    requested_frames = int(user_params.get("context_frames", DEFAULT_CONTEXT_FRAMES))
    requested_overlap = int(user_params.get("context_overlap", DEFAULT_CONTEXT_OVERLAP))

    if requested_frames < 1:
        logger.warning("context_frames=%s is invalid; using default %s.", requested_frames, DEFAULT_CONTEXT_FRAMES)
        requested_frames = DEFAULT_CONTEXT_FRAMES

    if requested_frames > MAX_CONTEXT_FRAMES:
        logger.warning(
            "context_frames=%s exceeds EchoMimic's supported maximum of %s; clamping.",
            requested_frames,
            MAX_CONTEXT_FRAMES,
        )
        requested_frames = MAX_CONTEXT_FRAMES

    max_overlap = max(0, requested_frames - 1)
    if requested_overlap < 0:
        logger.warning("context_overlap=%s is invalid; using default %s.", requested_overlap, DEFAULT_CONTEXT_OVERLAP)
        requested_overlap = DEFAULT_CONTEXT_OVERLAP

    if requested_overlap > max_overlap:
        logger.warning(
            "context_overlap=%s is too large for context_frames=%s; clamping to %s.",
            requested_overlap,
            requested_frames,
            max_overlap,
        )
        requested_overlap = max_overlap

    return requested_frames, requested_overlap

def preprocess_audio(input_path: str, output_path: str) -> str:
    """
    Normalise and resample audio to exactly what EchoMimic's Whisper model expects:
      • Mono (1 channel)
      • 16 000 Hz sample rate
      • loudnorm to -14 LUFS (prevents Whisper from being driven by silence or clipping)
    A bad sample rate / stereo mix / wrong loudness are the main causes of
    lip-sync drift.
    """
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", input_path,
            "-ac", "1",              # mono
            "-ar", "16000",          # 16 kHz — Whisper's native rate
            "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",  # broadcast loudness norm
            "-c:a", "pcm_s16le",     # 16-bit PCM WAV
            output_path,
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        logger.warning(f"Audio preprocessing failed: {result.stderr} — using original.")
        import shutil
        shutil.copy(input_path, output_path)
    else:
        logger.info(f"Audio preprocessed: mono 16kHz loudnorm → {output_path}")
    return output_path


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
    # infer_audio2vid.py reads ALL visual params from CLI args, NOT the yaml config.
    w    = int(user_params.get("target_size",       512))
    h    = int(user_params.get("target_size",       512))
    steps = int(user_params.get("inference_steps",   50))
    cfg  = float(user_params.get("cfg_scale",        3.0))  # 2.5=too mushy/no motion; 3.5=stiff; 3.0 is the sweet spot
    fps  = int(user_params.get("fps",                24))
    seed = int(user_params.get("seed",               42))
    # EchoMimic's temporal positional encoding supports up to 32 frames.
    # Larger values crash inside the motion module regardless of available VRAM.
    ctx_frames, ctx_overlap = _resolve_temporal_window(user_params)

    # facecrop_dilation_ratio: how much padding around the face box for the reference crop
    #   0.5 = default (tight); 1.2 = generous shoulder/hair room
    facecrop_dilation = float(user_params.get("face_expand_ratio",    0.5))

    # facemusk_dilation_ratio: padding around the face bbox for the animated mask.
    #   0.05 adds slight extra coverage so mouth corners aren't clipped at the mask edge.
    #   Increase to 0.1 if corners still clip; lower to 0.0 if mask bleeds into chin/neck.
    facemask_dilation = float(user_params.get("face_mask_dilation",   0.1))

    cmd = [
        "python", "-u", "infer_audio2vid.py",
        "--config", job_config_path,
        "-W",       str(w),
        "-H",       str(h),
        "--steps",  str(steps),
        "--cfg",    str(cfg),
        "--fps",    str(fps),
        "--seed",   str(seed),
        "--context_frames",        str(ctx_frames),
        "--context_overlap",       str(ctx_overlap),
        "--sample_rate",           "16000",
        "--facecrop_dilation_ratio", str(facecrop_dilation),
        "--facemusk_dilation_ratio", str(facemask_dilation),
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
        background_lock = float(input_data.get("background_lock", 0.0))
        job_id = str(uuid.uuid4())[:8]
        job_dir = os.path.join(WORKSPACE, job_id)
        os.makedirs(job_dir, exist_ok=True)

        if not input_data.get("avatar_image_url") or not input_data.get("audio_url"):
            return {"error": "avatar_image_url and audio_url are required"}

        # 1. Download Inputs
        raw_image_path  = os.path.join(job_dir, "avatar_raw.png")
        image_path      = os.path.join(job_dir, "avatar.png")
        background_ref_path = os.path.join(job_dir, "avatar_bg_ref.png")
        audio_path      = os.path.join(job_dir, "audio.wav")
        download_file(input_data["avatar_image_url"], raw_image_path)
        download_file(input_data["audio_url"], audio_path)

        if background_lock > 0.0:
            prepare_background_reference(raw_image_path, background_ref_path)

        # 1b. Preprocess audio: resample to 16kHz mono + loudnorm
        #     This is the primary fix for lip-sync drift.
        processed_audio_path = os.path.join(job_dir, "audio_processed.wav")
        audio_path = preprocess_audio(audio_path, processed_audio_path)

        # 1c. Preprocess image for maximum quality
        #     • Smart portrait crop   → proper talking-head framing
        #     • GFPGAN v1.4 restore   → sharpen/enhance face details (2× upscale)
        #     • White-balance fix     → neutralise colour casts
        skip_preprocess = input_data.get("skip_preprocess", False)
        if skip_preprocess:
            import shutil
            shutil.copy(raw_image_path, image_path)
            logger.info("Image preprocessing skipped (skip_preprocess=true).")
        else:
            preprocess_image(raw_image_path, image_path)

        # 2. Run Generation
        final_video_path = run_echomimic(image_path, audio_path, job_dir, input_data)

        # 2b. Face composite: paste only the animated face back onto the original
        #     reference image per frame.  This is the key HeyGen-style step —
        #     the diffusion model never touches the background, so there is zero
        #     drift and the original image quality is preserved everywhere except
        #     the mouth/face region.  background_lock is no longer needed.
        skip_composite = input_data.get("skip_composite", False)
        if not skip_composite:
            composite_path = os.path.join(job_dir, "output_composite.mp4")
            final_video_path = composite_face_video(
                final_video_path,
                image_path,
                composite_path,
                fps_override=float(input_data.get("fps", 24)),
            )

        # 2c. Post-process: Real-ESRGAN 2× upscale (512→1024) + H.264 CRF 16
        skip_enhance = input_data.get("skip_enhance", False)
        if not skip_enhance:
            from preprocess import enhance_video
            enhanced_path = os.path.join(job_dir, "output_enhanced.mp4")
            final_video_path = enhance_video(final_video_path, enhanced_path)

        if not isinstance(final_video_path, (str, os.PathLike)):
            raise RuntimeError(f"Invalid final video path returned: {final_video_path!r}")

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