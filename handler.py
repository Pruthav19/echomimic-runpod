"""
EchoMimicV3-Flash + LatentSync RunPod Serverless Handler.

Two-stage pipeline:
  1. EchoMimicV3-Flash  — generates the talking-head video (visual quality,
                           head motion, eye blinks, identity preservation)
  2. LatentSync 1.6      — post-processes the video to refine lip sync
                           (runs in a separate venv to avoid dependency
                           conflicts: torch 2.5.1 vs torch 2.2.2)

Models are loaded ONCE at first invocation (EchoMimicV3) and kept in GPU
memory. LatentSync runs as a subprocess per job using its own Python venv.
"""
import os
import sys
import uuid
import math
import logging
import subprocess

import numpy as np
import torch
import runpod
import boto3
import requests
import librosa
from PIL import Image
from moviepy import VideoFileClip, AudioFileClip
import pyloudnorm as pyln

# ── Add EchoMimicV3 to import path ──────────────────────────────
ECHOMIMIC_DIR = "/app/echomimic_v3"
sys.path.insert(0, ECHOMIMIC_DIR)

from omegaconf import OmegaConf
from safetensors.torch import load_file as load_safetensors
from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor
from einops import rearrange

from src.wan_vae import AutoencoderKLWan
from src.wan_image_encoder import CLIPModel
from src.wan_text_encoder import WanT5EncoderModel
from src.wan_transformer3d_audio_2512 import (
    WanTransformerAudioMask3DModel as WanTransformer,
)
from src.pipeline_wan_fun_inpaint_audio_2512 import WanFunInpaintAudioPipeline
from src.fm_solvers_unipc import FlowUniPCMultistepScheduler
from src.cache_utils import get_teacache_coefficients
from src.utils import (
    filter_kwargs,
    get_image_to_video_latent2,
    get_image_to_video_latent3,
    save_videos_grid,
)
from src.wav2vec2 import Wav2Vec2Model

# ── Constants ────────────────────────────────────────────────────
WORKSPACE = "/tmp/workspace"
MODEL_DIR = os.environ.get("MODEL_DIR", "/runpod-volume/models")
MODEL_BASE = os.path.join(MODEL_DIR, "echomimicv3-flash-pro")
MODEL_NAME = os.path.join(MODEL_BASE, "Wan2.1-Fun-V1.1-1.3B-InP")
TRANSFORMER_PATH = os.path.join(
    MODEL_BASE, "diffusion_pytorch_model.safetensors"
)
WAV2VEC_DIR = os.path.join(MODEL_BASE, "chinese-wav2vec2-base")
CONFIG_PATH = os.path.join(ECHOMIMIC_DIR, "config", "config.yaml")

# ── LatentSync constants ─────────────────────────────────────────
# LatentSync runs as a subprocess using its own isolated Python venv.
# It expects checkpoints at /app/latentsync/checkpoints/ (relative to CWD).
# We symlink that directory to the network volume at runtime.
LATENTSYNC_DIR = "/app/latentsync"
LATENTSYNC_PYTHON = "/app/latentsync_env/bin/python"
LATENTSYNC_CKPT_DIR = os.path.join(MODEL_DIR, "latentsync")
LATENTSYNC_UNET = os.path.join(LATENTSYNC_CKPT_DIR, "latentsync_unet.pt")
LATENTSYNC_CONFIG = "configs/unet/stage2_512.yaml"

S3_BUCKET = os.environ.get("S3_BUCKET", "your-bucket-name")
S3_REGION = os.environ.get("S3_REGION", "us-east-1")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY", "")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16
FPS = 25

# ── Default inference parameters (EchoMimicV3-Flash) ─────────────
# Flash is designed for 8-step fast inference with good visual quality.
# Lip sync is handled by LatentSync (post-processing), so we tune these
# for visual fidelity and smooth motion — not lip tracking.
DEFAULTS = {
    "num_inference_steps": 8,
    "guidance_scale": 6.0,
    "audio_guidance_scale": 2.5,
    "neg_scale": 1.0,
    "neg_steps": 0,
    "audio_scale": 1.0,  # NOTE: dead param in upstream pipeline, kept for API parity
    "sample_size": [768, 768],
    "seed": 43,
    "teacache_threshold": 0.1,
    "num_skip_start_steps": 8,   # == num_inference_steps → TeaCache disabled
    "riflex_k": 6,
    "shift": 5.0,
    # Chunking is THE source of face-structure drift. Every chunk boundary
    # re-conditions the model on drifted frames from the previous chunk,
    # and the bias compounds. Best fix: make chunks large enough that most
    # videos fit in ONE chunk (zero drift) and longer ones only cross one
    # boundary.
    #   81 frames (3.24s) → 10s video = 4 chunks (3 drift boundaries)
    #  241 frames (9.64s) → 10s video = 1 chunk (zero drift)
    #                     → 15s video = 2 chunks (1 drift boundary)
    # RiFlex (enable_riflex) handles length extension beyond trained seq.
    # H100 80 GB has plenty of headroom — 241 frames at 768×768 bf16 fits.
    "partial_video_length": 241,  # frames per chunk (241 = 9.64s @ 25fps)
    "overlap_video_length": 4,    # overlap frames blended between chunks
    # Prompt and negative_prompt feed T5 text encoder → cross-attention in
    # the transformer. This is EchoMimicV3's documented control surface for
    # motion/expression. The positive prompt describes *desired* dynamics;
    # the negative prompt (combined via CFG since guidance_scale > 1) pushes
    # the model *away* from static / stiff outputs. Adapted from official
    # infer_preview.py negative prompt.
    # Prompt kept minimal. Earlier versions with "engaged eye contact" and
    # "lively facial expressions" were pushing the model toward rapid
    # blinking and over-animated faces, especially in later chunks where
    # the effect compounds. Let the model use its default motion priors.
    "prompt": "A person is speaking naturally.",
    "negative_prompt": (
        "static, stiff, frozen, motionless, expressionless, blank stare, "
        "unnatural, robotic, rigid posture, bad hands, "
        "twisted fingers, blurry, low quality, rapid blinking, twitching."
    ),
    # LatentSync post-processing parameters (per-request tunable)
    "latentsync_steps": 20,       # denoising steps for LatentSync (20 = default)
    "latentsync_guidance": 1.5,   # CFG scale for LatentSync
    "latentsync_enabled": True,   # set False to skip LatentSync for debugging
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# LatentSync symlink setup (done once at process startup)
# ═══════════════════════════════════════════════════════════════════

_latentsync_ready = False


def _ensure_latentsync_checkpoints():
    """
    LatentSync's inference script hardcodes 'checkpoints/whisper/tiny.pt'
    relative to its CWD (/app/latentsync). We create a symlink from
    /app/latentsync/checkpoints → MODEL_DIR/latentsync so the script
    finds its weights on the network volume without copying 3GB into the image.
    """
    global _latentsync_ready
    if _latentsync_ready:
        return

    ckpt_link = os.path.join(LATENTSYNC_DIR, "checkpoints")
    if os.path.islink(ckpt_link):
        # Already a symlink — verify it points to the right place
        if os.readlink(ckpt_link) == LATENTSYNC_CKPT_DIR:
            _latentsync_ready = True
            return
        os.remove(ckpt_link)

    if os.path.isdir(ckpt_link):
        # Shouldn't happen in a clean container, but be safe
        logger.warning("checkpoints/ exists as a real dir — skipping symlink")
        _latentsync_ready = True
        return

    os.symlink(LATENTSYNC_CKPT_DIR, ckpt_link)
    logger.info(f"LatentSync checkpoints symlinked: {ckpt_link} → {LATENTSYNC_CKPT_DIR}")
    _latentsync_ready = True


# ═══════════════════════════════════════════════════════════════════
# Model loading — lazy singleton, loaded once per worker lifetime
# ═══════════════════════════════════════════════════════════════════

_models = None


def _load_models():
    """Load all EchoMimicV3-Flash models into GPU memory."""
    logger.info("Loading EchoMimicV3-Flash models (first job on this worker)...")

    config = OmegaConf.load(CONFIG_PATH)

    # ── Audio encoder (wav2vec) ──
    wav2vec_fe = Wav2Vec2FeatureExtractor.from_pretrained(WAV2VEC_DIR)
    audio_encoder = Wav2Vec2Model.from_pretrained(WAV2VEC_DIR)
    audio_encoder = audio_encoder.eval().to(DEVICE)
    logger.info("Wav2Vec audio encoder loaded.")

    # ── Transformer (Flash fine-tuned weights) ──
    transformer = WanTransformer.from_pretrained(
        MODEL_NAME,
        transformer_additional_kwargs=OmegaConf.to_container(
            config["transformer_additional_kwargs"]
        ),
        low_cpu_mem_usage=True,
        torch_dtype=DTYPE,
    )
    state_dict = load_safetensors(TRANSFORMER_PATH)
    m, u = transformer.load_state_dict(state_dict, strict=False)
    logger.info(f"Transformer loaded (missing={len(m)}, unexpected={len(u)}).")

    # ── VAE ──
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(
            MODEL_NAME, config["vae_kwargs"].get("vae_subpath", "vae")
        ),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    ).to(DTYPE)
    logger.info("VAE loaded.")

    # ── Tokenizer + Text encoder ──
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(
            MODEL_NAME,
            config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer"),
        ),
    )
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(
            MODEL_NAME,
            config["text_encoder_kwargs"].get(
                "text_encoder_subpath", "text_encoder"
            ),
        ),
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=DTYPE,
    ).eval()
    logger.info("Text encoder loaded.")

    # ── CLIP Image encoder ──
    clip_image_encoder = CLIPModel.from_pretrained(
        os.path.join(
            MODEL_NAME,
            config["image_encoder_kwargs"].get(
                "image_encoder_subpath", "image_encoder"
            ),
        ),
    ).to(DTYPE).eval()
    logger.info("CLIP image encoder loaded.")

    # ── Scheduler ──
    sched_kwargs = OmegaConf.to_container(config["scheduler_kwargs"])
    sched_kwargs["shift"] = 1  # required for Flow_Unipc
    scheduler = FlowUniPCMultistepScheduler(
        **filter_kwargs(FlowUniPCMultistepScheduler, sched_kwargs)
    )

    # ── Pipeline ──
    pipeline = WanFunInpaintAudioPipeline(
        transformer=transformer,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
        clip_image_encoder=clip_image_encoder,
    ).to(device=DEVICE)

    # ── TeaCache initialization ──
    # num_skip_start_steps == num_inference_steps → cache never fires.
    # Full computation on every step preserves audio cross-attention signal.
    coefficients = get_teacache_coefficients(MODEL_NAME)
    if coefficients is not None:
        pipeline.transformer.enable_teacache(
            coefficients,
            DEFAULTS["num_inference_steps"],
            DEFAULTS["teacache_threshold"],
            num_skip_start_steps=DEFAULTS["num_skip_start_steps"],
            offload=False,
        )
        logger.info(
            "TeaCache initialized but disabled "
            "(num_skip_start_steps == num_inference_steps)."
        )

    logger.info("EchoMimicV3-Flash models loaded. Worker is ready.")
    return {
        "pipeline": pipeline,
        "vae": vae,
        "wav2vec_fe": wav2vec_fe,
        "audio_encoder": audio_encoder,
        "config": config,
    }


def get_models():
    global _models
    if _models is None:
        _models = _load_models()
    return _models


# ═══════════════════════════════════════════════════════════════════
# Audio / video helper functions
# ═══════════════════════════════════════════════════════════════════


def loudness_norm(audio_array, sr=16000, lufs=-23):
    """Normalize audio to target LUFS (EchoMimicV3's expected level)."""
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio_array)
    if abs(loudness) > 100:
        return audio_array
    return pyln.normalize.loudness(audio_array, loudness, lufs)


def get_audio_embed(mel_input, wav2vec_fe, audio_encoder, video_length, sr=16000):
    """Extract wav2vec audio embeddings aligned to video frames."""
    audio_feature = np.squeeze(
        wav2vec_fe(mel_input, sampling_rate=sr).input_values
    )
    audio_feature = torch.from_numpy(audio_feature).float().to(device=DEVICE)
    audio_feature = audio_feature.unsqueeze(0)

    with torch.no_grad():
        embeddings = audio_encoder(
            audio_feature, seq_len=int(video_length), output_hidden_states=True
        )

    audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
    audio_emb = rearrange(audio_emb, "b s d -> s b d")
    return audio_emb.cpu().detach()


def align_to_vae(length, vae):
    """Align video length to VAE temporal compression ratio."""
    if length <= 1:
        return 1
    tcr = vae.config.temporal_compression_ratio
    return int((length - 1) // tcr * tcr) + 1


def get_sample_size(pil_img, sample_size):
    """Compute output resolution aligned to 16px, capped at sample_size area."""
    w, h = pil_img.size
    ori_a = w * h
    default_a = sample_size[0] * sample_size[1]
    if default_a < ori_a:
        ratio = math.sqrt(ori_a / default_a)
        w = int(w / ratio) // 16 * 16
        h = int(h / ratio) // 16 * 16
    else:
        w = w // 16 * 16
        h = h // 16 * 16
    return [int(h), int(w)]


# ═══════════════════════════════════════════════════════════════════
# LatentSync post-processing
# ═══════════════════════════════════════════════════════════════════


def run_latentsync(base_video, audio_path, output_path, job_dir, p):
    """
    Refine lip sync on base_video using LatentSync 1.6.

    Runs as a subprocess using the isolated /app/latentsync_env Python venv
    to avoid torch/diffusers version conflicts with EchoMimicV3's environment.

    LatentSync expects its model checkpoints at checkpoints/ relative to its
    CWD. We create a one-time symlink from /app/latentsync/checkpoints to
    the network volume directory where the weights were downloaded.
    """
    _ensure_latentsync_checkpoints()

    temp_dir = os.path.join(job_dir, "latentsync_temp")
    os.makedirs(temp_dir, exist_ok=True)

    cmd = [
        LATENTSYNC_PYTHON, "-m", "scripts.inference",
        "--unet_config_path", LATENTSYNC_CONFIG,
        "--inference_ckpt_path", LATENTSYNC_UNET,
        "--video_path", base_video,
        "--audio_path", audio_path,
        "--video_out_path", output_path,
        "--temp_dir", temp_dir,
        "--inference_steps", str(int(p["latentsync_steps"])),
        "--guidance_scale", str(float(p["latentsync_guidance"])),
        "--enable_deepcache",
    ]

    logger.info(f"Running LatentSync lip sync refinement → {output_path}")
    # Stream stdout/stderr live instead of capturing — LatentSync prints
    # tqdm progress bars and model loading status. Capturing makes the run
    # look "stuck" from the outside when it's actually loading weights
    # (Whisper tiny + UNet ~3 GB = 30–60 s on first invocation).
    result = subprocess.run(
        cmd,
        cwd=LATENTSYNC_DIR,
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"LatentSync failed (exit {result.returncode}). "
            "See streamed stderr above for details."
        )

    logger.info("LatentSync completed successfully.")
    return output_path


# ═══════════════════════════════════════════════════════════════════
# Inference — chunked for arbitrary-length audio
# ═══════════════════════════════════════════════════════════════════


def run_echomimic_v3(image_path, audio_path, job_dir, user_params):
    """
    Generate a talking-head video from a portrait image and audio.
    Processes audio in chunks of partial_video_length frames with overlap
    blending for seamless long-video output.
    """
    models = get_models()
    pipeline = models["pipeline"]
    vae = models["vae"]
    wav2vec_fe = models["wav2vec_fe"]
    audio_encoder = models["audio_encoder"]

    # Merge user params with defaults
    p = {**DEFAULTS, **{k: v for k, v in user_params.items() if k in DEFAULTS}}
    seed = int(p["seed"])
    partial_len = int(p["partial_video_length"])
    overlap_len = int(p["overlap_video_length"])

    # ── Load image ──
    ref_image = Image.open(image_path).convert("RGB")
    sample_h, sample_w = get_sample_size(ref_image, p["sample_size"])
    logger.info(f"Output resolution: {sample_w}x{sample_h}")

    # ── Load & normalize audio ──
    mel_input, sr = librosa.load(audio_path, sr=16000)
    mel_input = loudness_norm(mel_input, sr)

    audio_clip = AudioFileClip(audio_path)
    total_video_length = align_to_vae(int(audio_clip.duration * FPS), vae)
    logger.info(f"Total video: {total_video_length} frames ({total_video_length/FPS:.1f}s)")

    # Trim audio to match video length
    mel_trimmed = mel_input[: int(total_video_length / FPS * sr)]

    # Full audio embeddings
    audio_emb_full = get_audio_embed(
        mel_trimmed, wav2vec_fe, audio_encoder, total_video_length
    )
    audio_embeds = audio_emb_full.unsqueeze(0).to(device=DEVICE, dtype=DTYPE)

    # ── Build frame indices for audio embeddings ──
    indices = (torch.arange(2 * 2 + 1) - 2) * 1
    center_indices = (
        torch.arange(0, total_video_length, 1).unsqueeze(1) + indices.unsqueeze(0)
    )
    center_indices = torch.clamp(center_indices, min=0, max=audio_embeds.shape[1] - 1)
    audio_embeds = audio_embeds[:, center_indices.squeeze()]
    if audio_embeds.dim() == 3:
        audio_embeds = audio_embeds.unsqueeze(0)

    # ── RiFlex for long sequences ──
    partial_len_aligned = align_to_vae(partial_len, vae)
    latent_frames = (partial_len_aligned - 1) // vae.config.temporal_compression_ratio + 1
    pipeline.transformer.enable_riflex(k=int(p["riflex_k"]), L_test=latent_frames)

    prompt = p["prompt"]
    negative_prompt = p["negative_prompt"]

    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    # ── Chunked generation with overlap blending ──
    # current_ref: rolling window of last-N frames from the previous chunk.
    #              Feeds get_image_to_video_latent3 to maintain MOTION continuity
    #              across chunk boundaries (prevents the "reset to portrait"
    #              flicker every 3.24 seconds).
    # identity_clip_image: locked to the ORIGINAL portrait for every chunk.
    #              This prevents IDENTITY drift that would otherwise compound
    #              across chunks (chunk 3's clip_image would be conditioned on
    #              chunk 2's drifted frames, which was conditioned on chunk 1's,
    #              etc. — user-visible as "face gets sharper / changes character
    #              after ~6 seconds"). Keeping motion via input_video while
    #              anchoring identity via clip_image is the production fix.
    current_ref = ref_image
    identity_clip_image = None
    new_sample = None
    init_frames = 0

    while init_frames < total_video_length:
        remaining = total_video_length - init_frames
        chunk_len = min(partial_len_aligned, remaining)
        chunk_len = align_to_vae(chunk_len, vae)
        if chunk_len <= 0:
            break

        if init_frames == 0:
            input_video, input_video_mask, clip_image = get_image_to_video_latent2(
                current_ref, None,
                video_length=chunk_len,
                sample_size=[sample_h, sample_w],
            )
            # Capture the original portrait's CLIP conditioning once.
            identity_clip_image = clip_image
        else:
            input_video, input_video_mask, _ = get_image_to_video_latent3(
                current_ref, None,
                video_length=chunk_len,
                sample_size=[sample_h, sample_w],
            )
            # Override: use the ORIGINAL portrait's CLIP image, not the drifted
            # current_ref[0] that latent3 would have returned.
            clip_image = identity_clip_image

        chunk_audio = audio_embeds[:, init_frames: init_frames + chunk_len]

        sample = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=chunk_len,
            audio_embeds=chunk_audio,
            audio_scale=float(p["audio_scale"]),
            ip_mask=None,
            use_un_ip_mask=False,
            height=sample_h,
            width=sample_w,
            generator=generator,
            neg_scale=float(p["neg_scale"]),
            neg_steps=int(p["neg_steps"]),
            use_dynamic_cfg=False,
            use_dynamic_acfg=False,
            guidance_scale=float(p["guidance_scale"]),
            audio_guidance_scale=float(p["audio_guidance_scale"]),
            num_inference_steps=int(p["num_inference_steps"]),
            video=input_video,
            mask_video=input_video_mask,
            clip_image=clip_image,
            cfg_skip_ratio=0.0,
            shift=float(p["shift"]),
        ).videos

        # Blend with previous chunk using linear crossfade
        if init_frames != 0 and new_sample is not None:
            mix = torch.from_numpy(
                np.array(
                    [float(i) / float(overlap_len) for i in range(overlap_len)],
                    np.float32,
                )
            ).unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            new_sample[:, :, -overlap_len:] = (
                new_sample[:, :, -overlap_len:] * (1 - mix)
                + sample[:, :, :overlap_len] * mix
            )
            new_sample = torch.cat(
                [new_sample, sample[:, :, overlap_len:]], dim=2
            )
        else:
            new_sample = sample

        # Update reference to last overlap_len frames of this chunk
        current_ref = [
            Image.fromarray(
                (sample[0, :, i].permute(1, 2, 0).clamp(0, 1) * 255)
                .byte().cpu().numpy()
            )
            for i in range(-overlap_len, 0)
        ]

        init_frames += chunk_len - overlap_len
        if init_frames + overlap_len >= total_video_length:
            break

    # ── Save video ──
    tmp_video = os.path.join(job_dir, "tmp_output.mp4")
    save_videos_grid(
        new_sample[:, :, :total_video_length], tmp_video, fps=FPS
    )

    # ── Mux audio onto video ──
    base_video = os.path.join(job_dir, "base_output.mp4")
    video_clip = VideoFileClip(tmp_video)
    audio_clip_trimmed = audio_clip.subclipped(0, total_video_length / FPS)
    video_clip = video_clip.with_audio(audio_clip_trimmed)
    video_clip.write_videofile(
        base_video, codec="libx264", audio_codec="aac", threads=2
    )
    video_clip.close()
    audio_clip.close()
    os.remove(tmp_video)

    return base_video, p


# ═══════════════════════════════════════════════════════════════════
# S3 helpers
# ═══════════════════════════════════════════════════════════════════


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
    s3.upload_file(
        local_path,
        S3_BUCKET,
        s3_key,
        ExtraArgs={"ContentType": "video/mp4"},
    )
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": s3_key},
        ExpiresIn=3600,
    )


# ═══════════════════════════════════════════════════════════════════
# RunPod handler
# ═══════════════════════════════════════════════════════════════════


def handler(event):
    try:
        input_data = event.get("input", {})

        if not input_data.get("avatar_image_url") or not input_data.get("audio_url"):
            return {"error": "avatar_image_url and audio_url are required"}

        job_id = str(uuid.uuid4())[:8]
        job_dir = os.path.join(WORKSPACE, job_id)
        os.makedirs(job_dir, exist_ok=True)

        # ── 1. Download inputs ──
        image_path = os.path.join(job_dir, "avatar.png")
        audio_path = os.path.join(job_dir, "audio.wav")

        download_file(input_data["avatar_image_url"], image_path)
        download_file(input_data["audio_url"], audio_path)

        # ── 2. Preprocess audio (16kHz mono WAV) ──
        processed_audio = os.path.join(job_dir, "audio_16k.wav")
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", audio_path,
                "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
                processed_audio,
            ],
            capture_output=True,
            text=True,
        )
        audio_path = processed_audio

        # ── 3. Stage 1: EchoMimicV3-Flash — generate base talking-head video ──
        base_video, p = run_echomimic_v3(
            image_path, audio_path, job_dir, input_data
        )

        # ── 4. Upload the RAW EchoMimicV3 output (pre-LatentSync) ──
        # This lets us diagnose whether face-structure drift is coming from
        # EchoMimicV3 generation or from LatentSync's face-crop inpainting.
        base_s3_key = f"echomimic_v3/{job_id}_base.mp4"
        base_video_url = upload_to_s3(base_video, base_s3_key)

        # ── 5. Stage 2: LatentSync — refine lip sync on the generated video ──
        if p.get("latentsync_enabled", True):
            final_video = os.path.join(job_dir, "output_synced.mp4")
            run_latentsync(base_video, audio_path, final_video, job_dir, p)
            synced_s3_key = f"echomimic_v3/{job_id}_synced.mp4"
            synced_video_url = upload_to_s3(final_video, synced_s3_key)
        else:
            logger.info("LatentSync disabled via payload — skipping.")
            synced_video_url = base_video_url

        subprocess.run(["rm", "-rf", job_dir], capture_output=True)

        return {
            # Primary output — the LatentSync-refined video (or base if disabled)
            "video_url": synced_video_url,
            # Both stages exposed separately so face-drift source can be diagnosed:
            #   base_video_url  — raw EchoMimicV3 output, before lip refinement
            #   synced_video_url — same as video_url, after LatentSync refinement
            "base_video_url": base_video_url,
            "synced_video_url": synced_video_url,
            "job_id": job_id,
            "status": "success",
        }

    except Exception as e:
        logger.exception("Handler error")
        return {"error": str(e), "status": "failed"}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
