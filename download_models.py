import os
import sys
from huggingface_hub import snapshot_download, hf_hub_download

MODEL_DIR = os.environ.get("MODEL_DIR", "/runpod-volume/models")


def download_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Downloading models to {MODEL_DIR}...")

    # ── EchoMimicV3-Flash weights ────────────────────────────────
    # Flash variant (consistency-distilled, 8 steps). Repo layout:
    #   echomimicv3-flash-pro/config.json
    #   echomimicv3-flash-pro/diffusion_pytorch_model.safetensors
    flash_dir = os.path.join(MODEL_DIR, "echomimicv3-flash-pro")
    if not os.path.isfile(
        os.path.join(flash_dir, "diffusion_pytorch_model.safetensors")
    ):
        print("Downloading EchoMimicV3-Flash transformer from HuggingFace...")
        snapshot_download(
            repo_id="BadToBest/EchoMimicV3",
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False,
            allow_patterns=["echomimicv3-flash-pro/**"],
        )
        print("EchoMimicV3-Flash downloaded.")
    else:
        print("EchoMimicV3-Flash already present.")

    # ── Wan2.1-Fun-V1.1-1.3B-InP base model (~19 GB) ────────────
    wan_dir = os.path.join(flash_dir, "Wan2.1-Fun-V1.1-1.3B-InP")
    if not os.path.isfile(os.path.join(wan_dir, "config.json")):
        print("Downloading Wan2.1-Fun-V1.1-1.3B-InP base model (~19 GB)...")
        snapshot_download(
            repo_id="alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP",
            local_dir=wan_dir,
            local_dir_use_symlinks=False,
        )
        print("Wan2.1-Fun-V1.1-1.3B-InP downloaded.")
    else:
        print("Wan2.1-Fun-V1.1-1.3B-InP already present.")

    # ── chinese-wav2vec2-base audio encoder ──────────────────────
    wav2vec_dir = os.path.join(flash_dir, "chinese-wav2vec2-base")
    if not os.path.isfile(os.path.join(wav2vec_dir, "config.json")):
        print("Downloading chinese-wav2vec2-base audio encoder...")
        snapshot_download(
            repo_id="TencentGameMate/chinese-wav2vec2-base",
            local_dir=wav2vec_dir,
            local_dir_use_symlinks=False,
        )
        print("chinese-wav2vec2-base downloaded.")
    else:
        print("chinese-wav2vec2-base already present.")

    # ── LatentSync 1.6 checkpoints ───────────────────────────────
    # LatentSync runs as a post-processing pass to refine lip sync on the
    # EchoMimicV3-generated video. Two files needed:
    #   latentsync_unet.pt  — the diffusion U-Net (~2-3 GB)
    #   whisper/tiny.pt     — Whisper tiny audio encoder (~150 MB)
    #
    # LatentSync expects these at checkpoints/ relative to its CWD.
    # At runtime the handler creates a symlink:
    #   /app/latentsync/checkpoints → MODEL_DIR/latentsync
    latentsync_dir = os.path.join(MODEL_DIR, "latentsync")
    whisper_dir = os.path.join(latentsync_dir, "whisper")
    os.makedirs(latentsync_dir, exist_ok=True)
    os.makedirs(whisper_dir, exist_ok=True)

    if not os.path.isfile(os.path.join(latentsync_dir, "latentsync_unet.pt")):
        print("Downloading LatentSync 1.6 UNet checkpoint...")
        hf_hub_download(
            repo_id="ByteDance/LatentSync-1.6",
            filename="latentsync_unet.pt",
            local_dir=latentsync_dir,
            local_dir_use_symlinks=False,
        )
        print("LatentSync UNet downloaded.")
    else:
        print("LatentSync UNet already present.")

    if not os.path.isfile(os.path.join(whisper_dir, "tiny.pt")):
        print("Downloading Whisper tiny checkpoint for LatentSync...")
        hf_hub_download(
            repo_id="ByteDance/LatentSync-1.6",
            filename="whisper/tiny.pt",
            local_dir=latentsync_dir,
            local_dir_use_symlinks=False,
        )
        print("Whisper tiny downloaded.")
    else:
        print("Whisper tiny already present.")

    print("All models downloaded successfully.")


if __name__ == "__main__":
    try:
        download_models()
    except Exception as e:
        print(f"Download failed: {e}", file=sys.stderr)
        sys.exit(1)
