import os
import sys
from huggingface_hub import snapshot_download

MODEL_DIR = os.environ.get("MODEL_DIR", "/runpod-volume/models")


def download_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Downloading EchoMimicV3-Flash models to {MODEL_DIR}...")

    # EchoMimicV3 Flash weights — repo layout:
    #   echomimicv3-flash-pro/config.json
    #   echomimicv3-flash-pro/diffusion_pytorch_model.safetensors (fine-tuned weights)
    flash_dir = os.path.join(MODEL_DIR, "echomimicv3-flash-pro")
    if not os.path.isfile(
        os.path.join(flash_dir, "diffusion_pytorch_model.safetensors")
    ):
        print("Downloading EchoMimicV3-Flash from HuggingFace...")
        snapshot_download(
            repo_id="BadToBest/EchoMimicV3",
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False,
            allow_patterns=["echomimicv3-flash-pro/**"],
        )
        print("EchoMimicV3-Flash downloaded.")
    else:
        print("EchoMimicV3-Flash already present.")

    # Base Wan2.1-Fun-V1.1-1.3B-InP model (separate repo, ~19 GB)
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

    # chinese-wav2vec2-base audio encoder (separate HuggingFace repo)
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

    print("All models downloaded successfully.")


if __name__ == "__main__":
    try:
        download_models()
    except Exception as e:
        print(f"Download failed: {e}", file=sys.stderr)
        sys.exit(1)
