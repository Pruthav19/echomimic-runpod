import os
import sys
from huggingface_hub import snapshot_download

MODEL_DIR = os.environ.get("MODEL_DIR", "/runpod-volume/models")


def download_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Downloading EchoMimicV3-Preview models to {MODEL_DIR}...")

    # EchoMimicV3 Preview (non-Flash) weights — repo layout:
    #   transformer/config.json
    #   transformer/diffusion_pytorch_model.safetensors (~3.41 GB, full model)
    # The Preview variant is trained for 25 denoising steps, not distilled like
    # Flash. It produces higher-fidelity lip sync because fine mouth detail is
    # rendered in the last refinement steps that Flash compresses away.
    preview_dir = os.path.join(MODEL_DIR, "echomimicv3-preview")
    transformer_dir = os.path.join(preview_dir, "transformer")
    if not os.path.isfile(
        os.path.join(transformer_dir, "diffusion_pytorch_model.safetensors")
    ):
        print("Downloading EchoMimicV3-Preview transformer from HuggingFace...")
        snapshot_download(
            repo_id="BadToBest/EchoMimicV3",
            local_dir=preview_dir,
            local_dir_use_symlinks=False,
            allow_patterns=["transformer/**"],
        )
        print("EchoMimicV3-Preview transformer downloaded.")
    else:
        print("EchoMimicV3-Preview transformer already present.")

    # Base Wan2.1-Fun-V1.1-1.3B-InP model (separate repo, ~19 GB)
    wan_dir = os.path.join(preview_dir, "Wan2.1-Fun-V1.1-1.3B-InP")
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
    wav2vec_dir = os.path.join(preview_dir, "chinese-wav2vec2-base")
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
