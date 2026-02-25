import os
import sys
import urllib.request
from huggingface_hub import snapshot_download

MODEL_DIR = os.environ.get("MODEL_DIR", "/runpod-volume/echomimic_models")

def download_models():
    print(f"📥 Downloading EchoMimic models to {MODEL_DIR}...")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Main EchoMimic weights — MUST land directly in MODEL_DIR (not a subdir).
    #    start.sh symlinks MODEL_DIR → /app/EchoMimic/pretrained_weights, so
    #    files here map to ./pretrained_weights/<file> as the yaml config expects.
    #    This repo includes: denoising_unet.pth, reference_unet.pth,
    #    face_locator.pth, motion_module.pth, audio_processor/whisper_tiny.pt
    snapshot_download(repo_id="BadToBest/EchoMimic", local_dir=MODEL_DIR)

    # 2. SD Image Variations — backbone for the reference UNet
    snapshot_download(
        repo_id="lambdalabs/sd-image-variations-diffusers",
        local_dir=os.path.join(MODEL_DIR, "sd-image-variations-diffusers"),
    )

    # 3. Stable Diffusion VAE
    snapshot_download(repo_id="stabilityai/sd-vae-ft-mse", local_dir=os.path.join(MODEL_DIR, "sd-vae-ft-mse"))

    # 4. GFPGAN v1.4 — face restoration weights
    gfpgan_path = os.path.join(MODEL_DIR, "GFPGANv1.4.pth")
    if not os.path.exists(gfpgan_path):
        print("⬇️  Downloading GFPGANv1.4.pth…")
        urllib.request.urlretrieve(
            "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
            gfpgan_path,
        )
        print("✅ GFPGANv1.4.pth downloaded.")
    else:
        print("✅ GFPGANv1.4.pth already present.")

    print("✅ All models downloaded successfully!")

if __name__ == "__main__":
    try:
        download_models()
    except Exception as e:
        print(f"❌ Download failed: {e}", file=sys.stderr)
        sys.exit(1)