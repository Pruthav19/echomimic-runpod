import os
import sys
from huggingface_hub import snapshot_download

MODEL_DIR = os.environ.get("MODEL_DIR", "/runpod-volume/echomimic_models")

def download_models():
    print(f"📥 Downloading EchoMimic models to {MODEL_DIR}...")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Main EchoMimic Weights
    snapshot_download(repo_id="BadToBest/EchoMimic", local_dir=os.path.join(MODEL_DIR, "EchoMimic"))
    
    # 2. Stable Diffusion VAE
    snapshot_download(repo_id="stabilityai/sd-vae-ft-mse", local_dir=os.path.join(MODEL_DIR, "sd-vae-ft-mse"))
    
    # 3. Audio Processor (Whisper)
    snapshot_download(repo_id="openai/whisper-small", local_dir=os.path.join(MODEL_DIR, "whisper-small"))

    print("✅ All models downloaded successfully!")

if __name__ == "__main__":
    try:
        download_models()
    except Exception as e:
        print(f"❌ Download failed: {e}", file=sys.stderr)
        sys.exit(1)