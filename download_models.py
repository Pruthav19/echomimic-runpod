import os
import sys
from huggingface_hub import snapshot_download

MODEL_DIR = os.environ.get("MODEL_DIR", "/runpod-volume/models")


def download_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Downloading Hallo4 models to {MODEL_DIR}...")

    # Set this env var to your Hallo4 model repo (HF), e.g. org/repo-name
    repo_id = os.environ.get("HALLO4_HF_REPO", "")
    if not repo_id:
        print("HALLO4_HF_REPO is not set. Skipping automatic model download.")
        print("Mount models into MODEL_DIR or set HALLO4_HF_REPO for auto-download.")
        return

    snapshot_download(
        repo_id=repo_id,
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False,
    )
    print("Hallo4 models downloaded.")


if __name__ == "__main__":
    try:
        download_models()
    except Exception as e:
        print(f"Download failed: {e}", file=sys.stderr)
        sys.exit(1)
