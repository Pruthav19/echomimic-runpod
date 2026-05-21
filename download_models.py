import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_HALLO4_HF_REPO = "fudan-generative-ai/hallo4"
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/runpod-volume/models"))
HF_TOKEN_ENV_NAMES = (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "HF_HUB_TOKEN",
    "HUGGINGFACE_TOKEN",
    "HUGGING_FACE_TOKEN",
)


def required_model_path_groups():
    return [
        [
            MODEL_DIR / "hallo4" / "model_weight.ckpt",
            MODEL_DIR / "hallo4" / "model_weight.pth",
            MODEL_DIR / "pretrained_models" / "hallo4" / "model_weight.ckpt",
            MODEL_DIR / "pretrained_models" / "hallo4" / "model_weight.pth",
        ],
        [
            MODEL_DIR / "Wan2.1_Encoders",
            MODEL_DIR / "pretrained_models" / "Wan2.1_Encoders",
        ],
        [
            MODEL_DIR / "audio_separator" / "Kim_Vocal_2.onnx",
            MODEL_DIR / "pretrained_models" / "audio_separator" / "Kim_Vocal_2.onnx",
        ],
        [
            MODEL_DIR / "wav2vec2-base-960h",
            MODEL_DIR / "wav2vec" / "wav2vec2-base-960h",
            MODEL_DIR / "pretrained_models" / "wav2vec2-base-960h",
            MODEL_DIR / "pretrained_models" / "wav2vec" / "wav2vec2-base-960h",
        ],
    ]


def missing_model_paths():
    return [
        group[0]
        for group in required_model_path_groups()
        if not any(path.exists() for path in group)
    ]


def resolve_hf_token():
    for name in HF_TOKEN_ENV_NAMES:
        token = os.environ.get(name)
        if token:
            return name, token
    return None, None


def download_models():
    os.makedirs(str(MODEL_DIR), exist_ok=True)
    print(f"Downloading Hallo4 models to {MODEL_DIR}...")

    repo_id = os.environ.get("HALLO4_HF_REPO", DEFAULT_HALLO4_HF_REPO)
    token_name, token = resolve_hf_token()
    if not token:
        raise RuntimeError(
            f"{repo_id} is gated, but no Hugging Face token was found. "
            f"Set one of: {', '.join(HF_TOKEN_ENV_NAMES)}."
        )
    print(f"Using Hugging Face token from {token_name}.")

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(MODEL_DIR),
            local_dir_use_symlinks=False,
            token=token,
        )
    except Exception as e:
        message = str(e)
        if "401" in message or "gated repo" in message.lower() or "restricted" in message.lower():
            raise RuntimeError(
                f"Cannot access gated Hugging Face repo {repo_id}. The token from "
                f"{token_name} is missing access, expired, or belongs to an account "
                "that has not accepted the model terms. Accept access with that same "
                "Hugging Face account, create a read token, update the RunPod env, "
                "and restart the worker."
            ) from e
        raise

    missing = missing_model_paths()
    if missing:
        raise FileNotFoundError(
            "Hallo4 model download finished, but required files are still missing: "
            + ", ".join(str(path) for path in missing)
        )
    print("Hallo4 models downloaded and verified.")


if __name__ == "__main__":
    try:
        download_models()
    except Exception as e:
        print(f"Download failed: {e}", file=sys.stderr)
        print(
            "The official Hallo4 repo is gated. Accept access at "
            "https://huggingface.co/fudan-generative-ai/hallo4, then set HF_TOKEN "
            "or mount the snapshot into MODEL_DIR.",
            file=sys.stderr,
        )
        sys.exit(1)
