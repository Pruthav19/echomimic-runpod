"""
Preprocessing for EchoMimicV3-Flash RunPod handler.

Minimal — EchoMimicV3 handles face detection, audio feature extraction,
and loudness normalization internally. We just ensure clean input formats.
"""
import subprocess
import logging

logger = logging.getLogger(__name__)


def preprocess_audio(input_path: str, output_path: str) -> str:
    """
    Convert audio to 16 kHz mono WAV (pcm_s16le).

    EchoMimicV3's inference code applies its own pyloudnorm normalization
    (-23 LUFS) so we do NOT normalize here — just format conversion.
    """
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", input_path,
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "pcm_s16le",
            output_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.warning(f"Audio preprocessing failed: {result.stderr}")
        return input_path
    logger.info(f"Audio preprocessed → mono 16kHz → {output_path}")
    return output_path
