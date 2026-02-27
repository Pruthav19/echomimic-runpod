"""
Image preprocessing pipeline for EchoMimic — aimed at HeyGen-level output quality.

Steps applied in order:
  1. Smart portrait crop  – face-centred crop with proper headroom + shoulder room
                            so EchoMimic's internal MTCNN crop gets ideal input.
  2. GFPGAN v1.4          – face restoration / sharpening (biggest quality gain).
                            upscale=2 also rescues small / blurry photos.
  3. White-balance fix     – removes colour casts caused by artificial lighting.

The GFPGANer is initialised once (module-level singleton) to avoid paying the
model-load cost on every request inside the same serverless worker process.
"""

import os
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = os.environ.get("MODEL_DIR", "/runpod-volume/echomimic_models")
GFPGAN_MODEL_PATH = os.path.join(MODEL_DIR, "GFPGANv1.4.pth")

# ── Lazy singleton ────────────────────────────────────────────────────────────
_gfpgan_restorer = None


def _get_restorer():
    global _gfpgan_restorer
    if _gfpgan_restorer is None:
        from gfpgan import GFPGANer
        logger.info("Loading GFPGAN v1.4 model (one-time cost)…")
        _gfpgan_restorer = GFPGANer(
            model_path=GFPGAN_MODEL_PATH,
            upscale=2,          # output is 2× the cropped input — rescues low-res images
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,  # keep background sharpening separate for speed
        )
        logger.info("GFPGAN loaded.")
    return _gfpgan_restorer


# ── Step 1: Smart portrait crop ───────────────────────────────────────────────

def _detect_face_opencv(img_bgr: np.ndarray):
    """
    Returns (x, y, w, h) of the largest detected face, or None.
    Uses OpenCV's built-in DNN face detector (more robust than Haar cascades).
    Falls back to Haar if the DNN weights are not present.
    """
    # Try DNN detector first (ships with opencv-python-headless / opencv-python)
    prototxt = cv2.data.haarcascades + "../dnn/deploy.prototext"  # may not exist
    modelfile = cv2.data.haarcascades + "../dnn/res10_300x300_ssd_iter_140000_fp16.caffemodel"

    if os.path.exists(prototxt) and os.path.exists(modelfile):
        net = cv2.dnn.readNetFromCaffe(prototxt, modelfile)
        h, w = img_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(img_bgr, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0)
        )
        net.setInput(blob)
        detections = net.forward()
        best, best_conf = None, 0.5
        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf > best_conf:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                best = (x1, y1, x2 - x1, y2 - y1)
                best_conf = conf
        if best:
            return best

    # Haar fallback
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])  # largest face


def smart_portrait_crop(img_bgr: np.ndarray) -> np.ndarray:
    """
    Crops the image to a portrait frame suited for a talking-head video:
      • horizontal: face-centre ± 1× face-width  (generous side room)
      • top:        face-top    – 0.6× face-height (forehead + hair)
      • bottom:     face-bottom + 1.5× face-height (neck + shoulders)

    If no face is detected the original image is returned unchanged.
    """
    face = _detect_face_opencv(img_bgr)
    if face is None:
        logger.warning("No face detected — skipping portrait crop.")
        return img_bgr

    ih, iw = img_bgr.shape[:2]
    fx, fy, fw, fh = face
    cx, cy = fx + fw // 2, fy + fh // 2

    # Portrait framing constants (tuned for talking-head videos)
    PAD_TOP    = 0.6   # above face top  → hair / forehead room
    PAD_SIDES  = 1.0   # each side of face centre
    PAD_BOTTOM = 1.5   # below face bottom → neck + shoulder

    half_w = int(fw * PAD_SIDES)
    top    = max(0, fy - int(fh * PAD_TOP))
    bottom = min(ih, fy + fh + int(fh * PAD_BOTTOM))
    left   = max(0, cx - half_w)
    right  = min(iw, cx + half_w)

    # Make it square (pad the shorter axis) so resize to 512×512 is lossless
    crop_h = bottom - top
    crop_w = right - left
    if crop_h > crop_w:
        diff = crop_h - crop_w
        left  = max(0, left  - diff // 2)
        right = min(iw, right + diff // 2)
    elif crop_w > crop_h:
        diff = crop_w - crop_h
        top    = max(0, top    - diff // 2)
        bottom = min(ih, bottom + diff // 2)

    cropped = img_bgr[top:bottom, left:right]
    logger.info(f"Portrait crop: ({left},{top}) → ({right},{bottom}), face bbox {face}")
    return cropped


# ── Step 2: GFPGAN face restoration ──────────────────────────────────────────

def restore_face(img_bgr: np.ndarray) -> np.ndarray:
    """
    Runs GFPGAN v1.4 on the image.
    - only_center_face=True  → enhances the dominant face only
    - paste_back=True        → merges enhanced face back into the (upscaled) photo
    Returns the restored image in BGR uint8.
    """
    if not os.path.isfile(GFPGAN_MODEL_PATH):
        logger.warning(f"GFPGAN weights not found at {GFPGAN_MODEL_PATH} — skipping face restoration.")
        return img_bgr

    try:
        restorer = _get_restorer()
        _, _, restored = restorer.enhance(
            img_bgr,
            has_aligned=False,
            only_center_face=True,
            paste_back=True,
        )
        if restored is None:
            logger.warning("GFPGAN returned None — using original image.")
            return img_bgr
        return restored
    except Exception as e:
        logger.error(f"GFPGAN enhancement failed: {e} — using original image.")
        return img_bgr


# ── Step 3: White balance ─────────────────────────────────────────────────────

def white_balance(img_bgr: np.ndarray) -> np.ndarray:
    """
    Gray-world white balance correction.
    Removes colour casts caused by office / indoor lighting.
    """
    img_f = img_bgr.astype(np.float32)
    avg_b = np.mean(img_f[:, :, 0])
    avg_g = np.mean(img_f[:, :, 1])
    avg_r = np.mean(img_f[:, :, 2])
    avg   = (avg_b + avg_g + avg_r) / 3.0

    if avg_b > 0:
        img_f[:, :, 0] = np.clip(img_f[:, :, 0] * (avg / avg_b), 0, 255)
    if avg_g > 0:
        img_f[:, :, 1] = np.clip(img_f[:, :, 1] * (avg / avg_g), 0, 255)
    if avg_r > 0:
        img_f[:, :, 2] = np.clip(img_f[:, :, 2] * (avg / avg_r), 0, 255)

    return img_f.astype(np.uint8)


# ── Step 4: Real-ESRGAN video upscaling ──────────────────────────────────────

REALESRGAN_MODEL_PATH = os.path.join(MODEL_DIR, "RealESRGAN_x2plus.pth")
_realesrgan_upsampler = None


def _get_upsampler():
    global _realesrgan_upsampler
    if _realesrgan_upsampler is None:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        logger.info("Loading Real-ESRGAN x2 model (one-time cost)…")
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=2,
        )
        _realesrgan_upsampler = RealESRGANer(
            scale=2,
            model_path=REALESRGAN_MODEL_PATH,
            model=model,
            tile=256,        # tile to avoid OOM on large frames
            tile_pad=16,
            pre_pad=0,
            half=True,       # fp16 for speed on CUDA
        )
        logger.info("Real-ESRGAN loaded.")
    return _realesrgan_upsampler


def enhance_video(input_video_path: str, output_video_path: str) -> str:
    """
    Upscales every frame of *input_video_path* 2× using Real-ESRGAN
    (512×512 → 1024×1024), then muxes the original audio back via ffmpeg
    and re-encodes with high-quality H.264 (CRF 16).

    Falls back to copying the original video if the model is missing or
    any step fails, so inference output is never lost.
    """
    import subprocess as sp
    import shutil

    if not os.path.isfile(REALESRGAN_MODEL_PATH):
        logger.warning("RealESRGAN model not found — skipping video enhancement.")
        shutil.copy(input_video_path, output_video_path)
        return output_video_path

    tmp_video = output_video_path.replace(".mp4", "_noaudio.mp4")
    try:
        upsampler = _get_upsampler()

        cap = cv2.VideoCapture(input_video_path)
        fps        = cap.get(cv2.CAP_PROP_FPS) or 24
        orig_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out_w, out_h = orig_w * 2, orig_h * 2

        writer = cv2.VideoWriter(
            tmp_video,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (out_w, out_h),
        )

        logger.info(f"Upscaling {total} frames {orig_w}×{orig_h} → {out_w}×{out_h}…")
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            enhanced, _ = upsampler.enhance(frame, outscale=2)
            writer.write(enhanced)
            idx += 1
            if idx % 48 == 0:
                logger.info(f"  {idx}/{total} frames upscaled")

        cap.release()
        writer.release()

        # Re-encode with libx264 CRF 16 and mux original audio
        sp.run(
            [
                "ffmpeg", "-y",
                "-i", tmp_video,
                "-i", input_video_path,
                "-c:v", "libx264", "-crf", "16", "-preset", "fast",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k",
                "-map", "0:v:0", "-map", "1:a:0",
                "-shortest",
                output_video_path,
            ],
            check=True,
            capture_output=True,
        )
        os.remove(tmp_video)
        logger.info(f"Video enhanced → {output_video_path}")
        return output_video_path

    except Exception as e:
        logger.error(f"Video enhancement failed: {e} — using original video.")
        if os.path.exists(tmp_video):
            os.remove(tmp_video)
        shutil.copy(input_video_path, output_video_path)
        return output_video_path


# ── Public entry point ────────────────────────────────────────────────────────

def preprocess_image(input_path: str, output_path: str) -> str:
    """
    Full preprocessing pipeline. Reads *input_path*, applies all steps,
    writes the result to *output_path* and returns *output_path*.

    Steps:
      1. Smart portrait crop
      2. GFPGAN face restoration (upscales 2×)
      3. White balance correction
    """
    img = cv2.imread(input_path)
    if img is None:
        raise RuntimeError(f"Cannot read image: {input_path}")

    h0, w0 = img.shape[:2]
    logger.info(f"Preprocessing image {input_path} [{w0}×{h0}]")

    # 1. Portrait crop
    img = smart_portrait_crop(img)

    # 2. GFPGAN restoration (also upscales 2×)
    img = restore_face(img)

    # 3. White balance
    img = white_balance(img)

    h1, w1 = img.shape[:2]
    logger.info(f"Preprocessed image: [{w0}×{h0}] → [{w1}×{h1}], saved to {output_path}")

    cv2.imwrite(output_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    return output_path
