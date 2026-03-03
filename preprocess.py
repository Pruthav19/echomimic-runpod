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
import sys
import logging
import types

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── torchvision compat shim ───────────────────────────────────────────────────
# torchvision ≥ 0.17 removed `torchvision.transforms.functional_tensor`.
# basicsr / GFPGAN still import from it; create a proxy module that forwards
# everything to the current `torchvision.transforms.functional`.
try:
    import torchvision.transforms.functional_tensor  # noqa: F401 (exists in old builds)
except ModuleNotFoundError:
    import torchvision.transforms.functional as _tvf
    _proxy = types.ModuleType("torchvision.transforms.functional_tensor")
    _proxy.__dict__.update(
        {k: v for k, v in vars(_tvf).items() if not k.startswith("__")}
    )
    sys.modules["torchvision.transforms.functional_tensor"] = _proxy

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


# ── Step 4: Natural head motion synthesis ─────────────────────────────────────
# Adds organic head micro-motion (drift, breathing, sway, scale pulse)
# so the avatar doesn't look like a static image with a moving mouth.
# Eye blinks and facial expressions are handled by EchoMimic itself
# (the face mask now includes the full face region).


def add_natural_motion(input_video_path: str, output_video_path: str) -> str:
    """
    Post-processing pass: adds organic head micro-motion so the avatar
    doesn't look like a static image with a moving mouth.

    Components:
      - Gaussian random-walk drift  (±12 px, ±2°) for organic head sway
      - Sinusoidal breathing bob     (0.3 Hz, ±3 px vertical)
      - Sinusoidal side-sway         (0.18 Hz, ±2 px horizontal)
      - Slight periodic scale pulse  (0.25 Hz, ±0.5%) for breathing depth

    Eye blinks and facial expressions are now handled by EchoMimic itself
    (the face mask includes the full face region).

    Falls back to copying the original if any step fails.
    """
    import shutil
    import subprocess as sp

    try:
        from scipy.ndimage import gaussian_filter1d
    except ImportError as e:
        logger.warning(f"add_natural_motion: scipy missing ({e}) — skipping.")
        shutil.copy(input_video_path, output_video_path)
        return output_video_path

    try:
        cap = cv2.VideoCapture(input_video_path)
        fps    = cap.get(cv2.CAP_PROP_FPS) or 24.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        raw_frames = []
        while True:
            ret, f = cap.read()
            if not ret:
                break
            raw_frames.append(f)
        cap.release()
        if not raw_frames:
            raise RuntimeError("Could not read any frames.")

        n = len(raw_frames)
        logger.info(f"Natural motion: {n} frames @ {fps:.1f} fps")

        t_axis = np.arange(n) / fps  # seconds
        rng = np.random.default_rng(42)

        # ── Random organic drift ─────────────────────────────────────────
        raw_dx = np.cumsum(rng.normal(0, 1.2, n))
        raw_dy = np.cumsum(rng.normal(0, 0.9, n))
        raw_da = np.cumsum(rng.normal(0, 0.14, n))  # degrees
        dx = np.clip(gaussian_filter1d(raw_dx, sigma=16), -12, 12)
        dy = np.clip(gaussian_filter1d(raw_dy, sigma=16), -10, 10)
        da = np.clip(gaussian_filter1d(raw_da, sigma=22), -2.0, 2.0)

        # ── Periodic components ──────────────────────────────────────────
        # Breathing head-bob
        breathing = np.sin(2 * np.pi * 0.30 * t_axis) * 3.0   # ±3 px vertical
        # Gentle side-to-side sway (different frequency → organic feel)
        sway      = np.sin(2 * np.pi * 0.18 * t_axis) * 2.0   # ±2 px horizontal
        # Micro-nod (very slow)
        nod       = np.sin(2 * np.pi * 0.12 * t_axis) * 0.6   # ±0.6° rotation
        # Breathing scale pulse (chest/shoulder rise)
        scale_pulse = 1.0 + np.sin(2 * np.pi * 0.25 * t_axis) * 0.004  # ±0.4%

        dy += breathing
        dx += sway
        da += nod

        # Return to zero so video doesn't walk off-centre
        dx -= np.linspace(dx[0], dx[-1], n)
        dy -= np.linspace(dy[0], dy[-1], n)
        da -= np.linspace(da[0], da[-1], n)

        cx, cy = width / 2.0, height / 2.0

        # ── Apply per-frame warp ─────────────────────────────────────────
        processed = []
        for i, frame in enumerate(raw_frames):
            s = float(scale_pulse[i])
            M = cv2.getRotationMatrix2D((cx, cy), float(da[i]), s)
            M[0, 2] += float(dx[i])
            M[1, 2] += float(dy[i])
            f = cv2.warpAffine(
                frame, M, (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            processed.append(f)

        # ── Write frames then mux original audio ────────────────────────
        tmp_path = output_video_path.replace(".mp4", "_nat_noaudio.mp4")
        writer = cv2.VideoWriter(
            tmp_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        for f in processed:
            writer.write(f)
        writer.release()

        sp.run(
            [
                "ffmpeg", "-y",
                "-i", tmp_path,
                "-i", input_video_path,
                "-c:v", "libx264", "-crf", "15", "-preset", "fast",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k",
                "-map", "0:v:0", "-map", "1:a:0",
                "-shortest",
                output_video_path,
            ],
            check=True, capture_output=True,
        )
        os.remove(tmp_path)
        logger.info(f"Natural motion applied → {output_video_path}")
        return output_video_path

    except Exception as e:
        logger.error(f"add_natural_motion failed: {e} — using original video.")
        if os.path.exists(output_video_path.replace(".mp4", "_nat_noaudio.mp4")):
            os.remove(output_video_path.replace(".mp4", "_nat_noaudio.mp4"))
        shutil.copy(input_video_path, output_video_path)
        return output_video_path


# ── Step 5: GFPGAN video face restoration + upscale ──────────────────────────
# Applies GFPGAN to every frame of the output video.
# This fixes teeth artefacts, sharpens eyes/skin, and upscales 2× (512→1024)
# all in one pass — better face quality than Real-ESRGAN alone.


def restore_video_faces(input_video_path: str, output_video_path: str) -> str:
    """
    Runs GFPGAN face restoration on every frame of the video.
    - Fixes diffusion-model artefacts (teeth, eye detail, skin texture)
    - Upscales 2× (512→1024) via GFPGAN's built-in super-resolution
    - Muxes original audio back via ffmpeg

    Falls back to input video on any error.
    """
    import subprocess as sp
    import shutil

    if not os.path.isfile(GFPGAN_MODEL_PATH):
        logger.warning("GFPGAN model not found — skipping video face restoration.")
        shutil.copy(input_video_path, output_video_path)
        return output_video_path

    tmp_video = output_video_path.replace(".mp4", "_gfpgan_noaudio.mp4")
    try:
        restorer = _get_restorer()

        cap = cv2.VideoCapture(input_video_path)
        fps    = cap.get(cv2.CAP_PROP_FPS) or 24
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # GFPGAN upscales 2× internally
        out_w, out_h = orig_w * 2, orig_h * 2

        writer = cv2.VideoWriter(
            tmp_video,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (out_w, out_h),
        )

        logger.info(f"GFPGAN video restore: {total} frames {orig_w}×{orig_h} → {out_w}×{out_h}")
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            try:
                _, _, restored = restorer.enhance(
                    frame,
                    has_aligned=False,
                    only_center_face=True,
                    paste_back=True,
                )
                if restored is not None:
                    # Ensure output matches expected dimensions
                    if restored.shape[:2] != (out_h, out_w):
                        restored = cv2.resize(restored, (out_w, out_h),
                                              interpolation=cv2.INTER_LANCZOS4)
                    writer.write(restored)
                else:
                    # Fallback: simple upscale
                    writer.write(cv2.resize(frame, (out_w, out_h),
                                            interpolation=cv2.INTER_LANCZOS4))
            except Exception:
                writer.write(cv2.resize(frame, (out_w, out_h),
                                        interpolation=cv2.INTER_LANCZOS4))
            idx += 1
            if idx % 48 == 0:
                logger.info(f"  {idx}/{total} frames restored")

        cap.release()
        writer.release()

        # Re-encode with libx264 + mux original audio
        sp.run(
            [
                "ffmpeg", "-y",
                "-i", tmp_video,
                "-i", input_video_path,
                "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k",
                "-map", "0:v:0", "-map", "1:a:0",
                "-shortest",
                output_video_path,
            ],
            check=True, capture_output=True,
        )
        os.remove(tmp_video)
        logger.info(f"GFPGAN video restored → {output_video_path}")
        return output_video_path

    except Exception as e:
        logger.error(f"GFPGAN video restoration failed: {e} — using original.")
        if os.path.exists(tmp_video):
            os.remove(tmp_video)
        shutil.copy(input_video_path, output_video_path)
        return output_video_path


# ── Step 6: Real-ESRGAN video upscaling (legacy/fallback) ────────────────────


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

    # 2b. Sharpening pass — counteracts GFPGAN's slight softness
    #     Unsharp mask: subtract a blurred copy to boost high-frequency detail
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=2.0)
    img = cv2.addWeighted(img, 1.4, blurred, -0.4, 0)
    img = np.clip(img, 0, 255).astype(np.uint8)

    # 3. White balance
    img = white_balance(img)

    h1, w1 = img.shape[:2]
    logger.info(f"Preprocessed image: [{w0}×{h0}] → [{w1}×{h1}], saved to {output_path}")

    cv2.imwrite(output_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    return output_path
