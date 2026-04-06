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

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/models")
GFPGAN_MODEL_PATH = os.path.join(MODEL_DIR, "GFPGANv1.4.pth")

# ── Lazy singleton ────────────────────────────────────────────────────────────
_gfpgan_restorer = None


def _ensure_torchvision_compat():
    """
    GFPGAN / facexlib / Real-ESRGAN may import the old
    `torchvision.transforms.functional_tensor` module path, which was moved in
    newer torchvision releases. Register a compatibility alias when possible.
    """
    legacy_name = "torchvision.transforms.functional_tensor"
    if legacy_name in sys.modules:
        return

    try:
        from torchvision.transforms import _functional_tensor as functional_tensor
        sys.modules[legacy_name] = functional_tensor
        logger.info("Registered torchvision functional_tensor compatibility shim.")
    except Exception as e:
        logger.warning(f"Torchvision compatibility shim unavailable: {e}")


def _get_restorer():
    global _gfpgan_restorer
    if _gfpgan_restorer is None:
        _ensure_torchvision_compat()
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
        _ensure_torchvision_compat()
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


# Real-ESRGAN on an H100 runs ~0.15s/frame — 60s of video costs ~216s of GPU time,
# well within budget. Fallback to ffmpeg only for very long videos (>60s).
_MAX_REALESRGAN_FRAMES = 1440   # 60 s at 24 fps


def _ffmpeg_upscale(input_video_path: str, output_video_path: str) -> str:
    """Fast 2× upscale via ffmpeg lanczos + mild unsharp. CPU-only, no GPU cost."""
    import subprocess as sp

    sp.run(
        [
            "ffmpeg", "-y",
            "-i", input_video_path,
            "-vf", "scale=iw*2:ih*2:flags=lanczos,unsharp=5:5:0.8:3:3:0.0",
            "-c:v", "libx264", "-crf", "16", "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            output_video_path,
        ],
        check=True,
        capture_output=True,
    )
    logger.info(f"Video upscaled (ffmpeg lanczos) → {output_video_path}")
    return output_video_path


_SHARPEN_KERNEL = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0],
], dtype=np.float32)


def _sharpen_mouth_region(frame: np.ndarray, face: tuple) -> np.ndarray:
    """
    Apply targeted unsharp sharpening to the mouth area of an upscaled frame.
    `face` is (x, y, w, h) in the upscaled frame's coordinate space.
    Uses a soft elliptical blend mask so there's no hard edge.
    """
    fx, fy, fw, fh = face
    # Mouth region: centre-bottom 40% of the face bbox
    mx = fx + fw // 2
    my = fy + int(fh * 0.65)          # vertical centre of mouth zone
    mw = int(fw * 0.65)               # half-width of sharpening ellipse
    mh = int(fh * 0.22)               # half-height

    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(mask, (mx, my), (max(1, mw), max(1, mh)), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (31, 31), 0)[..., None]

    sharpened = cv2.filter2D(frame, -1, _SHARPEN_KERNEL)
    result = frame.astype(np.float32) * (1.0 - mask) + sharpened.astype(np.float32) * mask
    return np.clip(result, 0, 255).astype(np.uint8)


def _realesrgan_upscale(input_video_path: str, output_video_path: str) -> str:
    """Full Real-ESRGAN 2× upscale with targeted mouth sharpening."""
    import subprocess as sp

    upsampler = _get_upsampler()
    tmp_video = output_video_path.replace(".mp4", "_noaudio.mp4")

    cap = cv2.VideoCapture(input_video_path)
    fps      = cap.get(cv2.CAP_PROP_FPS) or 24
    orig_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_w, out_h = orig_w * 2, orig_h * 2

    writer = cv2.VideoWriter(
        tmp_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h)
    )
    logger.info(f"RealESRGAN: upscaling {total} frames {orig_w}×{orig_h} → {out_w}×{out_h}…")

    # Detect face once from first frame to get stable sharpening coords.
    # Re-detect every 48 frames to handle any drift.
    cached_face = None
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        enhanced, _ = upsampler.enhance(frame, outscale=2)

        if idx % 48 == 0:
            cached_face = _detect_face_opencv(enhanced)
            if idx > 0:
                logger.info(f"  {idx}/{total} frames upscaled")

        if cached_face is not None:
            enhanced = _sharpen_mouth_region(enhanced, cached_face)

        writer.write(enhanced)
        idx += 1

    cap.release()
    writer.release()

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
    logger.info(f"Video enhanced (RealESRGAN + mouth sharpen) → {output_video_path}")
    return output_video_path


def enhance_video(input_video_path: str, output_video_path: str) -> str:
    """
    2× upscale the output video (512×512 → 1024×1024).

    Strategy:
    - Short clips (≤ _MAX_REALESRGAN_FRAMES): Real-ESRGAN for maximum quality.
    - Long clips (> _MAX_REALESRGAN_FRAMES): ffmpeg lanczos + unsharp — visually
      close to ESRGAN but uses CPU only and runs ~20× faster, keeping cost in budget.
    - Falls back to the original video if anything fails.
    """
    import shutil

    # Probe frame count without decoding
    cap = cv2.VideoCapture(input_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    use_realesrgan = (
        os.path.isfile(REALESRGAN_MODEL_PATH)
        and total_frames <= _MAX_REALESRGAN_FRAMES
    )

    try:
        if use_realesrgan:
            return _realesrgan_upscale(input_video_path, output_video_path)
        else:
            if not os.path.isfile(REALESRGAN_MODEL_PATH):
                logger.info("RealESRGAN weights absent — using ffmpeg upscale.")
            else:
                logger.info(
                    f"Video has {total_frames} frames (>{_MAX_REALESRGAN_FRAMES} limit) "
                    "— using fast ffmpeg upscale to stay in budget."
                )
            return _ffmpeg_upscale(input_video_path, output_video_path)
    except Exception as e:
        logger.error(f"Video enhancement failed: {e} — keeping original video.")
        shutil.copy(input_video_path, output_video_path)
        return output_video_path


def composite_face_video(
    generated_video: str,
    reference_image: str,
    output_video: str,
    fps_override: float | None = None,
) -> str:
    """
    HeyGen-style face composite.

    The generated video (512×512, from EchoMimic) is always face-zoomed.
    The reference image is the original photo at its native resolution.

    Strategy:
      1. Detect face in both the reference image and the generated frame.
      2. For each frame: crop the face region out of the generated frame,
         warp it to match the face region in the reference image, and blend
         it in with a soft mask.  Everything outside the face comes from
         the original photo — zero background drift, original quality.
    """
    import shutil
    import subprocess as sp

    ref_orig = cv2.imread(reference_image)
    if ref_orig is None:
        logger.warning("composite_face_video: cannot read reference — skipping.")
        shutil.copy(generated_video, output_video)
        return output_video

    cap = cv2.VideoCapture(generated_video)
    fps   = float(fps_override) if fps_override else (cap.get(cv2.CAP_PROP_FPS) or 24)
    gen_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    gen_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read the first frame to detect the face in the generated video
    ret, first_frame = cap.read()
    if not ret:
        logger.warning("composite_face_video: could not read first frame — skipping.")
        cap.release()
        shutil.copy(generated_video, output_video)
        return output_video
    cap.release()

    # --- Detect face in the generated frame (512×512 space) ---
    gen_face = _detect_face_opencv(first_frame)
    if gen_face is None:
        logger.warning("composite_face_video: no face in generated frame — skipping.")
        shutil.copy(generated_video, output_video)
        return output_video
    gfx, gfy, gfw, gfh = gen_face

    # --- Detect face in the original reference image (native resolution) ---
    ref_face = _detect_face_opencv(ref_orig)
    if ref_face is None:
        logger.warning("composite_face_video: no face in reference image — skipping.")
        shutil.copy(generated_video, output_video)
        return output_video
    rfx, rfy, rfw, rfh = ref_face

    ref_H, ref_W = ref_orig.shape[:2]

    # Padding multiplier: how much context around the face bbox to include in the
    # composite patch. 0.55 = just over half a face-width of padding on each side.
    PAD = 0.55

    # --- Source patch coords in generated (512×512) frame ---
    gpad_x = int(gfw * PAD)
    gpad_y = int(gfh * PAD)
    gs_x1 = max(0, gfx - gpad_x)
    gs_y1 = max(0, gfy - gpad_y)
    gs_x2 = min(gen_W, gfx + gfw + gpad_x)
    gs_y2 = min(gen_H, gfy + gfh + gpad_y)

    # --- Destination patch coords in original reference (native res) ---
    rpad_x = int(rfw * PAD)
    rpad_y = int(rfh * PAD)
    rd_x1 = max(0, rfx - rpad_x)
    rd_y1 = max(0, rfy - rpad_y)
    rd_x2 = min(ref_W, rfx + rfw + rpad_x)
    rd_y2 = min(ref_H, rfy + rfh + rpad_y)
    dest_w = rd_x2 - rd_x1
    dest_h = rd_y2 - rd_y1

    # --- Build the blend mask in destination patch space ---
    # Ellipse centred on the lower half of the face (mouth/chin area gets full weight).
    mask_patch = np.zeros((dest_h, dest_w), dtype=np.float32)
    mc_x = dest_w // 2
    mc_y = int(dest_h * 0.45)
    cv2.ellipse(
        mask_patch,
        (mc_x, mc_y),
        (max(1, int(dest_w * 0.42)), max(1, int(dest_h * 0.46))),
        0, 0, 360, 1.0, -1,
    )
    feather = max(21, int(dest_w * 0.22) | 1)   # must be odd
    mask_patch = cv2.GaussianBlur(mask_patch, (feather, feather), 0)[..., None]

    # --- Output is at the reference image's native resolution ---
    out_W, out_H = ref_W, ref_H

    tmp_video = output_video.replace(".mp4", "_noaudio.mp4")
    writer = cv2.VideoWriter(
        tmp_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_W, out_H)
    )

    logger.info(
        f"Face composite: {total} frames, gen face {gen_face} → "
        f"ref face {ref_face}, dest patch {dest_w}×{dest_h}…"
    )

    ref_f = ref_orig.astype(np.float32)

    # Re-open the video to iterate from frame 0
    cap = cv2.VideoCapture(generated_video)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Start with the original reference photo
        canvas = ref_f.copy()

        # 2. Crop the face patch from the generated frame and scale to dest size
        src_patch = frame[gs_y1:gs_y2, gs_x1:gs_x2]
        src_resized = cv2.resize(src_patch, (dest_w, dest_h), interpolation=cv2.INTER_LANCZOS4)

        # 3. Blend into the reference at the correct location
        dest_region = canvas[rd_y1:rd_y2, rd_x1:rd_x2]
        blended_region = (
            dest_region * (1.0 - mask_patch)
            + src_resized.astype(np.float32) * mask_patch
        )
        canvas[rd_y1:rd_y2, rd_x1:rd_x2] = blended_region

        writer.write(np.clip(canvas, 0, 255).astype(np.uint8))
        idx += 1
        if idx % 96 == 0:
            logger.info(f"  composite {idx}/{total}")

    cap.release()
    writer.release()

    sp.run(
        [
            "ffmpeg", "-y",
            "-i", tmp_video,
            "-i", generated_video,
            "-c:v", "libx264", "-crf", "16", "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-map", "0:v:0", "-map", "1:a:0",
            "-shortest",
            output_video,
        ],
        check=True,
        capture_output=True,
    )
    os.remove(tmp_video)
    logger.info(f"Face composite done → {output_video}")
    return output_video


def stabilize_background(
    input_video_path: str,
    reference_image_path: str,
    output_video_path: str,
    lock_strength: float = 0.85,
    fps_override: float | None = None,
) -> str:
    """
    Reduces generative background drift by repainting background-like regions
    toward the original background color outside a soft head-and-shoulders mask.

    `lock_strength` range:
      0.0 = disabled
            1.0 = background fully pulled back to the reference background color

        This improves stability for static portraits while avoiding the ghosting
        that happens when the full reference portrait is blended back in.
    """
    import shutil
    import subprocess as sp

    lock_strength = float(np.clip(lock_strength, 0.0, 1.0))
    if lock_strength <= 0.0:
        shutil.copy(input_video_path, output_video_path)
        return output_video_path

    reference = cv2.imread(reference_image_path)
    if reference is None:
        logger.warning("Background lock skipped: cannot read reference image.")
        shutil.copy(input_video_path, output_video_path)
        return output_video_path

    cap = cv2.VideoCapture(input_video_path)
    fps = float(fps_override) if fps_override else (cap.get(cv2.CAP_PROP_FPS) or 24)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_w <= 0 or frame_h <= 0:
        logger.warning("Background lock skipped: invalid video dimensions.")
        cap.release()
        shutil.copy(input_video_path, output_video_path)
        return output_video_path

    reference = cv2.resize(reference, (frame_w, frame_h), interpolation=cv2.INTER_CUBIC)
    face = _detect_face_opencv(reference)
    if face is None:
        logger.warning("Background lock skipped: no face detected in reference image.")
        cap.release()
        shutil.copy(input_video_path, output_video_path)
        return output_video_path

    fx, fy, fw, fh = face
    cx = int(fx + fw / 2)
    cy = int(fy + fh * 1.0)

    mask = np.zeros((frame_h, frame_w), dtype=np.float32)
    # Protect more of the head / hair / shoulders so the reference image
    # does not bleed through as a translucent "shadow" around the subject.
    ellipse_axes = (max(1, int(fw * 1.75)), max(1, int(fh * 2.25)))
    cv2.ellipse(mask, (cx, cy), ellipse_axes, 0, 0, 360, 1.0, -1)

    shoulder_left = max(0, int(cx - fw * 1.9))
    shoulder_right = min(frame_w, int(cx + fw * 1.9))
    shoulder_top = max(0, int(fy + fh * 0.75))
    shoulder_bottom = min(frame_h, int(fy + fh * 3.2))
    cv2.rectangle(mask, (shoulder_left, shoulder_top), (shoulder_right, shoulder_bottom), 1.0, -1)

    blur_size = 41
    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)

    # Stronger lock near frame edges helps suppress drifting/tinted studio
    # backgrounds without freezing the central head-and-shoulders region.
    yy, xx = np.mgrid[0:frame_h, 0:frame_w]
    nx = ((xx / max(1, frame_w - 1)) - 0.5) * 2.0
    ny = ((yy / max(1, frame_h - 1)) - 0.5) * 2.0
    radial = np.clip(np.sqrt(nx * nx + ny * ny), 0.0, 1.0)
    edge_boost = np.clip((radial - 0.35) / 0.65, 0.0, 1.0).astype(np.float32)

    background_mix = np.maximum(
        (1.0 - mask) * lock_strength,
        edge_boost * min(1.0, lock_strength + 0.12) * 0.55,
    )

    # Only lock pixels that look like real background in the reference image.
    reference_hsv = cv2.cvtColor(reference, cv2.COLOR_BGR2HSV)
    reference_bg_mask = (
        (reference_hsv[:, :, 1] <= 28) &
        (reference_hsv[:, :, 2] >= 180)
    ).astype(np.float32)
    reference_bg_mask = cv2.GaussianBlur(reference_bg_mask, (31, 31), 0)

    base_background_mix = background_mix * reference_bg_mask

    bg_pixels = reference[reference_bg_mask > 0.65]
    if bg_pixels.size == 0:
        corner_size = max(8, min(frame_w, frame_h) // 12)
        corner_samples = [
            reference[:corner_size, :corner_size],
            reference[:corner_size, frame_w - corner_size:],
            reference[frame_h - corner_size:, :corner_size],
            reference[frame_h - corner_size:, frame_w - corner_size:],
        ]
        bg_pixels = np.concatenate([sample.reshape(-1, 3) for sample in corner_samples], axis=0)

    bg_color = np.mean(bg_pixels, axis=0).astype(np.float32)
    background_canvas = np.broadcast_to(bg_color, (frame_h, frame_w, 3)).astype(np.float32)

    tmp_video = output_video_path.replace(".mp4", "_noaudio.mp4")
    writer = cv2.VideoWriter(
        tmp_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_w, frame_h),
    )

    logger.info(
        f"Applying background lock (strength={lock_strength:.2f}) to {total} frames..."
    )

    try:
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            frame_bg_mask = (
                (frame_hsv[:, :, 1] <= 40) &
                (frame_hsv[:, :, 2] >= 150)
            ).astype(np.float32)
            frame_bg_mask = cv2.GaussianBlur(frame_bg_mask, (21, 21), 0)

            # Only replace areas that are background-like in BOTH the original
            # reference and the generated frame. This keeps the animated
            # subject intact and only repaints likely background pixels.
            frame_background_mix = (base_background_mix * frame_bg_mask)[..., None]

            blended = (
                frame.astype(np.float32) * (1.0 - frame_background_mix)
                + background_canvas * frame_background_mix
            )
            writer.write(np.clip(blended, 0, 255).astype(np.uint8))
            idx += 1
            if idx % 48 == 0:
                logger.info(f"  {idx}/{total} frames background-locked")

        cap.release()
        writer.release()

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
        logger.info(f"Background stabilized → {output_video_path}")
        return output_video_path
    except Exception as e:
        logger.error(f"Background lock failed: {e} — using original video.")
        cap.release()
        writer.release()
        if os.path.exists(tmp_video):
            os.remove(tmp_video)
        shutil.copy(input_video_path, output_video_path)
        return output_video_path

# ── Public entry point ────────────────────────────────────────────────────────

def _center_square_crop(img: np.ndarray) -> np.ndarray:
    """
    Crops the image to a centre square without zooming into the face.
    Preserves the original framing (person + background) — EchoMimic
    does its own internal face detection; we must NOT pre-crop to the face.
    """
    h, w = img.shape[:2]
    if h == w:
        return img
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return img[y0:y0 + side, x0:x0 + side]


def preprocess_image(input_path: str, output_path: str) -> str:
    """
    Preprocessing pipeline for EchoMimic input.

    Steps:
      1. Centre-square crop  – preserves original framing (no face zoom)
      2. GFPGAN face restoration (upscales 2×, sharpens face details)
      3. White balance correction
      4. Resize to 512×512  – EchoMimic's required input resolution

    NOTE: smart_portrait_crop was removed. It cropped tight to the face
    before EchoMimic, so the model generated a face-zoomed output and the
    final video looked like a giant close-up. EchoMimic's internal MTCNN
    handles face detection — our job is just to give it a clean square image
    at the right resolution with the original framing intact.
    """
    img = cv2.imread(input_path)
    if img is None:
        raise RuntimeError(f"Cannot read image: {input_path}")

    h0, w0 = img.shape[:2]
    logger.info(f"Preprocessing image {input_path} [{w0}×{h0}]")

    # 1. Centre-square crop (no face zoom)
    img = _center_square_crop(img)

    # 2. GFPGAN restoration (upscales 2×)
    img = restore_face(img)

    # 3. White balance
    img = white_balance(img)

    # 4. Resize to 512×512 for EchoMimic
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LANCZOS4)

    h1, w1 = img.shape[:2]
    logger.info(f"Preprocessed image: [{w0}×{h0}] → [{w1}×{h1}], saved to {output_path}")

    cv2.imwrite(output_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    return output_path


def prepare_background_reference(input_path: str, output_path: str) -> str:
    """
    Prepares a background reference aligned to the talking-head crop but without
    GFPGAN or white-balance changes. This helps preserve the original background
    color (e.g. clean white studio backdrops) for background stabilization.
    """
    img = cv2.imread(input_path)
    if img is None:
        raise RuntimeError(f"Cannot read image: {input_path}")

    img = smart_portrait_crop(img)
    cv2.imwrite(output_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    logger.info(f"Background reference prepared → {output_path}")
    return output_path
