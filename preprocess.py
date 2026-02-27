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


# ── Step 4: Natural motion synthesis ─────────────────────────────────────────
# Three sub-steps applied to the generated video before upscaling:
#   A. Eye blink injection  – mediapipe face_mesh locates the eyelids;
#                             blinks fire at natural Poisson intervals (~4s).
#   B. Micro head motion    – smooth random-walk affine warp (±4 px, ±0.4°).
#   C. Face temporal smooth – 3-frame weighted blend on the face region only,
#                             removes diffusion-noise flicker.

def _get_face_mesh():
    """Lazy mediapipe FaceMesh singleton."""
    import mediapipe as mp
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


# Mediapipe face-mesh landmark indices — full eye contours (468-pt model)
# Right eye (from viewer's perspective)
_RIGHT_EYE_ALL   = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
_RIGHT_UPPER_LID = [159, 158, 157, 173, 133, 160, 161]   # upper arc
_RIGHT_LOWER_LID = [145, 144, 153, 154, 155, 33, 7]      # lower arc
# Left eye
_LEFT_EYE_ALL    = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
_LEFT_UPPER_LID  = [386, 385, 384, 398, 362, 387, 388]   # upper arc
_LEFT_LOWER_LID  = [374, 373, 380, 381, 382, 263, 249]   # lower arc


def _blink_schedule(total_frames: int, fps: float, seed: int = 0) -> set:
    """
    Returns a set of frame indices where a blink starts.
    Blinks follow a Poisson process with mean interval 4 s.
    Each blink occupies 5 frames: 2 close + 1 hold + 2 open.
    """
    rng = np.random.default_rng(seed)
    avg_interval = int(fps * 4)   # ~4 seconds between blinks
    frames = set()
    f = avg_interval // 2         # first blink not at frame 0
    while f < total_frames - 6:
        frames.add(f)
        # Poisson interval: exponential inter-arrival
        interval = max(int(fps * 2), int(rng.exponential(avg_interval)))
        f += interval
    return frames


def _apply_blink(frame: np.ndarray, landmarks, h: int, w: int,
                 t: float) -> np.ndarray:
    """
    Closes both eyes at weight *t* (0=open, 1=fully closed).

    Technique: slide the *actual upper-eyelid skin texture* downward over
    the eye region.  At weight t the lid covers t×eye_height pixels from
    the top of the eye down.  This produces photo-realistic closure with
    the correct skin tone, wrinkles and lashes of that specific face.
    A feathered blend mask at the lid edge prevents hard lines.
    """
    if t <= 0.0:
        return frame
    frame = frame.copy()

    for all_ids in [_RIGHT_EYE_ALL, _LEFT_EYE_ALL]:
        pts = np.array(
            [[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in all_ids],
            dtype=np.int32,
        )
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        ew = max(x2 - x1, 1)
        eh = max(y2 - y1, 1)

        # ── Source texture: strip of skin directly above the eye ──────────
        # Use the same height as the eye so a full close maps 1:1.
        src_y1 = max(0, y1 - eh)
        src_y2 = y1
        src_x1 = max(0, x1)
        src_x2 = min(w, x2)
        src_w  = src_x2 - src_x1

        if src_w <= 0 or (src_y2 - src_y1) <= 0:
            continue  # safety guard

        lid_texture = frame[src_y1:src_y2, src_x1:src_x2].copy()  # shape (eh, ew)

        # ── How many pixel rows the lid covers this frame ─────────────────
        cover_rows = int(round(t * eh))
        if cover_rows <= 0:
            continue

        # ── Eye contour mask (keeps painting inside the eye opening only) ─
        eye_mask = np.zeros((h, w), dtype=np.float32)
        cv2.fillPoly(eye_mask, [pts], 1.0)
        # Soft feather
        eye_mask = cv2.GaussianBlur(eye_mask, (5, 5), 0)

        # ── Paste lid texture row-by-row into destination ─────────────────
        dst_y1 = y1
        dst_y2 = min(y1 + cover_rows, y2, h)
        if dst_y2 <= dst_y1:
            continue

        paste_h = dst_y2 - dst_y1
        # Scale lid texture to paste_h rows (stretch lid as it closes)
        block = cv2.resize(lid_texture, (src_w, paste_h),
                           interpolation=cv2.INTER_LINEAR)

        # Feathered alpha at the bottom edge of the lid (soften lid line)
        alpha = np.ones((paste_h, src_w, 1), dtype=np.float32)
        feather = max(2, paste_h // 4)
        alpha[-feather:] *= np.linspace(1, 0, feather, dtype=np.float32).reshape(-1, 1, 1)

        # Extract eye mask slice
        em_slice = eye_mask[dst_y1:dst_y2, src_x1:src_x2, np.newaxis]
        combined_alpha = np.clip(alpha * em_slice, 0, 1)

        orig = frame[dst_y1:dst_y2, src_x1:src_x2].astype(np.float32)
        blended = (block.astype(np.float32) * combined_alpha
                   + orig * (1 - combined_alpha)).astype(np.uint8)
        frame[dst_y1:dst_y2, src_x1:src_x2] = blended

    return frame


def _smooth_trajectory(values: np.ndarray, sigma: float = 8.0) -> np.ndarray:
    """Gaussian-smooth a 1D trajectory."""
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(values, sigma=sigma)


def add_natural_motion(input_video_path: str, output_video_path: str) -> str:
    """
    Post-processing pass that makes the generated video feel more natural:

      A. Eye blinks   – synthesised every ~4 s via mediapipe face landmarks.
      B. Micro motion – smooth random-walk head sway (±4 px, ±0.4°).
      C. Face smooth  – light 3-frame temporal blend on the face ROI
                        to remove diffusion-model flicker.

    Falls back to copying the original if mediapipe is unavailable or any
    step fails.
    """
    import shutil
    import subprocess as sp

    try:
        import mediapipe as mp
        from scipy.ndimage import gaussian_filter1d
    except ImportError as e:
        logger.warning(f"add_natural_motion: missing dependency ({e}) — skipping.")
        shutil.copy(input_video_path, output_video_path)
        return output_video_path

    try:
        cap = cv2.VideoCapture(input_video_path)
        fps    = cap.get(cv2.CAP_PROP_FPS) or 24.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # ── Read all frames into memory (videos are short, typically <30s) ──
        raw_frames = []
        while True:
            ret, f = cap.read()
            if not ret:
                break
            raw_frames.append(f)
        cap.release()
        if not raw_frames:
            raise RuntimeError("Could not read any frames.")

        logger.info(f"Natural motion: {len(raw_frames)} frames @ {fps:.1f} fps")

        # ── B. Micro head motion trajectory ─────────────────────────────────
        n = len(raw_frames)
        t_axis = np.arange(n) / fps   # time in seconds
        rng = np.random.default_rng(42)

        # Random organic drift (smoothed cumulative noise)
        raw_dx = np.cumsum(rng.normal(0, 1.0, n))
        raw_dy = np.cumsum(rng.normal(0, 0.7, n))
        raw_da = np.cumsum(rng.normal(0, 0.10, n))
        dx = np.clip(_smooth_trajectory(raw_dx, 14), -10, 10)
        dy = np.clip(_smooth_trajectory(raw_dy, 14), -8,   8)
        da = np.clip(_smooth_trajectory(raw_da, 20), -1.5, 1.5)

        # Add a gentle sinusoidal breathing / head-bob (0.3 Hz ≈ one breath per 3s)
        breathing = np.sin(2 * np.pi * 0.30 * t_axis) * 2.5   # ±2.5 px vertical
        side_sway = np.sin(2 * np.pi * 0.18 * t_axis) * 1.5   # ±1.5 px horizontal
        dy += breathing
        dx += side_sway

        # Drift back toward zero so video doesn't drift off-centre
        dx -= np.linspace(dx[0], dx[-1], n)
        dy -= np.linspace(dy[0], dy[-1], n)
        da -= np.linspace(da[0], da[-1], n)

        cx, cy = width / 2, height / 2

        # ── Detect face ROI and landmarks on first frame for blink & smooth ─
        mesh = _get_face_mesh()
        first_rgb = cv2.cvtColor(raw_frames[0], cv2.COLOR_BGR2RGB)
        mesh_result = mesh.process(first_rgb)

        has_landmarks = (
            mesh_result.multi_face_landmarks is not None
            and len(mesh_result.multi_face_landmarks) > 0
        )
        lm0 = mesh_result.multi_face_landmarks[0].landmark if has_landmarks else None

        # Face bounding box from first-frame landmarks (for temporal smooth ROI)
        if has_landmarks:
            xs = [int(l.x * width)  for l in lm0]
            ys = [int(l.y * height) for l in lm0]
            face_x1 = max(0, min(xs) - 20)
            face_y1 = max(0, min(ys) - 20)
            face_x2 = min(width,  max(xs) + 20)
            face_y2 = min(height, max(ys) + 20)
        else:
            face_x1, face_y1, face_x2, face_y2 = 0, 0, width, height

        # ── A. Blink schedule ────────────────────────────────────────────────
        blink_starts = _blink_schedule(len(raw_frames), fps, seed=42)
        # Build per-frame blink weight (0=open … 1=closed … 0=open)
        # Shape: 2 frames ramp up, 1 hold, 2 ramp down
        blink_weight = np.zeros(len(raw_frames), dtype=np.float32)
        # 7-frame blink envelope — smooth rise and fall, no hard edges
        blink_curve  = [0.2, 0.7, 1.0, 1.0, 0.6, 0.2, 0.05]
        for bs in blink_starts:
            for o, w_val in enumerate(blink_curve):
                if bs + o < len(raw_frames):
                    blink_weight[bs + o] = max(blink_weight[bs + o], w_val)

        # ── Process frames ───────────────────────────────────────────────────
        processed = []
        for i, frame in enumerate(raw_frames):
            f = frame.copy()

            # C. Temporal face-region smooth (3-frame blend, avoid first/last)
            if 1 <= i <= len(raw_frames) - 2:
                prev_f = raw_frames[i - 1]
                next_f = raw_frames[i + 1]
                blended = (
                    0.15 * prev_f.astype(np.float32)
                    + 0.70 * f.astype(np.float32)
                    + 0.15 * next_f.astype(np.float32)
                ).astype(np.uint8)
                # Apply blend only within face ROI
                f[face_y1:face_y2, face_x1:face_x2] = \
                    blended[face_y1:face_y2, face_x1:face_x2]

            # A. Eye blink
            if has_landmarks and blink_weight[i] > 0.01:
                # Re-detect landmarks for this frame for accuracy
                rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                res = mesh.process(rgb)
                if res.multi_face_landmarks:
                    lm = res.multi_face_landmarks[0].landmark
                    f = _apply_blink(f, lm, height, width, float(blink_weight[i]))

            # B. Micro head motion (affine warp)
            angle  = float(da[i])
            tx, ty = float(dx[i]), float(dy[i])
            M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
            M[0, 2] += tx
            M[1, 2] += ty
            f = cv2.warpAffine(
                f, M, (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )

            processed.append(f)

        mesh.close()

        # ── Write processed frames to temp file then mux audio ───────────────
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


# ── Step 5: Real-ESRGAN video upscaling ──────────────────────────────────────


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
