"""Video and Image preprocessing service for DeepGuard.

Optimized for presentation speed and maximum forensic accuracy:
Extracts 24 uniform keyframes to guarantee complete temporal coverage.
"""

from pathlib import Path
import urllib.request
import cv2
import numpy as np
from PIL import Image

class VideoProcessingError(Exception):
    pass

SUPPORTED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
SUPPORTED_EXTENSIONS = SUPPORTED_VIDEO_EXTENSIONS | SUPPORTED_IMAGE_EXTENSIONS

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
YUNET_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"

def ensure_yunet_model() -> Path:
    if YUNET_PATH.exists() and YUNET_PATH.stat().st_size > 100_000:
        return YUNET_PATH

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(YUNET_URL, str(YUNET_PATH))
    except Exception:
        pass

    if not YUNET_PATH.exists() or YUNET_PATH.stat().st_size < 100_000:
        raise VideoProcessingError("Could not access YuNet face detection model.")
    return YUNET_PATH

_detector = None
_detector_input_size = None

def get_face_detector(width: int, height: int):
    global _detector, _detector_input_size
    model_path = ensure_yunet_model()

    if _detector is None or _detector_input_size != (width, height):
        try:
            _detector = cv2.FaceDetectorYN_create(
                str(model_path), "", (width, height),
                score_threshold=0.45, nms_threshold=0.30, top_k=5000
            )
            _detector_input_size = (width, height)
        except Exception as e:
            raise VideoProcessingError(f"Failed to initialize YuNet: {e}")

    return _detector

def crop_face(frame_bgr: np.ndarray, margin: float = 0.30) -> tuple[np.ndarray, bool, list | None]:
    h_img, w_img, _ = frame_bgr.shape
    detector = get_face_detector(w_img, h_img)

    try:
        _, faces = detector.detect(frame_bgr)
        if faces is not None and len(faces) > 0:
            best_face = max(faces, key=lambda f: f[2] * f[3])
            bx, by, bw, bh = best_face[0:4].astype(int)
            conf = float(best_face[-1])
            if conf >= 0.45 and bw > 20 and bh > 20:
                cx = bx + bw // 2
                cy = by + bh // 2
                side = int(max(bw, bh) * (1.0 + margin * 2.0))

                x1 = max(0, cx - side // 2)
                y1 = max(0, cy - side // 2)
                x2 = min(w_img, cx + side // 2)
                y2 = min(h_img, cy + side // 2)

                crop = frame_bgr[y1:y2, x1:x2]
                if crop.size > 0:
                    return crop, True, [int(bx), int(by), int(bw), int(bh)]
    except Exception:
        pass

    # Center crop fallback
    side = int(min(h_img, w_img) * 0.85)
    cx, cy = w_img // 2, h_img // 2
    crop = frame_bgr[max(0, cy - side // 2):min(h_img, cy + side // 2),
                     max(0, cx - side // 2):min(w_img, cx + side // 2)]
    return crop, False, None

def extract_frames(video_path: str, sample_count: int = 24) -> tuple[list[dict], dict]:
    """Extracts 24 uniform keyframes across video duration for low latency and high accuracy."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoProcessingError(f"Unable to open video: {video_path}")

    try:
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise VideoProcessingError("Video contains zero frames.")

        duration_seconds = total_frames / video_fps
        num_samples = min(total_frames, max(16, sample_count))
        indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
        indices = sorted(list(set(indices)))

        processed_frames = []
        face_count = 0

        for f_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f_idx))
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            crop_bgr, has_face, bbox = crop_face(frame)
            if has_face:
                face_count += 1

            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(crop_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

            processed_frames.append({
                "image": Image.fromarray(resized),
                "raw_crop_bgr": crop_bgr,
                "has_face": has_face,
                "bbox": bbox,
                "frame_number": len(processed_frames),
                "timestamp": round(f_idx / video_fps, 2)
            })

        if not processed_frames:
            raise VideoProcessingError("No frames extracted.")

        metadata = {
            "fps": round(video_fps, 2),
            "total_frames": total_frames,
            "duration_seconds": round(duration_seconds, 2),
            "extracted_count": len(processed_frames),
            "faces_detected": face_count,
            "face_detection_ratio": round(face_count / len(processed_frames), 2),
            "media_type": "video"
        }

        return processed_frames, metadata

    finally:
        cap.release()

def process_image(image_path: str) -> tuple[list[dict], dict]:
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None or img_bgr.size == 0:
        raise VideoProcessingError(f"Unable to decode image: {image_path}")

    h, w, _ = img_bgr.shape
    crop_bgr, has_face, bbox = crop_face(img_bgr)

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(crop_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

    processed_frames = [{
        "image": Image.fromarray(resized),
        "raw_crop_bgr": crop_bgr,
        "has_face": has_face,
        "bbox": bbox,
        "frame_number": 0,
        "timestamp": 0.0
    }]

    metadata = {
        "width": w,
        "height": h,
        "total_frames": 1,
        "extracted_count": 1,
        "faces_detected": 1 if has_face else 0,
        "face_detection_ratio": 1.0 if has_face else 0.0,
        "media_type": "image"
    }

    return processed_frames, metadata