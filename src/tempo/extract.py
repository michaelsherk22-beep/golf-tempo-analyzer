from __future__ import annotations
import math
from dataclasses import dataclass
import imageio.v3 as iio
import numpy as np

@dataclass
class Frame:
    t: float
    image_bgr: np.ndarray

def get_video_meta(video_path: str) -> tuple[float, int]:
    meta = iio.immeta(video_path, plugin="FFMPEG") or {}
    fps_raw = meta.get("fps")
    try:
        fps = float(fps_raw) if fps_raw is not None else 30.0
    except Exception:
        fps = 30.0
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0

    n_raw = meta.get("nframes") or meta.get("n_frames") or meta.get("duration_frames")
    nframes = 0
    try:
        if n_raw is not None:
            n = float(n_raw)
            if math.isfinite(n) and n > 0:
                nframes = int(n)
    except Exception:
        nframes = 0

    if nframes <= 0:
        try:
            nframes = sum(1 for _ in iio.imiter(video_path, plugin="FFMPEG"))
        except Exception:
            nframes = 0

    return fps, nframes

def iter_frames(video_path: str, max_frames: int = 300) -> list[Frame]:
    fps, _ = get_video_meta(video_path)
    frames: list[Frame] = []
    for i, img in enumerate(iio.imiter(video_path, plugin="FFMPEG")):
        if i >= max_frames:
            break
        bgr = img[:, :, ::-1].copy() if img.ndim == 3 and img.shape[2] >= 3 else img
        frames.append(Frame(t=i / fps, image_bgr=bgr))
    return frames
