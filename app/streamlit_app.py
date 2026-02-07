# ruff: noqa: E402
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
    """
    Returns (fps, total_frames). Some containers/codecs report nframes=inf or 0.
    We guard against that and fall back to counting frames when needed.
    """
    meta = iio.immeta(video_path, plugin="FFMPEG") or {}

    # fps
    fps_raw = meta.get("fps")
    try:
        fps = float(fps_raw) if fps_raw is not None else 30.0
    except Exception:
        fps = 30.0
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0

    # nframes (can be inf/None/0)
    n_raw = meta.get("nframes") or meta.get("n_frames") or meta.get("duration_frames")
    nframes = 0
    try:
        if n_raw is not None:
            n = float(n_raw)
            if math.isfinite(n) and n > 0:
                nframes = int(n)
    except Exception:
        nframes = 0

    # fallback: count frames (slower, but reliable)
    if nframes <= 0:
        try:
            nframes = sum(1 for _ in iio.imiter(video_path, plugin="FFMPEG"))
        except Exception:
            nframes = 0

    return fps, nframes

