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

    fps = meta.get("fps")
    try:
        fps = float(fps) if fps is not None else 30.0
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0
    except Exception:
        fps = 30.0

    nframes = meta.get("nframes")
    total = 0
    try:
        # nframes can be None, 0, or inf depending on the file
        nf = float(nframes) if nframes is not None else 0.0
        if math.isfinite(nf) and nf > 0:
            total = int(nf)
        else:
            total = 0
    except Exception:
        total = 0

    # Fallback: count frames (reliable but can take a few seconds)
    if total == 0:
        try:
            total = 0
            for _ in iio.imiter(video_path, plugin="FFMPEG"):
                total += 1
        except Exception:
            total = 0

    return fps, total
