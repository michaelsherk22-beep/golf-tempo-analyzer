from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

import imageio.v3 as iio
import numpy as np


@dataclass(frozen=True)
class Frame:
    """A single video frame with timestamp (seconds) and BGR pixels."""
    t: float
    image_bgr: np.ndarray


def get_video_meta(video_path: str) -> tuple[float, int]:
    """
    Returns (fps, total_frames).

    Notes:
    - Some containers/codecs report fps as None/0/NaN.
    - Some report nframes as inf/0/None.
    - We guard against those and fall back to counting frames when needed.
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


def iter_frames(video_path: str, max_frames: int | None = None) -> Iterator[Frame]:
    """
    Yields frames as BGR numpy arrays plus timestamps in seconds.

    Uses imageio+ffmpeg (works on Streamlit Cloud without OpenCV).
    """
    fps, _ = get_video_meta(video_path)

    for idx, rgb in enumerate(iio.imiter(video_path, plugin="FFMPEG")):
        if max_frames is not None and idx >= max_frames:
            break

        # imageio returns RGB; convert to BGR to match OpenCV conventions
        if rgb.ndim == 3 and rgb.shape[-1] >= 3:
            bgr = rgb[..., :3][..., ::-1]
        else:
            # unexpected format, pass through as-is
            bgr = rgb

        t = float(idx) / float(fps)
        yield Frame(t=t, image_bgr=bgr)
