from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, Optional

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

    Some containers/codecs report fps as None/0/NaN.
    Some report nframes as inf/0/None.
    We guard against those and fall back to counting frames when needed.
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

    # nframes (can be inf/None/0 depending on container)
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


def _to_bgr(frame: np.ndarray) -> np.ndarray:
    """
    imageio returns RGB frames for FFMPEG. Convert to BGR to match prior OpenCV-style code.
    Handles grayscale by expanding to 3 channels.
    """
    if frame.ndim == 2:
        rgb = np.stack([frame, frame, frame], axis=-1)
    else:
        rgb = frame

    if rgb.ndim == 3 and rgb.shape[-1] >= 3:
        bgr = rgb[..., :3][..., ::-1]
        return np.ascontiguousarray(bgr)
    return np.ascontiguousarray(rgb)


def iter_frames(video_path: str, max_frames: Optional[int] = None) -> Iterator[Frame]:
    """
    Yields Frame(t, image_bgr) for each decoded frame.

    t is computed from index / fps (meta fps, with safe fallback).
    """
    fps, _ = get_video_meta(video_path)
    for idx, rgb in enumerate(iio.imiter(video_path, plugin="FFMPEG")):
        if max_frames is not None and idx >= max_frames:
            break
        t = float(idx) / float(fps)
        yield Frame(t=t, image_bgr=_to_bgr(np.asarray(rgb)))
