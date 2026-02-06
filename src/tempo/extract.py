from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import imageio.v3 as iio
import numpy as np


@dataclass
class Frame:
    image_bgr: np.ndarray
    t: float


def iter_frames(video_path: str, max_frames: Optional[int] = None) -> Iterator[Frame]:
    """
    Yield frames from a video using imageio (ffmpeg backend), avoiding OpenCV.
    Returns frames as BGR uint8 arrays plus timestamp seconds.
    """
    # Read metadata if available
    meta = {}
    try:
        meta = iio.immeta(video_path, plugin="ffmpeg") or {}
    except Exception:
        meta = {}

    fps = float(meta.get("fps") or 30.0)

    count = 0
    for idx, frame_rgb in enumerate(iio.imiter(video_path, plugin="ffmpeg")):
        # frame_rgb is RGB; convert to BGR for downstream code
        frame_rgb = np.asarray(frame_rgb)
        if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            continue
        frame_bgr = frame_rgb[..., ::-1].copy()
        t = idx / fps

        yield Frame(image_bgr=frame_bgr, t=t)

        count += 1
        if max_frames is not None and count >= max_frames:
            break
