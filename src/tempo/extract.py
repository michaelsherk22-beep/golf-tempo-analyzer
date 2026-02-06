from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple

import imageio.v3 as iio


@dataclass
class Frame:
    t: float
    image_rgb: object  # numpy array; keep loose type to avoid hard dependency


def get_video_meta(video_path: str) -> Tuple[float, int]:
    # imageio doesn't always expose fps reliably; fall back if missing.
    meta = iio.immeta(video_path, plugin="FFMPEG")
    fps = float(meta.get("fps") or 30.0)
    nframes = int(meta.get("nframes") or 0)
    return fps, nframes


def iter_frames(video_path: str, max_frames: int = 240) -> Iterator[Frame]:
    meta = iio.immeta(video_path, plugin="FFMPEG")
    fps = float(meta.get("fps") or 30.0)

    # Read frames lazily; imageio returns RGB frames
    for idx, frame in enumerate(iio.imiter(video_path, plugin="FFMPEG")):
        if idx >= max_frames:
            break
        yield Frame(t=idx / fps, image_rgb=frame)

