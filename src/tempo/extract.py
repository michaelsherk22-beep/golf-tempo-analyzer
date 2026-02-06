from __future__ import annotations
import cv2
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class Frame:
    idx: int
    t: float  # seconds
    image_bgr: "cv2.Mat"


def iter_frames(video_path: str, max_frames: int | None = None) -> Iterator[Frame]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        yield Frame(idx=idx, t=t, image_bgr=frame)
        idx += 1
        if max_frames is not None and idx >= max_frames:
            break

    cap.release()


def video_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return float(fps)
