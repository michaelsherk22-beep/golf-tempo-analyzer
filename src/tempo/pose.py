from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

# Some mediapipe Linux builds do not expose mp.solutions at top-level.
# This import path is stable across versions.
from mediapipe.python.solutions import pose as mp_pose


@dataclass
class PoseResult:
    landmarks: Optional[Any]  # NormalizedLandmarkList


class PoseEstimator:
    def __init__(self) -> None:
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def infer(self, image_bgr: np.ndarray) -> PoseResult:
        # MediaPipe expects RGB
        image_rgb = image_bgr[..., ::-1]
        res = self.pose.process(image_rgb)
        return PoseResult(landmarks=res.pose_landmarks)

    def close(self) -> None:
        try:
            self.pose.close()
        except Exception:
            pass


def wrist_y_series(poses: list[PoseResult], handedness: str = "right") -> np.ndarray:
    idx = 16 if handedness.lower().startswith("r") else 15  # right_wrist / left_wrist
    ys: list[float] = []
    for p in poses:
        if p.landmarks is None:
            ys.append(np.nan)
        else:
            ys.append(p.landmarks.landmark[idx].y)
    return np.array(ys, dtype=float)
