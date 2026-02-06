from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import mediapipe as mp


mp_pose = mp.solutions.pose


@dataclass
class PoseResult:
    """Minimal pose result we need for tempo."""
    landmarks: Optional[Any]  # mp.framework.formats.landmark_pb2.NormalizedLandmarkList


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
        """
        Accepts BGR (from our pipeline), converts to RGB for MediaPipe.
        Avoids cv2 entirely.
        """
        # BGR -> RGB using numpy
        image_rgb = image_bgr[..., ::-1]
        res = self.pose.process(image_rgb)
        return PoseResult(landmarks=res.pose_landmarks)

    def close(self) -> None:
        try:
            self.pose.close()
        except Exception:
            pass


def wrist_y_series(poses: list[PoseResult], handedness: str = "right") -> np.ndarray:
    """
    Return wrist Y series (normalized image coordinates, 0 top -> 1 bottom).
    If landmark missing, returns NaN for that frame.
    """
    # MediaPipe landmark indices:
    # 15 = left_wrist, 16 = right_wrist
    idx = 16 if handedness.lower().startswith("r") else 15

    ys = []
    for p in poses:
        if p.landmarks is None:
            ys.append(np.nan)
        else:
            ys.append(p.landmarks.landmark[idx].y)
    return np.array(ys, dtype=float)
