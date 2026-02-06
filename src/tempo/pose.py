from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose


@dataclass(frozen=True)
class PosePoint:
    x: float  # normalized [0..1]
    y: float  # normalized [0..1]
    vis: float


@dataclass(frozen=True)
class PoseResult:
    left_wrist: Optional[PosePoint]
    right_wrist: Optional[PosePoint]


def _get_landmark(landmarks, lm) -> Optional[PosePoint]:
    if landmarks is None:
        return None
    p = landmarks.landmark[lm]
    return PosePoint(x=float(p.x), y=float(p.y), vis=float(p.visibility))


class PoseEstimator:
    def __init__(self) -> None:
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def infer(self, image_bgr: "cv2.Mat") -> PoseResult:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        res = self.pose.process(image_rgb)
        lm = res.pose_landmarks
        left = _get_landmark(lm, mp_pose.PoseLandmark.LEFT_WRIST) if lm else None
        right = _get_landmark(lm, mp_pose.PoseLandmark.RIGHT_WRIST) if lm else None
        return PoseResult(left_wrist=left, right_wrist=right)

    def close(self) -> None:
        self.pose.close()


def choose_wrist(p: PoseResult) -> Optional[PosePoint]:
    lw, rw = p.left_wrist, p.right_wrist
    if lw and rw:
        return lw if lw.vis >= rw.vis else rw
    return lw or rw


def wrist_y_series(results: list[PoseResult]) -> np.ndarray:
    ys = []
    for r in results:
        w = choose_wrist(r)
        ys.append(w.y if w else np.nan)
    return np.asarray(ys, dtype=float)
