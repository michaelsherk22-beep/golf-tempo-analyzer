from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

try:
    import mediapipe as mp
    _HAS_MEDIAPIPE = True
except ImportError:
    _HAS_MEDIAPIPE = False

@dataclass
class PoseLandmarks:
    landmarks: list[tuple[float, float, float, float]] = field(default_factory=list)
    def get(self, idx: int) -> Optional[tuple[float, float, float, float]]:
        return self.landmarks[idx] if 0 <= idx < len(self.landmarks) else None

class PoseEstimator:
    LEFT_WRIST = 15
    RIGHT_WRIST = 16

    def __init__(self, model_complexity: int = 1, min_detection_confidence: float = 0.5):
        if not _HAS_MEDIAPIPE:
            raise ImportError("Run: pip install mediapipe opencv-python-headless")
        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(static_image_mode=True,
                                         model_complexity=model_complexity,
                                         min_detection_confidence=min_detection_confidence)

    def infer(self, image_bgr: np.ndarray) -> PoseLandmarks:
        import cv2
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)
        if result.pose_landmarks is None:
            return PoseLandmarks()
        return PoseLandmarks([(lm.x, lm.y, lm.z, lm.visibility)
                               for lm in result.pose_landmarks.landmark])

    def close(self) -> None:
        self._pose.close()

def wrist_y_series(poses: list[PoseLandmarks], handedness: str = "right") -> np.ndarray:
    idx = PoseEstimator.RIGHT_WRIST if handedness == "right" else PoseEstimator.LEFT_WRIST
    ys = [p.get(idx)[1] if p.get(idx) is not None else float("nan") for p in poses]
    return np.array(ys, dtype=float)
