from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.signal import argrelextrema

@dataclass
class SwingEvents:
    address_idx: int
    top_idx: int
    impact_idx: int

def detect_events_from_wrist_y(y: np.ndarray, order: int = 5) -> SwingEvents:
    n = len(y)
    if n < 3:
        raise ValueError("Not enough frames to detect swing events.")

    minima = argrelextrema(y, np.less, order=order)[0]
    maxima = argrelextrema(y, np.greater, order=order)[0]

    if len(minima) == 0 or len(maxima) == 0:
        return SwingEvents(0, n // 3, (2 * n) // 3)

    top_idx = int(minima[0])
    post_top_maxima = maxima[maxima > top_idx]
    impact_idx = int(post_top_maxima[0]) if len(post_top_maxima) > 0 else min(top_idx + order, n - 1)
    address_idx = max(0, top_idx - max(order, top_idx // 3))
    return SwingEvents(address_idx, top_idx, impact_idx)
