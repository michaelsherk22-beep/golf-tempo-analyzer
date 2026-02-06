from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SwingEvents:
    address_idx: int
    top_idx: int
    impact_idx: int


def _nan_smooth(x: np.ndarray, win: int = 7) -> np.ndarray:
    y = x.copy()
    isn = np.isnan(y)
    if np.all(isn):
        return y

    idx = np.where(~isn)[0]
    y[: idx[0]] = y[idx[0]]
    y[idx[-1] + 1 :] = y[idx[-1]]
    for i in range(1, len(idx)):
        a, b = idx[i - 1], idx[i]
        if b - a > 1:
            y[a:b] = np.interp(np.arange(a, b), [a, b], [y[a], y[b]])

    k = max(3, win | 1)
    pad = k // 2
    yp = np.pad(y, (pad, pad), mode="edge")
    kern = np.ones(k) / k
    return np.convolve(yp, kern, mode="valid")


def detect_events_from_wrist_y(wrist_y: np.ndarray) -> SwingEvents:
    n = len(wrist_y)
    if n < 30:
        raise ValueError("Video too short for event detection.")

    y = _nan_smooth(wrist_y, win=9)

    first = max(10, int(0.15 * n))
    v = np.abs(np.gradient(y[:first]))
    address_idx = int(np.argmin(v))

    start = address_idx + 5
    end = max(start + 10, int(0.70 * n))
    top_idx = int(np.nanargmin(y[start:end]) + start)

    addr_y = y[address_idx]
    tol = 0.03
    impact_idx = None
    for i in range(top_idx + 5, n):
        if abs(y[i] - addr_y) <= tol:
            impact_idx = i
            break

    if impact_idx is None:
        last_start = int(0.90 * n)
        v2 = np.abs(np.gradient(y[last_start:]))
        impact_idx = int(np.argmin(v2) + last_start)

    if not (address_idx < top_idx < impact_idx):
        raise ValueError("Could not detect a valid swing sequence (address < top < impact).")

    return SwingEvents(address_idx=address_idx, top_idx=top_idx, impact_idx=impact_idx)
