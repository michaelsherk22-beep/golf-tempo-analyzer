from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class TempoMetrics:
    backswing_s: float
    downswing_s: float
    ratio: float


def compute_tempo(address_t: float, top_t: float, impact_t: float) -> TempoMetrics:
    backswing = max(0.0, top_t - address_t)
    downswing = max(1e-9, impact_t - top_t)
    return TempoMetrics(
        backswing_s=backswing,
        downswing_s=downswing,
        ratio=backswing / downswing,
    )
