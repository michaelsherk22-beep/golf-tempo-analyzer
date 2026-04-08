from __future__ import annotations
from dataclasses import dataclass

@dataclass
class TempoMetrics:
    backswing_s: float
    downswing_s: float
    total_s: float
    ratio: float

    @property
    def rating(self) -> str:
        if self.downswing_s <= 0:
            return "Invalid"
        diff = abs(self.ratio - 3.0)
        if diff < 0.3:
            return "🟢 Excellent (near 3:1)"
        elif diff < 0.7:
            return "🟡 Good"
        elif diff < 1.2:
            return "🟠 Fair – work on rhythm"
        else:
            return "🔴 Needs improvement"

def compute_tempo(address_t: float, top_t: float, impact_t: float) -> TempoMetrics:
    backswing = top_t - address_t
    downswing = impact_t - top_t
    total = impact_t - address_t
    ratio = backswing / downswing if downswing > 0 else 0.0
    return TempoMetrics(backswing_s=backswing, downswing_s=downswing,
                        total_s=total, ratio=ratio)
