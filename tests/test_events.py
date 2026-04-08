import numpy as np
from tempo.events import detect_events_from_wrist_y

def test_synthetic_swing():
    y = np.cos(np.linspace(0, 2 * 3.14159, 200))
    e = detect_events_from_wrist_y(y, order=10)
    assert e.address_idx < e.top_idx < e.impact_idx

def test_fallback_short_series():
    e = detect_events_from_wrist_y(np.array([0.5, 0.4, 0.6]), order=1)
    assert e.address_idx < e.top_idx < e.impact_idx
