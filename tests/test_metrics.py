from tempo.metrics import compute_tempo

def test_compute_tempo_ratio():
    m = compute_tempo(0.0, 1.5, 2.0)
    assert abs(m.backswing_s - 1.5) < 1e-9
    assert abs(m.downswing_s - 0.5) < 1e-9
    assert abs(m.ratio - 3.0) < 1e-9
