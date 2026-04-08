from tempo.metrics import compute_tempo

def test_three_to_one():
    m = compute_tempo(0.0, 0.9, 1.2)
    assert abs(m.ratio - 3.0) < 1e-6

def test_rating_excellent():
    assert "Excellent" in compute_tempo(0.0, 0.9, 1.2).rating

def test_rating_needs_improvement():
    assert "Needs improvement" in compute_tempo(0.0, 2.4, 2.7).rating

def test_zero_downswing():
    assert compute_tempo(0.0, 1.0, 1.0).ratio == 0.0
