from app.models.health_scoring import compute_health_score


def test_health_score_is_bounded():
    assert compute_health_score(0.0, 1.0) == 1.0
    assert compute_health_score(2.0, 1.0) == 0.0


def test_health_score_mid_range():
    score = compute_health_score(0.5, 1.0)
    assert 0.49 <= score <= 0.51

