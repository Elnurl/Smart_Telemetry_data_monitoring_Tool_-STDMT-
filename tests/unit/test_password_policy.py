from app.security.password_policy import PasswordPolicy


def test_password_policy_rejects_short_password():
    policy = PasswordPolicy(min_length=12)
    ok, message = policy.validate("Aa1!")
    assert ok is False
    assert "at least" in message


def test_password_policy_accepts_strong_password():
    policy = PasswordPolicy(min_length=12)
    ok, message = policy.validate("StrongPass!123")
    assert ok is True
    assert "satisfied" in message.lower()

