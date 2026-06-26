import pandas as pd

from app.monitoring.obs_limits import evaluate_obs_condition, evaluate_obs_limits


def test_evaluate_obs_condition_supports_operators():
    assert evaluate_obs_condition(85.0, "value > 80")
    assert not evaluate_obs_condition(75.0, "value > 80")
    assert evaluate_obs_condition(10.0, "value <= 10")
    assert evaluate_obs_condition(10.0, "value == 10")


def test_evaluate_obs_condition_rejects_unsafe_expression():
    assert not evaluate_obs_condition(1.0, "__import__('os').system('echo')")
    assert not evaluate_obs_condition(1.0, "value > 80; print('x')")


def test_evaluate_obs_limits_detects_violation():
    df = pd.DataFrame({"cpu_temp": [70.0, 72.0, 91.0]})
    rules = [
        {
            "name": "High CPU Temperature",
            "parameter": "cpu_temp",
            "condition": "value > 80",
            "severity": 3,
        }
    ]
    result = evaluate_obs_limits(df, rules=rules)
    assert result["ok"] is False
    assert len(result["violations"]) == 1
    assert result["violations"][0]["parameter"] == "cpu_temp"
    assert result["violations"][0]["value"] == 91.0


def test_evaluate_obs_limits_empty_data_is_ok():
    result = evaluate_obs_limits(pd.DataFrame(), rules=[])
    assert result == {"ok": True, "violations": []}
