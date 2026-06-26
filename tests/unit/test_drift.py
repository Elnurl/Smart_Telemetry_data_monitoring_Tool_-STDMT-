import numpy as np
import pandas as pd

from app.models.drift import (
    TabConceptDriftChecker,
    compute_adaptive_threshold,
    compute_psi,
)


def test_compute_psi_detects_shift():
    ref = np.random.default_rng(0).normal(0.0, 1.0, 500)
    cur = np.random.default_rng(1).normal(2.0, 1.0, 500)
    stable = np.random.default_rng(2).normal(0.0, 1.0, 500)

    assert compute_psi(ref, cur) > compute_psi(ref, stable)


def test_tab_concept_drift_checker_initializes_reference():
    checker = TabConceptDriftChecker(reference_window_size=50)
    df = pd.DataFrame({"temp": np.linspace(0, 1, 120)})

    first = checker.check(df)
    assert first["status"] == "reference_initialized"
    assert not first["is_drift"]

    second = checker.check(df)
    assert "status" not in second or second.get("status") != "reference_initialized"
    assert checker.reference_window is not None


def test_tab_concept_drift_stable_reference_without_rebaseline():
    checker = TabConceptDriftChecker(
        reference_window_size=40,
        feature_ratio_threshold=0.5,
        ks_pvalue_threshold=0.05,
        psi_threshold=0.05,
    )
    ref_df = pd.DataFrame({"temp": np.random.default_rng(0).normal(0, 1, 200)})
    checker.check(ref_df)

    shifted = pd.DataFrame({"temp": np.random.default_rng(1).normal(5, 1, 200)})
    drift = checker.check(shifted, auto_rebaseline_on_drift=False)
    assert drift["is_drift"] is True
    assert checker.reference_window is not None
    assert np.isclose(checker.reference_window["temp"].mean(), ref_df.tail(40)["temp"].mean(), atol=0.5)


def test_compute_adaptive_threshold_grows_with_scores():
    history = [0.1] * 25
    threshold, updated = compute_adaptive_threshold(history, 0.95)
    assert len(updated) == 26
    assert threshold >= 0.35
