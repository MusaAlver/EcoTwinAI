
import numpy as np
import pandas as pd
import pytest

from src.model_health import (
    HealthThresholds,
    ModelHealthMonitor,
)


def make_reference(rows: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(42)

    return pd.DataFrame(
        {
            "total_power": rng.normal(
                50.0,
                5.0,
                rows,
            ),
            "outdoor_temp_avg": rng.normal(
                20.0,
                3.0,
                rows,
            ),
        }
    )


def test_identical_distribution_is_healthy():
    reference = make_reference()

    monitor = ModelHealthMonitor()
    monitor.fit_reference(reference)

    result = monitor.evaluate(
        reference.copy()
    )

    assert result["status"] == "HEALTHY"
    assert (
        result["summary"]["retrain_recommended"]
        is False
    )

    for feature in result["features"].values():
        assert feature["psi"] == pytest.approx(0.0)
        assert feature["severity"] == "HEALTHY"


def test_large_distribution_shift_is_critical():
    reference = make_reference()

    current = reference.copy()
    current["total_power"] += 40.0

    monitor = ModelHealthMonitor()
    monitor.fit_reference(reference)

    result = monitor.evaluate(current)

    assert (
        result["features"]["total_power"]["severity"]
        == "CRITICAL"
    )

    assert result["status"] == "CRITICAL"
    assert (
        result["summary"]["retrain_recommended"]
        is True
    )


def test_large_missingness_increase_is_critical():
    reference = make_reference()

    current = reference.copy()

    current.loc[
        current.index[:60],
        "outdoor_temp_avg",
    ] = np.nan

    monitor = ModelHealthMonitor()
    monitor.fit_reference(reference)

    result = monitor.evaluate(current)

    feature = result["features"][
        "outdoor_temp_avg"
    ]

    assert feature["current_missing_ratio"] == pytest.approx(
        0.20
    )

    assert feature["severity"] == "CRITICAL"
    assert result["status"] == "CRITICAL"


def test_performance_degradation_warning():
    reference = make_reference()

    actual = np.zeros(300)
    reference_prediction = np.ones(300)

    monitor = ModelHealthMonitor()

    monitor.fit_reference(
        reference,
        actual=actual,
        prediction=reference_prediction,
    )

    current_prediction = np.full(
        300,
        1.30,
    )

    result = monitor.evaluate(
        reference.copy(),
        actual=actual,
        prediction=current_prediction,
    )

    performance = result["performance"]

    assert performance["available"] is True
    assert performance["reference_mae"] == pytest.approx(1.0)
    assert performance["current_mae"] == pytest.approx(1.30)

    assert performance["relative_degradation"] == pytest.approx(
        0.30
    )

    assert performance["severity"] == "WARNING"
    assert result["status"] == "WARNING"


def test_severe_performance_degradation_triggers_retraining():
    reference = make_reference()

    actual = np.zeros(300)

    monitor = ModelHealthMonitor()

    monitor.fit_reference(
        reference,
        actual=actual,
        prediction=np.ones(300),
    )

    result = monitor.evaluate(
        reference.copy(),
        actual=actual,
        prediction=np.full(
            300,
            2.0,
        ),
    )

    assert (
        result["performance"]["severity"]
        == "CRITICAL"
    )

    assert result["status"] == "CRITICAL"

    assert (
        result["summary"]["retrain_recommended"]
        is True
    )


def test_feature_drift_works_without_outcomes():
    reference = make_reference()

    current = reference.copy()
    current["total_power"] += 30.0

    monitor = ModelHealthMonitor()
    monitor.fit_reference(reference)

    result = monitor.evaluate(current)

    assert result["performance"]["available"] is False
    assert result["status"] == "CRITICAL"


def test_missing_monitored_feature_is_rejected():
    reference = make_reference()

    monitor = ModelHealthMonitor()
    monitor.fit_reference(reference)

    current = reference.drop(
        columns=["outdoor_temp_avg"]
    )

    with pytest.raises(
        ValueError,
        match="Missing monitored features",
    ):
        monitor.evaluate(current)


def test_constant_reference_feature_change_is_critical():
    reference = pd.DataFrame(
        {
            "total_power": np.full(
                200,
                50.0,
            ),
        }
    )

    current = pd.DataFrame(
        {
            "total_power": np.full(
                200,
                51.0,
            ),
        }
    )

    monitor = ModelHealthMonitor()
    monitor.fit_reference(reference)

    result = monitor.evaluate(current)

    assert (
        result["features"]["total_power"]["severity"]
        == "CRITICAL"
    )

    assert result["status"] == "CRITICAL"


def test_reference_profile_round_trip(tmp_path):
    reference = make_reference()

    monitor = ModelHealthMonitor()

    original = monitor.fit_reference(
        reference,
        model_version="v-test",
    )

    path = tmp_path / "health_reference.json"

    monitor.save_reference(path)

    restored = ModelHealthMonitor()

    loaded = restored.load_reference(
        path
    )

    assert (
        loaded["feature_fingerprint"]
        == original["feature_fingerprint"]
    )

    assert loaded["model_version"] == "v-test"

    result = restored.evaluate(
        reference.copy()
    )

    assert result["status"] == "HEALTHY"


def test_too_small_monitoring_window_is_rejected():
    reference = make_reference()

    monitor = ModelHealthMonitor(
        thresholds=HealthThresholds(
            minimum_current_rows=30
        )
    )

    monitor.fit_reference(reference)

    with pytest.raises(
        ValueError,
        match="Current monitoring window is too small",
    ):
        monitor.evaluate(
            reference.iloc[:20]
        )
