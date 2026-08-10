
import numpy as np

from src.recommendation import (
    EcoTwinRecommendationEngine
)


FEATURES = [
    "total_power",
    "indoor_temp_avg",
    "outdoor_temp_avg",
    "relative_humidity_set_1",
]


def create_engine():

    return EcoTwinRecommendationEngine(
        FEATURES
    )


def test_severity_levels():

    engine = create_engine()

    assert engine.get_severity(0.8)[0] == "NORMAL"
    assert engine.get_severity(1.2)[0] == "WARNING"
    assert engine.get_severity(2.0)[0] == "HIGH"
    assert engine.get_severity(3.0)[0] == "CRITICAL"


def test_hvac_high_consumption():

    engine = create_engine()

    state = np.array(
        [50, 24, 10, 40],
        dtype=float
    )

    result = engine.build(
        cause="hvac_N",
        residual_kw=12,
        anomaly_score=1.8,
        state=state
    )

    assert result["severity"] == "HIGH"
    assert result["direction"] == "HIGH_CONSUMPTION"

    assert (
        "HVAC"
        in result["title"]
    )


def test_hvac_low_consumption():

    engine = create_engine()

    result = engine.build(
        cause="hvac_S",
        residual_kw=-10,
        anomaly_score=1.6
    )

    assert (
        result["direction"]
        ==
        "LOW_CONSUMPTION"
    )

    assert (
        "below expected"
        in result["title"]
    )


def test_lighting_recommendation():

    engine = create_engine()

    result = engine.build(
        cause="lig_S",
        residual_kw=8,
        anomaly_score=1.3
    )

    assert (
        result["severity"]
        ==
        "WARNING"
    )

    assert (
        "lighting"
        in result[
            "title"
        ].lower()
    )


def test_unexplained_load():

    engine = create_engine()

    result = engine.build(
        cause="UNEXPLAINED_LOAD",
        residual_kw=20,
        anomaly_score=3.2
    )

    assert (
        result["severity"]
        ==
        "CRITICAL"
    )

    assert (
        "Unexplained"
        in result["title"]
    )


def test_normal_operation():

    engine = create_engine()

    result = engine.build(
        cause="hvac_N",
        residual_kw=2,
        anomaly_score=0.5
    )

    assert (
        result["severity"]
        ==
        "NORMAL"
    )

    assert (
        result["action"]
        ==
        "Continue monitoring."
    )
