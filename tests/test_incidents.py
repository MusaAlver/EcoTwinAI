
import numpy as np
import pandas as pd

from src.incidents import (
    EcoTwinIncidentEngine
)


def alarm_row(
    time,
    residual=10,
    severity="WARNING",
    anomaly_score=1.2,
    cause="HVAC NORTH",
    strength=80,
):

    return {
        "alarm_time":
            pd.Timestamp(time),

        "residual_kw":
            residual,

        "severity":
            severity,

        "anomaly_score":
            anomaly_score,

        "root_cause":
            cause,

        "attribution_strength":
            strength,

        "observed_kw":
            60.0,

        "forecast_kw":
            50.0,
    }


def test_close_alarms_same_incident():

    df = pd.DataFrame(
        [
            alarm_row(
                "2020-01-01 10:00"
            ),
            alarm_row(
                "2020-01-01 10:15"
            ),
            alarm_row(
                "2020-01-01 10:30"
            ),
        ]
    )

    engine = (
        EcoTwinIncidentEngine()
    )

    source, incidents = (
        engine.build(df)
    )

    assert len(incidents) == 1

    assert (
        source[
            "incident_id"
        ].nunique()
        ==
        1
    )


def test_large_gap_new_incident():

    df = pd.DataFrame(
        [
            alarm_row(
                "2020-01-01 10:00"
            ),
            alarm_row(
                "2020-01-01 10:45"
            ),
        ]
    )

    engine = (
        EcoTwinIncidentEngine()
    )

    _, incidents = (
        engine.build(df)
    )

    assert len(incidents) == 2


def test_direction_change_new_incident():

    df = pd.DataFrame(
        [
            alarm_row(
                "2020-01-01 10:00",
                residual=10,
            ),

            alarm_row(
                "2020-01-01 10:15",
                residual=-10,
            ),
        ]
    )

    engine = (
        EcoTwinIncidentEngine()
    )

    _, incidents = (
        engine.build(df)
    )

    assert len(incidents) == 2


def test_duration_includes_observation():

    df = pd.DataFrame(
        [
            alarm_row(
                "2020-01-01 10:00"
            ),
            alarm_row(
                "2020-01-01 10:15"
            ),
            alarm_row(
                "2020-01-01 10:30"
            ),
        ]
    )

    engine = (
        EcoTwinIncidentEngine(
            observation_minutes=15
        )
    )

    _, incidents = (
        engine.build(df)
    )

    assert (
        incidents.iloc[0][
            "duration_minutes"
        ]
        ==
        45
    )


def test_weighted_dominant_cause():

    df = pd.DataFrame(
        [
            alarm_row(
                "2020-01-01 10:00",
                anomaly_score=1.1,
                cause="HVAC NORTH",
                strength=30,
            ),

            alarm_row(
                "2020-01-01 10:15",
                anomaly_score=2.0,
                cause="LIGHTING",
                strength=90,
            ),
        ]
    )

    engine = (
        EcoTwinIncidentEngine()
    )

    _, incidents = (
        engine.build(df)
    )

    assert (
        incidents.iloc[0][
            "dominant_root_cause"
        ]
        ==
        "LIGHTING"
    )


def test_peak_severity_priority():

    df = pd.DataFrame(
        [
            alarm_row(
                "2020-01-01 10:00",
                severity="WARNING",
                anomaly_score=4.0,
            ),

            alarm_row(
                "2020-01-01 10:15",
                severity="CRITICAL",
                anomaly_score=2.6,
            ),
        ]
    )

    engine = (
        EcoTwinIncidentEngine()
    )

    _, incidents = (
        engine.build(df)
    )

    assert (
        incidents.iloc[0][
            "peak_severity"
        ]
        ==
        "CRITICAL"
    )
