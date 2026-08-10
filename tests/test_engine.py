
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.engine import (
    EcoTwinEngine
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

FIXTURE_PATH = (
    PROJECT_ROOT
    / "tests"
    / "fixtures"
    / "ecotwin_30m_smoke_test.npz"
)


@pytest.fixture(scope="module")
def engine():

    return EcoTwinEngine(
        PROJECT_ROOT
    )


def initialize_engine(
    engine
):

    production_start = pd.Timestamp(
        "2020-08-16 05:30"
    )

    # ----------------------------------------
    # 30-day uncertainty calibration
    # ----------------------------------------

    calibration_dates = pd.date_range(
        end=
            production_start
            -
            pd.Timedelta(
                minutes=30
            ),

        periods=
            30 * 96,

        freq=
            "15min",
    )

    calibration_errors = np.full(
        len(
            calibration_dates
        ),
        5.0,
        dtype=float,
    )

    # ----------------------------------------
    # 28-day subsystem history
    # ----------------------------------------

    component_dates = pd.date_range(
        end=
            production_start
            -
            pd.Timedelta(
                minutes=15
            ),

        periods=
            28 * 96,

        freq=
            "15min",
    )

    component_values = np.tile(
        np.array(
            [
                10.0,  # HVAC NORTH
                12.0,  # HVAC SOUTH
                5.0,   # MELS NORTH
                6.0,   # MELS SOUTH
                3.0,   # LIGHTING
            ]
        ),

        (
            len(
                component_dates
            ),
            1,
        ),
    )

    result = engine.initialize(
        calibration_forecast_dates=
            calibration_dates,

        calibration_absolute_errors=
            calibration_errors,

        component_dates=
            component_dates,

        component_values=
            component_values,

        production_start=
            production_start,
    )

    return (
        production_start,
        result,
    )


def load_raw_fixture(
    engine
):

    fixture = np.load(
        FIXTURE_PATH
    )

    scaled_window = (
        fixture[
            "model_input"
        ][0]
    )

    raw_window = (
        engine
        .forecaster
        .x_scaler
        .inverse_transform(
            scaled_window
        )
    )

    return (
        fixture,
        raw_window,
    )


def test_engine_initialization(
    engine
):

    _, result = (
        initialize_engine(
            engine
        )
    )

    assert result[
        "initialized"
    ] is True

    assert result[
        "feature_count"
    ] == 23

    assert result[
        "component_count"
    ] == 5

    assert abs(
        result[
            "initial_threshold_kw"
        ]
        -
        5.0
    ) < 1e-12


def test_engine_forecast_reproducibility(
    engine
):

    production_start, _ = (
        initialize_engine(
            engine
        )
    )

    fixture, raw_window = (
        load_raw_fixture(
            engine
        )
    )

    result = (
        engine.create_forecast(
            forecast_time=
                production_start,

            raw_window=
                raw_window,

            persistence_kw=
                fixture[
                    "persistence"
                ].item(),
        )
    )

    expected = fixture[
        "expected_forecast"
    ].item()

    assert abs(
        result[
            "forecast_kw"
        ]
        -
        expected
    ) < 1e-5

    assert abs(
        result[
            "adaptive_threshold_kw"
        ]
        -
        5.0
    ) < 1e-12


def test_full_alarm_root_cause_flow(
    engine
):

    production_start, _ = (
        initialize_engine(
            engine
        )
    )

    fixture, raw_window = (
        load_raw_fixture(
            engine
        )
    )

    forecast = (
        engine.create_forecast(
            forecast_time=
                production_start,

            raw_window=
                raw_window,

            persistence_kw=
                fixture[
                    "persistence"
                ].item(),
        )
    )

    observed_state = (
        raw_window[-1]
        .copy()
    )

    # ----------------------------------------
    # Create a controlled HVAC-N anomaly
    # ----------------------------------------

    observed_state[
        engine.feature_index[
            "total_power"
        ]
    ] = (
        forecast[
            "forecast_kw"
        ]
        +
        20.0
    )

    observed_state[
        engine.feature_index[
            "hvac_N"
        ]
    ] = 25.0

    observed_state[
        engine.feature_index[
            "hvac_S"
        ]
    ] = 12.0

    observed_state[
        engine.feature_index[
            "mels_N"
        ]
    ] = 5.0

    observed_state[
        engine.feature_index[
            "mels_S"
        ]
    ] = 6.0

    observed_state[
        engine.feature_index[
            "lig_S"
        ]
    ] = 3.0

    result = (
        engine.process_outcome(
            forecast_id=
                forecast[
                    "forecast_id"
                ],

            observed_state=
                observed_state,
        )
    )

    assert result[
        "anomaly"
    ][
        "is_anomaly"
    ] is True

    assert result[
        "root_cause"
    ][
        "cause"
    ] == "hvac_N"

    assert result[
        "recommendation"
    ][
        "severity"
    ] == "CRITICAL"

    assert (
        len(
            engine.alarm_rows
        )
        ==
        1
    )


def test_engine_builds_incident(
    engine
):

    production_start, _ = (
        initialize_engine(
            engine
        )
    )

    fixture, raw_window = (
        load_raw_fixture(
            engine
        )
    )

    forecast = (
        engine.create_forecast(
            forecast_time=
                production_start,

            raw_window=
                raw_window,

            persistence_kw=
                fixture[
                    "persistence"
                ].item(),
        )
    )

    observed_state = (
        raw_window[-1]
        .copy()
    )

    observed_state[
        engine.feature_index[
            "total_power"
        ]
    ] = (
        forecast[
            "forecast_kw"
        ]
        +
        20.0
    )

    observed_state[
        engine.feature_index[
            "hvac_N"
        ]
    ] = 25.0

    observed_state[
        engine.feature_index[
            "hvac_S"
        ]
    ] = 12.0

    observed_state[
        engine.feature_index[
            "mels_N"
        ]
    ] = 5.0

    observed_state[
        engine.feature_index[
            "mels_S"
        ]
    ] = 6.0

    observed_state[
        engine.feature_index[
            "lig_S"
        ]
    ] = 3.0

    engine.process_outcome(
        forecast_id=
            forecast[
                "forecast_id"
            ],

        observed_state=
            observed_state,
    )

    alarm_source, incidents = (
        engine.build_incidents()
    )

    assert len(
        alarm_source
    ) == 1

    assert len(
        incidents
    ) == 1

    assert (
        incidents.iloc[0][
            "dominant_root_cause"
        ]
        ==
        "HVAC NORTH"
    )

    assert (
        incidents.iloc[0][
            "peak_severity"
        ]
        ==
        "CRITICAL"
    )
