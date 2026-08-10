
import numpy as np
import pandas as pd

from src.root_cause import (
    EcoTwinRootCauseEngine,
    COMPONENTS,
)


def create_engine():

    engine = EcoTwinRootCauseEngine(
        window_days=28
    )

    dates = pd.date_range(
        "2020-01-01",
        periods=28 * 96,
        freq="15min",
    )

    # Stable synthetic subsystem history
    components = np.tile(
        np.array(
            [
                10.0,  # HVAC N
                12.0,  # HVAC S
                5.0,   # MELS N
                6.0,   # MELS S
                3.0,   # Lighting
            ]
        ),
        (
            len(dates),
            1,
        ),
    )

    engine.initialize(
        dates=
            dates,

        component_values=
            components,

        production_start=
            dates[-1]
            +
            pd.Timedelta(
                minutes=15
            ),
    )

    return engine


def test_component_contract():

    assert len(
        COMPONENTS
    ) == 5


def test_hvac_north_positive_cause():

    engine = create_engine()

    expected = np.array(
        [10, 12, 5, 6, 3],
        dtype=float,
    )

    observed = np.array(
        [25, 12, 5, 6, 3],
        dtype=float,
    )

    scales = np.ones(
        5
    )

    result = engine.diagnose(
        observed_components=
            observed,

        expected_components=
            expected,

        component_scales=
            scales,

        total_error_kw=
            15,
    )

    assert (
        result["cause"]
        ==
        "hvac_N"
    )

    assert (
        result[
            "attribution_strength"
        ]
        > 90
    )


def test_hvac_south_negative_cause():

    engine = create_engine()

    expected = np.array(
        [10, 20, 5, 6, 3],
        dtype=float,
    )

    observed = np.array(
        [10, 5, 5, 6, 3],
        dtype=float,
    )

    scales = np.ones(
        5
    )

    result = engine.diagnose(
        observed,
        expected,
        scales,
        total_error_kw=-15,
    )

    assert (
        result["cause"]
        ==
        "hvac_S"
    )


def test_unexplained_load():

    engine = create_engine()

    expected = np.array(
        [10, 12, 5, 6, 3],
        dtype=float,
    )

    # Only tiny subsystem deviations,
    # despite large total building error.
    observed = np.array(
        [10.2, 12.1, 5.1, 6.1, 3.1],
        dtype=float,
    )

    scales = np.ones(
        5
    )

    result = engine.diagnose(
        observed,
        expected,
        scales,
        total_error_kw=20,
    )

    assert (
        result["cause"]
        ==
        "UNEXPLAINED_LOAD"
    )

    assert np.isnan(
        result[
            "attribution_strength"
        ]
    )


def test_expected_state_is_finite():

    engine = create_engine()

    expected, scales = (
        engine.expected_state(
            pd.Timestamp(
                "2020-01-29 00:00"
            )
        )
    )

    assert expected.shape == (5,)
    assert scales.shape == (5,)

    assert np.isfinite(
        expected
    ).all()

    assert np.isfinite(
        scales
    ).all()

    assert (
        scales >= 0.25
    ).all()


def test_evaluate_updates_history():

    engine = create_engine()

    now = pd.Timestamp(
        "2020-01-29 12:00"
    )

    key = engine.slot_key(
        now
    )

    before = len(
        engine.slot_history[
            key
        ]
    )

    observed = np.array(
        [10, 12, 5, 6, 3],
        dtype=float,
    )

    engine.evaluate(
        now=
            now,

        observed_components=
            observed,

        total_error_kw=
            0.5,

        update_history=True,
    )

    after = len(
        engine.slot_history[
            key
        ]
    )

    assert (
        after
        ==
        before + 1
    )
