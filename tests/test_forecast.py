
from pathlib import Path

import numpy as np

from src.forecast import EcoTwinForecaster


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


def get_forecaster():
    return EcoTwinForecaster(
        PROJECT_ROOT
    )


def test_model_contract():

    forecaster = get_forecaster()

    assert forecaster.history_timesteps == 16
    assert forecaster.feature_count == 23
    assert len(forecaster.features) == 23

    assert abs(
        forecaster.gate_lambda - 0.42
    ) < 1e-12

    assert (
        forecaster.model.input_shape[1:]
        ==
        (16, 23)
    )

    assert (
        forecaster.model.output_shape[-1]
        ==
        1
    )


def test_prediction_reproducibility():

    forecaster = get_forecaster()

    fixture = np.load(
        FIXTURE_PATH
    )

    # Fixture model_input ölçeklenmiş halde saklandı.
    scaled_window = fixture[
        "model_input"
    ][0]

    # Production fonksiyonu ham feature bekliyor.
    raw_window = (
        forecaster.x_scaler
        .inverse_transform(
            scaled_window
        )
    )

    persistence_kw = fixture[
        "persistence"
    ].item()

    expected_forecast = fixture[
        "expected_forecast"
    ].item()

    result = forecaster.predict_30m(
        raw_window,
        persistence_kw=persistence_kw
    )

    assert abs(
        result["forecast_kw"]
        -
        expected_forecast
    ) < 1e-5


def test_prediction_output_contract():

    forecaster = get_forecaster()

    fixture = np.load(
        FIXTURE_PATH
    )

    raw_window = (
        forecaster.x_scaler
        .inverse_transform(
            fixture["model_input"][0]
        )
    )

    result = forecaster.predict_30m(
        raw_window,
        persistence_kw=fixture[
            "persistence"
        ].item()
    )

    required_keys = {
        "forecast_horizon_minutes",
        "forecast_kw",
        "persistence_kw",
        "predicted_residual_kw",
        "gate_lambda",
        "applied_correction_kw",
        "method",
    }

    assert required_keys.issubset(
        result.keys()
    )

    assert (
        result[
            "forecast_horizon_minutes"
        ]
        ==
        30
    )

    assert (
        result["method"]
        ==
        "gated_residual_lstm"
    )

    assert np.isfinite(
        result["forecast_kw"]
    )


def test_wrong_input_shape_rejected():

    forecaster = get_forecaster()

    wrong_window = np.zeros(
        (10, 23),
        dtype=np.float32
    )

    try:

        forecaster.predict_30m(
            wrong_window
        )

        assert False, (
            "Invalid input shape "
            "should raise ValueError."
        )

    except ValueError:
        pass


def test_nan_input_rejected():

    forecaster = get_forecaster()

    bad_window = np.zeros(
        (16, 23),
        dtype=np.float32
    )

    bad_window[0, 0] = np.nan

    try:

        forecaster.predict_30m(
            bad_window
        )

        assert False, (
            "NaN input should "
            "raise ValueError."
        )

    except ValueError:
        pass
