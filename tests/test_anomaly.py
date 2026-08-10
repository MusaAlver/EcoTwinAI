
import numpy as np
import pandas as pd

from src.anomaly import (
    EcoTwinAnomalyDetector
)

from src.uncertainty import (
    AdaptiveConformalThreshold
)


def test_anomaly_normal():

    detector = (
        EcoTwinAnomalyDetector()
    )

    result = detector.evaluate(
        forecast_kw=50,
        observed_kw=55,
        threshold_kw=8
    )

    assert result[
        "is_anomaly"
    ] is False

    assert abs(
        result["anomaly_score"]
        -
        0.625
    ) < 1e-12


def test_anomaly_detected():

    detector = (
        EcoTwinAnomalyDetector()
    )

    result = detector.evaluate(
        forecast_kw=50,
        observed_kw=62,
        threshold_kw=8
    )

    assert result[
        "is_anomaly"
    ] is True

    assert abs(
        result["anomaly_score"]
        -
        1.5
    ) < 1e-12


def test_direction():

    detector = (
        EcoTwinAnomalyDetector()
    )

    high = detector.evaluate(
        50,
        60,
        8
    )

    low = detector.evaluate(
        50,
        40,
        8
    )

    assert (
        high["direction"]
        ==
        "HIGH_CONSUMPTION"
    )

    assert (
        low["direction"]
        ==
        "LOW_CONSUMPTION"
    )


def test_conformal_initialization():

    engine = (
        AdaptiveConformalThreshold(
            coverage=0.96,
            window_days=30,
            horizon_minutes=30
        )
    )

    dates = pd.date_range(
        "2020-01-01",
        periods=200,
        freq="15min"
    )

    errors = np.linspace(
        1,
        10,
        200
    )

    start = (
        dates[-1]
        + pd.Timedelta(
            minutes=30
        )
    )

    threshold = engine.initialize(
        dates,
        errors,
        start
    )

    assert threshold > 0

    assert np.isfinite(
        threshold
    )


def test_delayed_history_update():

    engine = (
        AdaptiveConformalThreshold(
            coverage=0.96,
            window_days=30,
            horizon_minutes=30
        )
    )

    dates = pd.date_range(
        "2020-01-01",
        periods=100,
        freq="15min"
    )

    errors = np.ones(
        100
    ) * 5

    start = (
        dates[-1]
        + pd.Timedelta(
            minutes=30
        )
    )

    engine.initialize(
        dates,
        errors,
        start
    )

    initial_history_size = len(
        engine.history
    )

    forecast_time = pd.Timestamp(
        "2020-01-03 12:00"
    )

    engine.register_error(
        forecast_time=
            forecast_time,

        absolute_error=
            100,

        threshold_used=
            5
    )

    # Forecast sonucu henüz gerçekleşmedi.
    engine.current_threshold(
        forecast_time
    )

    assert (
        len(engine.history)
        ==
        initial_history_size
    )

    # 30 dakika sonra sonuç biliniyor.
    engine.current_threshold(
        forecast_time
        +
        pd.Timedelta(
            minutes=30
        )
    )

    assert (
        len(engine.history)
        ==
        initial_history_size + 1
    )

    # 100 kW hata history'ye doğrudan
    # girmemeli; 5 kW'a clip edilmeli.
    assert (
        engine.history[-1][1]
        ==
        5
    )


def test_invalid_threshold_rejected():

    detector = (
        EcoTwinAnomalyDetector()
    )

    try:

        detector.evaluate(
            50,
            60,
            0
        )

        assert False

    except ValueError:
        pass
