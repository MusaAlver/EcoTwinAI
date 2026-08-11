
import numpy as np
import pandas as pd
import pytest

from src.preprocessing import BuildingDataPreprocessor


def test_canonicalize_aliases_and_order():
    df = pd.DataFrame(
        {
            "Date Time": [
                "2026-01-01 00:30:00",
                "2026-01-01 00:00:00",
                "2026-01-01 00:15:00",
            ],
            "Building Power": [
                "30.0",
                "20.0",
                "25.0",
            ],
            "Outside Temperature": [
                "8.0",
                "6.0",
                "7.0",
            ],
        }
    )

    preprocessor = BuildingDataPreprocessor()

    result, mapping = preprocessor.canonicalize(df)

    assert mapping["timestamp"] == "Date Time"
    assert mapping["total_power"] == "Building Power"
    assert (
        mapping["outdoor_temp_avg"]
        == "Outside Temperature"
    )

    assert list(result.columns) == [
        "timestamp",
        "total_power",
        "outdoor_temp_avg",
    ]

    assert result["timestamp"].is_monotonic_increasing

    assert result["total_power"].tolist() == [
        20.0,
        25.0,
        30.0,
    ]


def test_duplicate_timestamp_keeps_latest_record():
    df = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01 00:00:00",
                "2026-01-01 00:00:00",
                "2026-01-01 00:15:00",
            ],
            "total_power": [
                10.0,
                99.0,
                20.0,
            ],
        }
    )

    preprocessor = BuildingDataPreprocessor()

    result, _ = preprocessor.canonicalize(df)

    assert len(result) == 2

    assert result.iloc[0]["total_power"] == 99.0


def test_resample_to_15_minutes():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 00:00:00",
                    "2026-01-01 00:05:00",
                    "2026-01-01 00:15:00",
                ]
            ),
            "total_power": [
                10.0,
                20.0,
                30.0,
            ],
        }
    )

    preprocessor = BuildingDataPreprocessor(
        sampling_minutes=15
    )

    result = preprocessor.resample(df)

    assert len(result) == 2

    assert result.iloc[0]["total_power"] == 15.0
    assert result.iloc[1]["total_power"] == 30.0


def test_derived_features():
    timestamps = pd.date_range(
        "2026-01-01",
        periods=8,
        freq="15min",
    )

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "total_power": np.arange(
                10.0,
                18.0,
            ),
            "hvac_N": np.arange(
                1.0,
                9.0,
            ),
            "hvac_S": np.arange(
                2.0,
                10.0,
            ),
        }
    )

    result = (
        BuildingDataPreprocessor
        .add_derived_features(df)
    )

    assert result.iloc[0]["hvac_total"] == 3.0

    assert np.isnan(
        result.iloc[0]["power_lag_15m"]
    )

    assert (
        result.iloc[1]["power_lag_15m"]
        == 10.0
    )

    assert (
        result.iloc[4]["power_lag_60m"]
        == 10.0
    )

    assert (
        result.iloc[1]["power_delta_15m"]
        == 1.0
    )

    assert (
        result.iloc[4]["power_delta_60m"]
        == 4.0
    )

    for column in [
        "time_sin",
        "time_cos",
        "dow_sin",
        "dow_cos",
        "is_weekend",
    ]:
        assert column in result.columns


def test_prepare_returns_report():
    df = pd.DataFrame(
        {
            "datetime": pd.date_range(
                "2026-01-01",
                periods=8,
                freq="15min",
            ),
            "building_power": np.arange(
                20.0,
                28.0,
            ),
        }
    )

    preprocessor = BuildingDataPreprocessor()

    result, report = preprocessor.prepare(df)

    assert report["rows_input"] == 8
    assert report["rows_output"] == 8
    assert report["sampling_minutes"] == 15

    assert (
        report["column_mapping"]["timestamp"]
        == "datetime"
    )

    assert (
        report["column_mapping"]["total_power"]
        == "building_power"
    )

    assert "power_lag_15m" in result.columns
    assert "time_sin" in result.columns

    # Long historical lags are expected to be missing
    # when a short dataset is supplied.
    assert (
        report["missing_values"]["power_lag_7d"]
        == 8
    )


def test_missing_required_signal_fails():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=4,
                freq="15min",
            ),
            "humidity": [
                50,
                51,
                52,
                53,
            ],
        }
    )

    preprocessor = BuildingDataPreprocessor()

    with pytest.raises(
        ValueError,
        match="total_power",
    ):
        preprocessor.prepare(df)
