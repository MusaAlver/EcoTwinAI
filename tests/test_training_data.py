
import numpy as np
import pandas as pd
import pytest

from src.training_data import TimeSeriesDatasetBuilder


def make_dataset(
    rows: int = 400,
    freq: str = "15min",
):
    timestamp = pd.date_range(
        "2026-01-01",
        periods=rows,
        freq=freq,
    )

    values = np.arange(
        rows,
        dtype=float,
    )

    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "total_power": 40.0 + values * 0.05,
            "outdoor_temp_avg": 10.0 + values * 0.01,
        }
    )


def test_chronological_split_has_no_overlap():
    df = make_dataset()

    builder = TimeSeriesDatasetBuilder()

    splits = builder.split(df)

    train = splits["train"]
    validation = splits["validation"]
    test = splits["test"]

    assert train["timestamp"].max() < validation["timestamp"].min()
    assert validation["timestamp"].max() < test["timestamp"].min()

    assert len(train) == 280
    assert len(validation) == 60
    assert len(test) == 60


def test_sequence_shape_and_horizon():
    df = make_dataset(
        rows=120
    )

    builder = TimeSeriesDatasetBuilder(
        lookback_steps=16,
        horizon_minutes=30,
        sampling_minutes=15,
    )

    dataset = builder.build_sequences(
        df,
        feature_columns=[
            "total_power",
            "outdoor_temp_avg",
        ],
    )

    assert dataset.X.shape[1:] == (
        16,
        2,
    )

    assert dataset.y.shape[1:] == (
        1,
    )

    assert len(dataset.X) == len(dataset.y)

    horizon = (
        dataset.outcome_time
        - dataset.forecast_time
    ).astype(
        "timedelta64[m]"
    ).astype(int)

    assert np.all(
        horizon == 30
    )


def test_sequences_do_not_cross_split_boundaries():
    df = make_dataset()

    builder = TimeSeriesDatasetBuilder()

    result = builder.build(
        df,
        feature_columns=[
            "total_power",
            "outdoor_temp_avg",
        ],
    )

    for name in [
        "train",
        "validation",
        "test",
    ]:
        frame = result["splits"][name]
        dataset = result["datasets"][name]

        lower = np.datetime64(
            frame["timestamp"].min()
        )

        upper = np.datetime64(
            frame["timestamp"].max()
        )

        assert np.all(
            dataset.forecast_time >= lower
        )

        assert np.all(
            dataset.outcome_time <= upper
        )


def test_time_gap_sequences_are_skipped():
    df = make_dataset(
        rows=120
    )

    clean_builder = TimeSeriesDatasetBuilder()

    clean = clean_builder.build_sequences(
        df,
        feature_columns=[
            "total_power",
        ],
    )

    broken = df.drop(
        index=50
    ).reset_index(
        drop=True
    )

    gap_result = clean_builder.build_sequences(
        broken,
        feature_columns=[
            "total_power",
        ],
    )

    assert len(gap_result) < len(clean)

    for forecast, outcome in zip(
        gap_result.forecast_time,
        gap_result.outcome_time,
    ):
        assert outcome > forecast


def test_nan_windows_are_not_used():
    df = make_dataset(
        rows=120
    )

    builder = TimeSeriesDatasetBuilder()

    clean = builder.build_sequences(
        df,
        feature_columns=[
            "total_power",
            "outdoor_temp_avg",
        ],
    )

    df.loc[
        40,
        "outdoor_temp_avg",
    ] = np.nan

    result = builder.build_sequences(
        df,
        feature_columns=[
            "total_power",
            "outdoor_temp_avg",
        ],
    )

    assert len(result) < len(clean)

    assert np.isfinite(
        result.X
    ).all()


def test_duplicate_timestamps_are_rejected():
    df = make_dataset(
        rows=100
    )

    df.loc[
        20,
        "timestamp",
    ] = df.loc[
        19,
        "timestamp",
    ]

    builder = TimeSeriesDatasetBuilder()

    with pytest.raises(
        ValueError,
        match="duplicate timestamps",
    ):
        builder.build_sequences(
            df,
            feature_columns=[
                "total_power",
            ],
        )


def test_invalid_split_ratios_are_rejected():
    with pytest.raises(
        ValueError,
        match="must sum to 1",
    ):
        TimeSeriesDatasetBuilder(
            train_ratio=0.8,
            validation_ratio=0.15,
            test_ratio=0.10,
        )


def test_short_split_is_rejected():
    df = make_dataset(
        rows=60
    )

    builder = TimeSeriesDatasetBuilder(
        lookback_steps=16,
        horizon_minutes=30,
    )

    with pytest.raises(
        ValueError,
        match="split is too short",
    ):
        builder.split(df)
