
import numpy as np
import pandas as pd

from src.data_quality import BuildingDataQualityGate


def make_good_dataset(
    days: int = 31,
    freq: str = "15min",
):
    periods = int(
        pd.Timedelta(days=days)
        / pd.Timedelta(freq)
    ) + 1

    timestamp = pd.date_range(
        "2026-01-01",
        periods=periods,
        freq=freq,
    )

    power = (
        40
        + 8 * np.sin(
            np.arange(periods) / 20
        )
        + np.arange(periods) * 0.001
    )

    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "total_power": power,
        }
    )


def test_good_training_dataset_passes():
    gate = BuildingDataQualityGate()

    report = gate.evaluate(
        make_good_dataset(),
        mode="training",
    )

    assert report["status"] == "PASS"
    assert report["can_preprocess"] is True
    assert report["training_ready"] is True

    assert (
        report["metrics"]
        ["inferred_sampling_minutes"]
        == 15.0
    )

    assert report["issues"] == []


def test_missing_total_power_fails():
    df = make_good_dataset().drop(
        columns=["total_power"]
    )

    gate = BuildingDataQualityGate()

    report = gate.evaluate(df)

    assert report["status"] == "FAIL"
    assert report["can_preprocess"] is False

    codes = {
        issue["code"]
        for issue in report["issues"]
    }

    assert "MISSING_REQUIRED_SIGNALS" in codes


def test_bad_timestamp_parse_fails():
    df = make_good_dataset()

    df["timestamp"] = (
        df["timestamp"]
        .astype(str)
    )

    count = max(
        1,
        int(len(df) * 0.02),
    )

    df.loc[
        df.index[:count],
        "timestamp",
    ] = "not-a-date"

    gate = BuildingDataQualityGate()

    report = gate.evaluate(df)

    assert report["status"] == "FAIL"

    codes = {
        issue["code"]
        for issue in report["issues"]
    }

    assert "TIMESTAMP_PARSE_FAILURE" in codes


def test_high_power_missingness_fails():
    df = make_good_dataset()

    count = int(
        len(df) * 0.10
    )

    df.loc[
        df.index[:count],
        "total_power",
    ] = np.nan

    gate = BuildingDataQualityGate()

    report = gate.evaluate(df)

    assert report["status"] == "FAIL"

    codes = {
        issue["code"]
        for issue in report["issues"]
    }

    assert "POWER_MISSINGNESS_HIGH" in codes


def test_low_source_resolution_fails():
    df = make_good_dataset(
        days=31,
        freq="30min",
    )

    gate = BuildingDataQualityGate(
        target_sampling_minutes=15
    )

    report = gate.evaluate(df)

    assert report["status"] == "FAIL"

    codes = {
        issue["code"]
        for issue in report["issues"]
    }

    assert "SOURCE_RESOLUTION_TOO_LOW" in codes


def test_duplicate_timestamps_warn():
    df = make_good_dataset()

    duplicates = df.iloc[:60].copy()

    df = pd.concat(
        [
            df,
            duplicates,
        ],
        ignore_index=True,
    )

    gate = BuildingDataQualityGate()

    report = gate.evaluate(df)

    assert report["status"] == "WARN"

    codes = {
        issue["code"]
        for issue in report["issues"]
    }

    assert "DUPLICATE_TIMESTAMPS" in codes


def test_negative_power_warns_not_fails():
    df = make_good_dataset()

    df.loc[
        df.index[:10],
        "total_power",
    ] = -5.0

    gate = BuildingDataQualityGate()

    report = gate.evaluate(df)

    assert report["status"] == "WARN"
    assert report["can_preprocess"] is True

    codes = {
        issue["code"]
        for issue in report["issues"]
    }

    assert "NEGATIVE_POWER_VALUES" in codes


def test_short_training_history_fails():
    df = make_good_dataset(
        days=7
    )

    gate = BuildingDataQualityGate()

    report = gate.evaluate(
        df,
        mode="training",
    )

    assert report["status"] == "FAIL"

    codes = {
        issue["code"]
        for issue in report["issues"]
    }

    assert "INSUFFICIENT_HISTORY" in codes


def test_constant_power_signal_fails():
    df = make_good_dataset()

    df["total_power"] = 50.0

    gate = BuildingDataQualityGate()

    report = gate.evaluate(df)

    assert report["status"] == "FAIL"

    codes = {
        issue["code"]
        for issue in report["issues"]
    }

    assert "CONSTANT_POWER_SIGNAL" in codes


def test_short_history_is_allowed_for_inference():
    df = make_good_dataset(
        days=1
    )

    gate = BuildingDataQualityGate()

    report = gate.evaluate(
        df,
        mode="inference",
    )

    assert report["status"] == "PASS"
    assert report["can_preprocess"] is True
    assert report["training_ready"] is False
