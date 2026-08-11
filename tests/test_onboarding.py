
import pandas as pd

from src.onboarding import BuildingDataOnboarding


def make_timestamp():
    return pd.date_range(
        "2026-01-01",
        periods=8,
        freq="15min",
    )


def test_alias_mapping():
    df = pd.DataFrame(
        {
            "Date Time": make_timestamp(),
            "Building Power": range(8),
            "HVAC North": range(8),
            "Outside Temperature": range(8),
        }
    )

    onboarding = BuildingDataOnboarding()

    report = onboarding.inspect_dataframe(df)

    mapping = report["column_mapping"]

    assert mapping["timestamp"] == "Date Time"
    assert mapping["total_power"] == "Building Power"
    assert mapping["hvac_N"] == "HVAC North"
    assert mapping["outdoor_temp_avg"] == "Outside Temperature"

    assert (
        report["column_quality"]["timestamp"]
        ["inferred_sampling_minutes"]
        == 15.0
    )


def test_minimum_training_contract():
    df = pd.DataFrame(
        {
            "datetime": make_timestamp(),
            "electricity": [
                30.1,
                31.2,
                30.8,
                32.0,
                33.1,
                34.0,
                33.6,
                35.2,
            ],
        }
    )

    onboarding = BuildingDataOnboarding()

    report = onboarding.inspect_dataframe(df)

    assert report["missing_required"] == []
    assert (
        report["capabilities"]
        ["forecast_training"]
        ["available"]
        is True
    )

    assert report["compatibility_score"] == 60.0
    assert report["compatibility_status"] == "PARTIAL"


def test_missing_total_power_is_rejected():
    df = pd.DataFrame(
        {
            "timestamp": make_timestamp(),
            "humidity": range(8),
        }
    )

    onboarding = BuildingDataOnboarding()

    report = onboarding.inspect_dataframe(df)

    assert "total_power" in report["missing_required"]
    assert report["compatibility_status"] == "NO"

    assert (
        report["capabilities"]
        ["forecast_training"]
        ["available"]
        is False
    )


def test_full_signal_contract():
    timestamps = make_timestamp()

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "total_power": range(8),
            "hvac_N": range(8),
            "hvac_S": range(8),
            "mels_N": range(8),
            "mels_S": range(8),
            "lig_S": range(8),
            "indoor_temp_avg": range(8),
            "outdoor_temp_avg": range(8),
            "relative_humidity_set_1": range(8),
            "solar_radiation_set_1": range(8),
            "dew_point_temperature_set_1d": range(8),
        }
    )

    onboarding = BuildingDataOnboarding()

    report = onboarding.inspect_dataframe(df)

    assert report["compatibility_score"] == 100.0
    assert report["compatibility_status"] == "YES"

    assert (
        report["capabilities"]
        ["existing_building59_model"]
        ["compatible"]
        is True
    )
