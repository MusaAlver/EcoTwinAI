
import numpy as np
import pandas as pd
import pytest

from src.intake import BuildingDataIntake


def make_timestamps(rows=8):
    return pd.date_range(
        "2026-01-01",
        periods=rows,
        freq="15min",
    )


def test_canonical_total_power_is_accepted_as_kw():
    df = pd.DataFrame(
        {
            "timestamp": make_timestamps(),
            "total_power": np.arange(10.0, 18.0),
        }
    )

    intake = BuildingDataIntake()

    prepared, report = intake.prepare(
        df,
        resample=False,
        add_features=False,
    )

    np.testing.assert_allclose(
        prepared["total_power"],
        df["total_power"],
    )

    assert report["canonical_unit"] == "kW"
    assert (
        report["semantic_normalization"]["source_unit"]
        == "kW"
    )


def test_total_kw_is_safely_inferred_as_kw():
    df = pd.DataFrame(
        {
            "timestamp": make_timestamps(),
            "total_kw": np.arange(20.0, 28.0),
        }
    )

    intake = BuildingDataIntake()

    prepared, report = intake.prepare(
        df,
        resample=False,
        add_features=False,
    )

    np.testing.assert_allclose(
        prepared["total_power"],
        df["total_kw"],
    )

    assert (
        report["semantic_normalization"]["source_column"]
        == "total_kw"
    )


def test_building_power_without_unit_is_rejected():
    df = pd.DataFrame(
        {
            "timestamp": make_timestamps(),
            "building_power": np.arange(20.0, 28.0),
        }
    )

    intake = BuildingDataIntake()

    with pytest.raises(
        ValueError,
        match="Explicit unit is required",
    ):
        intake.prepare(
            df,
            resample=False,
            add_features=False,
        )


def test_building_power_in_watts_is_normalized():
    df = pd.DataFrame(
        {
            "timestamp": make_timestamps(),
            "building_power": np.arange(
                15000.0,
                23000.0,
                1000.0,
            ),
        }
    )

    intake = BuildingDataIntake()

    prepared, report = intake.prepare(
        df,
        power_unit="W",
        resample=False,
        add_features=False,
    )

    np.testing.assert_allclose(
        prepared["total_power"],
        np.arange(15.0, 23.0),
    )

    assert (
        report["semantic_normalization"]["source_unit"]
        == "W"
    )


def test_ambiguous_energy_column_is_not_auto_selected():
    df = pd.DataFrame(
        {
            "timestamp": make_timestamps(),
            "energy_consumption": np.repeat(
                3.75,
                8,
            ),
        }
    )

    intake = BuildingDataIntake()

    with pytest.raises(
        ValueError,
        match="No safe total_power mapping",
    ):
        intake.prepare(
            df,
            resample=False,
            add_features=False,
        )


def test_explicit_interval_energy_is_converted():
    df = pd.DataFrame(
        {
            "timestamp": make_timestamps(),
            "energy_consumption": np.repeat(
                3.75,
                8,
            ),
        }
    )

    intake = BuildingDataIntake()

    prepared, report = intake.prepare(
        df,
        total_power_column="energy_consumption",
        power_unit="kWh",
        meter_semantics="interval_energy",
        interval_minutes=15,
        resample=False,
        add_features=False,
    )

    np.testing.assert_allclose(
        prepared["total_power"],
        np.repeat(15.0, 8),
    )

    semantic = report["semantic_normalization"]

    assert semantic["source_column_confirmed"] is True
    assert semantic["unit_confirmed"] is True
    assert (
        semantic["conversion"]
        == "interval_energy_to_average_power"
    )


def test_explicit_cumulative_energy_is_converted_by_delta():
    df = pd.DataFrame(
        {
            "timestamp": make_timestamps(4),
            "meter_kwh": [
                100.0,
                103.75,
                107.50,
                111.25,
            ],
        }
    )

    intake = BuildingDataIntake()

    prepared, report = intake.prepare(
        df,
        total_power_column="meter_kwh",
        power_unit="kWh",
        meter_semantics="cumulative_energy",
        interval_minutes=15,
        resample=False,
        add_features=False,
    )

    values = prepared["total_power"].to_numpy()

    assert np.isnan(values[0])

    np.testing.assert_allclose(
        values[1:],
        [15.0, 15.0, 15.0],
    )

    assert (
        report["semantic_normalization"]["conversion"]
        == "cumulative_energy_delta_to_average_power"
    )


def test_conflicting_power_sources_are_rejected():
    df = pd.DataFrame(
        {
            "timestamp": make_timestamps(),
            "total_power": np.arange(10.0, 18.0),
            "meter_kw": np.arange(20.0, 28.0),
        }
    )

    intake = BuildingDataIntake()

    with pytest.raises(
        ValueError,
        match="contains total_power",
    ):
        intake.prepare(
            df,
            total_power_column="meter_kw",
            power_unit="kW",
            resample=False,
            add_features=False,
        )
