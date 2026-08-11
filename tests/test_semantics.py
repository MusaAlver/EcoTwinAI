
import numpy as np
import pandas as pd
import pytest

from src.semantics import SignalSemanticsValidator


def test_watts_are_converted_to_kw():
    validator = SignalSemanticsValidator()

    result = validator.normalize_total_power(
        pd.Series([15000.0, 20000.0]),
        unit="W",
    )

    np.testing.assert_allclose(
        result.values_kw.to_numpy(),
        [15.0, 20.0],
    )

    assert result.report["source_quantity"] == "power"
    assert result.report["canonical_unit"] == "kW"
    assert result.report["scale_factor"] == pytest.approx(0.001)


def test_megawatts_are_converted_to_kw():
    validator = SignalSemanticsValidator()

    result = validator.normalize_total_power(
        pd.Series([0.015, 0.020]),
        unit="MW",
    )

    np.testing.assert_allclose(
        result.values_kw.to_numpy(),
        [15.0, 20.0],
    )


def test_kw_requires_no_conversion():
    validator = SignalSemanticsValidator()

    result = validator.normalize_total_power(
        pd.Series([12.5, 18.25]),
        unit="kW",
    )

    np.testing.assert_allclose(
        result.values_kw.to_numpy(),
        [12.5, 18.25],
    )

    assert result.report["scale_factor"] == pytest.approx(1.0)


def test_interval_kwh_is_converted_to_average_kw():
    validator = SignalSemanticsValidator()

    result = validator.normalize_total_power(
        pd.Series([7.5, 10.0]),
        unit="kWh",
        meter_semantics="interval_energy",
        interval_minutes=30,
    )

    np.testing.assert_allclose(
        result.values_kw.to_numpy(),
        [15.0, 20.0],
    )

    assert (
        result.report["conversion"]
        == "interval_energy_to_average_power"
    )


def test_interval_wh_is_converted_to_average_kw():
    validator = SignalSemanticsValidator()

    result = validator.normalize_total_power(
        pd.Series([3750.0, 5000.0]),
        unit="Wh",
        meter_semantics="interval_energy",
        interval_minutes=15,
    )

    np.testing.assert_allclose(
        result.values_kw.to_numpy(),
        [15.0, 20.0],
    )


def test_cumulative_kwh_uses_meter_delta():
    validator = SignalSemanticsValidator()

    result = validator.normalize_total_power(
        pd.Series(
            [
                100.0,
                107.5,
                117.5,
            ]
        ),
        unit="kWh",
        meter_semantics="cumulative_energy",
        interval_minutes=30,
    )

    values = result.values_kw.to_numpy()

    assert np.isnan(values[0])

    np.testing.assert_allclose(
        values[1:],
        [15.0, 20.0],
    )

    assert (
        result.report["conversion"]
        == "cumulative_energy_delta_to_average_power"
    )

    assert result.report["introduced_nan_count"] == 1


def test_missing_unit_is_rejected():
    validator = SignalSemanticsValidator()

    with pytest.raises(
        ValueError,
        match="Explicit unit is required",
    ):
        validator.normalize_total_power(
            pd.Series([10.0, 20.0]),
            unit=None,
        )


def test_energy_without_meter_semantics_is_rejected():
    validator = SignalSemanticsValidator()

    with pytest.raises(
        ValueError,
        match="meter_semantics is required",
    ):
        validator.normalize_total_power(
            pd.Series([5.0, 6.0]),
            unit="kWh",
            interval_minutes=30,
        )


def test_energy_without_interval_is_rejected():
    validator = SignalSemanticsValidator()

    with pytest.raises(
        ValueError,
        match="interval_minutes is required",
    ):
        validator.normalize_total_power(
            pd.Series([5.0, 6.0]),
            unit="kWh",
            meter_semantics="interval_energy",
        )


def test_cumulative_meter_reset_is_rejected():
    validator = SignalSemanticsValidator()

    with pytest.raises(
        ValueError,
        match="meter reset or rollover",
    ):
        validator.normalize_total_power(
            pd.Series(
                [
                    100.0,
                    110.0,
                    5.0,
                ]
            ),
            unit="kWh",
            meter_semantics="cumulative_energy",
            interval_minutes=30,
        )


def test_power_must_not_receive_meter_semantics():
    validator = SignalSemanticsValidator()

    with pytest.raises(
        ValueError,
        match="must not be supplied",
    ):
        validator.normalize_total_power(
            pd.Series([15.0, 20.0]),
            unit="kW",
            meter_semantics="interval_energy",
        )


def test_non_numeric_signal_is_rejected():
    validator = SignalSemanticsValidator()

    with pytest.raises(
        ValueError,
        match="non-numeric",
    ):
        validator.normalize_total_power(
            pd.Series(
                [
                    10.0,
                    "broken",
                    20.0,
                ]
            ),
            unit="kW",
        )


def test_infinite_signal_is_rejected():
    validator = SignalSemanticsValidator()

    with pytest.raises(
        ValueError,
        match="infinite",
    ):
        validator.normalize_total_power(
            pd.Series(
                [
                    10.0,
                    np.inf,
                    20.0,
                ]
            ),
            unit="kW",
        )


def test_unsupported_unit_is_rejected():
    validator = SignalSemanticsValidator()

    with pytest.raises(
        ValueError,
        match="Unsupported unit",
    ):
        validator.normalize_total_power(
            pd.Series([10.0, 20.0]),
            unit="BTU/h",
        )
