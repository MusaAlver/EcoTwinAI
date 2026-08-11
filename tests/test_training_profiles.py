
import pandas as pd
import pytest

from src.training_profiles import TrainingProfileSelector


def make_core_frame(rows: int = 10) -> pd.DataFrame:
    selector = TrainingProfileSelector()

    return pd.DataFrame(
        {
            feature: [float(i) for i in range(rows)]
            for feature in selector.CORE_FEATURES
        }
    )


def test_core_profile_contains_only_core_features():
    selector = TrainingProfileSelector()
    frame = make_core_frame()

    report = selector.select(frame)

    assert report["profile"] == "CORE"
    assert report["feature_columns"] == selector.CORE_FEATURES
    assert report["feature_count"] == len(selector.CORE_FEATURES)

    assert report["capabilities"]["forecast_training"] is True
    assert report["capabilities"]["environment_context"] is False
    assert report["capabilities"]["full_root_cause"] is False


def test_environment_signal_upgrades_to_context():
    selector = TrainingProfileSelector()
    frame = make_core_frame()

    frame["outdoor_temp_avg"] = 20.0

    report = selector.select(frame)

    assert report["profile"] == "CONTEXT"
    assert "outdoor_temp_avg" in report["feature_columns"]
    assert report["capabilities"]["environment_context"] is True
    assert report["capabilities"]["full_root_cause"] is False


def test_partial_subsystem_signals_are_context_not_full():
    selector = TrainingProfileSelector()
    frame = make_core_frame()

    frame["hvac_N"] = 10.0
    frame["hvac_S"] = 12.0
    frame["hvac_total"] = 22.0

    report = selector.select(frame)

    assert report["profile"] == "CONTEXT"
    assert report["capabilities"]["full_root_cause"] is False

    assert "hvac_N" in report["feature_columns"]
    assert "hvac_S" in report["feature_columns"]
    assert "hvac_total" in report["feature_columns"]

    assert "mels_N" in report["missing_full_subsystem_signals"]


def test_all_subsystems_upgrade_to_full():
    selector = TrainingProfileSelector()
    frame = make_core_frame()

    for feature in selector.SUBSYSTEM_FEATURES:
        frame[feature] = 10.0

    report = selector.select(frame)

    assert report["profile"] == "FULL"
    assert report["capabilities"]["full_root_cause"] is True
    assert report["missing_full_subsystem_signals"] == []

    for feature in selector.SUBSYSTEM_FEATURES:
        assert feature in report["feature_columns"]


def test_full_profile_keeps_available_environment_context():
    selector = TrainingProfileSelector()
    frame = make_core_frame()

    for feature in selector.SUBSYSTEM_FEATURES:
        frame[feature] = 10.0

    frame["outdoor_temp_avg"] = 20.0
    frame["relative_humidity_set_1"] = 50.0

    report = selector.select(frame)

    assert report["profile"] == "FULL"
    assert report["capabilities"]["environment_context"] is True

    assert "outdoor_temp_avg" in report["feature_columns"]
    assert "relative_humidity_set_1" in report["feature_columns"]


def test_feature_order_and_fingerprint_are_deterministic():
    selector = TrainingProfileSelector()

    frame_a = make_core_frame()
    frame_a["hvac_S"] = 1.0
    frame_a["outdoor_temp_avg"] = 2.0

    frame_b = frame_a[
        list(reversed(frame_a.columns))
    ].copy()

    report_a = selector.select(frame_a)
    report_b = selector.select(frame_b)

    assert report_a["feature_columns"] == report_b["feature_columns"]
    assert (
        report_a["feature_fingerprint"]
        == report_b["feature_fingerprint"]
    )


def test_fingerprint_changes_when_feature_contract_changes():
    selector = TrainingProfileSelector()

    core = make_core_frame()

    context = core.copy()
    context["outdoor_temp_avg"] = 20.0

    core_report = selector.select(core)
    context_report = selector.select(context)

    assert (
        core_report["feature_fingerprint"]
        != context_report["feature_fingerprint"]
    )


def test_missing_core_feature_is_rejected():
    selector = TrainingProfileSelector()

    frame = make_core_frame().drop(
        columns=["power_lag_24h"]
    )

    with pytest.raises(
        ValueError,
        match="Missing required CORE training features",
    ):
        selector.select(frame)


def test_empty_training_frame_is_rejected():
    selector = TrainingProfileSelector()

    frame = make_core_frame(
        rows=0
    )

    with pytest.raises(
        ValueError,
        match="Training frame is empty",
    ):
        selector.select(frame)
