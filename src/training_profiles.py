from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


class TrainingProfileSelector:
    PROFILE_VERSION = "1.0"

    CORE_FEATURES = [
        "total_power",
        "power_lag_15m",
        "power_lag_60m",
        "power_lag_24h",
        "power_lag_7d",
        "power_delta_15m",
        "power_delta_60m",
        "time_sin",
        "time_cos",
        "dow_sin",
        "dow_cos",
        "is_weekend",
    ]

    SUBSYSTEM_FEATURES = [
        "hvac_N",
        "hvac_S",
        "mels_N",
        "mels_S",
        "lig_S",
    ]

    ENVIRONMENT_FEATURES = [
        "indoor_temp_avg",
        "outdoor_temp_avg",
        "relative_humidity_set_1",
        "solar_radiation_set_1",
        "dew_point_temperature_set_1d",
    ]

    OPTIONAL_DERIVED = [
        "hvac_total",
    ]

    def __init__(
        self,
        contract_path: str | Path = "config/building_data_contract.json",
    ):
        self.contract_path = Path(contract_path)

        self.contract = json.loads(
            self.contract_path.read_text(
                encoding="utf-8"
            )
        )

    @staticmethod
    def _fingerprint(
        feature_columns: list[str],
    ) -> str:
        payload = json.dumps(
            feature_columns,
            separators=(",", ":"),
        ).encode("utf-8")

        return hashlib.sha256(
            payload
        ).hexdigest()[:16]

    def select(
        self,
        frame: pd.DataFrame,
    ) -> dict[str, Any]:
        if frame.empty:
            raise ValueError(
                "Training frame is empty"
            )

        available = set(
            frame.columns
        )

        missing_core = [
            feature
            for feature in self.CORE_FEATURES
            if feature not in available
        ]

        if missing_core:
            raise ValueError(
                "Missing required CORE training features: "
                + ", ".join(missing_core)
            )

        subsystem_available = [
            feature
            for feature in self.SUBSYSTEM_FEATURES
            if feature in available
        ]

        environment_available = [
            feature
            for feature in self.ENVIRONMENT_FEATURES
            if feature in available
        ]

        derived_available = [
            feature
            for feature in self.OPTIONAL_DERIVED
            if feature in available
        ]

        full_subsystems = (
            len(subsystem_available)
            == len(self.SUBSYSTEM_FEATURES)
        )

        has_optional_context = bool(
            subsystem_available
            or environment_available
        )

        if full_subsystems:
            profile = "FULL"
        elif has_optional_context:
            profile = "CONTEXT"
        else:
            profile = "CORE"

        feature_columns = [
            *self.CORE_FEATURES,
            *subsystem_available,
            *derived_available,
            *environment_available,
        ]

        if len(feature_columns) != len(
            set(feature_columns)
        ):
            raise ValueError(
                "Duplicate features detected "
                "in training profile"
            )

        missing_full_subsystems = [
            feature
            for feature in self.SUBSYSTEM_FEATURES
            if feature not in available
        ]

        report = {
            "profile": profile,
            "profile_version": self.PROFILE_VERSION,
            "contract_version": self.contract.get(
                "contract_version"
            ),
            "feature_columns": feature_columns,
            "feature_count": len(feature_columns),
            "feature_fingerprint": self._fingerprint(
                feature_columns
            ),
            "available_subsystem_signals":
                subsystem_available,
            "available_environment_signals":
                environment_available,
            "missing_full_subsystem_signals":
                missing_full_subsystems,
            "capabilities": {
                "forecast_training": True,
                "environment_context": bool(
                    environment_available
                ),
                "full_root_cause": full_subsystems,
            },
        }

        return report
