from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.onboarding import BuildingDataOnboarding


class BuildingDataPreprocessor:
    def __init__(
        self,
        contract_path: str | Path = "config/building_data_contract.json",
        sampling_minutes: int = 15,
    ):
        self.contract_path = Path(contract_path)
        self.sampling_minutes = sampling_minutes

        with self.contract_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            self.contract = json.load(f)

        self.onboarding = BuildingDataOnboarding(
            contract_path=self.contract_path
        )

    def canonicalize(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, str]]:
        mapping = self.onboarding.match_columns(df)

        missing_required = {
            "timestamp",
            "total_power",
        } - set(mapping)

        if missing_required:
            raise ValueError(
                "Missing required signals: "
                + ", ".join(sorted(missing_required))
            )

        rename_map = {
            source: canonical
            for canonical, source in mapping.items()
        }

        result = df.rename(
            columns=rename_map
        ).copy()

        keep = list(mapping.keys())

        result = result[
            [
                column
                for column in keep
                if column in result.columns
            ]
        ]

        result["timestamp"] = pd.to_datetime(
            result["timestamp"],
            errors="coerce",
        )

        result = result.dropna(
            subset=["timestamp"]
        )

        numeric_columns = [
            column
            for column in result.columns
            if column != "timestamp"
        ]

        for column in numeric_columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

        result = (
            result
            .sort_values("timestamp")
            .drop_duplicates(
                subset=["timestamp"],
                keep="last",
            )
            .reset_index(drop=True)
        )

        return result, mapping

    def resample(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        frame = df.copy()

        frame = frame.set_index(
            "timestamp"
        )

        rule = f"{self.sampling_minutes}min"

        frame = frame.resample(
            rule
        ).mean()

        frame.index.name = "timestamp"

        return frame.reset_index()

    @staticmethod
    def add_derived_features(
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        frame = df.copy()

        if {
            "hvac_N",
            "hvac_S",
        }.issubset(frame.columns):
            frame["hvac_total"] = (
                frame["hvac_N"]
                + frame["hvac_S"]
            )

        if "total_power" in frame.columns:
            frame["power_lag_15m"] = (
                frame["total_power"].shift(1)
            )

            frame["power_lag_60m"] = (
                frame["total_power"].shift(4)
            )

            frame["power_lag_24h"] = (
                frame["total_power"].shift(96)
            )

            frame["power_lag_7d"] = (
                frame["total_power"].shift(672)
            )

            frame["power_delta_15m"] = (
                frame["total_power"].diff(1)
            )

            frame["power_delta_60m"] = (
                frame["total_power"].diff(4)
            )

        timestamp = frame["timestamp"]

        minutes = (
            timestamp.dt.hour * 60
            + timestamp.dt.minute
        )

        frame["time_sin"] = np.sin(
            2 * np.pi * minutes / 1440
        )

        frame["time_cos"] = np.cos(
            2 * np.pi * minutes / 1440
        )

        day = timestamp.dt.dayofweek

        frame["dow_sin"] = np.sin(
            2 * np.pi * day / 7
        )

        frame["dow_cos"] = np.cos(
            2 * np.pi * day / 7
        )

        frame["is_weekend"] = (
            day >= 5
        ).astype(int)

        return frame

    def prepare(
        self,
        df: pd.DataFrame,
        *,
        resample: bool = True,
        add_features: bool = True,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        canonical, mapping = self.canonicalize(
            df
        )

        rows_input = len(df)
        rows_canonical = len(canonical)

        if resample:
            canonical = self.resample(
                canonical
            )

        if add_features:
            canonical = self.add_derived_features(
                canonical
            )

        report = {
            "rows_input": int(rows_input),
            "rows_output": int(len(canonical)),
            "rows_after_canonicalization":
                int(rows_canonical),
            "sampling_minutes":
                self.sampling_minutes,
            "column_mapping": mapping,
            "output_columns":
                list(canonical.columns),
            "missing_values":
                {
                    column: int(value)
                    for column, value
                    in canonical.isna().sum().items()
                    if value > 0
                },
        }

        return canonical, report
