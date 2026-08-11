from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class BuildingDataOnboarding:
    def __init__(
        self,
        contract_path: str | Path = "config/building_data_contract.json",
    ):
        self.contract_path = Path(contract_path)

        with self.contract_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            self.contract = json.load(f)

        self.signals = self.contract["canonical_signals"]

    @staticmethod
    def _normalize_name(name: str) -> str:
        name = str(name).strip().lower()
        name = re.sub(r"[^a-z0-9]+", "_", name)
        return name.strip("_")

    def load(
        self,
        path: str | Path,
    ) -> pd.DataFrame:
        path = Path(path)

        suffix = path.suffix.lower()

        if suffix == ".csv":
            return pd.read_csv(path)

        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(path)

        raise ValueError(
            f"Unsupported dataset format: {suffix}"
        )

    def match_columns(
        self,
        df: pd.DataFrame,
    ) -> dict[str, str]:
        normalized_columns = {
            self._normalize_name(column): column
            for column in df.columns
        }

        mapping: dict[str, str] = {}

        for canonical, spec in self.signals.items():
            candidates = [
                canonical,
                *spec.get("aliases", []),
            ]

            for candidate in candidates:
                normalized = self._normalize_name(candidate)

                if normalized in normalized_columns:
                    mapping[canonical] = normalized_columns[normalized]
                    break

        return mapping

    @staticmethod
    def _numeric_quality(
        series: pd.Series,
    ) -> dict[str, Any]:
        converted = pd.to_numeric(
            series,
            errors="coerce",
        )

        non_null = series.notna().sum()

        if non_null == 0:
            parse_ratio = 0.0
        else:
            parse_ratio = (
                converted.notna().sum()
                / non_null
            )

        return {
            "parse_ratio": round(
                float(parse_ratio),
                4,
            ),
            "missing_ratio": round(
                float(series.isna().mean()),
                4,
            ),
            "min": (
                float(converted.min())
                if converted.notna().any()
                else None
            ),
            "max": (
                float(converted.max())
                if converted.notna().any()
                else None
            ),
        }

    @staticmethod
    def _timestamp_quality(
        series: pd.Series,
    ) -> dict[str, Any]:
        parsed = pd.to_datetime(
            series,
            errors="coerce",
        )

        valid = parsed.dropna()

        parse_ratio = float(
            parsed.notna().mean()
        )

        duplicates = int(
            valid.duplicated().sum()
        )

        monotonic = bool(
            valid.is_monotonic_increasing
        )

        sampling_minutes = None

        if len(valid) >= 2:
            ordered = (
                valid.sort_values()
                .drop_duplicates()
            )

            diffs = (
                ordered.diff()
                .dropna()
                .dt.total_seconds()
                .div(60)
            )

            positive = diffs[
                diffs > 0
            ]

            if not positive.empty:
                sampling_minutes = round(
                    float(
                        positive.median()
                    ),
                    3,
                )

        return {
            "parse_ratio": round(
                parse_ratio,
                4,
            ),
            "duplicate_timestamps": duplicates,
            "monotonic_increasing": monotonic,
            "inferred_sampling_minutes":
                sampling_minutes,
        }

    def _capability_report(
        self,
        mapping: dict[str, str],
    ) -> dict[str, Any]:
        available = set(mapping)

        required = set(
            self.contract[
                "minimum_training_contract"
            ]["required"]
        )

        subsystem = set(
            self.contract[
                "capabilities"
            ][
                "subsystem_root_cause"
            ]["recommended"]
        )

        environment = set(
            self.contract[
                "capabilities"
            ][
                "environment_context"
            ]["recommended"]
        )

        raw_model_signals = {
            key
            for key in self.signals
            if key != "timestamp"
        }

        return {
            "forecast_training": {
                "available":
                    required.issubset(
                        available
                    ),
                "missing":
                    sorted(
                        required - available
                    ),
            },
            "root_cause": {
                "available_signals":
                    len(
                        subsystem
                        & available
                    ),
                "total_signals":
                    len(subsystem),
                "missing":
                    sorted(
                        subsystem
                        - available
                    ),
            },
            "environment_context": {
                "available_signals":
                    len(
                        environment
                        & available
                    ),
                "total_signals":
                    len(environment),
                "missing":
                    sorted(
                        environment
                        - available
                    ),
            },
            "existing_building59_model": {
                "compatible":
                    raw_model_signals
                    .issubset(
                        available
                    ),
                "missing":
                    sorted(
                        raw_model_signals
                        - available
                    ),
            },
        }

    def _compatibility_score(
        self,
        mapping: dict[str, str],
    ) -> float:
        available = set(mapping)

        required = {
            "timestamp",
            "total_power",
        }

        subsystem = {
            "hvac_N",
            "hvac_S",
            "mels_N",
            "mels_S",
            "lig_S",
        }

        environment = {
            "indoor_temp_avg",
            "outdoor_temp_avg",
            "relative_humidity_set_1",
            "solar_radiation_set_1",
            "dew_point_temperature_set_1d",
        }

        required_score = (
            len(required & available)
            / len(required)
            * 60
        )

        subsystem_score = (
            len(subsystem & available)
            / len(subsystem)
            * 25
        )

        environment_score = (
            len(environment & available)
            / len(environment)
            * 15
        )

        return round(
            required_score
            + subsystem_score
            + environment_score,
            1,
        )

    def inspect_dataframe(
        self,
        df: pd.DataFrame,
    ) -> dict[str, Any]:
        if df.empty:
            raise ValueError(
                "Dataset is empty."
            )

        mapping = self.match_columns(df)

        column_quality = {}

        for canonical, source in mapping.items():
            spec = self.signals[canonical]

            if spec["type"] == "datetime":
                column_quality[canonical] = (
                    self._timestamp_quality(
                        df[source]
                    )
                )
            else:
                column_quality[canonical] = (
                    self._numeric_quality(
                        df[source]
                    )
                )

        required = set(
            self.contract[
                "minimum_training_contract"
            ]["required"]
        )

        missing_required = sorted(
            required - set(mapping)
        )

        score = self._compatibility_score(
            mapping
        )

        if missing_required:
            status = "NO"
        elif score >= 80:
            status = "YES"
        else:
            status = "PARTIAL"

        return {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "source_columns":
                list(map(str, df.columns)),
            "column_mapping": mapping,
            "unmapped_columns": [
                str(column)
                for column in df.columns
                if column
                not in mapping.values()
            ],
            "missing_required":
                missing_required,
            "column_quality":
                column_quality,
            "compatibility_score":
                score,
            "compatibility_status":
                status,
            "capabilities":
                self._capability_report(
                    mapping
                ),
        }

    def inspect_file(
        self,
        path: str | Path,
    ) -> dict[str, Any]:
        df = self.load(path)

        report = self.inspect_dataframe(
            df
        )

        report["source_file"] = str(
            Path(path)
        )

        return report
