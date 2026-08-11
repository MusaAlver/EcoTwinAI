from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.onboarding import BuildingDataOnboarding
from src.preprocessing import BuildingDataPreprocessor
from src.semantics import SignalSemanticsValidator


class BuildingDataIntake:
    def __init__(
        self,
        contract_path: str | Path = "config/building_data_contract.json",
        sampling_minutes: int = 15,
    ):
        self.contract_path = Path(contract_path)
        self.sampling_minutes = int(sampling_minutes)

        self.onboarding = BuildingDataOnboarding(
            contract_path=self.contract_path
        )

        self.semantics = SignalSemanticsValidator(
            contract_path=self.contract_path
        )

        self.preprocessor = BuildingDataPreprocessor(
            contract_path=self.contract_path,
            sampling_minutes=self.sampling_minutes,
        )

    @staticmethod
    def _normalize_name(name: str) -> str:
        normalized = str(name).strip().lower()
        normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
        return normalized.strip("_")

    def _resolve_power_source(
        self,
        df: pd.DataFrame,
        explicit_source: str | None,
    ) -> tuple[str, bool]:
        if explicit_source is not None:
            if explicit_source not in df.columns:
                raise ValueError(
                    f"Configured total power column does not exist: "
                    f"{explicit_source}"
                )

            return explicit_source, True

        mapping = self.onboarding.match_columns(df)

        source = mapping.get("total_power")

        if source is None:
            raise ValueError(
                "No safe total_power mapping was found. "
                "Provide total_power_column explicitly."
            )

        return source, False

    def _resolve_power_unit(
        self,
        source: str,
        explicit_unit: str | None,
    ) -> tuple[str, bool]:
        if explicit_unit is not None:
            return explicit_unit, True

        normalized = self._normalize_name(source)

        safe_unit_names = {
            "total_power": "kW",
            "total_kw": "kW",
        }

        inferred = safe_unit_names.get(normalized)

        if inferred is None:
            raise ValueError(
                f"Explicit unit is required for power column: {source}"
            )

        return inferred, False

    def prepare(
        self,
        df: pd.DataFrame,
        *,
        total_power_column: str | None = None,
        power_unit: str | None = None,
        meter_semantics: str | None = None,
        interval_minutes: float | None = None,
        resample: bool = True,
        add_features: bool = True,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        if df.empty:
            raise ValueError("Dataset is empty")

        source, source_confirmed = self._resolve_power_source(
            df,
            total_power_column,
        )

        unit, unit_confirmed = self._resolve_power_unit(
            source,
            power_unit,
        )

        normalized = self.semantics.normalize_total_power(
            df[source],
            unit=unit,
            meter_semantics=meter_semantics,
            interval_minutes=interval_minutes,
        )

        working = df.copy()

        if (
            source != "total_power"
            and "total_power" in working.columns
        ):
            raise ValueError(
                "Dataset contains total_power while another "
                "power source was explicitly selected"
            )

        working["total_power"] = normalized.values_kw

        prepared, preprocessing_report = (
            self.preprocessor.prepare(
                working,
                resample=resample,
                add_features=add_features,
            )
        )

        semantic_report = {
            "source_column": source,
            "source_column_confirmed": source_confirmed,
            "unit_confirmed": unit_confirmed,
            **normalized.report,
        }

        report = {
            "semantic_normalization": semantic_report,
            "preprocessing": preprocessing_report,
            "canonical_unit": "kW",
            "rows_input": int(len(df)),
            "rows_output": int(len(prepared)),
        }

        return prepared, report
