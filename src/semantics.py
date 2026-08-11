from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SemanticNormalizationResult:
    values_kw: pd.Series
    report: dict[str, Any]


class SignalSemanticsValidator:
    POWER_TO_KW = {
        "W": 0.001,
        "kW": 1.0,
        "MW": 1000.0,
    }

    ENERGY_TO_KWH = {
        "Wh": 0.001,
        "kWh": 1.0,
        "MWh": 1000.0,
    }

    METER_SEMANTICS = {
        "interval_energy",
        "cumulative_energy",
    }

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

        self._validate_contract()

    def _validate_contract(self) -> None:
        safety = self.contract.get(
            "semantic_safety"
        )

        if not safety:
            raise ValueError(
                "Contract does not define semantic_safety"
            )

        power = (
            self.contract
            .get("canonical_signals", {})
            .get("total_power")
        )

        if not power:
            raise ValueError(
                "Contract does not define total_power"
            )

        if power.get("unit") != "kW":
            raise ValueError(
                "total_power canonical unit must be kW"
            )

    @staticmethod
    def _numeric_series(
        values: pd.Series,
    ) -> pd.Series:
        numeric = pd.to_numeric(
            values,
            errors="coerce",
        ).astype(float)

        invalid = (
            values.notna()
            & numeric.isna()
        )

        if invalid.any():
            raise ValueError(
                "Signal contains non-numeric values"
            )

        finite = numeric.dropna()

        if not np.isfinite(
            finite.to_numpy()
        ).all():
            raise ValueError(
                "Signal contains infinite values"
            )

        return numeric

    @classmethod
    def _validate_unit(
        cls,
        unit: str | None,
    ) -> str:
        if unit is None:
            raise ValueError(
                "Explicit unit is required"
            )

        unit = str(unit).strip()

        supported = {
            *cls.POWER_TO_KW,
            *cls.ENERGY_TO_KWH,
        }

        if unit not in supported:
            raise ValueError(
                f"Unsupported unit: {unit}"
            )

        return unit

    @staticmethod
    def _validate_interval(
        interval_minutes: float | None,
    ) -> float:
        if interval_minutes is None:
            raise ValueError(
                "interval_minutes is required "
                "for energy-to-power conversion"
            )

        interval = float(
            interval_minutes
        )

        if (
            not np.isfinite(interval)
            or interval <= 0
        ):
            raise ValueError(
                "interval_minutes must be positive"
            )

        return interval

    def normalize_total_power(
        self,
        values: pd.Series,
        *,
        unit: str | None,
        meter_semantics: str | None = None,
        interval_minutes: float | None = None,
    ) -> SemanticNormalizationResult:
        unit = self._validate_unit(
            unit
        )

        numeric = self._numeric_series(
            values
        )

        if unit in self.POWER_TO_KW:
            if meter_semantics is not None:
                raise ValueError(
                    "meter_semantics must not be supplied "
                    "for instantaneous power"
                )

            factor = self.POWER_TO_KW[
                unit
            ]

            normalized = (
                numeric
                * factor
            )

            return SemanticNormalizationResult(
                values_kw=normalized,
                report={
                    "source_quantity": "power",
                    "source_unit": unit,
                    "canonical_unit": "kW",
                    "conversion": "scale",
                    "scale_factor": factor,
                    "meter_semantics": None,
                    "interval_minutes": None,
                },
            )

        if meter_semantics is None:
            raise ValueError(
                "meter_semantics is required "
                "for energy input"
            )

        if (
            meter_semantics
            not in self.METER_SEMANTICS
        ):
            raise ValueError(
                "Unsupported meter_semantics: "
                f"{meter_semantics}"
            )

        interval = self._validate_interval(
            interval_minutes
        )

        energy_kwh = (
            numeric
            * self.ENERGY_TO_KWH[
                unit
            ]
        )

        if meter_semantics == "interval_energy":
            normalized = (
                energy_kwh
                / (interval / 60.0)
            )

            conversion = (
                "interval_energy_to_average_power"
            )

        else:
            delta_kwh = (
                energy_kwh.diff()
            )

            negative_delta = (
                delta_kwh < 0
            )

            if negative_delta.any():
                raise ValueError(
                    "Cumulative energy decreased; "
                    "possible meter reset or rollover"
                )

            normalized = (
                delta_kwh
                / (interval / 60.0)
            )

            conversion = (
                "cumulative_energy_delta_to_average_power"
            )

        return SemanticNormalizationResult(
            values_kw=normalized,
            report={
                "source_quantity": "energy",
                "source_unit": unit,
                "canonical_unit": "kW",
                "conversion": conversion,
                "scale_to_kwh": self.ENERGY_TO_KWH[
                    unit
                ],
                "meter_semantics": meter_semantics,
                "interval_minutes": interval,
                "introduced_nan_count": int(
                    normalized.isna().sum()
                    - numeric.isna().sum()
                ),
            },
        )
