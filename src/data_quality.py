from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.onboarding import BuildingDataOnboarding


class BuildingDataQualityGate:
    def __init__(
        self,
        contract_path: str | Path = "config/building_data_contract.json",
        target_sampling_minutes: int = 15,
        min_training_days: int = 14,
        recommended_training_days: int = 30,
    ):
        self.onboarding = BuildingDataOnboarding(
            contract_path=contract_path
        )

        self.target_sampling_minutes = target_sampling_minutes
        self.min_training_days = min_training_days
        self.recommended_training_days = recommended_training_days

    @staticmethod
    def _issue(
        severity: str,
        code: str,
        message: str,
        **details: Any,
    ) -> dict[str, Any]:
        issue = {
            "severity": severity,
            "code": code,
            "message": message,
        }

        if details:
            issue["details"] = details

        return issue

    @staticmethod
    def _final_status(
        issues: list[dict[str, Any]],
    ) -> str:
        severities = {
            issue["severity"]
            for issue in issues
        }

        if "ERROR" in severities:
            return "FAIL"

        if "WARNING" in severities:
            return "WARN"

        return "PASS"

    def evaluate(
        self,
        df: pd.DataFrame,
        *,
        mode: str = "training",
    ) -> dict[str, Any]:
        if mode not in {"training", "inference"}:
            raise ValueError(
                "mode must be 'training' or 'inference'"
            )

        issues: list[dict[str, Any]] = []
        metrics: dict[str, Any] = {}

        if df.empty:
            return {
                "status": "FAIL",
                "mode": mode,
                "metrics": {},
                "issues": [
                    self._issue(
                        "ERROR",
                        "EMPTY_DATASET",
                        "Dataset contains no rows.",
                    )
                ],
                "can_preprocess": False,
                "training_ready": False,
            }

        mapping = self.onboarding.match_columns(df)

        missing_required = sorted(
            {
                "timestamp",
                "total_power",
            }
            - set(mapping)
        )

        if missing_required:
            issues.append(
                self._issue(
                    "ERROR",
                    "MISSING_REQUIRED_SIGNALS",
                    "Required building signals are missing.",
                    missing=missing_required,
                )
            )

        timestamp_source = mapping.get("timestamp")

        if timestamp_source is not None:
            raw_timestamp = df[timestamp_source]

            parsed_timestamp = pd.to_datetime(
                raw_timestamp,
                errors="coerce",
            )

            timestamp_parse_ratio = float(
                parsed_timestamp.notna().mean()
            )

            metrics["timestamp_parse_ratio"] = round(
                timestamp_parse_ratio,
                4,
            )

            if timestamp_parse_ratio < 0.99:
                issues.append(
                    self._issue(
                        "ERROR",
                        "TIMESTAMP_PARSE_FAILURE",
                        "Too many timestamp values could not be parsed.",
                        parse_ratio=round(
                            timestamp_parse_ratio,
                            4,
                        ),
                    )
                )

            elif timestamp_parse_ratio < 0.999:
                issues.append(
                    self._issue(
                        "WARNING",
                        "TIMESTAMP_PARSE_WARNING",
                        "Some timestamp values could not be parsed.",
                        parse_ratio=round(
                            timestamp_parse_ratio,
                            4,
                        ),
                    )
                )

            valid_timestamp = (
                parsed_timestamp
                .dropna()
                .sort_values()
            )

            duplicate_ratio = (
                float(valid_timestamp.duplicated().mean())
                if len(valid_timestamp)
                else 0.0
            )

            metrics["duplicate_timestamp_ratio"] = round(
                duplicate_ratio,
                4,
            )

            if duplicate_ratio > 0.01:
                issues.append(
                    self._issue(
                        "WARNING",
                        "DUPLICATE_TIMESTAMPS",
                        "Duplicate timestamps exceed 1% of valid observations.",
                        ratio=round(
                            duplicate_ratio,
                            4,
                        ),
                    )
                )

            unique_timestamp = (
                valid_timestamp
                .drop_duplicates()
            )

            if len(unique_timestamp) >= 2:
                diffs = (
                    unique_timestamp
                    .diff()
                    .dropna()
                    .dt.total_seconds()
                    .div(60)
                )

                positive_diffs = diffs[
                    diffs > 0
                ]

                if not positive_diffs.empty:
                    sampling = float(
                        positive_diffs.median()
                    )

                    tolerance = max(
                        1.0,
                        sampling * 0.05,
                    )

                    irregular_ratio = float(
                        (
                            np.abs(
                                positive_diffs
                                - sampling
                            )
                            > tolerance
                        ).mean()
                    )

                    metrics[
                        "inferred_sampling_minutes"
                    ] = round(
                        sampling,
                        3,
                    )

                    metrics[
                        "irregular_interval_ratio"
                    ] = round(
                        irregular_ratio,
                        4,
                    )

                    metrics[
                        "max_gap_minutes"
                    ] = round(
                        float(
                            positive_diffs.max()
                        ),
                        3,
                    )

                    if (
                        sampling
                        > self.target_sampling_minutes * 1.05
                    ):
                        issues.append(
                            self._issue(
                                "ERROR",
                                "SOURCE_RESOLUTION_TOO_LOW",
                                (
                                    "Source sampling interval is coarser "
                                    "than the current EcoTwin target."
                                ),
                                inferred_minutes=round(
                                    sampling,
                                    3,
                                ),
                                target_minutes=(
                                    self.target_sampling_minutes
                                ),
                            )
                        )

                    if irregular_ratio > 0.10:
                        issues.append(
                            self._issue(
                                "WARNING",
                                "IRREGULAR_SAMPLING",
                                (
                                    "More than 10% of timestamp intervals "
                                    "deviate from the inferred sampling rate."
                                ),
                                ratio=round(
                                    irregular_ratio,
                                    4,
                                ),
                            )
                        )

                duration_days = float(
                    (
                        unique_timestamp.max()
                        - unique_timestamp.min()
                    ).total_seconds()
                    / 86400
                )

                metrics["duration_days"] = round(
                    duration_days,
                    2,
                )

                if mode == "training":
                    if duration_days < self.min_training_days:
                        issues.append(
                            self._issue(
                                "ERROR",
                                "INSUFFICIENT_HISTORY",
                                (
                                    "Dataset history is too short for the "
                                    "current training pipeline."
                                ),
                                available_days=round(
                                    duration_days,
                                    2,
                                ),
                                minimum_days=(
                                    self.min_training_days
                                ),
                            )
                        )

                    elif (
                        duration_days
                        < self.recommended_training_days
                    ):
                        issues.append(
                            self._issue(
                                "WARNING",
                                "LIMITED_HISTORY",
                                (
                                    "Training is possible, but additional "
                                    "historical coverage is recommended."
                                ),
                                available_days=round(
                                    duration_days,
                                    2,
                                ),
                                recommended_days=(
                                    self.recommended_training_days
                                ),
                            )
                        )

        total_power_source = mapping.get(
            "total_power"
        )

        if total_power_source is not None:
            raw_power = df[
                total_power_source
            ]

            numeric_power = pd.to_numeric(
                raw_power,
                errors="coerce",
            )

            original_non_null = int(
                raw_power.notna().sum()
            )

            numeric_non_null = int(
                numeric_power.notna().sum()
            )

            parse_ratio = (
                numeric_non_null
                / original_non_null
                if original_non_null
                else 0.0
            )

            missing_ratio = float(
                numeric_power.isna().mean()
            )

            values = numeric_power.to_numpy(
                dtype=float
            )

            infinite_ratio = float(
                np.isinf(values).mean()
            )

            finite_power = numeric_power[
                np.isfinite(values)
            ]

            negative_ratio = (
                float(
                    (
                        finite_power < 0
                    ).mean()
                )
                if len(finite_power)
                else 0.0
            )

            metrics["total_power"] = {
                "parse_ratio": round(
                    parse_ratio,
                    4,
                ),
                "missing_ratio": round(
                    missing_ratio,
                    4,
                ),
                "infinite_ratio": round(
                    infinite_ratio,
                    4,
                ),
                "negative_ratio": round(
                    negative_ratio,
                    4,
                ),
            }

            if parse_ratio < 0.99:
                issues.append(
                    self._issue(
                        "ERROR",
                        "POWER_PARSE_FAILURE",
                        "Too many total-power values are non-numeric.",
                        parse_ratio=round(
                            parse_ratio,
                            4,
                        ),
                    )
                )

            if missing_ratio > 0.05:
                issues.append(
                    self._issue(
                        "ERROR",
                        "POWER_MISSINGNESS_HIGH",
                        "More than 5% of total-power values are missing.",
                        missing_ratio=round(
                            missing_ratio,
                            4,
                        ),
                    )
                )

            elif missing_ratio > 0.01:
                issues.append(
                    self._issue(
                        "WARNING",
                        "POWER_MISSINGNESS",
                        "Total-power missingness exceeds 1%.",
                        missing_ratio=round(
                            missing_ratio,
                            4,
                        ),
                    )
                )

            if infinite_ratio > 0:
                issues.append(
                    self._issue(
                        "ERROR",
                        "INFINITE_POWER_VALUES",
                        "Infinite total-power values were detected.",
                        ratio=round(
                            infinite_ratio,
                            4,
                        ),
                    )
                )

            if negative_ratio > 0.001:
                issues.append(
                    self._issue(
                        "WARNING",
                        "NEGATIVE_POWER_VALUES",
                        (
                            "Negative power values were detected. "
                            "These may be valid for net-export meters "
                            "and should be reviewed before training."
                        ),
                        ratio=round(
                            negative_ratio,
                            4,
                        ),
                    )
                )

            if (
                len(finite_power) > 1
                and float(finite_power.std()) == 0.0
            ):
                issues.append(
                    self._issue(
                        "ERROR",
                        "CONSTANT_POWER_SIGNAL",
                        "Total-power signal has no variation.",
                    )
                )

        for canonical, source in mapping.items():
            if canonical in {
                "timestamp",
                "total_power",
            }:
                continue

            spec = self.onboarding.signals[
                canonical
            ]

            if spec["type"] != "numeric":
                continue

            converted = pd.to_numeric(
                df[source],
                errors="coerce",
            )

            missing_ratio = float(
                converted.isna().mean()
            )

            if missing_ratio > 0.20:
                issues.append(
                    self._issue(
                        "WARNING",
                        "OPTIONAL_SIGNAL_MISSINGNESS",
                        (
                            f"Optional signal '{canonical}' "
                            "contains more than 20% missing values."
                        ),
                        signal=canonical,
                        missing_ratio=round(
                            missing_ratio,
                            4,
                        ),
                    )
                )

        status = self._final_status(
            issues
        )

        return {
            "status": status,
            "mode": mode,
            "rows": int(len(df)),
            "column_mapping": mapping,
            "metrics": metrics,
            "issues": issues,
            "can_preprocess": status != "FAIL",
            "training_ready": (
                mode == "training"
                and status != "FAIL"
            ),
        }
