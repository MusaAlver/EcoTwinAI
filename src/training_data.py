from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class SequenceDataset:
    X: np.ndarray
    y: np.ndarray
    forecast_time: np.ndarray
    outcome_time: np.ndarray

    def __len__(self) -> int:
        return len(self.X)


class TimeSeriesDatasetBuilder:
    def __init__(
        self,
        lookback_steps: int = 16,
        horizon_minutes: int = 30,
        sampling_minutes: int = 15,
        train_ratio: float = 0.70,
        validation_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ):
        self.lookback_steps = int(lookback_steps)
        self.horizon_minutes = int(horizon_minutes)
        self.sampling_minutes = int(sampling_minutes)

        self.train_ratio = float(train_ratio)
        self.validation_ratio = float(validation_ratio)
        self.test_ratio = float(test_ratio)

        self._validate_config()

    def _validate_config(self) -> None:
        if self.lookback_steps < 1:
            raise ValueError(
                "lookback_steps must be positive"
            )

        if self.sampling_minutes < 1:
            raise ValueError(
                "sampling_minutes must be positive"
            )

        if self.horizon_minutes < self.sampling_minutes:
            raise ValueError(
                "horizon_minutes must be at least one sampling interval"
            )

        if (
            self.horizon_minutes
            % self.sampling_minutes
            != 0
        ):
            raise ValueError(
                "horizon_minutes must be divisible by sampling_minutes"
            )

        ratio_sum = (
            self.train_ratio
            + self.validation_ratio
            + self.test_ratio
        )

        if not np.isclose(
            ratio_sum,
            1.0,
        ):
            raise ValueError(
                "train, validation and test ratios must sum to 1"
            )

        if min(
            self.train_ratio,
            self.validation_ratio,
            self.test_ratio,
        ) <= 0:
            raise ValueError(
                "split ratios must be greater than zero"
            )

    @property
    def horizon_steps(self) -> int:
        return (
            self.horizon_minutes
            // self.sampling_minutes
        )

    def prepare_frame(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        if "timestamp" not in df.columns:
            raise ValueError(
                "timestamp column is required"
            )

        if df.empty:
            raise ValueError(
                "Dataset is empty"
            )

        frame = df.copy()

        frame["timestamp"] = pd.to_datetime(
            frame["timestamp"],
            errors="coerce",
        )

        if frame["timestamp"].isna().any():
            raise ValueError(
                "Dataset contains invalid timestamps"
            )

        if frame["timestamp"].duplicated().any():
            raise ValueError(
                "Dataset contains duplicate timestamps"
            )

        frame = (
            frame
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        return frame

    def split(
        self,
        df: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        frame = self.prepare_frame(df)

        n = len(frame)

        train_end = int(
            np.floor(
                n * self.train_ratio
            )
        )

        validation_end = (
            train_end
            + int(
                np.floor(
                    n * self.validation_ratio
                )
            )
        )

        minimum_rows = (
            self.lookback_steps
            + self.horizon_steps
        )

        train = (
            frame.iloc[:train_end]
            .copy()
            .reset_index(drop=True)
        )

        validation = (
            frame.iloc[
                train_end:validation_end
            ]
            .copy()
            .reset_index(drop=True)
        )

        test = (
            frame.iloc[
                validation_end:
            ]
            .copy()
            .reset_index(drop=True)
        )

        splits = {
            "train": train,
            "validation": validation,
            "test": test,
        }

        for name, part in splits.items():
            if len(part) < minimum_rows:
                raise ValueError(
                    f"{name} split is too short: "
                    f"{len(part)} rows; "
                    f"minimum {minimum_rows}"
                )

        return splits

    def _is_contiguous(
        self,
        timestamps: pd.Series,
    ) -> bool:
        if len(timestamps) < 2:
            return True

        expected_seconds = (
            self.sampling_minutes
            * 60
        )

        actual = (
            timestamps
            .diff()
            .dropna()
            .dt.total_seconds()
            .to_numpy()
        )

        return bool(
            np.all(
                actual
                == expected_seconds
            )
        )

    def build_sequences(
        self,
        frame: pd.DataFrame,
        feature_columns: list[str],
        target_column: str = "total_power",
    ) -> SequenceDataset:
        required = {
            "timestamp",
            target_column,
            *feature_columns,
        }

        missing = sorted(
            required
            - set(frame.columns)
        )

        if missing:
            raise ValueError(
                "Missing sequence columns: "
                + ", ".join(missing)
            )

        frame = self.prepare_frame(frame)

        X: list[np.ndarray] = []
        y: list[float] = []
        forecast_times: list[np.datetime64] = []
        outcome_times: list[np.datetime64] = []

        last_input_index = (
            len(frame)
            - self.horizon_steps
        )

        for stop in range(
            self.lookback_steps,
            last_input_index + 1,
        ):
            start = (
                stop
                - self.lookback_steps
            )

            target_index = (
                stop - 1
                + self.horizon_steps
            )

            segment = frame.iloc[
                start:
                target_index + 1
            ]

            if not self._is_contiguous(
                segment["timestamp"]
            ):
                continue

            input_frame = frame.iloc[
                start:stop
            ][feature_columns]

            target_value = frame.iloc[
                target_index
            ][target_column]

            if input_frame.isna().any().any():
                continue

            if pd.isna(target_value):
                continue

            values = input_frame.to_numpy(
                dtype=np.float32
            )

            if not np.isfinite(values).all():
                continue

            target_value = float(
                target_value
            )

            if not np.isfinite(
                target_value
            ):
                continue

            X.append(values)
            y.append(target_value)

            forecast_times.append(
                np.datetime64(
                    frame.iloc[
                        stop - 1
                    ]["timestamp"]
                )
            )

            outcome_times.append(
                np.datetime64(
                    frame.iloc[
                        target_index
                    ]["timestamp"]
                )
            )

        if not X:
            raise ValueError(
                "No valid sequences could be created"
            )

        return SequenceDataset(
            X=np.stack(X),
            y=np.asarray(
                y,
                dtype=np.float32,
            ).reshape(-1, 1),
            forecast_time=np.asarray(
                forecast_times
            ),
            outcome_time=np.asarray(
                outcome_times
            ),
        )

    def build(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        target_column: str = "total_power",
    ) -> dict[str, Any]:
        splits = self.split(df)

        datasets = {
            name: self.build_sequences(
                frame,
                feature_columns=feature_columns,
                target_column=target_column,
            )
            for name, frame in splits.items()
        }

        return {
            "splits": splits,
            "datasets": datasets,
            "metadata": {
                "lookback_steps":
                    self.lookback_steps,
                "horizon_minutes":
                    self.horizon_minutes,
                "horizon_steps":
                    self.horizon_steps,
                "sampling_minutes":
                    self.sampling_minutes,
                "feature_columns":
                    list(feature_columns),
                "target_column":
                    target_column,
                "split_rows": {
                    name: len(frame)
                    for name, frame
                    in splits.items()
                },
                "sequence_counts": {
                    name: len(dataset)
                    for name, dataset
                    in datasets.items()
                },
            },
        }
