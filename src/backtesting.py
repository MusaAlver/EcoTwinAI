
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.training_data import SequenceDataset, TimeSeriesDatasetBuilder
from src.training_orchestrator import TrainingOrchestrator


@dataclass(frozen=True)
class WalkForwardFold:
    fold: int
    train_rows: int
    validation_rows: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    train_sequences: int
    validation_sequences: int


class WalkForwardBacktester:
    def __init__(
        self,
        *,
        dataset_builder: TimeSeriesDatasetBuilder | None = None,
        min_train_rows: int = 1500,
        validation_rows: int = 288,
        step_rows: int | None = None,
        max_folds: int = 5,
        minimum_improvement: float = 0.01,
    ):
        self.dataset_builder = (
            dataset_builder
            or TimeSeriesDatasetBuilder()
        )

        self.min_train_rows = int(min_train_rows)
        self.validation_rows = int(validation_rows)
        self.step_rows = int(
            step_rows
            if step_rows is not None
            else validation_rows
        )
        self.max_folds = int(max_folds)

        self.orchestrator = TrainingOrchestrator(
            dataset_builder=self.dataset_builder,
            minimum_improvement=minimum_improvement,
        )

        self._validate_config()

    def _validate_config(self) -> None:
        minimum_sequence_rows = (
            self.dataset_builder.lookback_steps
            + self.dataset_builder.horizon_steps
        )

        if self.min_train_rows < minimum_sequence_rows:
            raise ValueError(
                "min_train_rows is too small "
                "to build training sequences"
            )

        if self.validation_rows < 1:
            raise ValueError(
                "validation_rows must be positive"
            )

        if self.step_rows < self.validation_rows:
            raise ValueError(
                "step_rows must be at least validation_rows "
                "to prevent overlapping validation windows"
            )

        if self.max_folds < 1:
            raise ValueError(
                "max_folds must be positive"
            )

    @staticmethod
    def _subset_dataset(
        dataset: SequenceDataset,
        mask: np.ndarray,
    ) -> SequenceDataset:
        mask = np.asarray(
            mask,
            dtype=bool,
        )

        return SequenceDataset(
            X=dataset.X[mask],
            y=dataset.y[mask],
            forecast_time=dataset.forecast_time[mask],
            outcome_time=dataset.outcome_time[mask],
        )

    def _boundaries(
        self,
        rows: int,
    ) -> list[tuple[int, int]]:
        boundaries = []

        train_end = self.min_train_rows

        while (
            train_end + self.validation_rows
            <= rows
            and len(boundaries) < self.max_folds
        ):
            validation_end = (
                train_end
                + self.validation_rows
            )

            boundaries.append(
                (
                    train_end,
                    validation_end,
                )
            )

            train_end += self.step_rows

        if not boundaries:
            raise ValueError(
                "Dataset is too short for walk-forward backtesting"
            )

        return boundaries

    def _validation_dataset(
        self,
        frame: pd.DataFrame,
        *,
        train_end: int,
        validation_end: int,
        feature_columns: list[str],
        target_column: str,
    ) -> SequenceDataset:
        history_needed = (
            self.dataset_builder.lookback_steps
            + self.dataset_builder.horizon_steps
            - 1
        )

        context_start = max(
            0,
            train_end - history_needed,
        )

        context = (
            frame.iloc[
                context_start:validation_end
            ]
            .copy()
            .reset_index(drop=True)
        )

        dataset = (
            self.dataset_builder
            .build_sequences(
                context,
                feature_columns=feature_columns,
                target_column=target_column,
            )
        )

        validation_start_time = pd.Timestamp(
            frame.iloc[
                train_end
            ]["timestamp"]
        )

        validation_end_time = pd.Timestamp(
            frame.iloc[
                validation_end - 1
            ]["timestamp"]
        )

        outcome_time = pd.to_datetime(
            dataset.outcome_time
        )

        mask = (
            (outcome_time >= validation_start_time)
            & (outcome_time <= validation_end_time)
        )

        if not np.any(mask):
            raise ValueError(
                "No validation sequences were created for fold"
            )

        return self._subset_dataset(
            dataset,
            mask,
        )

    def build_folds(
        self,
        frame: pd.DataFrame,
        *,
        feature_columns: list[str],
        target_column: str = "total_power",
    ) -> list[
        tuple[
            WalkForwardFold,
            SequenceDataset,
            SequenceDataset,
            pd.DataFrame,
        ]
    ]:
        frame = self.dataset_builder.prepare_frame(
            frame
        )

        folds = []

        for fold_index, (
            train_end,
            validation_end,
        ) in enumerate(
            self._boundaries(len(frame)),
            start=1,
        ):
            train_frame = (
                frame.iloc[:train_end]
                .copy()
                .reset_index(drop=True)
            )

            train_dataset = (
                self.dataset_builder
                .build_sequences(
                    train_frame,
                    feature_columns=feature_columns,
                    target_column=target_column,
                )
            )

            validation_dataset = (
                self._validation_dataset(
                    frame,
                    train_end=train_end,
                    validation_end=validation_end,
                    feature_columns=feature_columns,
                    target_column=target_column,
                )
            )

            validation_start = pd.Timestamp(
                frame.iloc[
                    train_end
                ]["timestamp"]
            )

            if pd.Timestamp(
                train_dataset.outcome_time[-1]
            ) >= validation_start:
                raise RuntimeError(
                    "Training outcomes overlap validation period"
                )

            fold = WalkForwardFold(
                fold=fold_index,
                train_rows=len(train_frame),
                validation_rows=(
                    validation_end
                    - train_end
                ),
                train_start=pd.Timestamp(
                    train_frame.iloc[0]["timestamp"]
                ),
                train_end=pd.Timestamp(
                    train_frame.iloc[-1]["timestamp"]
                ),
                validation_start=validation_start,
                validation_end=pd.Timestamp(
                    frame.iloc[
                        validation_end - 1
                    ]["timestamp"]
                ),
                train_sequences=len(
                    train_dataset
                ),
                validation_sequences=len(
                    validation_dataset
                ),
            )

            history = (
                frame.iloc[:validation_end]
                .copy()
                .reset_index(drop=True)
            )

            folds.append(
                (
                    fold,
                    train_dataset,
                    validation_dataset,
                    history,
                )
            )

        return folds

    def run(
        self,
        frame: pd.DataFrame,
        *,
        feature_columns: list[str],
        fit_predict: Callable[
            [
                SequenceDataset,
                SequenceDataset,
                dict[str, Any],
            ],
            np.ndarray,
        ],
        target_column: str = "total_power",
    ) -> dict[str, Any]:
        folds = self.build_folds(
            frame,
            feature_columns=feature_columns,
            target_column=target_column,
        )

        fold_results = []
        all_actual = []
        all_prediction = []

        for (
            fold,
            train_dataset,
            validation_dataset,
            history,
        ) in folds:
            context = {
                "fold": fold.fold,
                "feature_columns": list(
                    feature_columns
                ),
                "target_column": target_column,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "validation_start":
                    fold.validation_start,
                "validation_end":
                    fold.validation_end,
            }

            prediction = np.asarray(
                fit_predict(
                    train_dataset,
                    validation_dataset,
                    context,
                ),
                dtype=float,
            ).reshape(-1, 1)

            if len(prediction) != len(
                validation_dataset
            ):
                raise ValueError(
                    "fit_predict returned an unexpected "
                    "number of validation predictions"
                )

            evaluation = (
                self.orchestrator
                .evaluate_candidate(
                    candidate_prediction=prediction,
                    dataset=validation_dataset,
                    full_frame=history,
                    target_column=target_column,
                )
            )

            fold_results.append(
                {
                    "fold": fold.fold,
                    "train_rows":
                        fold.train_rows,
                    "validation_rows":
                        fold.validation_rows,
                    "train_sequences":
                        fold.train_sequences,
                    "validation_sequences":
                        fold.validation_sequences,
                    "train_start":
                        fold.train_start.isoformat(),
                    "train_end":
                        fold.train_end.isoformat(),
                    "validation_start":
                        fold.validation_start.isoformat(),
                    "validation_end":
                        fold.validation_end.isoformat(),
                    **evaluation,
                }
            )

            all_actual.append(
                validation_dataset.y.reshape(-1)
            )

            all_prediction.append(
                prediction.reshape(-1)
            )

        aggregate = (
            self.orchestrator
            .evaluate_predictions(
                np.concatenate(all_actual),
                np.concatenate(all_prediction),
            )
            .to_dict()
        )

        accepted_folds = sum(
            result[
                "promotion_decision"
            ]["accepted"]
            for result in fold_results
        )

        return {
            "folds": fold_results,
            "summary": {
                "fold_count": len(
                    fold_results
                ),
                "accepted_folds":
                    int(accepted_folds),
                "rejected_folds":
                    int(
                        len(fold_results)
                        - accepted_folds
                    ),
                "promotion_rate":
                    float(
                        accepted_folds
                        / len(fold_results)
                    ),
                "aggregate_candidate":
                    aggregate,
                "validation_windows_overlap":
                    False,
            },
        }
