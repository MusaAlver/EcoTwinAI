
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.baselines import BaselineEvaluator
from src.model_registry import ModelRegistry
from src.training_data import SequenceDataset, TimeSeriesDatasetBuilder


@dataclass(frozen=True)
class EvaluationResult:
    samples: int
    mae: float
    rmse: float
    within_5kw: float
    within_10kw: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "samples": self.samples,
            "mae": self.mae,
            "rmse": self.rmse,
            "within_5kw": self.within_5kw,
            "within_10kw": self.within_10kw,
        }


class TrainingOrchestrator:
    def __init__(
        self,
        *,
        dataset_builder: TimeSeriesDatasetBuilder | None = None,
        baseline_evaluator: BaselineEvaluator | None = None,
        registry: ModelRegistry | None = None,
        minimum_improvement: float = 0.01,
    ):
        self.dataset_builder = (
            dataset_builder
            or TimeSeriesDatasetBuilder()
        )

        self.baseline_evaluator = (
            baseline_evaluator
            or BaselineEvaluator(
                sampling_minutes=self.dataset_builder.sampling_minutes,
                horizon_minutes=self.dataset_builder.horizon_minutes,
            )
        )

        self.registry = (
            registry
            or ModelRegistry()
        )

        self.minimum_improvement = float(
            minimum_improvement
        )

        if self.minimum_improvement < 0:
            raise ValueError(
                "minimum_improvement cannot be negative"
            )

    @staticmethod
    def evaluate_predictions(
        actual: np.ndarray,
        prediction: np.ndarray,
    ) -> EvaluationResult:
        actual = np.asarray(
            actual,
            dtype=float,
        ).reshape(-1)

        prediction = np.asarray(
            prediction,
            dtype=float,
        ).reshape(-1)

        if actual.shape != prediction.shape:
            raise ValueError(
                "actual and prediction must have identical shape"
            )

        mask = (
            np.isfinite(actual)
            & np.isfinite(prediction)
        )

        actual = actual[mask]
        prediction = prediction[mask]

        if len(actual) == 0:
            raise ValueError(
                "No valid prediction samples"
            )

        error = actual - prediction
        abs_error = np.abs(error)

        return EvaluationResult(
            samples=int(len(actual)),
            mae=float(
                np.mean(abs_error)
            ),
            rmse=float(
                np.sqrt(
                    np.mean(
                        error ** 2
                    )
                )
            ),
            within_5kw=float(
                np.mean(
                    abs_error <= 5.0
                )
            ),
            within_10kw=float(
                np.mean(
                    abs_error <= 10.0
                )
            ),
        )

    @staticmethod
    def _timestamp_power_map(
        frame: pd.DataFrame,
        target_column: str,
    ) -> dict[pd.Timestamp, float]:
        if target_column not in frame.columns:
            raise ValueError(
                f"Missing target column: {target_column}"
            )

        timestamps = pd.to_datetime(
            frame["timestamp"],
            errors="coerce",
        )

        power = pd.to_numeric(
            frame[target_column],
            errors="coerce",
        )

        return {
            pd.Timestamp(timestamp): float(value)
            for timestamp, value
            in zip(timestamps, power)
            if pd.notna(timestamp)
            and pd.notna(value)
            and np.isfinite(value)
        }

    def aligned_baselines(
        self,
        *,
        full_frame: pd.DataFrame,
        dataset: SequenceDataset,
        target_column: str = "total_power",
    ) -> dict[str, dict[str, Any]]:
        power_map = self._timestamp_power_map(
            full_frame,
            target_column,
        )

        actual = np.asarray(
            dataset.y,
            dtype=float,
        ).reshape(-1)

        forecast_times = pd.to_datetime(
            dataset.forecast_time
        )

        outcome_times = pd.to_datetime(
            dataset.outcome_time
        )

        sample_count = len(actual)

        if (
            len(forecast_times) != sample_count
            or len(outcome_times) != sample_count
        ):
            raise ValueError(
                "Dataset timestamps and targets are not aligned"
            )

        strategies = {
            "persistence": None,
            "daily_seasonal": pd.Timedelta(days=1),
            "weekly_seasonal": pd.Timedelta(days=7),
        }

        results: dict[str, dict[str, Any]] = {}

        for name, offset in strategies.items():
            predictions = np.full(
                sample_count,
                np.nan,
                dtype=float,
            )

            for index, outcome_time in enumerate(
                outcome_times
            ):
                if name == "persistence":
                    reference_time = forecast_times[index]
                else:
                    reference_time = outcome_time - offset

                reference_value = power_map.get(
                    pd.Timestamp(reference_time)
                )

                if (
                    reference_value is not None
                    and np.isfinite(reference_value)
                ):
                    predictions[index] = reference_value

            valid = (
                np.isfinite(actual)
                & np.isfinite(predictions)
            )

            available_samples = int(
                np.sum(valid)
            )

            if available_samples != sample_count:
                continue

            metrics = self.evaluate_predictions(
                actual,
                predictions,
            ).to_dict()

            metrics["coverage"] = 1.0
            metrics["available_samples"] = (
                available_samples
            )

            results[name] = metrics

        if not results:
            raise ValueError(
                "No baseline has full coverage "
                "for the candidate evaluation set"
            )

        return results

    def evaluate_candidate(
        self,
        *,
        candidate_prediction: np.ndarray,
        dataset: SequenceDataset,
        full_frame: pd.DataFrame,
        target_column: str = "total_power",
        metric: str = "mae",
    ) -> dict[str, Any]:
        candidate = self.evaluate_predictions(
            dataset.y,
            candidate_prediction,
        )

        baselines = self.aligned_baselines(
            full_frame=full_frame,
            dataset=dataset,
            target_column=target_column,
        )

        champion = (
            self.baseline_evaluator.select_best(
                baselines,
                metric=metric,
            )
        )

        decision = (
            self.baseline_evaluator.compare_candidate(
                candidate.to_dict(),
                champion,
                metric=metric,
                minimum_improvement=self.minimum_improvement,
            )
        )

        return {
            "candidate": candidate.to_dict(),
            "baselines": baselines,
            "baseline_champion": champion,
            "promotion_decision": decision,
        }

    @staticmethod
    def predict(
        predictor: Callable[
            [np.ndarray],
            np.ndarray,
        ],
        dataset: SequenceDataset,
    ) -> np.ndarray:
        prediction = predictor(
            dataset.X
        )

        prediction = np.asarray(
            prediction,
            dtype=float,
        ).reshape(-1, 1)

        if len(prediction) != len(dataset):
            raise ValueError(
                "Predictor returned an unexpected number of samples"
            )

        return prediction

    def prepare_training_data(
        self,
        df: pd.DataFrame,
        *,
        feature_columns: list[str],
        target_column: str = "total_power",
    ) -> dict[str, Any]:
        return self.dataset_builder.build(
            df,
            feature_columns=feature_columns,
            target_column=target_column,
        )

    def evaluate_validation(
        self,
        *,
        prepared: dict[str, Any],
        predictor: Callable[
            [np.ndarray],
            np.ndarray,
        ],
        target_column: str = "total_power",
    ) -> dict[str, Any]:
        validation = (
            prepared["datasets"]
            ["validation"]
        )

        prediction = self.predict(
            predictor,
            validation,
        )

        full_frame = pd.concat(
            [
                prepared["splits"]["train"],
                prepared["splits"]["validation"],
            ],
            ignore_index=True,
        )

        return self.evaluate_candidate(
            candidate_prediction=prediction,
            dataset=validation,
            full_frame=full_frame,
            target_column=target_column,
        )

    def evaluate_test(
        self,
        *,
        prepared: dict[str, Any],
        predictor: Callable[
            [np.ndarray],
            np.ndarray,
        ],
    ) -> dict[str, Any]:
        test = (
            prepared["datasets"]["test"]
        )

        prediction = self.predict(
            predictor,
            test,
        )

        metrics = self.evaluate_predictions(
            test.y,
            prediction,
        )

        return metrics.to_dict()
