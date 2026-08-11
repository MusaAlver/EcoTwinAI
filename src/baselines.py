from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BaselineResult:
    name: str
    samples: int
    mae: float
    rmse: float
    within_5kw: float
    within_10kw: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "samples": self.samples,
            "mae": self.mae,
            "rmse": self.rmse,
            "within_5kw": self.within_5kw,
            "within_10kw": self.within_10kw,
        }


class BaselineEvaluator:
    def __init__(
        self,
        sampling_minutes: int = 15,
        horizon_minutes: int = 30,
    ):
        self.sampling_minutes = int(
            sampling_minutes
        )
        self.horizon_minutes = int(
            horizon_minutes
        )

        if self.sampling_minutes <= 0:
            raise ValueError(
                "sampling_minutes must be positive"
            )

        if (
            self.horizon_minutes
            % self.sampling_minutes
            != 0
        ):
            raise ValueError(
                "horizon_minutes must be divisible "
                "by sampling_minutes"
            )

    @property
    def horizon_steps(self) -> int:
        return (
            self.horizon_minutes
            // self.sampling_minutes
        )

    @property
    def daily_steps(self) -> int:
        return int(
            24 * 60
            / self.sampling_minutes
        )

    @property
    def weekly_steps(self) -> int:
        return (
            self.daily_steps
            * 7
        )

    @staticmethod
    def _metrics(
        name: str,
        actual: np.ndarray,
        prediction: np.ndarray,
    ) -> BaselineResult:
        actual = np.asarray(
            actual,
            dtype=float,
        )

        prediction = np.asarray(
            prediction,
            dtype=float,
        )

        mask = (
            np.isfinite(actual)
            & np.isfinite(prediction)
        )

        actual = actual[mask]
        prediction = prediction[mask]

        if len(actual) == 0:
            raise ValueError(
                f"No valid samples for baseline: {name}"
            )

        error = actual - prediction
        abs_error = np.abs(error)

        return BaselineResult(
            name=name,
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

    def evaluate_persistence(
        self,
        power: pd.Series,
    ) -> BaselineResult:
        values = pd.to_numeric(
            power,
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        steps = self.horizon_steps

        if len(values) <= steps:
            raise ValueError(
                "Not enough data for persistence baseline"
            )

        prediction = values[:-steps]
        actual = values[steps:]

        return self._metrics(
            "persistence",
            actual,
            prediction,
        )

    def evaluate_seasonal(
        self,
        power: pd.Series,
        *,
        period_steps: int,
        name: str,
    ) -> BaselineResult:
        values = pd.to_numeric(
            power,
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        if period_steps <= 0:
            raise ValueError(
                "period_steps must be positive"
            )

        if period_steps <= self.horizon_steps:
            raise ValueError(
                "seasonal period must exceed "
                "forecast horizon"
            )

        if len(values) <= period_steps:
            raise ValueError(
                f"Not enough data for {name}"
            )

        actual = values[
            period_steps:
        ]

        prediction = values[
            :-period_steps
        ]

        return self._metrics(
            name,
            actual,
            prediction,
        )

    def evaluate_all(
        self,
        df: pd.DataFrame,
        target_column: str = "total_power",
    ) -> dict[str, dict[str, Any]]:
        if target_column not in df.columns:
            raise ValueError(
                f"Missing target column: {target_column}"
            )

        power = df[target_column]

        results: dict[
            str,
            BaselineResult
        ] = {}

        results["persistence"] = (
            self.evaluate_persistence(
                power
            )
        )

        if len(df) > self.daily_steps:
            results["daily_seasonal"] = (
                self.evaluate_seasonal(
                    power,
                    period_steps=self.daily_steps,
                    name="daily_seasonal",
                )
            )

        if len(df) > self.weekly_steps:
            results["weekly_seasonal"] = (
                self.evaluate_seasonal(
                    power,
                    period_steps=self.weekly_steps,
                    name="weekly_seasonal",
                )
            )

        return {
            name: result.to_dict()
            for name, result
            in results.items()
        }

    @staticmethod
    def select_best(
        results: dict[str, dict[str, Any]],
        metric: str = "mae",
    ) -> dict[str, Any]:
        if not results:
            raise ValueError(
                "No baseline results supplied"
            )

        if metric not in {
            "mae",
            "rmse",
        }:
            raise ValueError(
                "metric must be 'mae' or 'rmse'"
            )

        best_name = min(
            results,
            key=lambda name: results[name][metric],
        )

        return {
            "name": best_name,
            "metric": metric,
            "value": float(
                results[
                    best_name
                ][metric]
            ),
            "metrics": results[
                best_name
            ],
        }

    @staticmethod
    def compare_candidate(
        candidate_metrics: dict[str, float],
        baseline: dict[str, Any],
        *,
        metric: str = "mae",
        minimum_improvement: float = 0.01,
    ) -> dict[str, Any]:
        if metric not in candidate_metrics:
            raise ValueError(
                f"Candidate metric missing: {metric}"
            )

        baseline_value = float(
            baseline["value"]
        )

        candidate_value = float(
            candidate_metrics[metric]
        )

        if baseline_value <= 0:
            raise ValueError(
                "Baseline metric must be positive"
            )

        improvement = (
            baseline_value
            - candidate_value
        ) / baseline_value

        accepted = bool(
            improvement
            >= minimum_improvement
        )

        return {
            "accepted": accepted,
            "metric": metric,
            "candidate_value":
                candidate_value,
            "baseline_value":
                baseline_value,
            "relative_improvement":
                float(improvement),
            "minimum_improvement":
                float(minimum_improvement),
            "decision": (
                "PROMOTE"
                if accepted
                else "REJECT"
            ),
        }
