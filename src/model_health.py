
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HealthThresholds:
    psi_warning: float = 0.10
    psi_critical: float = 0.25

    missingness_warning: float = 0.05
    missingness_critical: float = 0.15

    mae_degradation_warning: float = 0.25
    mae_degradation_critical: float = 0.50

    minimum_reference_rows: int = 100
    minimum_current_rows: int = 30

    histogram_bins: int = 10


class ModelHealthMonitor:
    PROFILE_VERSION = "1.0"

    def __init__(
        self,
        thresholds: HealthThresholds | None = None,
    ):
        self.thresholds = (
            thresholds
            or HealthThresholds()
        )

        self.reference: dict[str, Any] | None = None

        self._validate_thresholds()

    def _validate_thresholds(self) -> None:
        t = self.thresholds

        if not (
            0 < t.psi_warning
            < t.psi_critical
        ):
            raise ValueError(
                "PSI thresholds are invalid"
            )

        if not (
            0 <= t.missingness_warning
            < t.missingness_critical
            <= 1
        ):
            raise ValueError(
                "Missingness thresholds are invalid"
            )

        if not (
            0 <= t.mae_degradation_warning
            < t.mae_degradation_critical
        ):
            raise ValueError(
                "MAE degradation thresholds are invalid"
            )

        if t.minimum_reference_rows < 1:
            raise ValueError(
                "minimum_reference_rows must be positive"
            )

        if t.minimum_current_rows < 1:
            raise ValueError(
                "minimum_current_rows must be positive"
            )

        if t.histogram_bins < 3:
            raise ValueError(
                "histogram_bins must be at least 3"
            )

    @staticmethod
    def _feature_fingerprint(
        feature_names: list[str],
    ) -> str:
        payload = json.dumps(
            feature_names,
            separators=(",", ":"),
        ).encode("utf-8")

        return hashlib.sha256(
            payload
        ).hexdigest()[:16]

    @staticmethod
    def _severity_rank(
        severity: str,
    ) -> int:
        return {
            "HEALTHY": 0,
            "WARNING": 1,
            "CRITICAL": 2,
        }[severity]

    @classmethod
    def _max_severity(
        cls,
        *severities: str,
    ) -> str:
        return max(
            severities,
            key=cls._severity_rank,
        )

    @staticmethod
    def _numeric_frame(
        data: pd.DataFrame | np.ndarray,
        feature_names: list[str] | None = None,
    ) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            frame = data.copy()

        else:
            values = np.asarray(
                data
            )

            if values.ndim != 2:
                raise ValueError(
                    "Health monitoring data must be 2-dimensional"
                )

            if feature_names is None:
                raise ValueError(
                    "feature_names are required for ndarray input"
                )

            if values.shape[1] != len(
                feature_names
            ):
                raise ValueError(
                    "feature_names do not match input width"
                )

            frame = pd.DataFrame(
                values,
                columns=feature_names,
            )

        if frame.empty:
            raise ValueError(
                "Health monitoring frame is empty"
            )

        converted = pd.DataFrame(
            index=frame.index
        )

        for column in frame.columns:
            numeric = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

            invalid = (
                frame[column].notna()
                & numeric.isna()
            )

            if invalid.any():
                raise ValueError(
                    f"Non-numeric values in feature: {column}"
                )

            converted[column] = (
                numeric.astype(float)
            )

        return converted

    @staticmethod
    def _mae(
        actual: np.ndarray,
        prediction: np.ndarray,
    ) -> float:
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

        valid = (
            np.isfinite(actual)
            & np.isfinite(prediction)
        )

        if not np.any(valid):
            raise ValueError(
                "No valid performance samples"
            )

        return float(
            np.mean(
                np.abs(
                    actual[valid]
                    - prediction[valid]
                )
            )
        )

    @staticmethod
    def _reference_bins(
        values: np.ndarray,
        bins: int,
    ) -> list[float]:
        values = values[
            np.isfinite(values)
        ]

        if len(values) == 0:
            return []

        unique = np.unique(
            values
        )

        if len(unique) < 2:
            return []

        quantiles = np.linspace(
            0.0,
            1.0,
            bins + 1,
        )

        edges = np.quantile(
            values,
            quantiles,
        )

        internal = np.unique(
            edges[1:-1]
        )

        minimum = float(
            np.min(values)
        )

        maximum = float(
            np.max(values)
        )

        internal = internal[
            (internal > minimum)
            & (internal < maximum)
        ]

        return [
            float(value)
            for value in internal
        ]

    @staticmethod
    def _distribution(
        values: np.ndarray,
        internal_edges: list[float],
    ) -> list[float]:
        values = values[
            np.isfinite(values)
        ]

        if len(values) == 0:
            return []

        edges = np.asarray(
            [
                -np.inf,
                *internal_edges,
                np.inf,
            ],
            dtype=float,
        )

        counts, _ = np.histogram(
            values,
            bins=edges,
        )

        return (
            counts
            / counts.sum()
        ).astype(float).tolist()

    @staticmethod
    def _psi(
        reference_distribution: list[float],
        current_distribution: list[float],
    ) -> float:
        if (
            not reference_distribution
            or not current_distribution
        ):
            return 0.0

        reference = np.asarray(
            reference_distribution,
            dtype=float,
        )

        current = np.asarray(
            current_distribution,
            dtype=float,
        )

        if reference.shape != current.shape:
            raise ValueError(
                "Histogram distributions are not aligned"
            )

        epsilon = 1e-6

        reference = np.clip(
            reference,
            epsilon,
            None,
        )

        current = np.clip(
            current,
            epsilon,
            None,
        )

        return float(
            np.sum(
                (current - reference)
                * np.log(
                    current
                    / reference
                )
            )
        )

    def _psi_severity(
        self,
        psi: float,
    ) -> str:
        if psi >= self.thresholds.psi_critical:
            return "CRITICAL"

        if psi >= self.thresholds.psi_warning:
            return "WARNING"

        return "HEALTHY"

    def _missingness_severity(
        self,
        increase: float,
    ) -> str:
        if (
            increase
            >= self.thresholds.missingness_critical
        ):
            return "CRITICAL"

        if (
            increase
            >= self.thresholds.missingness_warning
        ):
            return "WARNING"

        return "HEALTHY"

    def _performance_severity(
        self,
        degradation: float,
    ) -> str:
        if (
            degradation
            >= self.thresholds.mae_degradation_critical
        ):
            return "CRITICAL"

        if (
            degradation
            >= self.thresholds.mae_degradation_warning
        ):
            return "WARNING"

        return "HEALTHY"

    def fit_reference(
        self,
        data: pd.DataFrame | np.ndarray,
        *,
        feature_names: list[str] | None = None,
        actual: np.ndarray | None = None,
        prediction: np.ndarray | None = None,
        model_version: str | None = None,
    ) -> dict[str, Any]:
        frame = self._numeric_frame(
            data,
            feature_names,
        )

        if (
            len(frame)
            < self.thresholds.minimum_reference_rows
        ):
            raise ValueError(
                "Reference dataset is too small"
            )

        reference_features = {}

        for column in frame.columns:
            values = frame[
                column
            ].to_numpy(
                dtype=float
            )

            finite = values[
                np.isfinite(values)
            ]

            if len(finite) == 0:
                raise ValueError(
                    f"Reference feature has no finite values: {column}"
                )

            internal_edges = (
                self._reference_bins(
                    finite,
                    self.thresholds.histogram_bins,
                )
            )

            distribution = (
                self._distribution(
                    finite,
                    internal_edges,
                )
            )

            reference_features[
                column
            ] = {
                "mean": float(
                    np.mean(finite)
                ),
                "std": float(
                    np.std(finite)
                ),
                "minimum": float(
                    np.min(finite)
                ),
                "maximum": float(
                    np.max(finite)
                ),
                "missing_ratio": float(
                    np.mean(
                        ~np.isfinite(values)
                    )
                ),
                "constant": bool(
                    np.ptp(finite) == 0
                ),
                "histogram_edges":
                    internal_edges,
                "histogram_distribution":
                    distribution,
            }

        reference_mae = None

        if (
            actual is not None
            or prediction is not None
        ):
            if (
                actual is None
                or prediction is None
            ):
                raise ValueError(
                    "actual and prediction must be supplied together"
                )

            reference_mae = self._mae(
                actual,
                prediction,
            )

        feature_names = list(
            frame.columns
        )

        self.reference = {
            "profile_version":
                self.PROFILE_VERSION,
            "model_version":
                model_version,
            "feature_names":
                feature_names,
            "feature_fingerprint":
                self._feature_fingerprint(
                    feature_names
                ),
            "rows":
                int(len(frame)),
            "reference_mae":
                reference_mae,
            "features":
                reference_features,
            "thresholds":
                asdict(
                    self.thresholds
                ),
        }

        return self.reference

    def evaluate(
        self,
        data: pd.DataFrame | np.ndarray,
        *,
        feature_names: list[str] | None = None,
        actual: np.ndarray | None = None,
        prediction: np.ndarray | None = None,
    ) -> dict[str, Any]:
        if self.reference is None:
            raise RuntimeError(
                "Reference profile has not been fitted"
            )

        frame = self._numeric_frame(
            data,
            feature_names,
        )

        if (
            len(frame)
            < self.thresholds.minimum_current_rows
        ):
            raise ValueError(
                "Current monitoring window is too small"
            )

        expected_features = (
            self.reference[
                "feature_names"
            ]
        )

        missing_features = [
            feature
            for feature in expected_features
            if feature not in frame.columns
        ]

        unexpected_features = [
            feature
            for feature in frame.columns
            if feature not in expected_features
        ]

        if missing_features:
            raise ValueError(
                "Missing monitored features: "
                + ", ".join(
                    missing_features
                )
            )

        frame = frame[
            expected_features
        ]

        feature_results = {}
        overall = "HEALTHY"

        warning_features = []
        critical_features = []

        for feature in expected_features:
            reference = (
                self.reference[
                    "features"
                ][feature]
            )

            values = frame[
                feature
            ].to_numpy(
                dtype=float
            )

            finite = values[
                np.isfinite(values)
            ]

            current_missing = float(
                np.mean(
                    ~np.isfinite(values)
                )
            )

            missingness_increase = max(
                0.0,
                current_missing
                - reference[
                    "missing_ratio"
                ],
            )

            missing_severity = (
                self._missingness_severity(
                    missingness_increase
                )
            )

            if len(finite) == 0:
                psi = float("inf")
                drift_severity = "CRITICAL"
                current_mean = None
                current_std = None

            elif reference["constant"]:
                current_mean = float(
                    np.mean(finite)
                )

                current_std = float(
                    np.std(finite)
                )

                tolerance = max(
                    abs(
                        reference["mean"]
                    ) * 1e-6,
                    1e-9,
                )

                changed = bool(
                    np.any(
                        np.abs(
                            finite
                            - reference["mean"]
                        )
                        > tolerance
                    )
                )

                psi = (
                    float("inf")
                    if changed
                    else 0.0
                )

                drift_severity = (
                    "CRITICAL"
                    if changed
                    else "HEALTHY"
                )

            else:
                current_mean = float(
                    np.mean(finite)
                )

                current_std = float(
                    np.std(finite)
                )

                current_distribution = (
                    self._distribution(
                        finite,
                        reference[
                            "histogram_edges"
                        ],
                    )
                )

                psi = self._psi(
                    reference[
                        "histogram_distribution"
                    ],
                    current_distribution,
                )

                drift_severity = (
                    self._psi_severity(
                        psi
                    )
                )

            severity = (
                self._max_severity(
                    drift_severity,
                    missing_severity,
                )
            )

            overall = self._max_severity(
                overall,
                severity,
            )

            if severity == "WARNING":
                warning_features.append(
                    feature
                )

            if severity == "CRITICAL":
                critical_features.append(
                    feature
                )

            feature_results[
                feature
            ] = {
                "severity":
                    severity,
                "psi":
                    psi,
                "reference_mean":
                    reference["mean"],
                "current_mean":
                    current_mean,
                "reference_std":
                    reference["std"],
                "current_std":
                    current_std,
                "reference_missing_ratio":
                    reference[
                        "missing_ratio"
                    ],
                "current_missing_ratio":
                    current_missing,
                "missingness_increase":
                    missingness_increase,
            }

        performance = {
            "available": False,
            "severity": "HEALTHY",
            "reference_mae":
                self.reference[
                    "reference_mae"
                ],
            "current_mae": None,
            "relative_degradation": None,
        }

        if (
            actual is not None
            or prediction is not None
        ):
            if (
                actual is None
                or prediction is None
            ):
                raise ValueError(
                    "actual and prediction must be supplied together"
                )

            current_mae = self._mae(
                actual,
                prediction,
            )

            reference_mae = (
                self.reference[
                    "reference_mae"
                ]
            )

            if reference_mae is None:
                raise ValueError(
                    "Reference profile does not contain performance metrics"
                )

            if reference_mae == 0:
                degradation = (
                    0.0
                    if current_mae == 0
                    else float("inf")
                )
            else:
                degradation = (
                    current_mae
                    - reference_mae
                ) / reference_mae

            degradation = max(
                0.0,
                degradation,
            )

            performance_severity = (
                self._performance_severity(
                    degradation
                )
            )

            overall = self._max_severity(
                overall,
                performance_severity,
            )

            performance = {
                "available": True,
                "severity":
                    performance_severity,
                "reference_mae":
                    reference_mae,
                "current_mae":
                    current_mae,
                "relative_degradation":
                    degradation,
            }

        return {
            "status": overall,
            "rows": int(
                len(frame)
            ),
            "model_version":
                self.reference[
                    "model_version"
                ],
            "feature_fingerprint":
                self.reference[
                    "feature_fingerprint"
                ],
            "features":
                feature_results,
            "performance":
                performance,
            "summary": {
                "warning_features":
                    warning_features,
                "critical_features":
                    critical_features,
                "unexpected_features":
                    unexpected_features,
                "retrain_recommended":
                    overall == "CRITICAL",
            },
        }

    def save_reference(
        self,
        path: str | Path,
    ) -> Path:
        if self.reference is None:
            raise RuntimeError(
                "Reference profile has not been fitted"
            )

        path = Path(
            path
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary = path.with_suffix(
            path.suffix + ".tmp"
        )

        temporary.write_text(
            json.dumps(
                self.reference,
                indent=2,
                sort_keys=True,
                allow_nan=True,
            ),
            encoding="utf-8",
        )

        temporary.replace(
            path
        )

        return path

    def load_reference(
        self,
        path: str | Path,
    ) -> dict[str, Any]:
        path = Path(
            path
        )

        reference = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        required = {
            "profile_version",
            "feature_names",
            "feature_fingerprint",
            "features",
            "thresholds",
        }

        missing = required - set(
            reference
        )

        if missing:
            raise ValueError(
                "Invalid health reference profile: "
                + ", ".join(
                    sorted(missing)
                )
            )

        expected_fingerprint = (
            self._feature_fingerprint(
                reference[
                    "feature_names"
                ]
            )
        )

        if (
            reference[
                "feature_fingerprint"
            ]
            != expected_fingerprint
        ):
            raise ValueError(
                "Health reference feature fingerprint mismatch"
            )

        self.reference = reference

        return reference
