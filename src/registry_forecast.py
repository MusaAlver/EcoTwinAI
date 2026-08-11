
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import tensorflow as tf

from src.model_registry import ModelRegistry


class RegistryForecaster:
    def __init__(
        self,
        *,
        registry_root: str | Path = "models/registry",
        version: str | None = None,
    ):
        self.registry = ModelRegistry(registry_root)

        if version is None:
            manifest = self.registry.production()

            if manifest is None:
                raise RuntimeError(
                    "No production model is registered"
                )
        else:
            manifest = self.registry.get(version)

        self.version = manifest["version"]
        self.manifest = manifest

        verification = self.registry.verify(
            self.version
        )

        if not verification["valid"]:
            raise RuntimeError(
                "Model artifact integrity verification failed"
            )

        self.feature_contract = json.loads(
            self.registry.artifact_path(
                self.version,
                "feature_contract",
            ).read_text(
                encoding="utf-8"
            )
        )

        self.feature_columns = list(
            self.feature_contract[
                "feature_columns"
            ]
        )

        self._validate_contract()

        self.model = tf.keras.models.load_model(
            self.registry.artifact_path(
                self.version,
                "model",
            )
        )

        self.x_scaler = joblib.load(
            self.registry.artifact_path(
                self.version,
                "x_scaler",
            )
        )

        self.residual_scaler = joblib.load(
            self.registry.artifact_path(
                self.version,
                "residual_scaler",
            )
        )

        self.total_power_index = (
            self.feature_columns.index(
                "total_power"
            )
        )

        input_shape = self.model.input_shape

        if len(input_shape) != 3:
            raise RuntimeError(
                "Registered model must accept "
                "(batch, timesteps, features)"
            )

        self.lookback_steps = int(
            input_shape[1]
        )

        self.feature_count = int(
            input_shape[2]
        )

        if (
            self.feature_count
            != len(self.feature_columns)
        ):
            raise RuntimeError(
                "Model input width does not match "
                "registered feature contract"
            )

    @staticmethod
    def _fingerprint(
        feature_columns: list[str],
    ) -> str:
        payload = json.dumps(
            feature_columns,
            separators=(",", ":"),
        ).encode("utf-8")

        return hashlib.sha256(
            payload
        ).hexdigest()[:16]

    def _validate_contract(self) -> None:
        manifest_features = list(
            self.manifest["feature_columns"]
        )

        if (
            manifest_features
            != self.feature_columns
        ):
            raise RuntimeError(
                "Registry manifest and feature contract disagree"
            )

        if "total_power" not in self.feature_columns:
            raise RuntimeError(
                "Feature contract does not contain total_power"
            )

        expected = self._fingerprint(
            self.feature_columns
        )

        actual = self.feature_contract.get(
            "feature_fingerprint"
        )

        if actual != expected:
            raise RuntimeError(
                "Feature contract fingerprint mismatch"
            )

        manifest_fingerprint = (
            self.manifest
            .get("metadata", {})
            .get("feature_fingerprint")
        )

        if (
            manifest_fingerprint is not None
            and manifest_fingerprint != expected
        ):
            raise RuntimeError(
                "Registry feature fingerprint mismatch"
            )

    def _validate_input(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(
            X,
            dtype=np.float64,
        )

        if values.ndim == 2:
            values = values[
                np.newaxis,
                ...,
            ]

        if values.ndim != 3:
            raise ValueError(
                "Input must have shape "
                "(timesteps, features) or "
                "(batch, timesteps, features)"
            )

        if values.shape[1] != self.lookback_steps:
            raise ValueError(
                f"Expected {self.lookback_steps} timesteps, "
                f"received {values.shape[1]}"
            )

        if values.shape[2] != self.feature_count:
            raise ValueError(
                f"Expected {self.feature_count} features, "
                f"received {values.shape[2]}"
            )

        if not np.isfinite(values).all():
            raise ValueError(
                "Forecast input contains non-finite values"
            )

        return values

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        values = self._validate_input(
            X
        )

        shape = values.shape

        scaled = self.x_scaler.transform(
            values.reshape(
                -1,
                shape[-1],
            )
        ).reshape(
            shape
        ).astype(
            np.float32
        )

        scaled_residual = self.model.predict(
            scaled,
            verbose=0,
        )

        residual = (
            self.residual_scaler
            .inverse_transform(
                scaled_residual
            )
            .reshape(-1)
        )

        persistence = values[
            :,
            -1,
            self.total_power_index,
        ]

        forecast = (
            persistence
            + residual
        )

        return forecast.astype(
            np.float64
        )

    def status(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "status": self.manifest["status"],
            "profile": self.feature_contract["profile"],
            "feature_count": self.feature_count,
            "feature_columns": self.feature_columns,
            "feature_fingerprint":
                self.feature_contract[
                    "feature_fingerprint"
                ],
            "lookback_steps": self.lookback_steps,
            "canonical_power_unit": "kW",
            "artifact_integrity": True,
        }
