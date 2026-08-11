
from __future__ import annotations

import json
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from src.model_registry import ModelRegistry
from src.training_data import SequenceDataset, TimeSeriesDatasetBuilder
from src.training_orchestrator import TrainingOrchestrator
from src.training_profiles import TrainingProfileSelector


@dataclass(frozen=True)
class TrainerConfig:
    epochs: int = 60
    batch_size: int = 64
    learning_rate: float = 1e-3
    lstm_units: int = 64
    second_lstm_units: int = 32
    dense_units: int = 32
    dropout: float = 0.20
    patience: int = 8
    min_delta: float = 1e-4
    seed: int = 42
    verbose: int = 0


class ProductionTrainer:
    def __init__(
        self,
        *,
        config: TrainerConfig | None = None,
        dataset_builder: TimeSeriesDatasetBuilder | None = None,
        registry: ModelRegistry | None = None,
        minimum_improvement: float = 0.01,
    ):
        self.config = config or TrainerConfig()

        self.dataset_builder = (
            dataset_builder
            or TimeSeriesDatasetBuilder()
        )

        self.registry = registry or ModelRegistry()

        self.profile_selector = TrainingProfileSelector()

        self.orchestrator = TrainingOrchestrator(
            dataset_builder=self.dataset_builder,
            registry=self.registry,
            minimum_improvement=minimum_improvement,
        )

        self._validate_config()

    def _validate_config(self) -> None:
        if self.config.epochs < 1:
            raise ValueError("epochs must be positive")

        if self.config.batch_size < 1:
            raise ValueError("batch_size must be positive")

        if self.config.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        if self.config.lstm_units < 1:
            raise ValueError("lstm_units must be positive")

        if self.config.second_lstm_units < 1:
            raise ValueError(
                "second_lstm_units must be positive"
            )

        if self.config.dense_units < 1:
            raise ValueError("dense_units must be positive")

        if not 0 <= self.config.dropout < 1:
            raise ValueError(
                "dropout must be in [0, 1)"
            )

        if self.config.patience < 0:
            raise ValueError(
                "patience cannot be negative"
            )

    def _set_seed(self) -> bool:
        tf.keras.utils.set_random_seed(
            self.config.seed
        )

        try:
            tf.config.experimental.enable_op_determinism()
            return True
        except Exception:
            return False

    @staticmethod
    def _fit_x_scaler(
        dataset: SequenceDataset,
    ) -> StandardScaler:
        X = np.asarray(
            dataset.X,
            dtype=np.float64,
        )

        if X.ndim != 3:
            raise ValueError(
                "Training input must have shape "
                "(samples, timesteps, features)"
            )

        flattened = X.reshape(
            -1,
            X.shape[-1],
        )

        if not np.isfinite(flattened).all():
            raise ValueError(
                "Training input contains non-finite values"
            )

        scaler = StandardScaler()
        scaler.fit(flattened)

        return scaler

    @staticmethod
    def _transform_X(
        X: np.ndarray,
        scaler: StandardScaler,
    ) -> np.ndarray:
        values = np.asarray(
            X,
            dtype=np.float64,
        )

        if values.ndim != 3:
            raise ValueError(
                "Input must have shape "
                "(samples, timesteps, features)"
            )

        shape = values.shape

        transformed = scaler.transform(
            values.reshape(
                -1,
                shape[-1],
            )
        )

        return transformed.reshape(
            shape
        ).astype(np.float32)

    @staticmethod
    def _persistence(
        dataset: SequenceDataset,
        total_power_index: int,
    ) -> np.ndarray:
        return np.asarray(
            dataset.X[
                :,
                -1,
                total_power_index,
            ],
            dtype=np.float64,
        ).reshape(-1)

    @classmethod
    def _residual_targets(
        cls,
        dataset: SequenceDataset,
        total_power_index: int,
    ) -> np.ndarray:
        actual = np.asarray(
            dataset.y,
            dtype=np.float64,
        ).reshape(-1)

        persistence = cls._persistence(
            dataset,
            total_power_index,
        )

        residual = actual - persistence

        if not np.isfinite(residual).all():
            raise ValueError(
                "Residual target contains non-finite values"
            )

        return residual.reshape(-1, 1)

    @staticmethod
    def _fit_residual_scaler(
        residual: np.ndarray,
    ) -> StandardScaler:
        scaler = StandardScaler()
        scaler.fit(residual)
        return scaler

    def _build_model(
        self,
        input_shape: tuple[int, int],
    ) -> tf.keras.Model:
        inputs = tf.keras.Input(
            shape=input_shape,
            name="history",
        )

        x = tf.keras.layers.LSTM(
            self.config.lstm_units,
            return_sequences=True,
            name="lstm_1",
        )(inputs)

        x = tf.keras.layers.Dropout(
            self.config.dropout,
            name="dropout_1",
        )(x)

        x = tf.keras.layers.LSTM(
            self.config.second_lstm_units,
            name="lstm_2",
        )(x)

        x = tf.keras.layers.Dropout(
            self.config.dropout,
            name="dropout_2",
        )(x)

        x = tf.keras.layers.Dense(
            self.config.dense_units,
            activation="gelu",
            name="dense",
        )(x)

        outputs = tf.keras.layers.Dense(
            1,
            name="residual",
        )(x)

        model = tf.keras.Model(
            inputs=inputs,
            outputs=outputs,
            name="ecotwin_dynamic_residual_lstm",
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config.learning_rate
            ),
            loss=tf.keras.losses.Huber(),
            metrics=[
                tf.keras.metrics.MeanAbsoluteError(
                    name="mae"
                )
            ],
        )

        return model

    @staticmethod
    def _history_payload(
        history: tf.keras.callbacks.History,
    ) -> dict[str, list[float]]:
        return {
            key: [
                float(value)
                for value in values
            ]
            for key, values
            in history.history.items()
        }

    def train(
        self,
        frame,
        *,
        target_column: str = "total_power",
    ) -> dict[str, Any]:
        profile = self.profile_selector.select(
            frame
        )

        feature_columns = profile[
            "feature_columns"
        ]

        if target_column not in feature_columns:
            raise ValueError(
                "total_power must be present "
                "in the training feature contract"
            )

        prepared = (
            self.orchestrator
            .prepare_training_data(
                frame,
                feature_columns=feature_columns,
                target_column=target_column,
            )
        )

        train_data = prepared[
            "datasets"
        ]["train"]

        validation_data = prepared[
            "datasets"
        ]["validation"]

        total_power_index = (
            feature_columns.index(
                target_column
            )
        )

        x_scaler = self._fit_x_scaler(
            train_data
        )

        train_residual = (
            self._residual_targets(
                train_data,
                total_power_index,
            )
        )

        residual_scaler = (
            self._fit_residual_scaler(
                train_residual
            )
        )

        X_train = self._transform_X(
            train_data.X,
            x_scaler,
        )

        X_validation = self._transform_X(
            validation_data.X,
            x_scaler,
        )

        y_train = residual_scaler.transform(
            train_residual
        ).astype(np.float32)

        validation_residual = (
            self._residual_targets(
                validation_data,
                total_power_index,
            )
        )

        y_validation = (
            residual_scaler.transform(
                validation_residual
            )
            .astype(np.float32)
        )

        deterministic_ops = self._set_seed()

        model = self._build_model(
            input_shape=(
                X_train.shape[1],
                X_train.shape[2],
            )
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.config.patience,
                min_delta=self.config.min_delta,
                restore_best_weights=True,
                mode="min",
            )
        ]

        history = model.fit(
            X_train,
            y_train,
            validation_data=(
                X_validation,
                y_validation,
            ),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            shuffle=False,
            callbacks=callbacks,
            verbose=self.config.verbose,
        )

        def predictor(
            X: np.ndarray,
        ) -> np.ndarray:
            scaled_X = self._transform_X(
                X,
                x_scaler,
            )

            scaled_residual = model.predict(
                scaled_X,
                verbose=0,
            )

            residual = (
                residual_scaler
                .inverse_transform(
                    scaled_residual
                )
                .reshape(-1)
            )

            persistence = np.asarray(
                X[
                    :,
                    -1,
                    total_power_index,
                ],
                dtype=np.float64,
            )

            return (
                persistence
                + residual
            ).reshape(-1, 1)

        validation_evaluation = (
            self.orchestrator
            .evaluate_validation(
                prepared=prepared,
                predictor=predictor,
                target_column=target_column,
            )
        )

        decision = validation_evaluation[
            "promotion_decision"
        ]

        test_metrics = None

        # Test is touched only after the validation gate accepts the model.
        if decision["accepted"]:
            test_metrics = (
                self.orchestrator
                .evaluate_test(
                    prepared=prepared,
                    predictor=predictor,
                )
            )

        run_id = uuid.uuid4().hex[:12]

        with tempfile.TemporaryDirectory(
            prefix=f"ecotwin-{run_id}-"
        ) as temporary:
            artifact_dir = Path(temporary)

            model_path = (
                artifact_dir
                / "model.keras"
            )

            x_scaler_path = (
                artifact_dir
                / "x_scaler.pkl"
            )

            residual_scaler_path = (
                artifact_dir
                / "residual_scaler.pkl"
            )

            profile_path = (
                artifact_dir
                / "feature_contract.json"
            )

            training_path = (
                artifact_dir
                / "training_config.json"
            )

            history_path = (
                artifact_dir
                / "history.json"
            )

            model.save(
                model_path
            )

            joblib.dump(
                x_scaler,
                x_scaler_path,
            )

            joblib.dump(
                residual_scaler,
                residual_scaler_path,
            )

            profile_path.write_text(
                json.dumps(
                    profile,
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            training_payload = {
                **asdict(self.config),
                "lookback_steps":
                    self.dataset_builder.lookback_steps,
                "horizon_minutes":
                    self.dataset_builder.horizon_minutes,
                "sampling_minutes":
                    self.dataset_builder.sampling_minutes,
                "target_column":
                    target_column,
            }

            training_path.write_text(
                json.dumps(
                    training_payload,
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            history_payload = (
                self._history_payload(
                    history
                )
            )

            history_path.write_text(
                json.dumps(
                    history_payload,
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            registry_metrics = {
                "validation":
                    validation_evaluation[
                        "candidate"
                    ],
                "baseline_champion":
                    validation_evaluation[
                        "baseline_champion"
                    ],
                "test": test_metrics,
            }

            manifest = self.registry.register(
                artifacts={
                    "model": model_path,
                    "x_scaler": x_scaler_path,
                    "residual_scaler":
                        residual_scaler_path,
                    "feature_contract":
                        profile_path,
                    "training_config":
                        training_path,
                    "history":
                        history_path,
                },
                metrics=registry_metrics,
                feature_columns=feature_columns,
                training_config=training_payload,
                metadata={
                    "run_id": run_id,
                    "profile":
                        profile["profile"],
                    "profile_version":
                        profile[
                            "profile_version"
                        ],
                    "feature_fingerprint":
                        profile[
                            "feature_fingerprint"
                        ],
                    "deterministic_ops":
                        deterministic_ops,
                    "epochs_completed":
                        len(
                            history.history[
                                "loss"
                            ]
                        ),
                    "selection_dataset":
                        "validation",
                    "test_used_for_selection":
                        False,
                },
            )

        if decision["accepted"]:
            manifest = self.registry.promote(
                manifest["version"],
                approval=decision,
            )

        return {
            "version": manifest["version"],
            "status": manifest["status"],
            "profile": profile,
            "validation":
                validation_evaluation,
            "test": test_metrics,
            "training": {
                "epochs_completed": len(
                    history.history["loss"]
                ),
                "deterministic_ops":
                    deterministic_ops,
            },
        }
