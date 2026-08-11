
import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from src.model_registry import ModelRegistry
from src.registry_forecast import RegistryForecaster


class FakeLoadedModel:
    def __init__(
        self,
        timesteps=16,
        features=2,
        residual=0.0,
    ):
        self.input_shape = (
            None,
            timesteps,
            features,
        )

        self.residual = float(
            residual
        )

    def predict(
        self,
        X,
        verbose=0,
    ):
        return np.full(
            (
                len(X),
                1,
            ),
            self.residual,
            dtype=np.float32,
        )


def create_registry_model(
    tmp_path,
    *,
    feature_columns=None,
    contract_features=None,
    promote=True,
):
    feature_columns = (
        feature_columns
        or [
            "total_power",
            "time_sin",
        ]
    )

    contract_features = (
        contract_features
        or feature_columns
    )

    registry = ModelRegistry(
        tmp_path / "registry"
    )

    artifacts = (
        tmp_path
        / "artifacts"
    )

    artifacts.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        artifacts
        / "model.keras"
    )

    model_path.write_bytes(
        b"fake-model"
    )

    x_scaler = StandardScaler()

    x_scaler.fit(
        np.array(
            [
                [10.0, 0.0],
                [20.0, 1.0],
                [30.0, -1.0],
            ]
        )
    )

    x_scaler_path = (
        artifacts
        / "x_scaler.pkl"
    )

    joblib.dump(
        x_scaler,
        x_scaler_path,
    )

    residual_scaler = (
        StandardScaler()
    )

    residual_scaler.fit(
        np.array(
            [
                [-2.0],
                [0.0],
                [2.0],
            ]
        )
    )

    residual_scaler_path = (
        artifacts
        / "residual_scaler.pkl"
    )

    joblib.dump(
        residual_scaler,
        residual_scaler_path,
    )

    fingerprint = (
        RegistryForecaster
        ._fingerprint(
            contract_features
        )
    )

    contract = {
        "profile": "CORE",
        "profile_version": "1.0",
        "feature_columns":
            contract_features,
        "feature_count":
            len(contract_features),
        "feature_fingerprint":
            fingerprint,
    }

    contract_path = (
        artifacts
        / "feature_contract.json"
    )

    contract_path.write_text(
        json.dumps(
            contract,
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest = registry.register(
        artifacts={
            "model":
                model_path,
            "x_scaler":
                x_scaler_path,
            "residual_scaler":
                residual_scaler_path,
            "feature_contract":
                contract_path,
        },
        metrics={
            "validation": {
                "mae": 1.0,
            }
        },
        feature_columns=
            feature_columns,
        training_config={
            "lookback_steps": 16,
            "horizon_minutes": 30,
        },
        metadata={
            "profile": "CORE",
            "feature_fingerprint":
                RegistryForecaster
                ._fingerprint(
                    feature_columns
                ),
        },
        version="test-v1",
    )

    if promote:
        registry.promote(
            manifest["version"],
            approval={
                "accepted": True,
                "decision": "PROMOTE",
            },
        )

    return registry


def test_production_model_loads_from_registry(
    tmp_path,
    monkeypatch,
):
    registry = create_registry_model(
        tmp_path
    )

    monkeypatch.setattr(
        "src.registry_forecast."
        "tf.keras.models.load_model",
        lambda path: FakeLoadedModel(
            features=2
        ),
    )

    forecaster = RegistryForecaster(
        registry_root=registry.root
    )

    status = forecaster.status()

    assert status["version"] == "test-v1"
    assert status["status"] == "production"
    assert status["profile"] == "CORE"
    assert status["feature_count"] == 2
    assert status["lookback_steps"] == 16
    assert status["artifact_integrity"] is True


def test_prediction_reconstructs_persistence_plus_residual(
    tmp_path,
    monkeypatch,
):
    registry = create_registry_model(
        tmp_path
    )

    monkeypatch.setattr(
        "src.registry_forecast."
        "tf.keras.models.load_model",
        lambda path: FakeLoadedModel(
            features=2,
            residual=0.0,
        ),
    )

    forecaster = RegistryForecaster(
        registry_root=registry.root
    )

    X = np.zeros(
        (
            1,
            16,
            2,
        ),
        dtype=float,
    )

    X[
        0,
        -1,
        0,
    ] = 42.5

    prediction = forecaster.predict(
        X
    )

    assert prediction.shape == (1,)

    assert prediction[0] == pytest.approx(
        42.5
    )


def test_two_dimensional_single_window_is_supported(
    tmp_path,
    monkeypatch,
):
    registry = create_registry_model(
        tmp_path
    )

    monkeypatch.setattr(
        "src.registry_forecast."
        "tf.keras.models.load_model",
        lambda path: FakeLoadedModel(
            features=2
        ),
    )

    forecaster = RegistryForecaster(
        registry_root=registry.root
    )

    X = np.zeros(
        (
            16,
            2,
        )
    )

    X[
        -1,
        0,
    ] = 33.0

    result = forecaster.predict(
        X
    )

    assert result.shape == (1,)
    assert result[0] == pytest.approx(
        33.0
    )


def test_wrong_input_shape_is_rejected(
    tmp_path,
    monkeypatch,
):
    registry = create_registry_model(
        tmp_path
    )

    monkeypatch.setattr(
        "src.registry_forecast."
        "tf.keras.models.load_model",
        lambda path: FakeLoadedModel(
            features=2
        ),
    )

    forecaster = RegistryForecaster(
        registry_root=registry.root
    )

    with pytest.raises(
        ValueError,
        match="Expected 16 timesteps",
    ):
        forecaster.predict(
            np.zeros(
                (
                    10,
                    2,
                )
            )
        )

    with pytest.raises(
        ValueError,
        match="Expected 2 features",
    ):
        forecaster.predict(
            np.zeros(
                (
                    16,
                    3,
                )
            )
        )


def test_non_finite_forecast_input_is_rejected(
    tmp_path,
    monkeypatch,
):
    registry = create_registry_model(
        tmp_path
    )

    monkeypatch.setattr(
        "src.registry_forecast."
        "tf.keras.models.load_model",
        lambda path: FakeLoadedModel(
            features=2
        ),
    )

    forecaster = RegistryForecaster(
        registry_root=registry.root
    )

    X = np.zeros(
        (
            16,
            2,
        )
    )

    X[5, 0] = np.nan

    with pytest.raises(
        ValueError,
        match="non-finite",
    ):
        forecaster.predict(X)


def test_corrupted_artifact_is_rejected_before_loading(
    tmp_path,
    monkeypatch,
):
    registry = create_registry_model(
        tmp_path
    )

    model_path = (
        registry.artifact_path(
            "test-v1",
            "model",
        )
    )

    model_path.write_bytes(
        b"tampered-model"
    )

    called = {
        "load_model": False
    }

    def forbidden_load(path):
        called["load_model"] = True
        raise AssertionError(
            "Corrupted model must not be loaded"
        )

    monkeypatch.setattr(
        "src.registry_forecast."
        "tf.keras.models.load_model",
        forbidden_load,
    )

    with pytest.raises(
        RuntimeError,
        match="integrity verification failed",
    ):
        RegistryForecaster(
            registry_root=registry.root
        )

    assert (
        called["load_model"]
        is False
    )


def test_manifest_and_feature_contract_mismatch_is_rejected(
    tmp_path,
    monkeypatch,
):
    registry = create_registry_model(
        tmp_path,
        feature_columns=[
            "total_power",
            "time_sin",
        ],
        contract_features=[
            "total_power",
            "time_cos",
        ],
    )

    monkeypatch.setattr(
        "src.registry_forecast."
        "tf.keras.models.load_model",
        lambda path: FakeLoadedModel(
            features=2
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="manifest and feature contract disagree",
    ):
        RegistryForecaster(
            registry_root=registry.root
        )


def test_registry_without_production_model_is_rejected(
    tmp_path,
):
    registry = ModelRegistry(
        tmp_path / "registry"
    )

    with pytest.raises(
        RuntimeError,
        match="No production model",
    ):
        RegistryForecaster(
            registry_root=registry.root
        )
