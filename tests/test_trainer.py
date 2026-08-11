
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.model_registry import ModelRegistry
from src.trainer import ProductionTrainer, TrainerConfig
from src.training_data import SequenceDataset
from src.training_profiles import TrainingProfileSelector


class FakeHistory:
    history = {
        "loss": [1.0],
        "val_loss": [1.1],
        "mae": [0.8],
        "val_mae": [0.9],
    }


class FakeModel:
    def __init__(self):
        self.fit_kwargs = None

    def fit(self, *args, **kwargs):
        self.fit_kwargs = kwargs
        return FakeHistory()

    def predict(self, X, verbose=0):
        return np.zeros((len(X), 1), dtype=np.float32)

    def save(self, path):
        Path(path).write_bytes(b"fake-keras-model")


def make_core_frame(rows=300):
    selector = TrainingProfileSelector()
    index = np.arange(rows, dtype=float)

    data = {
        "timestamp": pd.date_range(
            "2026-01-01",
            periods=rows,
            freq="15min",
        ),
    }

    for position, feature in enumerate(selector.CORE_FEATURES):
        if feature == "total_power":
            data[feature] = 50.0 + index * 0.05
        else:
            data[feature] = index * 0.01 + position

    return pd.DataFrame(data)


def install_fake_model(trainer):
    models = []

    def build_model(input_shape):
        model = FakeModel()
        models.append((input_shape, model))
        return model

    trainer._build_model = build_model
    return models


def accepted_evaluation():
    return {
        "candidate": {
            "samples": 10,
            "mae": 1.0,
            "rmse": 1.2,
            "within_5kw": 1.0,
            "within_10kw": 1.0,
        },
        "baselines": {
            "persistence": {
                "samples": 10,
                "mae": 2.0,
                "rmse": 2.2,
            }
        },
        "baseline_champion": {
            "name": "persistence",
            "metric": "mae",
            "value": 2.0,
        },
        "promotion_decision": {
            "accepted": True,
            "decision": "PROMOTE",
            "metric": "mae",
            "candidate_value": 1.0,
            "baseline_value": 2.0,
            "relative_improvement": 0.5,
            "minimum_improvement": 0.01,
        },
    }


def rejected_evaluation():
    result = accepted_evaluation()
    result["promotion_decision"] = {
        "accepted": False,
        "decision": "REJECT",
        "metric": "mae",
        "candidate_value": 2.1,
        "baseline_value": 2.0,
        "relative_improvement": -0.05,
        "minimum_improvement": 0.01,
    }
    return result


def test_x_scaler_is_fit_only_on_training_data():
    X = np.zeros((4, 16, 3), dtype=np.float32)

    train = SequenceDataset(
        X=X,
        y=np.zeros((4, 1), dtype=np.float32),
        forecast_time=np.arange(4),
        outcome_time=np.arange(4),
    )

    scaler = ProductionTrainer._fit_x_scaler(train)

    np.testing.assert_allclose(
        scaler.mean_,
        [0.0, 0.0, 0.0],
    )

    validation_outlier = np.full(
        (2, 16, 3),
        1_000_000.0,
        dtype=np.float32,
    )

    assert not np.allclose(
        scaler.mean_,
        validation_outlier.mean(axis=(0, 1)),
    )


def test_residual_target_uses_last_observed_power():
    X = np.zeros((2, 16, 2), dtype=np.float32)

    X[0, -1, 0] = 10.0
    X[1, -1, 0] = 20.0

    dataset = SequenceDataset(
        X=X,
        y=np.array([[12.0], [25.0]], dtype=np.float32),
        forecast_time=np.arange(2),
        outcome_time=np.arange(2),
    )

    residual = ProductionTrainer._residual_targets(
        dataset,
        total_power_index=0,
    )

    np.testing.assert_allclose(
        residual.reshape(-1),
        [2.0, 5.0],
    )


def test_dynamic_lstm_accepts_profile_feature_count(tmp_path):
    trainer = ProductionTrainer(
        config=TrainerConfig(
            lstm_units=4,
            second_lstm_units=2,
            dense_units=4,
        ),
        registry=ModelRegistry(tmp_path / "registry"),
    )

    model = trainer._build_model((16, 12))

    assert model.input_shape == (None, 16, 12)
    assert model.output_shape == (None, 1)


def test_invalid_dropout_is_rejected(tmp_path):
    with pytest.raises(
        ValueError,
        match="dropout must be",
    ):
        ProductionTrainer(
            config=TrainerConfig(dropout=1.0),
            registry=ModelRegistry(tmp_path / "registry"),
        )


def test_rejected_candidate_never_touches_test_set(tmp_path):
    registry = ModelRegistry(tmp_path / "registry")

    trainer = ProductionTrainer(
        config=TrainerConfig(epochs=1),
        registry=registry,
    )

    install_fake_model(trainer)

    trainer.orchestrator.evaluate_validation = (
        lambda **kwargs: rejected_evaluation()
    )

    def forbidden_test_evaluation(**kwargs):
        raise AssertionError(
            "Test set must not be evaluated after rejection"
        )

    trainer.orchestrator.evaluate_test = forbidden_test_evaluation

    result = trainer.train(make_core_frame())

    assert result["status"] == "candidate"
    assert result["test"] is None
    assert registry.production() is None


def test_accepted_candidate_evaluates_test_and_promotes(tmp_path):
    registry = ModelRegistry(tmp_path / "registry")

    trainer = ProductionTrainer(
        config=TrainerConfig(epochs=1),
        registry=registry,
    )

    install_fake_model(trainer)

    trainer.orchestrator.evaluate_validation = (
        lambda **kwargs: accepted_evaluation()
    )

    calls = {"test": 0}

    def evaluate_test(**kwargs):
        calls["test"] += 1
        return {
            "samples": 20,
            "mae": 1.1,
            "rmse": 1.3,
            "within_5kw": 1.0,
            "within_10kw": 1.0,
        }

    trainer.orchestrator.evaluate_test = evaluate_test

    result = trainer.train(make_core_frame())

    assert calls["test"] == 1
    assert result["status"] == "production"
    assert result["test"]["mae"] == pytest.approx(1.1)

    production = registry.production()

    assert production is not None
    assert production["version"] == result["version"]


def test_training_keeps_time_series_order(tmp_path):
    trainer = ProductionTrainer(
        config=TrainerConfig(epochs=1),
        registry=ModelRegistry(tmp_path / "registry"),
    )

    models = install_fake_model(trainer)

    trainer.orchestrator.evaluate_validation = (
        lambda **kwargs: rejected_evaluation()
    )

    trainer.train(make_core_frame())

    _, model = models[0]

    assert model.fit_kwargs["shuffle"] is False
    assert "validation_data" in model.fit_kwargs


def test_registered_artifacts_pass_integrity_check(tmp_path):
    registry = ModelRegistry(tmp_path / "registry")

    trainer = ProductionTrainer(
        config=TrainerConfig(epochs=1),
        registry=registry,
    )

    install_fake_model(trainer)

    trainer.orchestrator.evaluate_validation = (
        lambda **kwargs: rejected_evaluation()
    )

    result = trainer.train(make_core_frame())

    verification = registry.verify(result["version"])

    assert verification["valid"] is True

    manifest = registry.get(result["version"])

    artifact_names = {
        artifact["name"]
        for artifact in manifest["artifacts"]
    }

    assert artifact_names == {
        "model",
        "x_scaler",
        "residual_scaler",
        "feature_contract",
        "training_config",
        "history",
    }

    assert manifest["metadata"]["selection_dataset"] == "validation"
    assert manifest["metadata"]["test_used_for_selection"] is False


def test_context_profile_is_recorded_in_registry(tmp_path):
    registry = ModelRegistry(tmp_path / "registry")

    trainer = ProductionTrainer(
        config=TrainerConfig(epochs=1),
        registry=registry,
    )

    install_fake_model(trainer)

    trainer.orchestrator.evaluate_validation = (
        lambda **kwargs: rejected_evaluation()
    )

    frame = make_core_frame()
    frame["outdoor_temp_avg"] = 20.0

    result = trainer.train(frame)

    assert result["profile"]["profile"] == "CONTEXT"

    manifest = registry.get(result["version"])

    assert manifest["metadata"]["profile"] == "CONTEXT"
    assert (
        manifest["metadata"]["feature_fingerprint"]
        == result["profile"]["feature_fingerprint"]
    )
