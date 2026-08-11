
from pathlib import Path

import numpy as np
import pytest

from src.model_registry import ModelRegistry


def make_artifact(
    directory: Path,
    name: str = "model.keras",
    content: bytes = b"model-v1",
) -> Path:
    path = directory / name
    path.write_bytes(content)
    return path


def register_model(
    registry: ModelRegistry,
    artifact: Path,
    version: str,
):
    return registry.register(
        version=version,
        artifacts={
            "model": artifact,
        },
        metrics={
            "mae": np.float64(2.5),
            "rmse": 3.4,
        },
        feature_columns=[
            "total_power",
            "outdoor_temp_avg",
        ],
        training_config={
            "lookback_steps": 16,
            "horizon_minutes": 30,
        },
    )


def test_register_creates_candidate_and_copies_artifact(
    tmp_path,
):
    registry = ModelRegistry(
        tmp_path / "registry"
    )

    artifact = make_artifact(
        tmp_path
    )

    manifest = register_model(
        registry,
        artifact,
        version="v1",
    )

    assert manifest["version"] == "v1"
    assert manifest["status"] == "candidate"
    assert manifest["metrics"]["mae"] == 2.5

    stored = registry.artifact_path(
        "v1",
        "model",
    )

    assert stored.exists()
    assert stored.read_bytes() == b"model-v1"
    assert stored != artifact


def test_registered_artifact_passes_integrity_check(
    tmp_path,
):
    registry = ModelRegistry(
        tmp_path / "registry"
    )

    artifact = make_artifact(
        tmp_path
    )

    register_model(
        registry,
        artifact,
        version="v1",
    )

    verification = registry.verify(
        "v1"
    )

    assert verification["valid"] is True
    assert (
        verification["artifacts"][0]["valid"]
        is True
    )


def test_corrupted_artifact_fails_integrity_check(
    tmp_path,
):
    registry = ModelRegistry(
        tmp_path / "registry"
    )

    artifact = make_artifact(
        tmp_path
    )

    register_model(
        registry,
        artifact,
        version="v1",
    )

    stored = registry.artifact_path(
        "v1",
        "model",
    )

    stored.write_bytes(
        b"tampered-model"
    )

    verification = registry.verify(
        "v1"
    )

    assert verification["valid"] is False
    assert (
        verification["artifacts"][0]["valid"]
        is False
    )


def test_rejected_candidate_cannot_be_promoted(
    tmp_path,
):
    registry = ModelRegistry(
        tmp_path / "registry"
    )

    artifact = make_artifact(
        tmp_path
    )

    register_model(
        registry,
        artifact,
        version="v1",
    )

    with pytest.raises(
        ValueError,
        match="accepted evaluation",
    ):
        registry.promote(
            "v1",
            approval={
                "accepted": False,
                "decision": "REJECT",
            },
        )

    assert registry.production() is None
    assert (
        registry.get("v1")["status"]
        == "candidate"
    )


def test_verified_candidate_can_become_production(
    tmp_path,
):
    registry = ModelRegistry(
        tmp_path / "registry"
    )

    artifact = make_artifact(
        tmp_path
    )

    register_model(
        registry,
        artifact,
        version="v1",
    )

    promoted = registry.promote(
        "v1",
        approval={
            "accepted": True,
            "decision": "PROMOTE",
            "metric": "mae",
            "relative_improvement": 0.12,
        },
    )

    assert promoted["status"] == "production"
    assert promoted["promotion"]["decision"] == "PROMOTE"

    production = registry.production()

    assert production is not None
    assert production["version"] == "v1"


def test_corrupted_candidate_cannot_be_promoted(
    tmp_path,
):
    registry = ModelRegistry(
        tmp_path / "registry"
    )

    artifact = make_artifact(
        tmp_path
    )

    register_model(
        registry,
        artifact,
        version="v1",
    )

    stored = registry.artifact_path(
        "v1",
        "model",
    )

    stored.write_bytes(
        b"corrupted"
    )

    with pytest.raises(
        ValueError,
        match="integrity verification",
    ):
        registry.promote(
            "v1",
            approval={
                "accepted": True,
            },
        )

    assert registry.production() is None


def test_new_production_archives_previous_model(
    tmp_path,
):
    registry = ModelRegistry(
        tmp_path / "registry"
    )

    artifact_v1 = make_artifact(
        tmp_path,
        name="model-v1.keras",
        content=b"version-one",
    )

    artifact_v2 = make_artifact(
        tmp_path,
        name="model-v2.keras",
        content=b"version-two",
    )

    register_model(
        registry,
        artifact_v1,
        version="v1",
    )

    registry.promote(
        "v1",
        approval={
            "accepted": True,
            "decision": "PROMOTE",
        },
    )

    register_model(
        registry,
        artifact_v2,
        version="v2",
    )

    registry.promote(
        "v2",
        approval={
            "accepted": True,
            "decision": "PROMOTE",
        },
    )

    assert (
        registry.get("v1")["status"]
        == "archived"
    )

    assert (
        registry.get("v2")["status"]
        == "production"
    )

    assert (
        registry.production()["version"]
        == "v2"
    )


def test_failed_registration_is_cleaned_up(
    tmp_path,
):
    registry_root = (
        tmp_path
        / "registry"
    )

    registry = ModelRegistry(
        registry_root
    )

    missing_artifact = (
        tmp_path
        / "does-not-exist.keras"
    )

    with pytest.raises(
        FileNotFoundError,
        match="Artifact not found",
    ):
        registry.register(
            version="broken-v1",
            artifacts={
                "model": missing_artifact,
            },
            metrics={
                "mae": 2.0,
            },
            feature_columns=[
                "total_power",
            ],
            training_config={},
        )

    assert not (
        registry_root
        / "broken-v1"
    ).exists()
