from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


class ModelRegistry:
    VALID_STATUSES = {
        "candidate",
        "production",
        "archived",
    }

    def __init__(
        self,
        root: str | Path = "models/registry",
    ):
        self.root = Path(root)
        self.root.mkdir(
            parents=True,
            exist_ok=True,
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()

        with path.open("rb") as file:
            for chunk in iter(
                lambda: file.read(1024 * 1024),
                b"",
            ):
                digest.update(chunk)

        return digest.hexdigest()

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): ModelRegistry._json_safe(item)
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple)):
            return [
                ModelRegistry._json_safe(item)
                for item in value
            ]

        if isinstance(value, np.ndarray):
            return value.tolist()

        if isinstance(value, np.generic):
            return value.item()

        if isinstance(value, Path):
            return str(value)

        return value

    @staticmethod
    def _safe_artifact_name(name: str) -> str:
        cleaned = "".join(
            character
            if character.isalnum() or character in {"-", "_"}
            else "_"
            for character in name
        ).strip("_")

        if not cleaned:
            raise ValueError(
                "Artifact name cannot be empty"
            )

        return cleaned

    def _manifest_path(
        self,
        version: str,
    ) -> Path:
        return (
            self.root
            / version
            / "manifest.json"
        )

    def _write_json(
        self,
        path: Path,
        payload: dict[str, Any],
    ) -> None:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary = path.with_suffix(
            path.suffix + ".tmp"
        )

        temporary.write_text(
            json.dumps(
                self._json_safe(payload),
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        temporary.replace(path)

    def _new_version(self) -> str:
        timestamp = datetime.now(
            timezone.utc
        ).strftime(
            "%Y%m%dT%H%M%SZ"
        )

        suffix = uuid.uuid4().hex[:8]

        return f"{timestamp}-{suffix}"

    def register(
        self,
        *,
        artifacts: dict[str, str | Path],
        metrics: dict[str, Any],
        feature_columns: list[str],
        training_config: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        version: str | None = None,
    ) -> dict[str, Any]:
        if not artifacts:
            raise ValueError(
                "At least one artifact is required"
            )

        if not feature_columns:
            raise ValueError(
                "feature_columns cannot be empty"
            )

        version = (
            version
            or self._new_version()
        )

        version_dir = (
            self.root
            / version
        )

        if version_dir.exists():
            raise ValueError(
                f"Model version already exists: {version}"
            )

        artifact_dir = (
            version_dir
            / "artifacts"
        )

        artifact_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        artifact_records = []
        safe_names = set()

        try:
            for logical_name, source in artifacts.items():
                source = Path(source)

                if not source.exists():
                    raise FileNotFoundError(
                        f"Artifact not found: {source}"
                    )

                if not source.is_file():
                    raise ValueError(
                        f"Artifact must be a file: {source}"
                    )

                safe_name = self._safe_artifact_name(
                    logical_name
                )

                if safe_name in safe_names:
                    raise ValueError(
                        "Artifact names collide after normalization"
                    )

                safe_names.add(safe_name)

                extension = "".join(
                    source.suffixes
                )

                destination = (
                    artifact_dir
                    / f"{safe_name}{extension}"
                )

                shutil.copy2(
                    source,
                    destination,
                )

                artifact_records.append(
                    {
                        "name": logical_name,
                        "filename": destination.name,
                        "size_bytes": destination.stat().st_size,
                        "sha256": self._sha256(
                            destination
                        ),
                    }
                )

            manifest = {
                "version": version,
                "created_at": datetime.now(
                    timezone.utc
                ).isoformat(),
                "status": "candidate",
                "feature_columns": list(
                    feature_columns
                ),
                "metrics": metrics,
                "training_config": training_config,
                "metadata": metadata or {},
                "artifacts": artifact_records,
            }

            self._write_json(
                version_dir / "manifest.json",
                manifest,
            )

            return manifest

        except Exception:
            shutil.rmtree(
                version_dir,
                ignore_errors=True,
            )
            raise

    def get(
        self,
        version: str,
    ) -> dict[str, Any]:
        manifest_path = self._manifest_path(
            version
        )

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Unknown model version: {version}"
            )

        return json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )

    def list_versions(
        self,
    ) -> list[dict[str, Any]]:
        manifests = []

        for directory in self.root.iterdir():
            if not directory.is_dir():
                continue

            manifest_path = (
                directory
                / "manifest.json"
            )

            if not manifest_path.exists():
                continue

            manifests.append(
                json.loads(
                    manifest_path.read_text(
                        encoding="utf-8"
                    )
                )
            )

        return sorted(
            manifests,
            key=lambda item: item["created_at"],
            reverse=True,
        )

    def verify(
        self,
        version: str,
    ) -> dict[str, Any]:
        manifest = self.get(
            version
        )

        artifact_dir = (
            self.root
            / version
            / "artifacts"
        )

        checks = []

        for artifact in manifest["artifacts"]:
            path = (
                artifact_dir
                / artifact["filename"]
            )

            exists = path.exists()

            actual_hash = (
                self._sha256(path)
                if exists
                else None
            )

            valid = (
                exists
                and actual_hash
                == artifact["sha256"]
            )

            checks.append(
                {
                    "name": artifact["name"],
                    "exists": exists,
                    "expected_sha256": artifact["sha256"],
                    "actual_sha256": actual_hash,
                    "valid": valid,
                }
            )

        return {
            "version": version,
            "valid": all(
                check["valid"]
                for check in checks
            ),
            "artifacts": checks,
        }

    def production(
        self,
    ) -> dict[str, Any] | None:
        pointer = (
            self.root
            / "production.json"
        )

        if not pointer.exists():
            return None

        data = json.loads(
            pointer.read_text(
                encoding="utf-8"
            )
        )

        return self.get(
            data["version"]
        )

    def artifact_path(
        self,
        version: str,
        name: str,
    ) -> Path:
        manifest = self.get(
            version
        )

        for artifact in manifest["artifacts"]:
            if artifact["name"] == name:
                return (
                    self.root
                    / version
                    / "artifacts"
                    / artifact["filename"]
                )

        raise KeyError(
            f"Artifact not registered: {name}"
        )

    def promote(
        self,
        version: str,
        *,
        approval: dict[str, Any],
    ) -> dict[str, Any]:
        if not approval.get(
            "accepted",
            False,
        ):
            raise ValueError(
                "Model cannot be promoted without an accepted evaluation"
            )

        verification = self.verify(
            version
        )

        if not verification["valid"]:
            raise ValueError(
                "Model artifacts failed integrity verification"
            )

        manifest = self.get(
            version
        )

        if manifest["status"] not in {
            "candidate",
            "production",
        }:
            raise ValueError(
                f"Cannot promote model with status: "
                f"{manifest['status']}"
            )

        current = self.production()

        if (
            current is not None
            and current["version"] != version
        ):
            current["status"] = "archived"

            self._write_json(
                self._manifest_path(
                    current["version"]
                ),
                current,
            )

        manifest["status"] = "production"
        manifest["promotion"] = {
            **approval,
            "promoted_at": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        self._write_json(
            self._manifest_path(version),
            manifest,
        )

        self._write_json(
            self.root / "production.json",
            {
                "version": version,
                "updated_at": datetime.now(
                    timezone.utc
                ).isoformat(),
            },
        )

        return manifest
