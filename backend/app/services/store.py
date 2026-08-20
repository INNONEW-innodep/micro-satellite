from __future__ import annotations

import threading
import zipfile
from pathlib import Path

from pydantic import ValidationError

from ..schemas import PredictionResult


class ResultNotFoundError(KeyError):
    pass


class ArtifactNotFoundError(KeyError):
    pass


class FileResultStore:
    """Thread-safe in-memory index backed by one directory per prediction."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._items: dict[str, PredictionResult] = {}
        self.load_errors: list[str] = []
        self._load_existing()

    def create_directory(self, prediction_id: str) -> Path:
        directory = self.root / prediction_id
        directory.mkdir(parents=False, exist_ok=False)
        return directory

    def save(self, result: PredictionResult) -> None:
        directory = self.root / result.id
        directory.mkdir(parents=True, exist_ok=True)
        manifest = directory / "result.json"
        temporary = directory / ".result.json.tmp"
        temporary.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(manifest)
        with self._lock:
            self._items[result.id] = result

    def get(self, prediction_id: str) -> PredictionResult:
        with self._lock:
            result = self._items.get(prediction_id)
        if result is None:
            raise ResultNotFoundError(prediction_id)
        return result

    def list(self) -> list[PredictionResult]:
        with self._lock:
            return sorted(
                self._items.values(), key=lambda item: item.created_at, reverse=True
            )

    def artifact_path(self, prediction_id: str, filename: str) -> tuple[Path, str]:
        result = self.get(prediction_id)
        if filename == "result.json":
            manifest = self.root / prediction_id / filename
            if not manifest.is_file():
                raise ArtifactNotFoundError(filename)
            return manifest, "application/json"
        artifact = next(
            (item for item in result.artifacts if item.name == filename), None
        )
        if artifact is None:
            raise ArtifactNotFoundError(filename)
        path = self.root / prediction_id / artifact.name
        if not path.is_file():
            raise ArtifactNotFoundError(filename)
        return path, artifact.media_type

    def bundle_path(self, prediction_id: str) -> Path:
        result = self.get(prediction_id)
        directory = self.root / prediction_id
        bundle = directory / f"prediction_{prediction_id}.zip"
        with (
            self._lock,
            zipfile.ZipFile(
                bundle, mode="w", compression=zipfile.ZIP_DEFLATED
            ) as archive,
        ):
            manifest = directory / "result.json"
            archive.write(manifest, arcname="result.json")
            for artifact in result.artifacts:
                path = directory / artifact.name
                if path.is_file():
                    archive.write(path, arcname=artifact.name)
        return bundle

    def _load_existing(self) -> None:
        for manifest in sorted(self.root.glob("*/result.json")):
            try:
                result = PredictionResult.model_validate_json(
                    manifest.read_text(encoding="utf-8")
                )
            except (OSError, ValidationError) as exc:
                self.load_errors.append(f"{manifest}: {exc}")
                continue
            self._items[result.id] = result
