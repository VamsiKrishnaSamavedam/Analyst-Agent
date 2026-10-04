"""Local, UUID-addressed storage for uploaded and cleaned datasets."""

from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path

import pandas as pd

from ai_agent.models import DatasetProfile, ValidationReport
from ai_agent.services.quality import read_dataset, write_dataset


class DatasetStore:
    """Persist datasets locally without trusting user-provided path fragments."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def create_from_file(self, source: Path, original_name: str) -> tuple[str, pd.DataFrame]:
        dataset_id = str(uuid.uuid4())
        target = self._folder(dataset_id) / f"raw{source.suffix.lower()}"
        target.parent.mkdir(parents=True)
        target.write_bytes(source.read_bytes())
        frame = read_dataset(target)
        self._write_metadata(dataset_id, {"original_name": Path(original_name).name, "raw": target.name, "active": target.name})
        return dataset_id, frame

    def create_from_frame(self, frame: pd.DataFrame, original_name: str) -> tuple[str, pd.DataFrame]:
        dataset_id = str(uuid.uuid4())
        folder = self._folder(dataset_id)
        folder.mkdir(parents=True)
        target = folder / "raw.csv"
        write_dataset(frame, target)
        self._write_metadata(dataset_id, {"original_name": Path(original_name).name, "raw": target.name, "active": target.name})
        return dataset_id, frame

    def active_frame(self, dataset_id: str) -> pd.DataFrame:
        metadata = self._metadata(dataset_id)
        return read_dataset(self._folder(dataset_id) / metadata["active"])

    def raw_frame(self, dataset_id: str) -> pd.DataFrame:
        """Return the originally received dataset, even after cleaned output exists."""

        metadata = self._metadata(dataset_id)
        raw_name = str(metadata.get("raw", metadata["active"]))
        return read_dataset(self._folder(dataset_id) / raw_name)

    def save_cleaned(self, dataset_id: str, frame: pd.DataFrame, report: ValidationReport) -> None:
        folder = self._folder(dataset_id)
        output = folder / "cleaned.csv"
        write_dataset(frame, output)
        metadata = self._metadata(dataset_id)
        metadata.update({"active": output.name, "validation": report.model_dump(mode="json")})
        self._write_metadata(dataset_id, metadata)

    def save_profile(self, dataset_id: str, dataset_profile: DatasetProfile) -> None:
        metadata = self._metadata(dataset_id)
        metadata["profile"] = dataset_profile.model_dump(mode="json")
        self._write_metadata(dataset_id, metadata)

    def metadata(self, dataset_id: str) -> dict[str, object]:
        return self._metadata(dataset_id)

    def download_path(self, dataset_id: str) -> Path:
        metadata = self._metadata(dataset_id)
        return self._folder(dataset_id) / str(metadata["active"])

    def purge_older_than(self, seconds: int) -> int:
        """Delete expired UUID dataset folders; never follow user-provided paths."""

        cutoff = time.time() - seconds
        removed = 0
        for folder in self.root.iterdir():
            if not folder.is_dir() or folder.stat().st_mtime >= cutoff:
                continue
            try:
                uuid.UUID(folder.name)
            except ValueError:
                continue
            shutil.rmtree(folder)
            removed += 1
        return removed

    def _folder(self, dataset_id: str) -> Path:
        try:
            return self.root / str(uuid.UUID(dataset_id))
        except ValueError as error:
            raise ValueError("Dataset was not found.") from error

    def _metadata(self, dataset_id: str) -> dict[str, object]:
        metadata_path = self._folder(dataset_id) / "metadata.json"
        if not metadata_path.is_file():
            raise ValueError("Dataset was not found.")
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    def _write_metadata(self, dataset_id: str, content: dict[str, object]) -> None:
        (self._folder(dataset_id) / "metadata.json").write_text(json.dumps(content, indent=2), encoding="utf-8")
