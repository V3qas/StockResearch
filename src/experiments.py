from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

from src.config import PROJECT_ROOT


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


class ExperimentRun:
    """Isolated artifacts; publish one pointer only after successful completion.

    An interrupted/failed run remains inspectable in its own directory. Readers
    resolve latest.json once and then use only files inside that run.
    """

    def __init__(self, runs_dir: Path, metadata: dict, source_root: Path = PROJECT_ROOT):
        self.runs_dir = runs_dir
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:12]
        self.path = runs_dir / self.run_id
        self.raw = self.path / "raw"
        self.processed = self.path / "processed"
        self.predictions = self.path / "predictions"
        self.source_root = source_root
        self.manifest = {
            "schema_version": 3,
            "run_id": self.run_id,
            "status": "running",
            "started_at": _utc_now(),
            "parameters": metadata,
            "python_version": platform.python_version(),
            "versions": {name: version(name) for name in (
                "pandas", "numpy", "yfinance", "scikit-learn", "xgboost", "pyarrow", "exchange-calendars",
            )},
        }

    def __enter__(self) -> ExperimentRun:
        self.path.mkdir(parents=True, exist_ok=False)
        _atomic_json(self.path / "manifest.json", self.manifest)
        try:
            for directory in (self.raw, self.processed, self.predictions):
                directory.mkdir()
            source_files = [self.source_root / "main.py", self.source_root / "requirements.txt"]
            source_files.extend(sorted((self.source_root / "src").rglob("*.py")))
            for source in source_files:
                destination = self.path / "source" / source.relative_to(self.source_root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
        except BaseException as error:
            self._record_failure(error)
            raise
        return self

    def _record_failure(self, error: BaseException) -> None:
        self.manifest.update(status="failed", finished_at=_utc_now(), error=f"{type(error).__name__}: {error}")
        try:
            _atomic_json(self.path / "manifest.json", self.manifest)
        except OSError:
            # Preserve the original error if the filesystem itself is failing.
            pass

    def __exit__(self, exc_type, error, traceback) -> bool:
        if error is not None:
            self._record_failure(error)
            return False
        try:
            artifacts = {
                path.relative_to(self.path).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(self.path.rglob("*"))
                if path.is_file() and path != self.path / "manifest.json"
            }
            self.manifest.update(status="complete", finished_at=_utc_now(), artifacts_sha256=artifacts)
            _atomic_json(self.path / "manifest.json", self.manifest)
            _atomic_json(self.runs_dir / "latest.json", {"run_id": self.run_id})
        except BaseException as error:
            self._record_failure(error)
            raise
        return False
