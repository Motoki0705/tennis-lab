"""Hash-verified stage caches and atomic receipts for unattended reconstruction."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from src.utils.checksum import dual_sha256


def json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported artifact value {type(value).__name__}")


def document_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, suffix=".partial", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(json_value(value), handle, ensure_ascii=False, sort_keys=True, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


class PipelineArtifactStore:
    def __init__(self, root: Path, *, source: str = "execute", overwrite: bool = False) -> None:
        if source not in ("execute", "load"):
            raise ValueError("Pipeline cache source must be execute or load")
        self.root, self.source, self.overwrite = root, source, overwrite
        self.references: dict[str, dict[str, str]] = {}

    def _paths(self, stage: str) -> tuple[Path, Path]:
        if not stage or Path(stage).name != stage or stage in (".", ".."):
            raise ValueError("Stage cache name must be one path component")
        return self.root / f"{stage}.npz", self.root / f"{stage}.json"

    def load(self, stage: str, identity: Any) -> dict[str, Any] | None:
        arrays_path, receipt_path = self._paths(stage)
        if self.overwrite and self.source == "execute":
            return None
        if not receipt_path.exists():
            if self.source == "load":
                raise FileNotFoundError(f"Completed stage cache missing: {receipt_path}")
            if arrays_path.exists():
                raise ValueError(f"Incomplete stage cache; explicitly overwrite: {arrays_path}")
            return None
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("schema") != "tennis_scene_stage_v2" or receipt.get("identity_sha256") != document_digest(identity):
            raise ValueError(f"Stale stage cache; explicitly overwrite: {receipt_path}")
        if not arrays_path.is_file() or dual_sha256(arrays_path) != receipt.get("arrays_sha256"):
            raise ValueError(f"Stage cache content digest mismatch: {arrays_path}")
        with np.load(arrays_path, allow_pickle=False) as archive:
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
        def unpack(value: Any) -> Any:
            if isinstance(value, dict):
                if set(value) == {"array"}:
                    return arrays[value["array"]]
                return {k: unpack(v) for k, v in value.items()}
            if isinstance(value, list):
                return [unpack(v) for v in value]
            return value
        result = unpack(receipt["payload"])
        if not isinstance(result, dict):
            raise ValueError("Stage cache payload must be a mapping")
        self.references[stage] = {"path": str(arrays_path), "sha256": receipt["arrays_sha256"], "identity_sha256": receipt["identity_sha256"]}
        return result

    def save(self, stage: str, identity: Any, payload: dict[str, Any]) -> None:
        if self.source == "load":
            raise RuntimeError("A load-only stage cache cannot execute missing work")
        identity_digest = document_digest(identity)
        identity_document = json_value(identity)
        arrays_path, receipt_path = self._paths(stage)
        self.root.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        def pack(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                if value.dtype.hasobject:
                    raise TypeError("Object arrays are forbidden in stage caches")
                name = f"array_{len(arrays)}"
                arrays[name] = value
                return {"array": name}
            if is_dataclass(value) and not isinstance(value, type):
                return {field.name: pack(getattr(value, field.name)) for field in fields(value)}
            if isinstance(value, dict):
                return {str(k): pack(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [pack(v) for v in value]
            return json_value(value)
        document = pack(payload)
        with tempfile.NamedTemporaryFile("wb", dir=self.root, suffix=".partial", delete=False) as handle:
            temporary = Path(handle.name)
            try:
                np.savez_compressed(handle, allow_pickle=False, **cast(dict[str, Any], arrays))
                handle.flush()
                os.fsync(handle.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        os.replace(temporary, arrays_path)
        digest = dual_sha256(arrays_path)
        receipt = {"schema": "tennis_scene_stage_v2", "identity_sha256": identity_digest, "identity": identity_document, "arrays_sha256": digest, "payload": document}
        write_json_atomic(receipt_path, receipt)
        self.references[stage] = {"path": str(arrays_path), "sha256": digest, "identity_sha256": receipt["identity_sha256"]}
