"""Explicit offline materialization of source-neutral dense Court targets."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from src.tasks.court_detection.data.contracts import (
    CourtDenseTargetKind,
    CourtSampleRecord,
    CourtSourceSplit,
)
from src.tasks.court_detection.data.inputs.contract import CourtInput
from src.tasks.court_detection.data.target_generation.line import (
    generate_line_target,
)
from src.tasks.court_detection.data.target_generation.segmentation import (
    generate_segmentation_target,
)
from src.tasks.court_detection.data.target_generation.store import (
    LINE_TARGET_SCHEMA,
    SEGMENTATION_TARGET_SCHEMA,
    CourtDerivedTargetStore,
)


@dataclass(frozen=True, slots=True)
class CourtMaterializationResult:
    split: CourtSourceSplit
    target_kind: CourtDenseTargetKind
    written: int


class CourtTargetMaterializer:
    """Generate selected dense targets; Dataset/DataModule never call this class."""

    def __init__(
        self,
        *,
        input_layer: CourtInput,
        target_store: CourtDerivedTargetStore,
    ) -> None:
        self.input_layer = input_layer
        self.target_store = target_store

    def materialize(
        self,
        *,
        splits: tuple[CourtSourceSplit, ...],
        target_kinds: tuple[CourtDenseTargetKind, ...],
    ) -> tuple[CourtMaterializationResult, ...]:
        if not splits or not target_kinds:
            raise ValueError("Court materialization requires splits and dense targets.")
        if (
            len(set(splits)) != len(splits)
            or len(set(target_kinds)) != len(target_kinds)
        ):
            raise ValueError("Court materialization selections must be unique.")
        results: list[CourtMaterializationResult] = []
        for split in splits:
            records = self.input_layer.records(split)
            for kind in target_kinds:
                written = 0
                for record in records:
                    self._materialize_record(record, kind=kind)
                    written += 1
                results.append(
                    CourtMaterializationResult(
                        split=split,
                        target_kind=kind,
                        written=written,
                    )
                )
        return tuple(results)

    def _materialize_record(
        self,
        record: CourtSampleRecord,
        *,
        kind: CourtDenseTargetKind,
    ) -> None:
        raw = self.input_layer.load(record)
        height, width = raw.image.height, raw.image.width
        if kind == "seg":
            array = generate_segmentation_target(
                height=height,
                width=width,
                instances=raw.court_instances,
            )
            schema = SEGMENTATION_TARGET_SCHEMA
        elif kind == "line":
            array = generate_line_target(
                height=height,
                width=width,
                instances=raw.court_instances,
            )
            schema = LINE_TARGET_SCHEMA
        else:  # pragma: no cover - type and selection validation
            raise ValueError(f"Unsupported dense Court target: {kind!r}.")
        path = record.dense_target_refs[kind]
        self._write_png(path, array)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        metadata = {
            "schema": schema,
            "target_kind": kind,
            "source_kind": raw.metadata.source_kind,
            "source_schema": raw.metadata.source_schema,
            "source_sample_id": raw.metadata.source_sample_id,
            "stable_sample_id": raw.sample_id,
            "width": width,
            "height": height,
            "sha256": digest,
        }
        self._write_json(self.target_store.metadata_path(path), metadata)

    @staticmethod
    def _write_png(path: Path, array: np.ndarray) -> None:
        if array.dtype != np.uint8 or array.ndim != 2:
            raise ValueError("Derived Court PNG must be uint8 [H,W].")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp.png",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
            Image.fromarray(array, mode="L").save(temporary, format="PNG")
            os.replace(temporary, path)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @staticmethod
    def _write_json(path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(payload, sort_keys=True, indent=2) + "\n"
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


__all__ = [
    "CourtMaterializationResult",
    "CourtTargetMaterializer",
]
