"""Create small distribution files; transfer immutable payloads separately."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from src.tasks.ball_detection.data.store import (
    FRAME_FIELDS,
    INSTANCE_FIELDS,
    SCHEMA_VERSION,
    BallFrameStore,
    shard_name,
)
from src.tasks.ball_detection.data.teacher_masks import (
    EXPLICIT_MASK_SCHEMA,
    MASK_FIELDS,
)
from src.utils.checksum import dual_sha256
from src.utils.data.verification import file_version

from .manifest import DISTRIBUTION_FILE, DISTRIBUTION_SCHEMA, relative_file


def prepare_distribution(
    source_directory: Path, directory: Path, *, version: str,
    clip_ids: Sequence[str], inputs: Mapping[str, Path],
) -> dict[str, Any]:
    """Publish metadata/index/annotations/inputs with a complete content manifest.

    JPEG shards and video payloads are hashed in place, never copied or linked.
    ``inputs`` maps distribution-relative paths below ``inputs/`` to caller-owned
    small files (for example split rules and window-manifest inputs).
    """
    if not source_directory.is_absolute() or not directory.is_absolute():
        raise ValueError("Distribution source and destination must be absolute")
    if directory.exists():
        raise FileExistsError(directory)
    if directory.is_relative_to(source_directory) or source_directory.is_relative_to(directory):
        raise ValueError("Distribution and source directories must be separate")
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", version):
        raise ValueError("Distribution needs an explicit dataset version name")
    tracked = {source_directory / name: file_version(source_directory / name) for name in ("metadata.json", "index.npz")}
    identity = {path.name: dual_sha256(path) for path in tracked}
    source = BallFrameStore(source_directory)
    if source.metadata["schema_version"] not in (SCHEMA_VERSION, EXPLICIT_MASK_SCHEMA):
        raise ValueError("Distribution preparation supports JPEG-backed v1/v2 stores only")
    source_version = source.metadata.get("version")
    if version == source_version:
        raise ValueError("Distribution version must differ from its source")
    if not clip_ids or len(set(clip_ids)) != len(clip_ids):
        raise ValueError("Distribution needs a nonempty unique clip selection")
    selected = sorted((source.clip_by_id(value) for value in clip_ids), key=lambda clip: clip.index)
    for value in inputs:
        if relative_file(value).parts[0] != "inputs":
            raise ValueError("Additional distribution inputs must be below inputs/")
    frame_rows = np.concatenate([source.clip_rows(clip) for clip in selected])
    instance_rows = np.concatenate([
        np.arange(int(source.frames["inst_start"][rows[0]]),
                  int(source.frames["inst_start"][rows[-1]] + source.frames["inst_count"][rows[-1]]))
        for clip in selected for rows in [source.clip_rows(clip)]
    ])
    with np.load(source_directory / "index.npz") as archive:
        expected = set(FRAME_FIELDS) | set(INSTANCE_FIELDS) | {"xy"}
        if source.teacher_masks is not None:
            expected |= set(MASK_FIELDS)
        if set(archive.files) != expected:
            raise ValueError("Unsupported index columns; refusing to drop unknown data")
        columns = {name: archive[name][frame_rows if name in FRAME_FIELDS or name in MASK_FIELDS else instance_rows].copy()
                   for name in archive.files}
    columns["clip"] = np.repeat(np.arange(len(selected), dtype=np.int32), [c.frame_count for c in selected])
    columns["inst_start"] = np.concatenate([[0], np.cumsum(columns["inst_count"], dtype=np.int64)[:-1]])
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ball-distribution-", dir=directory.parent) as temporary:
        staging = Path(temporary) / "store"
        staging.mkdir()
        metadata: dict[str, Any] = dict(
            schema_version=source.metadata["schema_version"], version=version,
            annotation_path_base="store", clips=[],
        )
        for key in ("jpeg_quality", "max_height", "new_clip_encoding"):
            if key in source.metadata:
                metadata[key] = source.metadata[key]
        files: dict[str, Path] = {}
        mapping = []
        for index, clip in enumerate(selected):
            record = asdict(clip)
            record["index"] = index
            payload = source.image_path(clip)
            tracked[payload] = file_version(payload)
            target = f"shards/{shard_name(index)}"
            files[target] = payload
            mapping.append(dict(index=index, source_index=clip.index, clip_id=clip.clip_id,
                                payload=target, source_payload=payload.relative_to(source_directory).as_posix()))
            if clip.source == "chat_annotation":
                annotation = source.annotation_path(clip)
                tracked[annotation] = file_version(annotation)
                if dual_sha256(annotation) != clip.annotation_sha256:
                    raise ValueError("Original Chat annotation checksum differs from its source store")
                target = f"annotations/clip-{index:05d}.json"
                (staging / "annotations").mkdir(exist_ok=True)
                shutil.copyfile(annotation, staging / target)
                record["annotation_path"] = target
            metadata["clips"].append(record)
        (staging / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        savez: Any = np.savez
        savez(staging / "index.npz", **columns)
        for target, path in inputs.items():
            if not path.is_absolute() or not path.is_file():
                raise ValueError("Distribution inputs must be existing absolute file paths")
            tracked[path] = file_version(path)
            destination = staging / relative_file(target)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
        files.update({p.relative_to(staging).as_posix(): p for p in staging.rglob("*") if p.is_file()})
        manifest = dict(
            schema_version=DISTRIBUTION_SCHEMA,
            source=dict(version=source_version, hashes=identity), clips=mapping,
            files=[dict(path=name, size_bytes=path.stat().st_size, sha256=dual_sha256(path))
                   for name, path in sorted(files.items())],
        )
        (staging / DISTRIBUTION_FILE).write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n")
        BallFrameStore(staging, external_payloads={c.clip_id: source.image_path(c) for c in selected})
        if any(file_version(path) != before for path, before in tracked.items()):
            raise ValueError("A distribution source changed during preparation")
        if directory.exists():
            raise FileExistsError(directory)
        os.rename(staging, directory)
    return manifest
