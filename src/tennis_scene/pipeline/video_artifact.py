"""Explicit acquisition and immutable JSON objects for fixed-camera video products."""
from __future__ import annotations

import fcntl
import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.tennis_scene.generate_dataset.manifest import validate_id_component
from src.tennis_scene.pipeline.artifacts import document_digest, write_json_atomic
from src.utils.configuration import PathResolver, PathRole


def validate_digest(value: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Invalid video artifact content digest')
    return value


def _object_path(root: Path, digest: str) -> Path:
    path = root / 'objects' / f'{validate_digest(digest)}.json'
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Video artifact object escapes its store')
    return path


def load_document(root: Path, digest: str) -> dict[str, Any]:
    path = _object_path(root, digest)
    document = json.loads(path.read_text())
    if not isinstance(document, dict) or document_digest(document) != digest:
        raise ValueError('Video artifact content changed')
    return document


def publish_document(root: Path, document: dict[str, Any], *, repair: bool = False) -> str:
    digest: str = document_digest(document)
    target = _object_path(root, digest)
    if target.exists():
        try:
            load_document(root, digest)
        except ValueError:
            if not repair:
                raise
            write_json_atomic(target, document)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(target, document)
    return digest


@dataclass(frozen=True)
class VideoArtifactConfig:
    action: str
    directory: Path
    dataset_id: str
    video_id: str
    request: Path | None
    request_sha256: str | None
    artifact_digest: str | None
    rebuild: bool

    def __post_init__(self) -> None:
        if self.action not in {'build', 'load'} or not self.dataset_id or not self.video_id:
            raise ValueError('video_shared requires explicit build/load and dataset/video IDs')
        validate_id_component(self.dataset_id, field_name='dataset_id')
        validate_id_component(self.video_id, field_name='video_id')
        for value in (self.request_sha256, self.artifact_digest):
            if value is not None:
                validate_digest(value)
        if self.action == 'build':
            if self.request is None or self.request_sha256 is None or self.artifact_digest is not None:
                raise ValueError('Shared video build requires a pinned request, without a load digest')
        elif self.artifact_digest is None or self.request is not None or self.request_sha256 is not None or self.rebuild:
            raise ValueError('Shared video load requires only a pinned artifact, and forbids rebuild')

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], resolver: PathResolver) -> VideoArtifactConfig:
        if any(value[k] is None for k in ('action', 'directory', 'dataset_id', 'video_id')):
            raise ValueError('video_shared acquisition and scope must be explicit')
        return cls(value['action'], resolver.resolve(PathRole.ARTIFACT, value['directory']), value['dataset_id'], value['video_id'],
                   None if value['request'] is None else resolver.resolve(PathRole.ARTIFACT, value['request']),
                   value['request_sha256'], value['artifact_digest'], value['rebuild'])


@contextmanager
def store_lock(root: Path, name: str) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    with (root / name).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
