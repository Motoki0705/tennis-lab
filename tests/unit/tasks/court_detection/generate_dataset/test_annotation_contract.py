"""Strict persisted court-annotation format and DATA path contract."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from src.tasks.court_detection.generate_dataset.annotation_session import (
    AnnotationSessionConfig,
    find_image_for_entry,
    read_annotation_document,
)
from src.utils.configuration import PathContractError, PathResolver, RuntimePathRoots


def _resolver(tmp_path: Path) -> PathResolver:
    return PathResolver(
        RuntimePathRoots(
            project_root=tmp_path / "project",
            data_root=tmp_path / "data",
            checkpoint_root=tmp_path / "checkpoint",
            artifact_root=tmp_path / "artifact",
            output_root=tmp_path / "output",
            cache_root=tmp_path / "cache",
            external_asset_root=tmp_path / "external",
        )
    )


def test_annotation_document_rejects_legacy_list(tmp_path: Path) -> None:
    path = tmp_path / "legacy.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="object containing 'items'"):
        read_annotation_document(path)


def test_annotation_document_accepts_only_wrapped_object_items(tmp_path: Path) -> None:
    path = tmp_path / "annotations.json"
    path.write_text(
        json.dumps({"schema_name": "court_youtube_keypoints_v2", "items": []}),
        encoding="utf-8",
    )

    document = read_annotation_document(path)

    assert document.items == []
    assert document.metadata == {"schema_name": "court_youtube_keypoints_v2"}


def test_entry_image_path_is_required_and_data_role_bound(tmp_path: Path) -> None:
    resolver = _resolver(tmp_path)
    image = resolver.roots.data_root / "court/youtube/frames/frame.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    config = cast(
        AnnotationSessionConfig,
        SimpleNamespace(
            resolver=resolver,
            root_fragment="court/youtube",
            image_path_key="image_path",
        ),
    )

    assert find_image_for_entry(
        {"image_path": "frames/frame.jpg"}, "frame", config
    ) == image
    with pytest.raises(ValueError, match="missing"):
        find_image_for_entry({}, "frame", config)
    with pytest.raises(PathContractError, match="must be relative"):
        find_image_for_entry({"image_path": "/etc/passwd"}, "frame", config)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        find_image_for_entry({"image_path": "frames/missing.jpg"}, "frame", config)
