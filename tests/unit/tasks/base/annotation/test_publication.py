"""Ownership and immutability at the shared directory publication boundary."""

from pathlib import Path

import pytest

from src.tasks.base.annotation.artifacts import write_document
from src.tasks.base.annotation.contracts import Document
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.publication import publish_directory


class Snapshot(Document):
    value: int


def test_foreign_or_unowned_staging_is_never_deleted(tmp_path: Path):
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    intent = write_document(
        layout, layout.path("campaigns", "intent.json"), Snapshot(value=1)
    )
    staging = tmp_path / "data/player_detection/.example.staging"
    staging.mkdir(parents=True)
    sentinel = staging / "keep.txt"
    sentinel.write_text("another publication")

    def build(*_):
        pytest.fail("foreign staging must not run a builder")

    def validate(*_):
        pytest.fail("foreign staging must not be read as a snapshot")

    with pytest.raises(ValueError, match="no verifiable owner"):
        publish_directory(
            layout,
            "player_detection/example",
            intent,
            kind=Snapshot,
            build=build,
            validate=validate,
        )
    assert sentinel.read_text() == "another publication"
    owner = intent.model_copy(update={"digest": "f" * 64})
    (staging / "owner.json").write_bytes(owner.canonical_bytes())
    with pytest.raises(ValueError, match="owned by another"):
        publish_directory(
            layout,
            "player_detection/example",
            intent,
            kind=Snapshot,
            build=build,
            validate=validate,
        )
    assert sentinel.exists()


def test_completed_snapshot_tamper_is_not_rebuilt(tmp_path: Path):
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    intent = write_document(
        layout, layout.path("campaigns", "intent.json"), Snapshot(value=1)
    )
    builds = []

    def build(*_):
        builds.append(True)
        return Snapshot(value=1)

    def validate(value, _):
        if value.value != 1:
            raise ValueError("snapshot changed")

    reference, _ = publish_directory(
        layout,
        "player_detection/example",
        intent,
        kind=Snapshot,
        build=build,
        validate=validate,
    )
    layout.artifact_path(reference).write_bytes(Snapshot(value=2).canonical_bytes())
    with pytest.raises(ValueError, match="snapshot changed"):
        publish_directory(
            layout,
            "player_detection/example",
            intent,
            kind=Snapshot,
            build=build,
            validate=validate,
        )
    assert builds == [True]


def test_symlinked_staging_is_rejected(tmp_path: Path):
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    intent = write_document(
        layout, layout.path("campaigns", "intent.json"), Snapshot(value=1)
    )
    parent = tmp_path / "data/player_detection"
    parent.mkdir(parents=True)
    outside = tmp_path / "untouched"
    outside.mkdir()
    (parent / ".example.staging").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        publish_directory(
            layout,
            "player_detection/example",
            intent,
            kind=Snapshot,
            build=lambda *_: Snapshot(value=1),
            validate=lambda *_: None,
        )
    assert list(outside.iterdir()) == []
