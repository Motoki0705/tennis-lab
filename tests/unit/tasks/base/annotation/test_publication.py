"""Ownership and immutability at the shared directory publication boundary."""

from pathlib import Path

import pytest

from src.tasks.base.annotation.artifacts import write_document
from src.tasks.base.annotation.contracts import (
    ArtifactRef,
    Document,
    PublicationReceipt,
)
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.publication import publish_directory


class Snapshot(Document):
    value: int


def test_publication_receipt_requires_one_explicit_teacher_input_and_preserves_legacy_bytes() -> None:
    reference = ArtifactRef(role="annotation", path="annotations/original.json", digest="a" * 64, size_bytes=10)
    payload = dict(task="ball", product="frames", request_id="b" * 64, base_snapshot=None,
                   snapshot=reference, adoptions=(), added_clips=("c" * 64,), reused_clips=())
    with pytest.raises(ValueError, match="adoption receipts or saved annotations"):
        PublicationReceipt(**payload)
    direct = PublicationReceipt(**dict(payload, annotations=(reference,)))
    assert direct.adoptions == ()
    assert PublicationReceipt.model_validate_json(direct.canonical_bytes()) == direct
    with pytest.raises(ValueError, match="one explicit teacher input mode"):
        PublicationReceipt(**dict(payload, annotations=(reference,), adoptions=(reference,)))
    with pytest.raises(ValueError, match="annotation revisions must be unique"):
        PublicationReceipt(**dict(payload, annotations=(reference, reference)))
    legacy = PublicationReceipt(**dict(payload, adoptions=(reference,)))
    assert "annotations" not in legacy.model_dump(mode="json")
    assert PublicationReceipt.model_validate_json(legacy.canonical_bytes()).canonical_bytes() == legacy.canonical_bytes()


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
