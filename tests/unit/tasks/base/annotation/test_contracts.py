from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from src.tasks.base.annotation.contracts import (
    ArtifactRef,
    ClipManifest,
    ProductOutcome,
    ReviewEvidence,
)


def reference() -> ArtifactRef:
    return ArtifactRef(
        role="annotation",
        path="media/clips/a/video.mp4",
        digest="a" * 64,
        size_bytes=12,
    )


def manifest_payload() -> dict[str, object]:
    media = reference().model_dump()
    return {
        "source": {
            "provider": "youtube",
            "channel_id": "channel",
            "video_id": "video",
            "media": media,
            "origin_url": "https://example.test/video",
        },
        "media": media,
        "preparation_digest": "b" * 64,
        "width": 64,
        "height": 32,
        "time_base": "1/90000",
        "nominal_fps": "30",
        "source_range": {"start": 40, "stop": 44},
        "target_range": {"start": 1, "stop": 3},
        "frames": [
            {
                "frame_index": i,
                "source_frame_index": 40 + i,
                "source_pts": 120000 + 3000 * i,
                "clip_pts": 3000 * i,
                "duration_pts": 3000,
            }
            for i in range(4)
        ],
    }


def test_clip_identity_survives_reencoding_but_manifest_digest_changes() -> None:
    clip = ClipManifest.model_validate_json(json.dumps(manifest_payload()))
    payload = clip.model_dump(mode="json")
    payload["media"]["digest"] = "c" * 64
    changed = ClipManifest.model_validate_json(json.dumps(payload))
    assert clip.clip_id == changed.clip_id
    assert clip.sha256 != changed.sha256
    payload["target_range"]["start"] = 0
    changed_target = ClipManifest.model_validate_json(json.dumps(payload))
    assert changed_target.clip_id == clip.clip_id
    assert changed_target.sha256 != clip.sha256


@pytest.mark.parametrize("mutation", ["missing", "pts", "duration", "order", "target"])
def test_clip_rejects_inconsistent_frame_mapping(mutation: str) -> None:
    # Decode a valid document to avoid duplicating production validation in this test.
    payload = ClipManifest.model_validate_json(
        json.dumps(manifest_payload())
    ).model_dump(mode="json")
    if mutation == "missing":
        payload["frames"].pop()
    elif mutation == "pts":
        payload["frames"][2]["clip_pts"] += 1
    elif mutation == "duration":
        payload["frames"][1]["duration_pts"] -= 1
    elif mutation == "order":
        payload["frames"][0]["frame_index"] = 1
    else:
        payload["target_range"]["stop"] = 5
    with pytest.raises(ValidationError):
        ClipManifest.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize(
    "path", ["../outside", "/absolute", "a/../b", "a//b", ".", "a\\b"]
)
def test_artifact_rejects_ambiguous_or_escaping_paths(path: str) -> None:
    with pytest.raises(ValidationError):
        ArtifactRef(role="annotation", path=path, digest="a" * 64, size_bytes=0)


def test_presentation_and_adoption_do_not_imply_review() -> None:
    review = ReviewEvidence.model_validate_json(
        json.dumps(
            {
                "clip_id": "b" * 64,
                "prediction": reference().model_dump(),
                "revision": reference().model_dump(),
                "presented_ranges": [{"start": 0, "stop": 10}],
                "reviewed_ranges": [],
                "unresolved_ranges": [{"start": 0, "stop": 10}],
                "evidence": [],
            }
        )
    )
    assert not review.reviewed_ranges
    with pytest.raises(ValidationError):
        review.reviewed_ranges = review.presented_ranges


@pytest.mark.parametrize("status", ["pending", "blocked", "failed"])
def test_incomplete_products_cannot_complete_annotation(status: str) -> None:
    outcome = ProductOutcome.model_validate_json(
        json.dumps(
            {
                "product": "pose",
                "status": status,
                "publication": None,
                "reason": "needs input",
            }
        )
    )
    assert not outcome.annotation_complete


def test_publication_and_exclusion_need_different_evidence() -> None:
    with pytest.raises(ValidationError):
        ProductOutcome(
            product="pose", status="published", publication=None, reason=None
        )
    with pytest.raises(ValidationError):
        ProductOutcome(
            product="pose", status="not_applicable", publication=None, reason=" "
        )
    assert ProductOutcome(
        product="pose",
        status="not_applicable",
        publication=None,
        reason="No approved observations",
    ).annotation_complete
