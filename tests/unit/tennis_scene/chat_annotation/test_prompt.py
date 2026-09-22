from __future__ import annotations

from src.tennis_scene.chat_annotation.prompt import compact_manifest, expand_manifest
from src.tennis_scene.chat_annotation.runtime.contracts import ClipManifest


def test_cfr_manifest_round_trip_is_compact(manifest: ClipManifest) -> None:
    value = compact_manifest(manifest)
    assert value["frame_runs"] == [[0, 12, 0, 1]]
    assert "frames" not in value
    assert expand_manifest(value) == manifest


def test_vfr_manifest_round_trip_preserves_every_timestamp(
    manifest: ClipManifest,
) -> None:
    position = 9
    for index, frame in enumerate(manifest.frames):
        frame.source_pts = position
        frame.clip_pts = position - 9
        frame.duration_pts = 2 if index in (2, 3, 9) else 1
        position += frame.duration_pts
    manifest = ClipManifest.model_validate(manifest.model_dump())
    value = compact_manifest(manifest)
    assert len(value["frame_runs"]) == 5
    assert expand_manifest(value) == manifest
