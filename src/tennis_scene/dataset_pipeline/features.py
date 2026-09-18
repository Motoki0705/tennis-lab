"""Reject stale RGB feature caches before an incremental dataset build."""

from __future__ import annotations

import json
from dataclasses import asdict

from src.tasks.slcs.data.dino_tokens import (
    DinoTokenSpec,
    dino_dir,
    load_dino_tokens,
)
from src.tennis_scene.generate_dataset.manifest import ClipManifest


def validated_feature_cache(
    clip: ClipManifest,
    spec: DinoTokenSpec,
    identity: dict[str, object],
) -> bool:
    marker = dino_dir(clip.clip_dir) / "annotation.json"
    if not marker.exists():
        return False
    saved = json.loads(marker.read_text())
    expected = {
        "generator": identity,
        "spec": asdict(spec),
        "input_manifest_digest": clip.digest(),
        "camera_ids": sorted(clip.camera_ids),
    }
    actual = {**saved, "camera_ids": sorted(saved["cameras"])}
    changed = {
        key: {"saved": actual.get(key), "expected": value}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if changed:
        raise ValueError(
            f"Stale RGB features for {clip.clip_id}; use a new dataset version; "
            f"mismatches={changed}"
        )
    for camera in clip.camera_ids:
        load_dino_tokens(clip, camera, expected_spec=spec)
    return True
