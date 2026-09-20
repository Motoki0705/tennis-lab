from pathlib import Path

import pytest

from src.tasks.slcs.data.annotation import load_slcs_annotation
from src.tennis_scene.archive import save_scene_result
from src.tennis_scene.dataset_pipeline.provenance import validated_scene_cache
from src.tennis_scene.generate_dataset.manifest import ClipManifest
from tests.support.tasks.slcs.dataset import (
    SLCSFixtureDatasetConfig,
    build_slcs_dataset_fixture,
)


def test_completed_scene_rejects_another_teacher_or_unidentified_cache(
    tmp_path: Path,
) -> None:
    index = build_slcs_dataset_fixture(tmp_path, SLCSFixtureDatasetConfig())
    clip = ClipManifest.load(index.clip_dir(index.clips[0]))
    identity = {
        "checkpoints": {"plcs": "first", "blcs": "ball"},
        "observations": {"ball": "same"},
    }
    with pytest.raises(ValueError, match="Stale reconstruction"):
        validated_scene_cache(clip, identity)
    scene = load_slcs_annotation(clip)
    scene.metadata["dataset_producer_identity"] = identity
    scene.metadata["checkpoints"] = {
        "plcs": {"sha256": "first"},
        "blcs": {"sha256": "ball"},
    }
    save_scene_result(scene, clip.clip_dir / "annotations/tennis_scene/scene.npz")
    assert validated_scene_cache(clip, identity)
    with pytest.raises(ValueError, match="Stale reconstruction"):
        validated_scene_cache(clip, {**identity, "checkpoints": {"plcs": "second"}})
    with pytest.raises(ValueError, match="Stale reconstruction"):
        validated_scene_cache(clip, {**identity, "observations": {"ball": "changed"}})


@pytest.mark.parametrize("name", ["human_kp_vis", "court_vis"])
@pytest.mark.parametrize("value", [1.046875, -0.01, float("nan"), float("inf")])
def test_cached_scene_rejects_invalid_visibility(tmp_path, name, value):
    index = build_slcs_dataset_fixture(
        tmp_path, SLCSFixtureDatasetConfig(videos=("video_000",))
    )
    clip = ClipManifest.load(index.clip_dir(index.clips[0]))
    identity = {"checkpoints": {"plcs": "first", "blcs": "ball"}}
    scene = load_slcs_annotation(clip)
    scene.metadata["dataset_producer_identity"] = identity
    scene.metadata["checkpoints"] = {
        "plcs": {"sha256": "first"},
        "blcs": {"sha256": "ball"},
    }
    getattr(scene, name).flat[0] = value
    save_scene_result(scene, clip.clip_dir / "annotations/tennis_scene/scene.npz")
    with pytest.raises(
        ValueError,
        match=f"Invalid reconstruction visibility {name}.*new dataset version",
    ):
        validated_scene_cache(clip, identity)
