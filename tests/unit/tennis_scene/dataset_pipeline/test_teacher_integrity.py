"""Teacher receipts must agree before publication and when reusing old archives."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.tasks.slcs.data.annotation import load_slcs_annotation
from src.tennis_scene.archive import save_scene_result
from src.tennis_scene.configuration import ReferenceClipPaths
from src.tennis_scene.dataset_pipeline import build, provenance
from src.tennis_scene.dataset_pipeline.checkpoint_warning import (
    checkpoint_warning_policy,
)
from src.tennis_scene.generate_dataset.manifest import ClipManifest
from tests.support.tasks.slcs.dataset import (
    SLCSFixtureDatasetConfig,
    build_slcs_dataset_fixture,
)


@pytest.fixture
def teacher(tmp_path):
    index = build_slcs_dataset_fixture(tmp_path, SLCSFixtureDatasetConfig())
    clip = ClipManifest.load(index.clip_dir(index.clips[0]))
    scene = load_slcs_annotation(clip)
    identity = {"checkpoints": {"plcs": "a" * 64, "blcs": "b" * 64}}
    scene.metadata["checkpoints"] = {
        role: {"sha256": digest} for role, digest in identity["checkpoints"].items()
    }
    pins = dict.fromkeys(("court", "dino", "vitpose", "dinov3"), "c" * 64)
    pins.update(identity["checkpoints"])
    return clip, scene, identity, pins


@pytest.mark.parametrize("role", ["plcs", "blcs"])
@pytest.mark.parametrize("pinned", [False, True])
def test_runner_rejects_raw_mismatch_before_publication(
    teacher, tmp_path, monkeypatch, role, pinned
):
    clip, scene, identity, pins = teacher
    scene.metadata["checkpoints"][role]["sha256"] = "d" * 64
    before = copy.deepcopy(scene.metadata)
    reconstruct = MagicMock(return_value=scene)
    evaluate = MagicMock(
        side_effect=AssertionError("must fail before quality/publication")
    )
    monkeypatch.setattr(build, "reconstruct", reconstruct)
    monkeypatch.setattr(build, "evaluate_reconstruction", evaluate)
    cfg = OmegaConf.create({"coordinate_mode": "reference"})
    if pinned:
        cfg.checkpoint_sha256 = pins
    # The detector/pose exception must never allow a teacher mismatch.
    with checkpoint_warning_policy(pins, ("dino", "vitpose"), None):
        runner = build._scene_runner(
            cfg,
            cast(ReferenceClipPaths, object()),
            clip,
            tmp_path,
            np.empty(0),
            np.empty(0),
            tmp_path,
            identity,
        )
        with pytest.raises(
            ValueError, match=f"Raw teacher checkpoint mismatch: {role}"
        ):
            runner([clip.media_path(c) for c in clip.camera_ids], clip.camera_ids)
    evaluate.assert_not_called()
    assert scene.metadata == before
    assert not (tmp_path / "refined_scene.npz").exists()


@pytest.mark.parametrize("role", ["plcs", "blcs"])
def test_runner_rejects_identity_that_disagrees_with_pins(
    teacher, tmp_path, monkeypatch, role
):
    clip, _, identity, pins = teacher
    identity["checkpoints"][role] = "d" * 64
    reconstruct = MagicMock()
    monkeypatch.setattr(build, "reconstruct", reconstruct)
    with pytest.raises(ValueError, match=f"Teacher checkpoint pin mismatch: {role}"):
        build._scene_runner(
            OmegaConf.create({"checkpoint_sha256": pins}),
            cast(ReferenceClipPaths, object()),
            clip,
            tmp_path,
            np.empty(0),
            np.empty(0),
            tmp_path,
            identity,
        )
    reconstruct.assert_not_called()


@pytest.mark.parametrize("pinned", [False, True])
def test_runner_returns_valid_teacher_without_changing_raw_receipts(
    teacher, tmp_path, monkeypatch, pinned
):
    clip, scene, identity, pins = teacher
    raw_receipts = copy.deepcopy(scene.metadata["checkpoints"])
    monkeypatch.setattr(build, "reconstruct", MagicMock(return_value=scene))
    monkeypatch.setattr(
        build, "evaluate_reconstruction", MagicMock(return_value=({}, {}))
    )
    refinement: dict[str, object] = dict.fromkeys(
        build.RefinementSettings.__dataclass_fields__, 0.0
    )
    refinement.update(enabled=False, player_root_view_support="hips")
    cfg = OmegaConf.create(
        {
            "coordinate_mode": "reference",
            "view_half_turns": [False],
            "refinement": refinement,
        }
    )
    if pinned:
        cfg.checkpoint_sha256 = pins
    (tmp_path / "court.json").write_text(json.dumps({"identity": "court"}))
    runner = build._scene_runner(
        cfg,
        cast(ReferenceClipPaths, object()),
        clip,
        tmp_path,
        np.empty(0),
        np.empty(0),
        tmp_path,
        identity,
    )
    result = runner([clip.media_path(c) for c in clip.camera_ids], clip.camera_ids)
    assert result.metadata["checkpoints"] == raw_receipts
    assert result.metadata["dataset_producer_identity"] == identity


@pytest.mark.parametrize("role", [None, "plcs", "blcs", "missing"])
def test_legacy_cache_checks_preserved_raw_receipts_without_rewriting(teacher, role):
    clip, scene, identity, _ = teacher
    scene.metadata["dataset_producer_identity"] = identity
    if role == "missing":
        del scene.metadata["checkpoints"]
    elif role is not None:
        scene.metadata["checkpoints"][role]["sha256"] = "d" * 64
    archive = clip.clip_dir / "annotations/tennis_scene/scene.npz"
    save_scene_result(scene, archive)
    sidecar = archive.with_suffix(".metadata.json")
    before = sidecar.read_bytes()
    if role is None:
        assert provenance.validated_scene_cache(clip, identity)
    else:
        with pytest.raises(ValueError, match="Raw teacher"):
            provenance.validated_scene_cache(clip, identity)
    assert sidecar.read_bytes() == before


@pytest.mark.parametrize("pinned", [False, True])
@pytest.mark.parametrize("role", [None, "plcs", "blcs"])
def test_scene_identity_checks_existing_hash_results_against_pins(
    teacher, tmp_path, monkeypatch, pinned, role
):
    clip, _, identity, pins = teacher
    fields = (
        "seed",
        "coordinate_mode",
        "reference_camera",
        "view_half_turns",
        "sample_stride",
        "window_size",
        "window_overlap",
        "pose_visibility_threshold",
        "ball_source",
        "refinement",
    )
    cfg = OmegaConf.create(dict.fromkeys(fields))
    if pinned:
        cfg.checkpoint_sha256 = pins
    digests = dict(identity["checkpoints"])
    if role is not None:
        digests[role] = "d" * 64
    digest = MagicMock(side_effect=lambda path: digests.get(path.name, "c" * 64))
    monkeypatch.setattr(provenance, "sha256", digest)
    paths = cast(
        ReferenceClipPaths,
        SimpleNamespace(plcs_checkpoint=Path("plcs"), blcs_checkpoint=Path("blcs")),
    )
    if pinned and role is not None:
        with pytest.raises(
            ValueError, match=f"Teacher checkpoint pin mismatch: {role}"
        ):
            provenance.scene_identity(cfg, paths, clip, tmp_path)
    else:
        actual = provenance.scene_identity(cfg, paths, clip, tmp_path)
        assert actual["checkpoints"] == digests
    for task in ("plcs", "blcs"):
        assert sum(call.args[0] == Path(task) for call in digest.call_args_list) == 1
