"""Tests for persistent BLCS Gaussian identities, transforms, and projections."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from src.synthetic_data_generation.blcs.assets import BallAssetRegistry
from src.synthetic_data_generation.blcs.planner import (
    build_blcs_gaussian_plan_from_scene,
    load_blcs_gaussian_plan,
    verify_blcs_gaussian_plan_output,
    write_blcs_gaussian_plan,
)
from src.synthetic_data_generation.scene_contract import (
    SceneCamera,
    SimilarityTransform,
)


@dataclass
class _Scene:
    scene_id: str
    ball_pos_world: torch.Tensor
    ball_vel_world: torch.Tensor
    ball_present: torch.Tensor | None
    fps_out: int = 50
    num_balls: int = 1


def _scene_from_court() -> SimilarityTransform:
    return SimilarityTransform(
        scale=2.0,
        rotation=(1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0),
        translation=(0.0, 0.0, 10.0),
    )


def _camera(camera_id: str = "camera-0") -> SceneCamera:
    return SceneCamera(
        camera_id=camera_id,
        source_camera_id=camera_id,
        image_uri=f"artifact://unit/{camera_id}.png",
        source_frame_index=0,
        group_id=0,
        width=640,
        height=480,
        intrinsics=(100.0, 0.0, 320.0, 0.0, 100.0, 240.0, 0.0, 0.0, 1.0),
        camera_to_scene=(
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ),
    )


def _single_scene() -> _Scene:
    return _Scene(
        scene_id="single-rally",
        ball_pos_world=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.5]],
            dtype=torch.float32,
        ),
        ball_vel_world=torch.tensor(
            [[10.0, 0.0, 5.0], [9.5, 0.0, 4.5]],
            dtype=torch.float32,
        ),
        ball_present=None,
    )


def _multi_scene() -> _Scene:
    return _Scene(
        scene_id="multi-rally",
        ball_pos_world=torch.tensor(
            [
                [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                [[2.0, 0.0, 0.0], [2.0, 2.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            dtype=torch.float32,
        ),
        ball_vel_world=torch.ones((3, 3, 3), dtype=torch.float32),
        ball_present=torch.tensor(
            [[True, False, False], [True, True, False], [False, True, False]],
            dtype=torch.bool,
        ),
        num_balls=2,
    )


def test_single_plan_applies_sim3_and_projects_opencv_consistently(
    ball_registry: BallAssetRegistry,
) -> None:
    plan = build_blcs_gaussian_plan_from_scene(
        _single_scene(),
        registry=ball_registry,
        seed=42,
        scene_from_court=_scene_from_court(),
        cameras=(_camera(),),
    )

    np.testing.assert_allclose(
        plan.positions_scene,
        np.asarray([[[0.0, 0.0, 10.0]], [[2.0, 0.0, 9.0]]]),
    )
    np.testing.assert_allclose(plan.camera_uv[0, 0, 0], [320.0, 240.0])
    np.testing.assert_allclose(
        plan.camera_uv[0, 1, 0],
        [320.0 + 200.0 / 9.0, 240.0],
    )
    assert plan.camera_geometric_visible.all()
    assert plan.instance_ids.tolist() == [1]
    assert not plan.positions_scene.flags.writeable


def test_multi_plan_keeps_column_identity_through_birth_and_death(
    ball_registry: BallAssetRegistry,
) -> None:
    plan = build_blcs_gaussian_plan_from_scene(
        _multi_scene(),
        registry=ball_registry,
        seed=24,
        scene_from_court=_scene_from_court(),
        cameras=(_camera(),),
    )

    assert plan.num_objects == 2
    assert plan.instance_ids.tolist() == [1, 2]
    assert [instance.instance_id for instance in plan.instances_at(0)] == [1]
    assert [instance.instance_id for instance in plan.instances_at(1)] == [1, 2]
    assert [instance.instance_id for instance in plan.instances_at(2)] == [2]
    assert plan.assignments[0].selection.selection_sha256 != ""
    np.testing.assert_allclose(
        plan.instances_at(1)[1].scene_from_asset.matrix(),
        plan.scene_from_asset[1, 1],
    )
    assert not plan.camera_geometric_visible[0, 0, 1]
    assert not plan.camera_geometric_visible[0, 2, 0]


def test_repeated_plan_publication_is_byte_identical_and_verified(
    tmp_path: Path,
    ball_registry: BallAssetRegistry,
) -> None:
    plan = build_blcs_gaussian_plan_from_scene(
        _multi_scene(),
        registry=ball_registry,
        seed=1729,
        scene_from_court=_scene_from_court(),
        cameras=(_camera(), _camera("camera-1")),
    )
    first = tmp_path / "plan-a"
    second = tmp_path / "plan-b"

    write_blcs_gaussian_plan(first, plan)
    write_blcs_gaussian_plan(second, plan)
    first_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(first.iterdir())
    }
    second_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(second.iterdir())
    }

    assert first_hashes == second_hashes
    loaded = load_blcs_gaussian_plan(first)
    assert loaded.plan_fingerprint == plan.plan_fingerprint
    np.testing.assert_array_equal(loaded.present, plan.present)
    assert verify_blcs_gaussian_plan_output(first) == {
        "plan_fingerprint": plan.plan_fingerprint,
        "num_frames": 3,
        "num_objects": 2,
        "num_cameras": 2,
        "geometric_visible_count": 8,
        "render_stage_complete": False,
    }
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_blcs_gaussian_plan(first, plan)


def test_multi_plan_requires_explicit_presence(
    ball_registry: BallAssetRegistry,
) -> None:
    scene = replace(_multi_scene(), ball_present=None)

    with pytest.raises(ValueError, match="require explicit ball_present"):
        build_blcs_gaussian_plan_from_scene(
            scene,
            registry=ball_registry,
            seed=0,
            scene_from_court=_scene_from_court(),
            cameras=(_camera(),),
        )


def test_output_verifier_rejects_tampered_label_bytes(
    tmp_path: Path,
    ball_registry: BallAssetRegistry,
) -> None:
    plan = build_blcs_gaussian_plan_from_scene(
        _single_scene(),
        registry=ball_registry,
        seed=0,
        scene_from_court=_scene_from_court(),
        cameras=(_camera(),),
    )
    output = tmp_path / "plan"
    write_blcs_gaussian_plan(output, plan)
    with (output / "camera_uv.npy").open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(ValueError, match="size mismatch"):
        verify_blcs_gaussian_plan_output(output)
