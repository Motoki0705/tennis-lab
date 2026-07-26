"""Unit tests for deterministic BLCS/scene single-frame pilot publication."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

import src.synthetic_data_generation.dataset.single_frame_pilot as pilot_module
from src.synthetic_data_generation.dataset.single_frame_pilot import (
    PilotProvenance,
    SingleFramePilotConfig,
    load_and_validate_single_frame_pilot,
    publish_single_frame_pilot,
)
from src.synthetic_data_generation.rendering.cpu_fake_renderer import (
    CpuSceneFrame,
    DeterministicCpuSphereRenderer,
)
from src.synthetic_data_generation.scene_contract import (
    AcceptedAlignment,
    ArtifactRef,
    SceneCamera,
    SceneContract,
    SimilarityTransform,
)
from src.tasks.blcs.generate_dataset.scene_generator import BLCSSceneData


def test_publish_pilot_transforms_once_and_preserves_negative(tmp_path: Path) -> None:
    contract, camera = _contract()
    rgb: NDArray[np.uint8] = np.full(
        (camera.height, camera.width, 3), 80, dtype=np.uint8
    )
    renderer = DeterministicCpuSphereRenderer(
        scene_fingerprint=contract.scene_fingerprint,
        frames={
            camera.camera_id: CpuSceneFrame(
                rgb=rgb,
                depth=np.full(
                    (camera.height, camera.width),
                    np.inf,
                    dtype=np.float32,
                ),
            )
        },
    )

    output = publish_single_frame_pilot(
        scene=_scene(),
        scene_contract=contract,
        renderer=renderer,
        config=SingleFramePilotConfig(
            camera_id=camera.camera_id,
            trajectory_frame_index=0,
            ball_radius_m=0.2,
            supersampling=4,
        ),
        provenance=_provenance(),
        output_root=tmp_path,
    )
    manifest = load_and_validate_single_frame_pilot(output)

    assert output.name == manifest["dataset_fingerprint"]
    assert manifest["identity"]["court_position_m"] == [0.0, 0.0, 0.0]
    assert manifest["identity"]["scene_position"] == [0.0, 0.0, 4.0]
    assert manifest["identity"]["ball_radius_scene_units"] == pytest.approx(0.1)
    assert manifest["frames"][0]["state"] == "fully_visible"
    assert manifest["frames"][0]["visibility"] == 1.0
    assert manifest["frames"][1]["state"] == "absent"
    assert manifest["label_statistics"]["negative_frames"] == 1
    assert (output / "splits" / "train.txt").read_text() == "train/Clip1\n"


def test_strict_reload_rejects_tampered_frame(tmp_path: Path) -> None:
    contract, camera = _contract()
    renderer = DeterministicCpuSphereRenderer(
        scene_fingerprint=contract.scene_fingerprint,
        frames={
            camera.camera_id: CpuSceneFrame(
                rgb=np.zeros((camera.height, camera.width, 3), dtype=np.uint8),
                depth=np.full(
                    (camera.height, camera.width),
                    np.inf,
                    dtype=np.float32,
                ),
            )
        },
    )
    output = publish_single_frame_pilot(
        scene=_scene(),
        scene_contract=contract,
        renderer=renderer,
        config=SingleFramePilotConfig(
            camera_id=camera.camera_id,
            trajectory_frame_index=0,
            ball_radius_m=0.2,
        ),
        provenance=_provenance(),
        output_root=tmp_path,
    )
    frame_path = output / "train" / "Clip1" / "0000.jpg"
    frame_path.write_bytes(frame_path.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="size mismatch"):
        load_and_validate_single_frame_pilot(output)


def test_pipeline_dependency_boundary() -> None:
    tree = ast.parse(inspect.getsource(pilot_module))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    forbidden = (
        "gsplat",
        "src.tasks.ball_detection",
        "src.tasks.blcs.generate_dataset.io",
    )

    assert not any(
        module == prefix or module.startswith(f"{prefix}.")
        for module in imports
        for prefix in forbidden
    )
    assert "BLCSDatasetWriter" not in inspect.getsource(pilot_module)
    assert "sys.path" not in inspect.getsource(pilot_module)


def _scene() -> BLCSSceneData:
    return BLCSSceneData(
        scene_id="test-rally",
        initial_from_cell=0,
        initial_from_side="near",
        rally_length=1,
        end_reason="test",
        winner_side=None,
        shots=[],
        ball_pos_world=torch.tensor([[0.0, 0.0, 0.0]]),
        ball_pos_norm=torch.tensor([[0.0, 0.0, 0.0]]),
        ball_vel_world=torch.tensor([[1.0, 0.0, 0.0]]),
        cameras=[],
        num_cameras_sampled=0,
        fps_out=30,
        sim_fps=240,
        physics_config_dict={"gravity": 9.81},
        court_config_dict={"net_post_offset_x": 0.914},
    )


def _contract() -> tuple[SceneContract, SceneCamera]:
    camera = _camera("fit-0", group_id=0)
    holdout = _camera("holdout-0", group_id=1)
    transform = SimilarityTransform(
        scale=0.5,
        rotation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        translation=(0.0, 0.0, 4.0),
    )
    alignment = AcceptedAlignment(
        alignment_id="test-alignment",
        accepted=True,
        selected_court_cluster="court-0",
        selected_symmetry="identity",
        fit_camera_ids=(camera.camera_id,),
        holdout_camera_ids=(holdout.camera_id,),
        scene_from_court=transform,
        court_from_scene=transform.inverse(),
        manifest=_artifact("alignment"),
    )
    return (
        SceneContract.create(
            scene_id="test-scene",
            provider_backend="test-provider",
            artifacts=(_artifact("scene"),),
            cameras=(camera, holdout),
            alignment=alignment,
        ),
        camera,
    )


def _camera(camera_id: str, *, group_id: int) -> SceneCamera:
    return SceneCamera(
        camera_id=camera_id,
        source_camera_id="source",
        image_uri=f"images/{camera_id}.png",
        source_frame_index=group_id,
        group_id=group_id,
        width=64,
        height=48,
        intrinsics=(50.0, 0.0, 31.5, 0.0, 50.0, 23.5, 0.0, 0.0, 1.0),
        camera_to_scene=tuple(float(value) for value in np.eye(4).ravel()),
    )


def _artifact(artifact_id: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=artifact_id,
        uri=f"artifact://{artifact_id}",
        sha256="a" * 64,
        size_bytes=1,
    )


def _provenance() -> PilotProvenance:
    return PilotProvenance(
        seed=1,
        scene_contract_uri="scene.json",
        scene_contract_sha256="b" * 64,
        static_scene_uri="scene.npz",
        static_scene_sha256="c" * 64,
        static_scene_request_fingerprint="d" * 64,
        git_revision="e" * 40,
        git_dirty=True,
        code_diff_sha256="f" * 64,
    )
