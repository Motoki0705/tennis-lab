"""Tests for grouped full-scale BLCS/3DGS dataset publication."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import src.synthetic_data_generation.dataset.full_scale_dataset as dataset_module
from src.synthetic_data_generation.dataset.full_scale_dataset import (
    FullScaleDatasetConfig,
    FullScaleProvenance,
    StaticSceneProvenance,
    TrajectorySamplingSpec,
    load_and_validate_full_scale_dataset,
    publish_full_scale_dataset,
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
from src.tasks.ball_detection.data.tracknet_datamodule import TrackNetDataModule
from src.tasks.blcs.generate_dataset.scene_generator import BLCSSceneData


def test_publish_grouped_dataset_and_tracknet_smoke(tmp_path: Path) -> None:
    contract, cameras = _contract()
    specs = _specs()
    scenes = tuple(_scene(spec) for spec in specs)
    renderer = _renderer(contract, cameras)
    config = _config(cameras, specs)

    output = publish_full_scale_dataset(
        scenes=scenes,
        scene_contract=contract,
        renderer=renderer,
        config=config,
        provenance=_provenance(cameras),
        output_root=tmp_path,
    )
    manifest = load_and_validate_full_scale_dataset(output)

    assert manifest["publication"]["frame_count"] == 32
    assert manifest["publication"]["clip_count"] == 4
    assert manifest["label_statistics"]["positive_frames"] == 16
    assert manifest["label_statistics"]["negative_frames"] == 16
    assert manifest["diversity"]["camera_group_count"] == 2
    assert manifest["diversity"]["trajectory_count"] == 2
    assert manifest["diversity"]["visible_ball_displacement"]["count"] == 12
    assert (
        manifest["diversity"]["projected_center_displacement"]["count"]
        > manifest["diversity"]["visible_ball_displacement"]["count"]
    )
    assert manifest["identity"]["court_to_scene_scale"] == pytest.approx(0.5)
    assert (output / "splits" / "val.txt").read_text().startswith("# synthetic")

    data_module = TrackNetDataModule(
        OmegaConf.create(
            {
                "model": {"num_frames": 8},
                "data": {
                    "data_dir": str(output),
                    "sample_stride": 8,
                    "split": {"train_file": "splits/train.txt"},
                },
            }
        )
    )
    windows = data_module.create_windows(
        split_name="train",
        split_file="splits/train.txt",
    )
    sample = data_module.create_dataset(
        split_name="train",
        split_file="splits/train.txt",
        augmentation=None,
    )[0]

    assert len(windows) == 4
    assert sample["images"].shape == (8, 3, 288, 512)
    assert sample["heatmaps"].shape == (8, 144, 256)
    assert float(sample["visibility"].sum()) == pytest.approx(4.0)


def test_strict_reload_rejects_tampered_full_scale_frame(tmp_path: Path) -> None:
    contract, cameras = _contract()
    specs = _specs()
    output = publish_full_scale_dataset(
        scenes=tuple(_scene(spec) for spec in specs),
        scene_contract=contract,
        renderer=_renderer(contract, cameras),
        config=_config(cameras, specs),
        provenance=_provenance(cameras),
        output_root=tmp_path,
    )
    frame = next(output.glob("train/**/000000.jpg"))
    frame.write_bytes(frame.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="payload size mismatch"):
        load_and_validate_full_scale_dataset(output)


def test_full_scale_pipeline_dependency_boundary() -> None:
    source = inspect.getsource(dataset_module)
    tree = ast.parse(source)
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
    assert "BLCSDatasetWriter" not in source
    assert "sys.path" not in source


def _config(
    cameras: tuple[SceneCamera, ...],
    specs: tuple[TrajectorySamplingSpec, ...],
) -> FullScaleDatasetConfig:
    return FullScaleDatasetConfig(
        camera_ids=tuple(camera.camera_id for camera in cameras),
        trajectories=specs,
        clip_length=8,
        ball_radius_m=0.2,
        jpeg_quality=90,
    )


def _specs() -> tuple[TrajectorySamplingSpec, ...]:
    return (
        TrajectorySamplingSpec(
            seed=10,
            from_cell=0,
            side="near",
            scene_id="rally-0",
        ),
        TrajectorySamplingSpec(
            seed=11,
            from_cell=1,
            side="far",
            scene_id="rally-1",
        ),
    )


def _scene(spec: TrajectorySamplingSpec) -> BLCSSceneData:
    positions = torch.tensor(
        [[0.0, 0.0, 0.0]] * 4 + [[100.0, 0.0, 0.0]] * 4,
        dtype=torch.float32,
    )
    velocities = torch.tensor([[1.0, 0.0, 0.0]] * 8, dtype=torch.float32)
    return BLCSSceneData(
        scene_id=spec.scene_id,
        initial_from_cell=spec.from_cell,
        initial_from_side=spec.side,
        rally_length=1,
        end_reason="test",
        winner_side=None,
        shots=[],
        ball_pos_world=positions,
        ball_pos_norm=positions,
        ball_vel_world=velocities,
        cameras=[],
        num_cameras_sampled=0,
        fps_out=30,
        sim_fps=240,
        physics_config_dict={"gravity": 9.81},
        court_config_dict={"net_post_offset_x": 0.914},
    )


def _renderer(
    contract: SceneContract,
    cameras: tuple[SceneCamera, ...],
) -> DeterministicCpuSphereRenderer:
    frames = {
        camera.camera_id: CpuSceneFrame(
            rgb=np.full((camera.height, camera.width, 3), 80, dtype=np.uint8),
            depth=np.full((camera.height, camera.width), np.inf, dtype=np.float32),
        )
        for camera in cameras
    }
    return DeterministicCpuSphereRenderer(
        scene_fingerprint=contract.scene_fingerprint,
        frames=frames,
    )


def _contract() -> tuple[SceneContract, tuple[SceneCamera, ...]]:
    cameras = (
        _camera("camera-0", group_id=0),
        _camera("camera-1", group_id=1),
    )
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
        fit_camera_ids=(cameras[0].camera_id,),
        holdout_camera_ids=(cameras[1].camera_id,),
        scene_from_court=transform,
        court_from_scene=transform.inverse(),
        manifest=_artifact("alignment"),
    )
    return (
        SceneContract.create(
            scene_id="test-scene",
            provider_backend="test-provider",
            artifacts=(_artifact("scene"),),
            cameras=cameras,
            alignment=alignment,
        ),
        cameras,
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


def _provenance(cameras: tuple[SceneCamera, ...]) -> FullScaleProvenance:
    return FullScaleProvenance(
        scene_contract_uri="scene.json",
        scene_contract_sha256="b" * 64,
        static_scenes=tuple(
            StaticSceneProvenance(
                camera_id=camera.camera_id,
                uri=f"{camera.camera_id}.npz",
                sha256=f"{index + 1:x}" * 64,
                request_fingerprint=f"{index + 3:x}" * 64,
            )
            for index, camera in enumerate(cameras)
        ),
        git_revision="e" * 40,
        git_dirty=True,
        code_diff_sha256="f" * 64,
    )
