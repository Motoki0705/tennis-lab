"""Cross-task CPU smoke coverage for CourtKP20 model-frame provenance."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from numpy.typing import NDArray
from omegaconf import DictConfig
from torch import Tensor

from src.tasks.base.generate_dataset import (
    COURT_KEYPOINT_METADATA_KEY,
    CourtKeypointContract,
    CourtKeypointContractMetadata,
    CourtReferenceFrameProvenance,
    MissingCourtKeypointMetadataError,
    apply_court_view_record,
    build_court_view_record,
    build_reference_frame_provenance,
    court_points_physical_to_target,
    resolve_court_keypoint_contract,
)
from src.tasks.blcs.configuration import (
    TrackQueryReferenceModelConfig,
    parse_model_config,
)
from src.tasks.blcs.data.dataset import (
    BallTrajectoryDataset,
    collate_multiview_trajectories,
)
from src.tasks.blcs.generate_dataset.io.dataset_io import BLCSDatasetWriter
from src.tasks.blcs.generate_dataset.scene_generator import (
    BLCSSceneData,
)
from src.tasks.blcs.generate_dataset.scene_generator import (
    CameraData as BLCSCameraData,
)
from src.tasks.blcs.model_io import (
    BLCSTrajectoryPrediction,
    blcs_trajectory_prediction_to_physical,
    compose_blcs_trajectory_model_io,
)
from src.tasks.blcs.models.blcs_track_query_reference_model import (
    BLCSTrackQueryReferenceModel,
)
from src.tasks.blcs.training.lightning_module import BLCSLightningModule
from src.tasks.plcs.configuration import PLCSModelConfig
from src.tasks.plcs.court_keypoint_contract import (
    headings_target_to_physical,
    normalized_points_target_to_physical,
)
from src.tasks.plcs.data.dataset import SceneDataset, collate_plcs_batch
from src.tasks.plcs.generate_dataset.io.dataset_io import PLCSDatasetWriter
from src.tasks.plcs.generate_dataset.scene_generator import (
    CameraData as PLCSCameraData,
)
from src.tasks.plcs.generate_dataset.scene_generator import SceneData as PLCSSceneData
from src.tasks.plcs.model_io import PLCSDecodedPrediction, PLCSPhysicalPrediction
from src.tasks.plcs.models.plcs_track_query_reference_model import (
    PLCSTrackQueryReferenceModel,
)
from src.tasks.plcs.training.lightning_module import PLCSLightningModule
from src.tennis_scene.pipeline.components.blcs import BLCSModule
from src.tennis_scene.pipeline.components.plcs import PLCSModule
from src.utils.schema.court import COURT_KP20_HALF_TURN_INDEX
from src.utils.schema.court_normalization import (
    denormalize_court_position,
    normalize_court_position,
    normalize_court_velocity,
)
from tests.unit.tennis_scene.pipeline.config_factories import (
    make_blcs_config,
    make_plcs_config,
)

pytestmark = pytest.mark.integration


class _BLCSProcessInputs(TypedDict):
    ball_uv: NDArray[np.float32]
    court_kp: NDArray[np.float32]
    ball_vis: NDArray[np.bool_]
    court_vis: NDArray[np.float32]


class _PLCSProcessInputs(TypedDict):
    human_kp_2d: NDArray[np.float32]
    court_kp: NDArray[np.float32]
    human_kp_vis: NDArray[np.float32]
    court_vis: NDArray[np.float32]
    track_ids: NDArray[np.int32]


_BLCS_CONFIG_DIR = Path("src/tasks/blcs/configs").resolve()
_PLCS_CONFIG_DIR = Path("src/tasks/plcs/configs").resolve()
_PHYSICAL_COURT_UV = np.linspace(0.05, 0.95, 40, dtype=np.float32).reshape(20, 2)
_CENTERS = ((0.5, 12.0, -5.0), (-0.75, -12.0, -5.0))


def _contract_document(contract: CourtKeypointContract) -> dict[str, object]:
    return {
        COURT_KEYPOINT_METADATA_KEY: CourtKeypointContractMetadata.from_contract(
            contract
        ).to_dict()
    }


def _blcs_camera(
    index: int,
    contract: CourtKeypointContract,
) -> BLCSCameraData:
    center = _CENTERS[index]
    view = build_court_view_record(
        camera_id=f"cam_{index}",
        camera_center_court_m=center,
        contract=contract,
    )
    court = apply_court_view_record(
        _PHYSICAL_COURT_UV,
        view,
        keypoint_axis=0,
    )
    assert isinstance(court, np.ndarray)
    return BLCSCameraData(
        camera_params={
            "R": np.eye(3, dtype=np.float32).tolist(),
            "C": list(center),
            "f": 100.0,
            "cx": 50.0,
            "cy": 40.0,
            "w": 100,
            "h": 80,
        },
        ball_uv=np.full((2, 2), 0.4 + index * 0.1, dtype=np.float32),
        ball_vis=np.ones(2, dtype=np.bool_),
        ball_visibility_ratio=1.0,
        court_kp_uv=court,
        court_kp_vis=np.ones(20, dtype=np.bool_),
        court_visibility_count=20.0,
        court_view=view,
    )


def _write_blcs_dataset(
    root: Path,
    contract: CourtKeypointContract,
) -> None:
    physical_position = torch.tensor(
        [[1.0, 2.0, 0.5], [1.5, 2.5, 0.75]], dtype=torch.float32
    )
    normalized = normalize_court_position(physical_position)
    physical_velocity = torch.tensor(
        [[0.5, 1.0, 0.25], [0.75, 1.25, 0.0]], dtype=torch.float32
    )
    scene = BLCSSceneData(
        scene_id="scene_000000",
        initial_from_cell=0,
        initial_from_side="near",
        rally_length=1,
        end_reason="finished",
        winner_side=None,
        shots=[],
        ball_pos_world=physical_position,
        ball_pos_norm=normalized,
        ball_vel_world=physical_velocity,
        ball_vel_norm=normalize_court_velocity(physical_velocity),
        cameras=[_blcs_camera(0, contract), _blcs_camera(1, contract)],
        num_cameras_sampled=2,
        fps_out=30,
        sim_fps=120,
        physics_config_dict={},
        court_config_dict={},
        num_balls=1,
    )
    writer = BLCSDatasetWriter(
        root,
        court_keypoint_contract=contract,
    )
    writer.save_scene(scene)
    writer.save_meta_json()
    (root / "test.txt").write_text("scene_000000\n", encoding="utf-8")


def _plcs_camera(
    index: int,
    contract: CourtKeypointContract,
) -> PLCSCameraData:
    center = _CENTERS[index]
    view = build_court_view_record(
        camera_id=f"camera_{index}",
        camera_center_court_m=center,
        contract=contract,
    )
    court = apply_court_view_record(
        _PHYSICAL_COURT_UV,
        view,
        keypoint_axis=0,
    )
    assert isinstance(court, np.ndarray)
    return PLCSCameraData(
        camera_params={
            "R": np.eye(3, dtype=np.float32).tolist(),
            "C": list(center),
            "f": 100.0,
            "cx": 50.0,
            "cy": 40.0,
            "w": 100,
            "h": 80,
        },
        human_kp_uv=np.full((2, 17, 2), 0.3 + index * 0.1, dtype=np.float32),
        court_kp_uv=np.repeat(court[None], 2, axis=0),
        human_kp_vis=np.ones((2, 17), dtype=np.bool_),
        court_kp_vis=np.ones((2, 20), dtype=np.bool_),
        human_visibility_ratio=1.0,
        court_visibility_count=20.0,
        court_view=view,
    )


def _write_plcs_dataset(
    root: Path,
    contract: CourtKeypointContract,
) -> None:
    physical_position = np.array([[1.0, 2.0, 0.5], [1.5, 2.5, 0.75]], dtype=np.float32)
    normalized = normalize_court_position(physical_position)
    world_joints = np.repeat(physical_position[:, None], 17, axis=1)
    scene = PLCSSceneData(
        meta={
            "scene_id": "scene_000000",
            "motion_source": "integration",
            "motion_category": "smoke",
            "gender": "neutral",
            "fps": 30,
            "num_frames": 2,
            "initial_position": [1.0, 2.0],
            "initial_yaw": 0.0,
            "num_cameras_sampled": 2,
        },
        position=normalized.astype(np.float32),
        rotation=np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        canonical_pose_3d=np.full((2, 17, 3), 0.125, dtype=np.float32),
        cameras=[_plcs_camera(0, contract), _plcs_camera(1, contract)],
        num_persons=1,
        human_kp_3d=world_joints,
        court_keypoint_contract=contract,
    )
    writer = PLCSDatasetWriter(
        root,
        court_keypoint_contract=contract,
    )
    writer.save_meta_json()
    writer.save_scene(scene)
    writer.save_meta_json()
    (root / "test.txt").write_text("scene_000000\n", encoding="utf-8")


def _blcs_config(selector: str) -> DictConfig:
    overrides = [
        f"court_keypoints={selector}",
        "model=multiview_axial_small",
        "model.hidden_dim=16",
        "model.num_layers=1",
        "model.num_heads=4",
        "model.ffn_dim=32",
        "model.rope_dim=4",
        "model.camera_layers_per_stage=[1]",
        "model.time_layers_per_stage=[1]",
        "model.time_global_stage_mask=[false]",
        "model.max_seq_len=2",
        "model.max_num_cameras=2",
        "model.num_court_tokens=20",
        "model.dropout=0.0",
        "data.seq_len_range=[2,2]",
        "data.num_views_range=[2,2]",
        "data.camera_mode=first",
        "data.num_court_kp=20",
        "data.batch_size=1",
        "data.num_workers=0",
        "training.compile.enabled=false",
    ]
    with initialize_config_dir(
        version_base="1.3",
        config_dir=str(_BLCS_CONFIG_DIR),
    ):
        return compose(config_name="train", overrides=overrides)


def _plcs_config(selector: str) -> DictConfig:
    overrides = [
        f"court_keypoints={selector}",
        "model=multiview_axial_base",
        "loss=no_canonical",
        "model.hidden_dim=16",
        "model.num_layers=1",
        "model.num_heads=4",
        "model.ffn_dim=32",
        "model.rope_dim=4",
        "model.max_seq_len=2",
        "model.max_views=2",
        "model.dropout=0.0",
        "data.seq_len_range=[2,2]",
        "data.num_views_range=[2,2]",
        "data.min_cameras=2",
        "data.camera_mode=first",
        "data.num_court_kp=20",
        "data.batch_size=1",
        "data.num_workers=0",
        "training.compile.enabled=false",
    ]
    with initialize_config_dir(
        version_base="1.3",
        config_dir=str(_PLCS_CONFIG_DIR),
    ):
        return compose(config_name="train", overrides=overrides)


def _assert_projection_round_trip(
    sample: dict[str, Any],
    provenance: CourtReferenceFrameProvenance,
) -> None:
    physical_point = torch.tensor([1.25, -0.5, 0.75], dtype=torch.float32)
    target_point = court_points_physical_to_target(physical_point, provenance)
    assert isinstance(target_point, Tensor)
    for index, center in enumerate(_CENTERS):
        physical_camera_point = physical_point - torch.tensor(center)
        target_camera_point = sample["camera_R"][index] @ (
            target_point - sample["camera_C"][index]
        )
        torch.testing.assert_close(target_camera_point, physical_camera_point)
        assert target_camera_point[2] > 0


@pytest.mark.parametrize("selector", ["physical_v1", "camera_view_v2"])
def test_standard_datasets_models_losses_metrics_and_physical_predictions(
    tmp_path: Path,
    selector: str,
) -> None:
    """Exercise both CourtKP contracts with the fixed isotropic normalization."""
    torch.manual_seed(799)
    contract = resolve_court_keypoint_contract(selector)

    blcs_root = tmp_path / "blcs"
    _write_blcs_dataset(blcs_root, contract)
    blcs_config = _blcs_config(selector)
    blcs_dataset = BallTrajectoryDataset(
        scene_dir=blcs_root,
        split_file="test.txt",
        config=blcs_config,
        augment=False,
        reference_camera_id=("cam_0" if selector == "camera_view_v2" else None),
    )
    blcs_sample = cast("dict[str, Any]", blcs_dataset[0])
    blcs_provenance = cast(
        "CourtReferenceFrameProvenance",
        blcs_sample["court_reference_provenance"],
    )
    assert blcs_sample["court_kp"].shape == (2, 2, 20, 2)
    torch.testing.assert_close(
        blcs_sample["court_kp"][0],
        blcs_sample["court_kp"][1],
    )
    expected_order = (
        COURT_KP20_HALF_TURN_INDEX if selector == "camera_view_v2" else tuple(range(20))
    )
    torch.testing.assert_close(
        blcs_sample["court_kp"][0, 0],
        torch.from_numpy(_PHYSICAL_COURT_UV[np.asarray(expected_order)]),
    )
    reference_sign = torch.tensor(
        [-1.0, -1.0, 1.0] if selector == "camera_view_v2" else [1.0, 1.0, 1.0]
    )
    expected_blcs_position = (
        torch.tensor(
            [[1.0, 2.0, 0.5], [1.5, 2.5, 0.75]],
            dtype=torch.float32,
        )
        * reference_sign
    )
    expected_blcs_velocity = (
        torch.tensor(
            [[0.5, 1.0, 0.25], [0.75, 1.25, 0.0]],
            dtype=torch.float32,
        )
        * reference_sign
    )
    torch.testing.assert_close(
        blcs_sample["position_3d"],
        normalize_court_position(expected_blcs_position),
    )
    torch.testing.assert_close(
        blcs_sample["velocity_3d"],
        normalize_court_velocity(expected_blcs_velocity),
    )
    _assert_projection_round_trip(blcs_sample, blcs_provenance)

    blcs_batch = cast(
        "dict[str, Any]",
        dict(collate_multiview_trajectories([blcs_dataset[0]])),
    )
    blcs_module = (
        BLCSLightningModule(
            blcs_config,
            model_io=compose_blcs_trajectory_model_io(blcs_config),
        )
        .cpu()
        .eval()
    )
    blcs_module.test_metrics.reset()
    with torch.no_grad():
        blcs_result = blcs_module._compute_supervised_result(blcs_batch, "test")
    blcs_output = cast("BLCSTrajectoryPrediction", blcs_result["outputs"])
    assert torch.isfinite(cast("Tensor", blcs_result["loss"]))
    assert blcs_module.test_metrics.compute()
    output_m = denormalize_court_position(blcs_output.position)
    physical_blcs = blcs_trajectory_prediction_to_physical(
        BLCSTrajectoryPrediction(
            position=output_m,
            velocity=None,
            court_reference_provenance=(blcs_provenance,),
            coordinates_in_metres=True,
        )
    )
    assert physical_blcs.position.shape == (1, 2, 3)
    assert torch.isfinite(physical_blcs.position).all()

    plcs_root = tmp_path / "plcs"
    _write_plcs_dataset(plcs_root, contract)
    plcs_config = _plcs_config(selector)
    plcs_dataset = SceneDataset(
        scene_dir=plcs_root,
        split_file="test.txt",
        config=plcs_config,
        augment=False,
        reference_camera_id=("camera_0" if selector == "camera_view_v2" else None),
    )
    plcs_sample = plcs_dataset[0]
    plcs_provenance = cast(
        "CourtReferenceFrameProvenance",
        plcs_sample["court_reference_provenance"],
    )
    assert plcs_sample["court_kp"].shape == (2, 2, 20, 2)
    torch.testing.assert_close(
        plcs_sample["court_kp"][0],
        plcs_sample["court_kp"][1],
    )
    torch.testing.assert_close(
        plcs_sample["human_kp"][0],
        torch.full((2, 17, 2), 0.3),
    )
    torch.testing.assert_close(
        plcs_sample["position"],
        normalize_court_position(expected_blcs_position),
    )
    torch.testing.assert_close(
        plcs_sample["rotation"],
        torch.tensor(
            [[reference_sign[0].item(), 0.0]] * 2,
            dtype=torch.float32,
        ),
    )
    torch.testing.assert_close(
        plcs_sample["human_kp_3d"][:, 0],
        expected_blcs_position,
    )
    _assert_projection_round_trip(plcs_sample, plcs_provenance)

    plcs_batch = cast("dict[str, Any]", dict(collate_plcs_batch([plcs_dataset[0]])))
    torch.manual_seed(799)
    plcs_module = PLCSLightningModule(plcs_config).cpu().eval()
    plcs_module.test_metrics.reset()
    with torch.no_grad():
        plcs_result = plcs_module._compute_supervised_result(plcs_batch, "test")
    plcs_output = cast("PLCSDecodedPrediction", plcs_result["outputs"])
    assert torch.isfinite(cast("Tensor", plcs_result["loss"]))
    assert plcs_module.test_metrics.compute()
    physical_plcs = normalized_points_target_to_physical(
        plcs_output.position,
        plcs_provenance,
    )
    physical_heading = headings_target_to_physical(
        plcs_output.rotation,
        plcs_provenance,
    )
    assert physical_plcs.shape == (1, 2, 3)
    assert physical_heading.shape == (1, 2, 2)
    assert torch.isfinite(physical_plcs).all()


def test_single_view_uses_selected_camera_as_the_complete_reference_frame(
    tmp_path: Path,
) -> None:
    contract = resolve_court_keypoint_contract("camera_view_v2")
    half_turn = torch.diag(torch.tensor([-1.0, -1.0, 1.0]))
    physical_position = torch.tensor(
        [[1.0, 2.0, 0.5], [1.5, 2.5, 0.75]],
        dtype=torch.float32,
    )
    expected_position = normalize_court_position(physical_position @ half_turn.T)
    expected_center = torch.tensor([-0.5, -12.0, -5.0])

    blcs_root = tmp_path / "blcs_single"
    _write_blcs_dataset(blcs_root, contract)
    blcs_config = _blcs_config("camera_view_v2")
    blcs_config.data.num_views_range = [1, 1]
    blcs_sample = cast(
        "dict[str, Any]",
        BallTrajectoryDataset(
            scene_dir=blcs_root,
            split_file="test.txt",
            config=blcs_config,
            augment=False,
        )[0],
    )
    blcs_provenance = cast(
        "CourtReferenceFrameProvenance",
        blcs_sample["court_reference_provenance"],
    )
    assert blcs_provenance.reference_camera_id == "cam_0"
    assert blcs_provenance.reference_camera_local_index == 0
    assert blcs_sample["court_kp"].shape == (1, 2, 20, 2)
    torch.testing.assert_close(
        blcs_sample["court_kp"][0, 0],
        torch.from_numpy(_PHYSICAL_COURT_UV[np.asarray(COURT_KP20_HALF_TURN_INDEX)]),
    )
    torch.testing.assert_close(blcs_sample["position_3d"], expected_position)
    torch.testing.assert_close(blcs_sample["camera_C"][0], expected_center)
    torch.testing.assert_close(blcs_sample["camera_R"][0], half_turn)

    plcs_root = tmp_path / "plcs_single"
    _write_plcs_dataset(plcs_root, contract)
    plcs_config = _plcs_config("camera_view_v2")
    plcs_config.data.num_views_range = [1, 1]
    plcs_sample = SceneDataset(
        scene_dir=plcs_root,
        split_file="test.txt",
        config=plcs_config,
        augment=False,
    )[0]
    plcs_provenance = cast(
        "CourtReferenceFrameProvenance",
        plcs_sample["court_reference_provenance"],
    )
    assert plcs_provenance.reference_camera_id == "camera_0"
    assert plcs_provenance.reference_camera_local_index == 0
    assert plcs_sample["selected_camera_ids"] == ("camera_0",)
    assert plcs_sample["court_kp"].shape == (1, 2, 20, 2)
    torch.testing.assert_close(
        plcs_sample["court_kp"][0, 0],
        torch.from_numpy(_PHYSICAL_COURT_UV[np.asarray(COURT_KP20_HALF_TURN_INDEX)]),
    )
    torch.testing.assert_close(plcs_sample["position"], expected_position)
    torch.testing.assert_close(
        plcs_sample["rotation"],
        torch.tensor([[-1.0, 0.0], [-1.0, 0.0]]),
    )
    torch.testing.assert_close(
        plcs_sample["human_kp_3d"][:, 0],
        physical_position @ half_turn.T,
    )
    torch.testing.assert_close(
        plcs_sample["human_kp"][0],
        torch.full((2, 17, 2), 0.3),
    )
    torch.testing.assert_close(plcs_sample["camera_C"][0], expected_center)
    torch.testing.assert_close(plcs_sample["camera_R"][0], half_turn)


def _blcs_tracking_model() -> BLCSTrackQueryReferenceModel:
    config = parse_model_config(
        {
            "model": {
                "name": "blcs_track_query_reference",
                "hidden_dim": 24,
                "num_heads": 4,
                "num_stages": 4,
                "ffn_dim": 32,
                "ffn_type": "swiglu",
                "num_queries": 2,
                "rope_dim": 6,
                "dropout": 0.0,
                "invisible_init_std": 0.02,
                "target_frame_contract": "reference_camera_court_rzpi_v1",
                "track_query_rope_contract": "time_camera_reference_selector_v1",
                "reference_selector_mode": "reference",
                "mhc": {
                    "coefficient_dim": 8,
                    "sinkhorn_iters": 2,
                    "eps": 1.0e-6,
                    "residual_identity_bias": 4.0,
                    "update_scale_init": 0.0,
                },
                "cswa": {
                    "compression_ratio": 2,
                    "window_radius": 1,
                    "backend": "reference",
                },
            }
        }
    )
    assert isinstance(config, TrackQueryReferenceModelConfig)
    model = BLCSTrackQueryReferenceModel(config)
    model.eval()
    return model


def _plcs_tracking_model() -> PLCSTrackQueryReferenceModel:
    config = PLCSModelConfig.from_mapping(
        {
            "name": "plcs_track_query_reference",
            "hidden_dim": 24,
            "num_heads": 4,
            "ffn_dim": 32,
            "num_queries": 2,
            "num_stages": 4,
            "num_joints": 17,
            "rope_dim": 6,
            "rope_theta": 10_000.0,
            "ffn_type": "swiglu",
            "dropout": 0.0,
            "invisible_init_std": 0.02,
            "target_frame_contract": "reference_camera_court_rzpi_v1",
            "track_query_rope_contract": "time_camera_reference_selector_v1",
            "reference_selector_mode": "reference",
            "mhc": {
                "coefficient_dim": 8,
                "sinkhorn_iters": 2,
                "eps": 1.0e-6,
                "residual_identity_bias": 4.0,
                "update_scale_init": 0.0,
            },
            "cswa": {
                "compression_ratio": 2,
                "window_radius": 1,
                "backend": "reference",
            },
        }
    )
    model = PLCSTrackQueryReferenceModel(config)
    model.eval()
    return model


def test_tracking_models_consume_reference_aligned_first14(
    tmp_path: Path,
) -> None:
    contract = resolve_court_keypoint_contract("camera_view_v2")
    _write_blcs_dataset(tmp_path / "blcs", contract)
    blcs_dataset = BallTrajectoryDataset(
        scene_dir=tmp_path / "blcs",
        split_file="test.txt",
        config=_blcs_config("camera_view_v2"),
        augment=False,
        reference_camera_id="cam_1",
    )
    blcs_sample = cast("dict[str, Tensor]", blcs_dataset[0])
    court = blcs_sample["court_kp"][:, :, :14]
    ball = blcs_sample["ball_uv"].unsqueeze(2).expand(-1, -1, 2, -1)
    with torch.no_grad():
        output = _blcs_tracking_model()(
            ball.unsqueeze(0),
            torch.ones(1, 2, 2, 2, dtype=torch.bool),
            court.unsqueeze(0),
            torch.ones(1, 2, 2, 14, dtype=torch.bool),
            torch.zeros(1, 2, 2, dtype=torch.bool),
            blcs_sample["reference_view_index"].reshape(1),
        )
    assert output["position"].shape == (1, 2, 2, 3)

    _write_plcs_dataset(tmp_path / "plcs", contract)
    plcs_dataset = SceneDataset(
        scene_dir=tmp_path / "plcs",
        split_file="test.txt",
        config=_plcs_config("camera_view_v2"),
        augment=False,
        reference_camera_id="camera_1",
    )
    plcs_sample = plcs_dataset[0]
    court = plcs_sample["court_kp"][:, :, :14]
    human = plcs_sample["human_kp"].unsqueeze(2).expand(-1, -1, 2, -1, -1)
    with torch.no_grad():
        output = _plcs_tracking_model()(
            human.unsqueeze(0),
            torch.ones(1, 2, 2, 2, 17, dtype=torch.bool),
            court.unsqueeze(0),
            torch.ones(1, 2, 2, 14, dtype=torch.bool),
            torch.zeros(1, 2, 2, dtype=torch.bool),
            plcs_sample["reference_view_index"].reshape(1),
        )
    assert output["position"].shape == (1, 2, 2, 3)


class _ReferenceBLCS:
    input_profile = "multiview"

    def __init__(self, provenance: CourtReferenceFrameProvenance) -> None:
        self.provenance = provenance
        self.calls = 0

    def predict_multiview_arrays(self, **kwargs: Any) -> BLCSTrajectoryPrediction:
        self.calls += 1
        frames = cast("np.ndarray", kwargs["ball_uv"]).shape[1]
        return BLCSTrajectoryPrediction(
            position=torch.tensor([[[-2.0, -3.0, 1.0]]]).expand(1, frames, 3),
            velocity=None,
            court_reference_provenance=(self.provenance,),
            coordinates_in_metres=True,
        )


class _PhysicalPLCS:
    def __init__(self, provenance: CourtReferenceFrameProvenance) -> None:
        self.provenance = provenance
        self.calls = 0

    def require_input_profile(self, profile: str) -> None:
        assert profile == "multiview"

    def predict_multiview_observations(self, **kwargs: Any) -> PLCSPhysicalPrediction:
        self.calls += 1
        human = cast("np.ndarray", kwargs["human_kp"])
        players, _, frames = human.shape[:3]
        return PLCSPhysicalPrediction(
            position_meters=np.full((players, frames, 3), [2.0, 3.0, 1.0], np.float32),
            yaw_radians=np.zeros((players, frames), dtype=np.float32),
            court_reference_provenance=(self.provenance,),
        )


def test_tennis_scene_requires_v2_markers_and_publishes_physical_results(
    tmp_path: Path,
) -> None:
    contract = resolve_court_keypoint_contract("camera_view_v2")
    reference_view = build_court_view_record(
        camera_id="camera_positive",
        camera_center_court_m=(0.0, 12.0, 5.0),
        contract=contract,
    )
    provenance = build_reference_frame_provenance(
        (reference_view,),
        reference_camera_id=reference_view.camera_id,
    )
    document = _contract_document(contract)

    blcs = BLCSModule(
        replace(make_blcs_config(tmp_path), court_keypoint_contract=contract)
    )
    blcs_predictor = _ReferenceBLCS(provenance)
    blcs._predictor = cast(Any, blcs_predictor)
    inputs: _BLCSProcessInputs = {
        "ball_uv": np.full((1, 2, 2), 0.5, dtype=np.float32),
        "court_kp": np.full((1, 2, 20, 2), 0.5, dtype=np.float32),
        "ball_vis": np.ones((1, 2), dtype=np.bool_),
        "court_vis": np.ones((1, 2, 20), dtype=np.float32),
    }
    with pytest.raises(MissingCourtKeypointMetadataError):
        blcs.process(**inputs)
    assert blcs_predictor.calls == 0
    blcs_result = blcs.process(
        **inputs,
        court_keypoint_document=document,
        court_reference_provenance=provenance,
    )
    np.testing.assert_array_equal(
        blcs_result.ball_3d,
        np.full((2, 3), [2.0, 3.0, 1.0], dtype=np.float32),
    )
    assert blcs_result.court_reference_provenance == provenance
    assert (
        blcs_result.to_dict(contract)[COURT_KEYPOINT_METADATA_KEY]
        == (document[COURT_KEYPOINT_METADATA_KEY])
    )

    plcs = PLCSModule(
        replace(make_plcs_config(tmp_path), court_keypoint_contract=contract)
    )
    plcs_predictor = _PhysicalPLCS(provenance)
    plcs._predictor = cast(Any, plcs_predictor)
    plcs_inputs: _PLCSProcessInputs = {
        "human_kp_2d": np.full((1, 1, 2, 17, 2), 0.5, dtype=np.float32),
        "court_kp": np.full((1, 2, 20, 2), 0.5, dtype=np.float32),
        "human_kp_vis": np.ones((1, 1, 2, 17), dtype=np.float32),
        "court_vis": np.ones((1, 2, 20), dtype=np.float32),
        "track_ids": np.array([7], dtype=np.int32),
    }
    with pytest.raises(MissingCourtKeypointMetadataError):
        plcs.process(**plcs_inputs)
    assert plcs_predictor.calls == 0
    plcs_result = plcs.process(
        **plcs_inputs,
        court_keypoint_document=document,
        court_reference_provenance=provenance,
    )
    np.testing.assert_array_equal(
        plcs_result.position,
        np.full((1, 2, 3), [2.0, 3.0, 1.0], dtype=np.float32),
    )
    assert plcs_result.court_reference_provenance == provenance
    assert (
        plcs_result.to_dict(contract)[COURT_KEYPOINT_METADATA_KEY]
        == document[COURT_KEYPOINT_METADATA_KEY]
    )
