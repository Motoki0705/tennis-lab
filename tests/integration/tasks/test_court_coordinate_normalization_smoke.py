"""Cross-task CPU smoke tests for court-coordinate normalization v1/v2."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

from src.tasks.base.data.court_coordinate_contract import (
    COURT_COORDINATE_NORMALIZATION_METADATA_KEY,
)
from src.tasks.base.data.court_coordinate_materializer import (
    CourtCoordinateMaterializationConfig,
    materialize_court_coordinate_normalization_dataset,
)
from src.tasks.base.visualization.style import SceneStyleConfig
from src.tasks.blcs.models.components.differentiable_projection import (
    DifferentiableProjection,
)
from src.tasks.blcs.training.losses import trajectory_position_loss
from src.tasks.blcs.training.metrics import BLCSMetrics
from src.tasks.plcs.training.losses import PLCSLoss, PLCSLossConfig
from src.tasks.plcs.training.metrics import PLCSMetrics
from src.tasks.plcs.visualization.contracts import PoseRenderScene
from src.tasks.plcs.visualization.rendering import PLCSSceneRenderer
from src.tennis_scene.pipeline.components.blcs import BLCSResult
from src.tennis_scene.pipeline.components.plcs import PLCSResult
from src.tennis_scene.schema import (
    SceneResult,
    attach_scene_result_court_coordinate_provenance,
)
from src.utils.paths import PROJECT_ROOT
from src.utils.schema.court_normalization import resolve_court_coordinate_normalization

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    ("relative_config_dir", "config_name"),
    [
        ("src/tasks/blcs/configs", "train"),
        ("src/tasks/blcs/configs", "generate_dataset"),
        ("src/tasks/blcs/configs", "visualize"),
        ("src/tasks/plcs/configs", "train"),
        ("src/tasks/plcs/configs", "generate_dataset"),
        ("src/tasks/plcs/configs", "visualize"),
        ("src/tasks/slcs/configs", "train"),
        ("src/tasks/slcs/configs", "evaluate"),
        ("src/tasks/slcs/configs", "predict_clip"),
        ("src/tennis_scene/configs", "pipeline"),
    ],
)
def test_hydra_boundaries_default_to_v1_and_explicitly_compose_v2(
    relative_config_dir: str,
    config_name: str,
) -> None:
    config_dir = PROJECT_ROOT / relative_config_dir
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        default = compose(config_name=config_name)
        selected = compose(
            config_name=config_name,
            overrides=["court_coordinate_normalization=v2"],
        )

    assert default.court_coordinate_normalization.version == "v1"
    assert selected.court_coordinate_normalization.version == "v2"


def _write_legacy_scene_dataset(
    root: Path,
    *,
    kind: str,
    physical: np.ndarray,
) -> tuple[Path, Path]:
    scene = root / "scenes" / "scene_000001"
    scene.mkdir(parents=True)
    (scene / "meta.json").write_text(
        json.dumps({"scene_id": "scene_000001", "num_frames": len(physical)}),
        encoding="utf-8",
    )
    v1 = resolve_court_coordinate_normalization("v1")
    normalized_name = "ball_pos_norm.npy" if kind == "blcs" else "position.npy"
    np.save(scene / normalized_name, v1.normalize_position(physical))
    if kind == "blcs":
        np.save(scene / "ball_pos_world.npy", physical)
    (root / "train.txt").write_text("scene_000001\n", encoding="utf-8")
    return scene, scene / normalized_name


@pytest.mark.parametrize("kind", ["blcs", "plcs"])
def test_v2_materialization_is_non_overwriting_and_round_trips_legacy_root_without_meta(
    tmp_path: Path,
    kind: str,
) -> None:
    physical = np.array(
        [[-4.0, -10.0, 0.5], [2.5, 7.0, 3.25]], dtype=np.float32
    )
    source = tmp_path / f"{kind}_legacy"
    source_scene, source_normalized = _write_legacy_scene_dataset(
        source,
        kind=kind,
        physical=physical,
    )
    source_bytes = source_normalized.read_bytes()
    output = tmp_path / f"{kind}_fixture_norm_v2"
    config = CourtCoordinateMaterializationConfig(
        dataset_kind=kind,  # type: ignore[arg-type]
        source_dir=source,
        output_dir=output,
        source_contract=resolve_court_coordinate_normalization("v1"),
        target_contract=resolve_court_coordinate_normalization("v2"),
        max_abs_round_trip_error_m=1.0e-5,
    )

    result = materialize_court_coordinate_normalization_dataset(config)

    assert result.scene_count == 1
    assert result.max_abs_round_trip_error_m <= 1.0e-5
    assert not (source / "meta.json").exists()
    assert source_normalized.read_bytes() == source_bytes
    output_name = "ball_pos_norm.npy" if kind == "blcs" else "position.npy"
    normalized_v2 = np.load(output / "scenes" / source_scene.name / output_name)
    restored = resolve_court_coordinate_normalization("v2").denormalize_position(
        normalized_v2
    )
    np.testing.assert_allclose(restored, physical, atol=1.0e-5, rtol=0.0)

    root_metadata = json.loads((output / "meta.json").read_text(encoding="utf-8"))
    scene_metadata = json.loads(
        (output / "scenes" / source_scene.name / "meta.json").read_text(
            encoding="utf-8"
        )
    )
    assert root_metadata[COURT_COORDINATE_NORMALIZATION_METADATA_KEY] == (
        scene_metadata[COURT_COORDINATE_NORMALIZATION_METADATA_KEY]
    )
    assert root_metadata[COURT_COORDINATE_NORMALIZATION_METADATA_KEY]["version"] == "v2"
    assert result.manifest_path.is_file()
    with pytest.raises(FileExistsError, match="overwrite"):
        materialize_court_coordinate_normalization_dataset(config)


def _plcs_loss_config() -> PLCSLossConfig:
    return PLCSLossConfig(
        position_weight=1.0,
        rotation_weight=0.0,
        angle_weight=0.0,
        position_smoothness_weight=0.0,
        canonical_pose_weight=0.0,
        joint_angle_weight=0.0,
        torsion_angle_weight=0.0,
        torso_twist_weight=0.0,
        bone_length_weight=0.0,
        joint_angle_velocity_weight=0.0,
        torsion_angle_velocity_weight=0.0,
        torso_twist_velocity_weight=0.0,
        joint_angle_velocity_angle_weights=None,
        torsion_angle_velocity_angle_weights=None,
    )


def _style() -> SceneStyleConfig:
    return SceneStyleConfig(
        theme="light",
        show_shadow=False,
        show_trail=True,
        trail_length=4,
        show_hud=False,
        show_minimap=False,
    )


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_blcs_plcs_cpu_flow_stays_meter_valued_through_projection_and_render(
    tmp_path: Path,
    version: str,
) -> None:
    contract = resolve_court_coordinate_normalization(version)
    physical = np.array(
        [[1.0, -2.0, 1.2], [2.0, -1.0, 1.5], [3.0, 0.0, 1.8]],
        dtype=np.float32,
    )
    dataset_path = tmp_path / f"positions_{version}.npy"
    np.save(dataset_path, contract.normalize_position(physical))
    dataset_position = torch.from_numpy(np.load(dataset_path)).unsqueeze(0)

    model = torch.nn.Identity().cpu()
    prediction = model(dataset_position)
    perturbed = prediction.clone()
    perturbed[..., 0] += 0.5 / contract.scale_xyz[0]

    blcs_loss = trajectory_position_loss(
        perturbed,
        prediction,
        torch.ones(1, 3, dtype=torch.bool),
        axis_weights=torch.ones(3),
        beta=(1.0 if version == "v1" else 1.0 / contract.scale_xyz[0]),
    )
    assert torch.isfinite(blcs_loss)
    blcs_metric = BLCSMetrics(
        position_threshold_m=0.3,
        endpoint_threshold_m=0.5,
        normalization=contract,
    ).update(perturbed, prediction)
    assert blcs_metric["x_error_m"] == pytest.approx(0.5)
    blcs_meters = contract.denormalize_position(prediction).squeeze(0).numpy()
    np.testing.assert_allclose(blcs_meters, physical, atol=1.0e-5, rtol=0.0)

    camera_r = torch.eye(3).view(1, 1, 3, 3)
    projector = DifferentiableProjection(normalization=contract)
    uv, visible = projector(
        prediction,
        camera_r,
        torch.tensor([[[0.0, 0.0, -20.0]]]),
        torch.tensor([[1000.0]]),
        torch.tensor([[500.0]]),
        torch.tensor([[500.0]]),
        torch.tensor([[1000.0]]),
        torch.tensor([[1000.0]]),
    )
    assert torch.isfinite(uv).all()
    assert visible.all()

    plcs_loss_fn = PLCSLoss(_plcs_loss_config(), normalization=contract)
    prepared = plcs_loss_fn.prepare_inputs(
        pred_position=perturbed,
        pred_rotation=torch.tensor([[[1.0, 0.0]]]).expand(1, 3, 2),
        target_position=prediction,
        target_rotation=torch.tensor([[[1.0, 0.0]]]).expand(1, 3, 2),
        pred_canonical_pose=None,
        target_human_kp_3d=None,
        padding_mask=torch.zeros(1, 3, dtype=torch.bool),
    )
    assert torch.isfinite(plcs_loss_fn(prepared)["total"])
    plcs_metric = PLCSMetrics(
        position_threshold_m=0.3,
        angle_threshold_deg=10.0,
        normalization=contract,
    ).update(
        perturbed,
        torch.tensor([[[1.0, 0.0]]]).expand(1, 3, 2),
        prediction,
        torch.tensor([[[1.0, 0.0]]]).expand(1, 3, 2),
    )
    assert plcs_metric["x_error_m"] == pytest.approx(0.5)

    canonical: np.ndarray = np.zeros((3, 17, 3), dtype=np.float32)
    canonical[:, :, 2] = np.linspace(0.0, 1.6, 17, dtype=np.float32)
    render_scene = PoseRenderScene(
        position=np.asarray(dataset_position.squeeze(0)),
        rotation=np.tile(np.array([1.0, 0.0], dtype=np.float32), (3, 1)),
        canonical_pose_3d=canonical,
        meta={"num_frames": 3},
    )
    renderer = PLCSSceneRenderer(style=_style(), normalization=contract)
    np.testing.assert_allclose(
        renderer._world_positions(render_scene),
        physical,
        atol=1.0e-5,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        renderer._compute_world_pose(render_scene, 0),
        canonical[0] + physical[0],
        atol=1.0e-5,
        rtol=0.0,
    )

    blcs_result = BLCSResult(
        ball_3d=blcs_meters.astype(np.float32),
        visibility=np.ones(3, dtype=np.bool_),
    )
    plcs_result = PLCSResult(
        position=physical[None],
        yaw=np.zeros((1, 3), dtype=np.float32),
        track_ids=np.array([7], dtype=np.int32),
    )
    integrated = SceneResult(
        num_frames=3,
        fps=30.0,
        width=1920,
        height=1080,
        court_kp=np.zeros((1, 3, 14, 2), dtype=np.float32),
        court_vis=np.ones((1, 3, 14), dtype=np.float32),
        player_position=plcs_result.position,
        player_yaw=plcs_result.yaw,
        smpl_body_pose=np.zeros((1, 3, 63), dtype=np.float32),
        smpl_global_orient=np.zeros((1, 3, 3), dtype=np.float32),
        smpl_betas=np.zeros((1, 10), dtype=np.float32),
        ball_3d=blcs_result.ball_3d,
    )
    original_player = integrated.player_position.copy()
    original_ball = integrated.ball_3d.copy()
    attach_scene_result_court_coordinate_provenance(integrated, contract)
    np.testing.assert_array_equal(integrated.player_position, original_player)
    np.testing.assert_array_equal(integrated.ball_3d, original_ball)
