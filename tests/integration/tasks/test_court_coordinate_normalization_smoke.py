"""Cross-task CPU smoke tests for court-coordinate normalization v1/v2."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from torch import Tensor

from src.tasks.base.data.court_coordinate_contract import (
    COURT_COORDINATE_NORMALIZATION_METADATA_KEY,
    CourtCoordinateNormalizationMetadata,
)
from src.tasks.base.data.court_coordinate_materializer import (
    CourtCoordinateMaterializationConfig,
    materialize_court_coordinate_normalization_dataset,
)
from src.tasks.base.visualization.style import SceneStyleConfig
from src.tasks.blcs.data.dataset import (
    BallTrajectoryDataset,
    collate_multiview_trajectories,
)
from src.tasks.blcs.model_io import (
    BLCSTrajectoryPrediction,
    compose_blcs_trajectory_model_io,
)
from src.tasks.blcs.models.blcs_multiview_axial_model import (
    BLCSMultiViewAxialModel,
)
from src.tasks.blcs.models.components.differentiable_projection import (
    DifferentiableProjection,
)
from src.tasks.blcs.training.lightning_module import BLCSLightningModule
from src.tasks.plcs.data.dataset import SceneDataset, collate_plcs_batch
from src.tasks.plcs.model_io import PLCSDecodedPrediction
from src.tasks.plcs.models.plcs_multiview_model import PLCSMultiViewModel
from src.tasks.plcs.training.lightning_module import PLCSLightningModule
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

_LEGACY_FIXTURE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "issue_786"
    / "legacy_v1_representative"
    / "datasets"
)


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


def _style() -> SceneStyleConfig:
    return SceneStyleConfig(
        theme="light",
        show_shadow=False,
        show_trail=True,
        trail_length=4,
        show_hud=False,
        show_minimap=False,
    )


def _metadata(version: str) -> dict[str, object]:
    metadata: dict[str, object] = CourtCoordinateNormalizationMetadata.from_contract(
        resolve_court_coordinate_normalization(version)
    ).to_dict()
    return metadata


def _versioned_dataset(
    tmp_path: Path,
    *,
    task: str,
    version: str,
) -> Path:
    source = _LEGACY_FIXTURE_ROOT / f"{task}_legacy_v1"
    destination = tmp_path / f"{task}_{version}"
    shutil.copytree(source, destination)
    scene = next((destination / "scenes").iterdir())
    v1 = resolve_court_coordinate_normalization("v1")
    contract = resolve_court_coordinate_normalization(version)
    if task == "blcs":
        physical = np.load(scene / "ball_pos_world.npy")
        np.save(scene / "ball_pos_norm.npy", contract.normalize_position(physical))
    else:
        legacy_position = np.load(scene / "position.npy")
        physical = v1.denormalize_position(legacy_position)
        np.save(scene / "position.npy", contract.normalize_position(physical))

    metadata = _metadata(version)
    (destination / "meta.json").write_text(
        json.dumps({COURT_COORDINATE_NORMALIZATION_METADATA_KEY: metadata}),
        encoding="utf-8",
    )
    scene_document = json.loads((scene / "meta.json").read_text(encoding="utf-8"))
    assert isinstance(scene_document, dict)
    scene_document[COURT_COORDINATE_NORMALIZATION_METADATA_KEY] = metadata
    (scene / "meta.json").write_text(
        json.dumps(scene_document),
        encoding="utf-8",
    )
    return destination


def _compose_blcs_smoke_config(version: str) -> DictConfig:
    overrides = [
        f"court_coordinate_normalization={version}",
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
        "model.max_num_cameras=1",
        "model.num_court_tokens=14",
        "model.dropout=0.0",
        "data.seq_len_range=[2,2]",
        "data.num_views_range=[1,1]",
        "data.camera_mode=first",
        "data.num_court_kp=14",
        "data.batch_size=1",
        "data.num_workers=0",
        "training.compile.enabled=false",
    ]
    with initialize_config_dir(
        version_base="1.3",
        config_dir=str(PROJECT_ROOT / "src/tasks/blcs/configs"),
    ):
        config = compose(config_name="train", overrides=overrides)
    return config


def _compose_plcs_smoke_config(version: str) -> DictConfig:
    overrides = [
        f"court_coordinate_normalization={version}",
        "model=multiview",
        "loss=no_canonical",
        "model.hidden_dim=16",
        "model.num_layers=1",
        "model.num_heads=4",
        "model.ffn_dim=32",
        "model.rope_dim=4",
        "model.max_seq_len=2",
        "model.max_views=1",
        "model.dropout=0.0",
        "data.seq_len_range=[2,2]",
        "data.num_views_range=[1,1]",
        "data.min_cameras=1",
        "data.camera_mode=first",
        "data.num_court_kp=14",
        "data.batch_size=1",
        "data.num_workers=0",
        "training.compile.enabled=false",
    ]
    with initialize_config_dir(
        version_base="1.3",
        config_dir=str(PROJECT_ROOT / "src/tasks/plcs/configs"),
    ):
        config = compose(config_name="train", overrides=overrides)
    return config


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_actual_blcs_plcs_cpu_flow_reaches_projection_and_render(
    tmp_path: Path,
    version: str,
) -> None:
    contract = resolve_court_coordinate_normalization(version)
    torch.manual_seed(786)

    blcs_root = _versioned_dataset(
        tmp_path,
        task="blcs",
        version=version,
    )
    blcs_config = _compose_blcs_smoke_config(version)
    blcs_dataset = BallTrajectoryDataset(
        scene_dir=blcs_root,
        split_file=blcs_root / "test.txt",
        config=blcs_config,
        augment=False,
    )
    blcs_dataset.rng = np.random.default_rng(786)
    blcs_batch = dict(collate_multiview_trajectories([blcs_dataset[0]]))
    blcs_binding = compose_blcs_trajectory_model_io(blcs_config)
    blcs_module = BLCSLightningModule(
        blcs_config,
        model_io=blcs_binding,
    ).cpu().eval()
    assert isinstance(blcs_dataset, BallTrajectoryDataset)
    assert isinstance(blcs_module.model, BLCSMultiViewAxialModel)
    blcs_module.test_metrics.reset()
    with torch.no_grad():
        blcs_result = blcs_module._compute_supervised_result(blcs_batch, "test")
    blcs_output = blcs_result["outputs"]
    assert isinstance(blcs_output, BLCSTrajectoryPrediction)
    assert torch.isfinite(cast("Tensor", blcs_result["loss"]))
    assert blcs_module.test_metrics.compute()
    blcs_meters = contract.denormalize_position(blcs_output.position)
    assert isinstance(blcs_meters, Tensor)
    assert blcs_meters.shape == (1, 2, 3)
    assert torch.isfinite(blcs_meters).all()

    projector = DifferentiableProjection(normalization=contract)
    uv, visible = projector(
        blcs_output.position,
        blcs_batch["camera_R"],
        blcs_batch["camera_C"],
        blcs_batch["camera_f"],
        blcs_batch["camera_cx"],
        blcs_batch["camera_cy"],
        blcs_batch["camera_w"],
        blcs_batch["camera_h"],
    )
    assert uv.shape == (1, 1, 2, 2)
    assert visible.shape == (1, 1, 2)
    assert torch.isfinite(uv).all()
    torch.testing.assert_close(
        projector.scale_xyz,
        torch.tensor(contract.scale_xyz, dtype=torch.float32),
    )

    plcs_root = _versioned_dataset(
        tmp_path,
        task="plcs",
        version=version,
    )
    plcs_config = _compose_plcs_smoke_config(version)
    plcs_dataset = SceneDataset(
        scene_dir=plcs_root,
        split_file=plcs_root / "test.txt",
        config=plcs_config,
        augment=False,
    )
    plcs_dataset.rng = np.random.default_rng(786)
    plcs_batch = cast("dict[str, Tensor]", dict(collate_plcs_batch([plcs_dataset[0]])))
    torch.manual_seed(786)
    plcs_module = PLCSLightningModule(plcs_config).cpu().eval()
    assert isinstance(plcs_dataset, SceneDataset)
    assert isinstance(plcs_module.model, PLCSMultiViewModel)
    plcs_module.test_metrics.reset()
    with torch.no_grad():
        plcs_result = plcs_module._compute_supervised_result(plcs_batch, "test")
    plcs_output = plcs_result["outputs"]
    assert isinstance(plcs_output, PLCSDecodedPrediction)
    assert torch.isfinite(cast("Tensor", plcs_result["loss"]))
    assert plcs_module.test_metrics.compute()
    plcs_meters = contract.denormalize_position(plcs_output.position)
    assert isinstance(plcs_meters, Tensor)
    assert plcs_meters.shape == (1, 2, 3)
    assert torch.isfinite(plcs_meters).all()

    canonical: np.ndarray = np.zeros((2, 17, 3), dtype=np.float32)
    canonical[:, :, 2] = np.linspace(0.0, 1.6, 17, dtype=np.float32)
    render_scene = PoseRenderScene(
        position=plcs_output.position[0].detach().cpu().numpy(),
        rotation=plcs_output.rotation[0].detach().cpu().numpy(),
        canonical_pose_3d=canonical,
        meta={"num_frames": 2},
    )
    renderer = PLCSSceneRenderer(style=_style(), normalization=contract)
    rendered_meters = renderer._world_positions(render_scene)
    np.testing.assert_allclose(
        rendered_meters,
        plcs_meters[0].numpy(),
        atol=1.0e-5,
        rtol=0.0,
    )
    figure = plt.figure(figsize=(4, 3))
    axes_3d = figure.add_subplot(111, projection="3d")
    renderer._render_3d_frame(
        axes_3d,
        [(render_scene, "red", "Prediction")],
        0,
        2,
        30.0,
        title=f"PLCS {version}",
    )
    figure.canvas.draw()
    assert axes_3d.has_data()
    plt.close(figure)

    figure_2d, axes_2d = plt.subplots(figsize=(4, 3))
    renderer._render_2d_subplot(axes_2d, render_scene, 0)
    figure_2d.canvas.draw()
    assert axes_2d.has_data()
    plt.close(figure_2d)

    blcs_result = BLCSResult(
        ball_3d=blcs_meters[0].numpy().astype(np.float32),
        visibility=np.ones(2, dtype=np.bool_),
    )
    plcs_result = PLCSResult(
        position=plcs_meters.numpy().astype(np.float32),
        yaw=np.zeros((1, 2), dtype=np.float32),
        track_ids=np.array([7], dtype=np.int32),
    )
    integrated = SceneResult(
        num_frames=2,
        fps=30.0,
        width=1920,
        height=1080,
        court_kp=np.zeros((1, 2, 14, 2), dtype=np.float32),
        court_vis=np.ones((1, 2, 14), dtype=np.float32),
        player_position=plcs_result.position,
        player_yaw=plcs_result.yaw,
        smpl_body_pose=np.zeros((1, 2, 63), dtype=np.float32),
        smpl_global_orient=np.zeros((1, 2, 3), dtype=np.float32),
        smpl_betas=np.zeros((1, 10), dtype=np.float32),
        ball_3d=blcs_result.ball_3d,
    )
    original_player = integrated.player_position.copy()
    assert integrated.ball_3d is not None
    original_ball = integrated.ball_3d.copy()
    attach_scene_result_court_coordinate_provenance(integrated, contract)
    np.testing.assert_array_equal(integrated.player_position, original_player)
    np.testing.assert_array_equal(integrated.ball_3d, original_ball)
