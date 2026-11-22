"""Explicit typed configuration factories for tennis-scene unit tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, TypeVar

from src.submodules.configuration import (
    BackendConfig,
    BatchedRuntimeConfig,
    DinoDetectorRuntimeConfig,
    Hmr2RuntimeConfig,
    SmplAssetPaths,
    SubmoduleRuntimeConfig,
    TrackingRuntimeConfig,
    ViTPoseRuntimeConfig,
)
from src.submodules.vendor.gvhmr.vitpose.heatmap_head import ViTPoseHeadConfig
from src.tasks.ball_detection.inference.trajectory_gate import TrajectoryGateConfig
from src.tasks.ball_detection.model_io.contracts import BallCandidateConfig
from src.tasks.court_detection.inference.regions import CourtRegionSearchConfig
from src.tennis_scene.pipeline.components.ball_detection import BallDetectionConfig
from src.tennis_scene.pipeline.components.court_kp import (
    CourtKPConfig,
    CourtKPModule,
    CourtKPPostprocessConfig,
)
from src.tennis_scene.pipeline.execution_plan import ComponentNode
from src.tennis_scene.pipeline.inference_runtime import InferenceComponent, ModelSpec
from src.tennis_scene.pipeline.model_assets import PeopleModelConfig
from src.utils.configuration import PathResolver, PathRole, RuntimePathRoots

ComponentT = TypeVar("ComponentT", bound=InferenceComponent[Any, Any])


def model_component(node: ComponentNode) -> InferenceComponent[Any, Any]:
    assert isinstance(node.component, InferenceComponent)
    return node.component


def fake_pose_spec() -> ModelSpec[Any]:
    """Explicit no-assets provenance for saved CPU feature fixtures; never loaded."""
    return ModelSpec('vitpose', {}, 'fixture_coco17', 'cpu', 'float32', {}, object)


def make_resolver(root: Path) -> PathResolver:
    """Build all seven explicit path roots beneath one temporary project."""
    project = root.resolve()
    return PathResolver(
        RuntimePathRoots(
            project_root=project,
            data_root=project / "data",
            checkpoint_root=project / "ckpt",
            artifact_root=project / "artifacts",
            output_root=project / "outputs",
            cache_root=project / ".cache",
            external_asset_root=project / "third_party",
        )
    )


def make_submodule_runtime(
    *,
    dino_confidence: float = 0.35,
) -> SubmoduleRuntimeConfig:
    """Return the complete typed model-runtime contract used by GVHMR."""
    backend = BackendConfig(precision="fp32", engines=())
    return SubmoduleRuntimeConfig(
        device="cpu",
        tracking=TrackingRuntimeConfig(
            bbox_enlarge=1.2,
        ),
        dino_detector=DinoDetectorRuntimeConfig(
            confidence=dino_confidence,
            short_side=800,
            max_long_side=1333,
            precision="fp32",
            batch_size=1,
            engines=(),
        ),
        vitpose=ViTPoseRuntimeConfig(
            flip_test=True,
            batch_size=8,
            precision="fp32",
            engines=(),
            head=ViTPoseHeadConfig(
                in_channels=1280,
                out_channels=17,
                num_deconv_layers=2,
                num_deconv_filters=(256, 256),
                num_deconv_kernels=(4, 4),
                final_conv_kernel=1,
                num_conv_layers=0,
                num_conv_kernels=(),
            ),
        ),
        hmr2=Hmr2RuntimeConfig(batch_size=8, backend=backend),
        clip_reid=BatchedRuntimeConfig(batch_size=64, backend=backend),
        gvhmr=BatchedRuntimeConfig(batch_size=1, backend=backend),
        static_cam=True,
    )


def make_ball_config(root: Path) -> BallDetectionConfig:
    resolver = make_resolver(root)
    return BallDetectionConfig(
        model_type="conv_next_unet", input_channels=2, num_frames=8, precision="fp32", compile_mode="off",
        checkpoint=resolver.resolve(PathRole.CHECKPOINT, "ball.ckpt"),
        batch_size=2,
        device="cpu",
        image_size=(360, 640),
        normalize_imagenet=False,
        score_threshold=0.1,
        subpixel_refine=False,
        checkpoint_strict=True,
        checkpoint_weights_only=False,
        prefetch_batches=1,
        window_stride=None,
        tail_policy="backfill",
        overlap_aggregation="max_score",
        pin_memory=False,
        trajectory_gate=TrajectoryGateConfig(
            enabled=True,
            max_residual_px=50.0,
            k_support=2,
            max_support_gap=4,
            max_passes=2,
        ),
        resolver=resolver,
        candidates=BallCandidateConfig(),
    )


def make_court_kp_config(root: Path) -> CourtKPConfig:
    resolver = make_resolver(root)
    return CourtKPConfig(
        checkpoint=resolver.resolve(PathRole.CHECKPOINT, "court.ckpt"),
        device="cpu",
        subpixel_refine=False,
        postprocess=CourtKPPostprocessConfig(),
        region_search=CourtRegionSearchConfig(),
        resolver=resolver,
    )


def make_x3d_ball_config(root: Path) -> BallDetectionConfig:
    return replace(make_ball_config(root), model_type="x3d_csn_dpt", input_channels=5, num_frames=32,
                   precision="fp32", image_size=(720, 1280), checkpoint_weights_only=True,
                   window_stride=16, overlap_aggregation="nearest_window_centre_then_earlier_start")


def make_people_config(root: Path, *, dino_confidence: float = .35) -> PeopleModelConfig:
    resolver = make_resolver(root)
    return PeopleModelConfig(
        gvhmr_checkpoint=resolver.resolve(PathRole.CHECKPOINT, "gvhmr.ckpt"),
        dino_checkpoint=resolver.resolve(PathRole.CHECKPOINT, "dino.pth"),
        dino_repository=resolver.resolve(PathRole.EXTERNAL_ASSET, "DINO"),
        vitpose_checkpoint=resolver.resolve(PathRole.CHECKPOINT, "vitpose.pth"),
        hmr2_checkpoint=resolver.resolve(PathRole.CHECKPOINT, "hmr2.ckpt"),
        body_models_dir=resolver.resolve(PathRole.CHECKPOINT, "body_models"),
        smpl_assets=SmplAssetPaths(
            hmr2_mean_params=resolver.resolve(PathRole.CHECKPOINT, "hmr2_mean.npz"),
            smplx_to_smpl=resolver.resolve(PathRole.CHECKPOINT, "smplx_to_smpl.pkl"),
            smpl_coco17_regressor=resolver.resolve(PathRole.CHECKPOINT, "smpl_coco17.npy"),
            smplx_verts437=resolver.resolve(PathRole.CHECKPOINT, "smplx_verts437.npy"),
            smpl_neutral_joint_regressor=resolver.resolve(PathRole.CHECKPOINT, "smplx_neutral_joints.npy")),
        runtime=make_submodule_runtime(dino_confidence=dino_confidence))


def materialize_model_weights(component: ComponentT) -> ComponentT:
    """Dummy content identities for tests replacing model loading, never real weights."""
    if isinstance(component, CourtKPModule):
        materialize_court_assets(component.config)
    from src.tennis_scene.pipeline.components.person_detection import (
        PersonDetectionModule,
    )
    if isinstance(component, PersonDetectionModule):
        materialize_dino_sources(component.config.dino_repository)
    for spec in component.model_specs().values():
        for path in spec.weights.values():
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(b"fake-model-identity")
    return component


def materialize_dino_sources(repository: Path) -> None:
    for relative in ("models/__init__.py", "util/__init__.py"):
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# test-owned external DINO source\n")


def materialize_court_assets(config: CourtKPConfig) -> None:
    """Real inference metadata and dummy external assets; no Court model build."""
    import torch
    from omegaconf import OmegaConf

    from src.tasks.court_detection.data.bundle_state import serialize_target_bundle
    from src.tasks.court_detection.inference.checkpoint import CourtInferenceSpec
    from tests.unit.tasks.court_detection.inference.test_pose_output_predictors import (
        _bundle,
    )
    from tests.unit.tasks.court_detection.test_configuration import _compose

    saved = OmegaConf.to_container(_compose("synthetic_court"), resolve=True)
    bundle = serialize_target_bundle(_bundle())
    config.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"hyper_parameters": {"config": saved, "target_bundle_state": bundle}}, config.checkpoint)
    spec = CourtInferenceSpec.from_checkpoint_config(saved, bundle, resolver=config.resolver)
    backbone, repository = spec.model.encoder.checkpoint_path, spec.model.encoder.repository_path
    assert backbone is not None and repository is not None
    backbone.parent.mkdir(parents=True, exist_ok=True)
    backbone.write_bytes(b"fake-backbone")
    repository.mkdir(parents=True, exist_ok=True)
    (repository / "hubconf.py").write_text("# test-owned DINOv3 implementation\n")


def fake_encoder_spec(factory: Callable[[], Any]) -> ModelSpec[Any]:
    return ModelSpec("appearance", {}, "fake-encoder", "cpu", "float32", {}, factory)
