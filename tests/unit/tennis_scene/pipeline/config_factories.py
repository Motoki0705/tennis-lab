"""Explicit typed configuration factories for tennis-scene unit tests."""

from __future__ import annotations

from pathlib import Path

from src.submodules.configuration import (
    BundledModelAssetPaths,
    DinoDetectorRuntimeConfig,
    Hmr2RuntimeConfig,
    SubmoduleRuntimeConfig,
    TrackingRuntimeConfig,
    ViTPoseRuntimeConfig,
)
from src.submodules.vendor.gvhmr.vitpose.heatmap_head import ViTPoseHeadConfig
from src.tasks.ball_detection.inference.trajectory_gate import TrajectoryGateConfig
from src.tasks.court_detection.inference.regions import CourtRegionSearchConfig
from src.tennis_scene.pipeline.components.ball_detection import BallDetectionConfig
from src.tennis_scene.pipeline.components.court_kp import (
    CourtKPConfig,
    CourtKPPostprocessConfig,
)
from src.tennis_scene.pipeline.model_assets import PeopleModelConfig
from src.utils.configuration import PathResolver, PathRole, RuntimePathRoots


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
    return SubmoduleRuntimeConfig(
        device="cpu",
        tracking=TrackingRuntimeConfig(
            yolo_confidence=0.25,
            bbox_enlarge=1.2,
        ),
        dino_detector=DinoDetectorRuntimeConfig(
            confidence=dino_confidence,
            short_side=800,
            max_long_side=1333,
        ),
        vitpose=ViTPoseRuntimeConfig(
            flip_test=True,
            batch_size=8,
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
        hmr2=Hmr2RuntimeConfig(batch_size=8),
        static_cam=True,
    )


def make_ball_config(root: Path) -> BallDetectionConfig:
    resolver = make_resolver(root)
    return BallDetectionConfig(
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








def make_people_config(root: Path, *, detector: str = "dino", dino_confidence: float = .35) -> PeopleModelConfig:
    resolver = make_resolver(root)
    return PeopleModelConfig(detector=detector,
        gvhmr_checkpoint=resolver.resolve(PathRole.CHECKPOINT, "gvhmr.ckpt"),
        yolo_checkpoint=resolver.resolve(PathRole.CHECKPOINT, "yolo.pt"),
        dino_checkpoint=resolver.resolve(PathRole.CHECKPOINT, "dino.pth"),
        dino_repository=resolver.resolve(PathRole.EXTERNAL_ASSET, "DINO"),
        vitpose_checkpoint=resolver.resolve(PathRole.CHECKPOINT, "vitpose.pth"),
        hmr2_checkpoint=resolver.resolve(PathRole.CHECKPOINT, "hmr2.ckpt"),
        body_models_dir=resolver.resolve(PathRole.CHECKPOINT, "body_models"),
        bundled_assets=BundledModelAssetPaths(
            hmr2_mean_params=resolver.resolve(PathRole.PROJECT, "hmr2_mean.npz"),
            smplx_to_smpl=resolver.resolve(PathRole.PROJECT, "smplx_to_smpl.pkl"),
            smpl_coco17_regressor=resolver.resolve(PathRole.PROJECT, "smpl_coco17.npy"),
            smplx_verts437=resolver.resolve(PathRole.PROJECT, "smplx_verts437.npy"),
            smpl_neutral_joint_regressor=resolver.resolve(PathRole.PROJECT, "smplx_neutral_joints.npy")),
        runtime=make_submodule_runtime(dino_confidence=dino_confidence))
