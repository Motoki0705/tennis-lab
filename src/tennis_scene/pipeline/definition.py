"""The standard recipe: explicit bindings, replaceable components and input builders."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from functools import lru_cache, partial
from typing import TYPE_CHECKING, Any

from src.tasks.person_tracking.all_person import ALL_PERSON_BOTSORT_SETTINGS
from src.tasks.person_tracking.court_linking import LinkingConfig
from src.tasks.player_association.appearance.encoders import build_encoder
from src.tasks.player_association.appearance.sampling import CropSamplingConfig
from src.tennis_scene.pipeline.artifacts import json_value
from src.tennis_scene.pipeline.ball_refiner_recipe import ball_refiner_definition
from src.tennis_scene.pipeline.components.ball_points import BallPointsModule
from src.tennis_scene.pipeline.components.body_placement import BodyPlacementModule
from src.tennis_scene.pipeline.components.body_view_selection import (
    BodyViewSelectionModule,
)
from src.tennis_scene.pipeline.components.camera_alignment import CameraAlignmentModule
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationModule,
)
from src.tennis_scene.pipeline.components.court_kp import CourtKPModule
from src.tennis_scene.pipeline.components.court_side import CourtSideModule
from src.tennis_scene.pipeline.components.gvhmr import GVHMRModule
from src.tennis_scene.pipeline.components.identity import PlayerAssociationModule
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionModule
from src.tennis_scene.pipeline.components.person_tracking import (
    PersonTrackingModule,
)
from src.tennis_scene.pipeline.components.player_selection import PlayerSelectionModule
from src.tennis_scene.pipeline.components.pose_estimation import PoseEstimationModule
from src.tennis_scene.pipeline.components.scene_assembly import SceneAssemblyModule
from src.tennis_scene.pipeline.components.triangulation import (
    BallTriangulationModule,
    PlayerTriangulationModule,
)
from src.tennis_scene.pipeline.contracts import AssemblyContext, ClipSource
from src.tennis_scene.pipeline.input_assembly.ball_points import (
    BallPointsInputAssembler,
)
from src.tennis_scene.pipeline.input_assembly.body import (
    BodyPlacementInputAssembler,
    BodyViewSelectionInputAssembler,
    GVHMRInputAssembler,
)
from src.tennis_scene.pipeline.input_assembly.preprocessing import (
    CourtCalibrationInputAssembler,
    CourtDetectionInputAssembler,
    PersonDetectionInputAssembler,
    PersonTrackingInputAssembler,
    PlayerSelectionInputAssembler,
    PoseEstimationInputAssembler,
)
from src.tennis_scene.pipeline.input_assembly.reconstruction import (
    BallTriangulationInputAssembler,
    CameraAlignmentInputAssembler,
    CourtSideInputAssembler,
    PlayerAssociationInputAssembler,
    PlayerTriangulationInputAssembler,
)
from src.tennis_scene.pipeline.input_assembly.scene import SceneAssemblyInputAssembler
from src.tennis_scene.pipeline.runner import ComponentNode
from src.utils.checksum import dual_sha256

if TYPE_CHECKING:
    from pathlib import Path

    from src.tennis_scene.configuration import PipelineRuntimeConfig


@lru_cache(maxsize=128)
def _asset_digest(path: Path, stat_identity: tuple[int, ...]) -> str:
    digest: str = dual_sha256(path)
    return digest


def file_identity(path: Path) -> dict[str, Any]:
    """Content identity of a required asset; a missing asset stops the build."""
    if not path.is_file():
        raise FileNotFoundError(f"Required pipeline asset is missing: {path}")
    stat = path.stat()
    return {"path": str(path), "sha256": _asset_digest(path, (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))}


def asset_identities(enabled: bool, paths: Mapping[str, Path]) -> dict[str, Any]:
    """Identities of a feature's assets; a disabled feature neither needs nor records them."""
    return {name: file_identity(path) for name, path in paths.items()} if enabled else {"enabled": False}


def standard_definition(cfg: PipelineRuntimeConfig, source: ClipSource, *, code_identity: str,
                        overrides: dict[str, Any] | None = None) -> tuple[ComponentNode, ...]:
    """Overrides select implementations through the same declarations, including tests.

    ``settings`` of a standard component are built lazily so that asset
    identities (which require every enabled asset to exist) are computed only
    for components that are not overridden.
    """
    ids = source.camera_ids
    nodes: list[ComponentNode] = []
    overrides = {} if overrides is None else overrides

    def add(name: str, component: Any, assembler: Any, bindings: dict[str, str], settings: Callable[[], Any],
            *, camera: str | None = None) -> None:
        stage = name.split("/")[0]
        if name in overrides:
            component = overrides[name]
            resolved = {"override": f"{type(component).__module__}.{type(component).__qualname__}"}
        else:
            resolved = settings()
        mode = "load" if cfg.cache_source == "load" else cfg.component_sources[stage]
        nodes.append(ComponentNode(name, component, component.io, assembler, bindings,
            AssemblyContext(source, camera), json_value(resolved), code_identity, mode))

    people_enabled = cfg.enabled["person_observations"]
    selection = LinkingConfig(max_candidates=cfg.max_tracks_per_camera)
    association = cfg.player_association
    weights = cfg.association_encoder_weights
    encoder_name = None if association.appearance is None else association.appearance.encoder
    sampling = CropSamplingConfig()
    encoder = None if encoder_name is None else partial(build_encoder, encoder_name,
        checkpoint_root=cfg.roots.checkpoint_root, external_root=cfg.roots.external_asset_root, device=cfg.device)
    body_enabled = cfg.enabled["gvhmr"]
    tracking_encoder = partial(build_encoder, cfg.tracking.encoder, checkpoint_root=cfg.roots.checkpoint_root,
                               external_root=cfg.roots.external_asset_root, device=cfg.device)
    tracking_assets = ({'pose': cfg.people.vitpose_checkpoint, 'encoder': cfg.tracking_encoder_weights}
                       if cfg.tracking.method != 'all_person_botsort' else {})
    if cfg.tracking.offline:
        tracking_assets['aflink'] = cfg.aflink_checkpoint

    def body_assets() -> dict[str, Any]:
        return asset_identities(body_enabled, cfg.people.body_assets())

    for camera in ids:
        add(f"court_detection/{camera}", CourtKPModule(cfg.court_kp), CourtDetectionInputAssembler(), {},
            lambda: {"config": cfg.processing_settings["court_kp"], "checkpoint": file_identity(cfg.court_kp.checkpoint)}, camera=camera)
    if not cfg.enabled["ball_detection"] and cfg.cache_source != "load":
        if cfg.component_sources["court_side"] != "load" and "court_side" not in overrides:
            raise ValueError("court_side decides sides from the ball alone; it requires ball_detection.enabled or execution.court_side=load")
        if any(cfg.component_sources[key] != "load" for key in ("ball_detection", "ball_refiner_2d", "ball_points")):
            raise ValueError("Disabled ball detection requires explicit load for the entire refiner/point chain")
    ball_nodes = ball_refiner_definition(
        source, detector_config=cfg.ball_detection, bundle_directory=cfg.ball_refiner.bundle,
        batch_size=cfg.ball_refiner.batch_size, code_identity=code_identity, execution_source=cfg.cache_source,
        ball_path=cfg.ball_refiner.path, calibration_artifact=cfg.ball_refiner.calibration_artifact,
    )
    for node in ball_nodes:
        component = overrides.get(node.name, node.component)
        mode = "load" if cfg.cache_source == "load" else cfg.component_sources[node.io.name]
        nodes.append(replace(node, component=component, io=component.io, source=mode))
        if node.io.name == "ball_refiner_2d":
            add(f"ball_points/{node.context.camera_id}",
                BallPointsModule(cfg.ball_confidence, distribution_version=component.io.version),
                BallPointsInputAssembler(), {"distribution": node.name},
                lambda: {"rule": cfg.ball_confidence, "point": "maximum_weight_mean",
                         "region": "conditional_second_moment_ellipse_90"}, camera=node.context.camera_id)
    add("court_calibration", CourtCalibrationModule(ids, cfg.camera_geometry, roi_margins=cfg.person_roi_margins), CourtCalibrationInputAssembler(),
        {c: f"court_detection/{c}" for c in ids}, lambda: {"geometry": cfg.camera_geometry, "roi": cfg.person_roi_margins, "enabled": people_enabled})
    for camera in ids:
        add(f"person_detection/{camera}", PersonDetectionModule(cfg.people, enabled=people_enabled,
            merge_duplicates=cfg.merge_duplicate_person_boxes), PersonDetectionInputAssembler(),
            {}, lambda: {"detector": cfg.people.detector,
             "assets": asset_identities(people_enabled, {"checkpoint": cfg.people.detector_checkpoint}),
             "runtime": cfg.people.runtime.dino_detector, "yolo_confidence": cfg.people.runtime.tracking.yolo_confidence,
             "scope": "full_frame", "enabled": people_enabled, "merge_duplicates": cfg.merge_duplicate_person_boxes,
             "merge_rule": "greedy_iou_ge_0.8_score_desc_source_row_asc"}, camera=camera)
        add(f"person_tracking/{camera}", PersonTrackingModule(cfg.tracking, people=cfg.people, encoder=tracking_encoder,
            aflink_checkpoint=cfg.aflink_checkpoint, enabled=people_enabled), PersonTrackingInputAssembler(),
            {"detections": f"person_detection/{camera}"}, lambda: {"profile": cfg.tracking.identity(),
                "baseline": ALL_PERSON_BOTSORT_SETTINGS if cfg.tracking.method == 'all_person_botsort' else None,
                "assets": asset_identities(people_enabled, tracking_assets), "pose_runtime": cfg.people.runtime.vitpose,
                "pose_precision": "float32", "boxes": "raw_detection_rows", "enabled": people_enabled}, camera=camera)
        add(f"player_selection/{camera}", PlayerSelectionModule(selection, footpoints=association.footpoints,
            sampling=sampling, encoder=encoder, encoder_name=encoder_name, device=cfg.device, enabled=people_enabled),
            PlayerSelectionInputAssembler(), {"tracks": f"person_tracking/{camera}", "calibration": "court_calibration"},
            lambda: {"rule": selection, "footpoints": association.footpoints, "sampling": sampling, "encoder": encoder_name,
                "enabled": people_enabled, "assets": asset_identities(people_enabled and weights is not None,
                    {} if weights is None else {"encoder": weights})}, camera=camera)
        add(f"pose_estimation/{camera}", PoseEstimationModule(cfg.people,
            require_evidence=people_enabled and cfg.tracking.method != 'all_person_botsort'), PoseEstimationInputAssembler(),
            {"selection": f"player_selection/{camera}"}, lambda: {"assets": asset_identities(people_enabled, {"checkpoint": cfg.people.vitpose_checkpoint}),
             "runtime": cfg.people.runtime.vitpose, "bbox_enlarge": cfg.people.runtime.tracking.bbox_enlarge}, camera=camera)
    poses = {f"pose_{c}": f"pose_estimation/{c}" for c in ids}
    balls = {f"ball_{c}": f"ball_points/{c}" for c in ids}
    observations = {"calibration": "court_calibration", **poses}
    add("court_side", CourtSideModule(ids, cfg.court_side, max_frames=cfg.sampling_max_frames), CourtSideInputAssembler(0.0),
        {"calibration": "court_calibration", **balls}, lambda: {"config": cfg.court_side, "max_frames": cfg.sampling_max_frames,
        "ball_threshold": 0.0})
    add("player_association", PlayerAssociationModule(ids, association, sampling=sampling, device=cfg.device, enabled=people_enabled,
        encoder=encoder),
        PlayerAssociationInputAssembler(), {"calibration": "court_calibration", "side": "court_side", **{f"tracks_{c}": f"player_selection/{c}" for c in ids}},
        lambda: {"config": association, "sampling": sampling, "enabled": people_enabled,
                 "assets": asset_identities(people_enabled and weights is not None, {} if weights is None else {"encoder": weights})})
    identified = {**observations, "identities": "player_association"}
    add("camera_alignment", CameraAlignmentModule(ids, cfg.camera_geometry, player_reprojection_px=cfg.player_reprojection_px,
        ball_reprojection_px=cfg.ball_reprojection_px, joint_confidence=cfg.joint_confidence, max_frames=cfg.sampling_max_frames),
        CameraAlignmentInputAssembler(cfg.human_vis_threshold, 0.0), {**identified, **balls, "side": "court_side"},
        lambda: {"config": cfg.camera_geometry, "player_reprojection": cfg.player_reprojection_px, "ball_reprojection": cfg.ball_reprojection_px,
         "joint_confidence": cfg.joint_confidence, "visibility": cfg.human_vis_threshold, "max_frames": cfg.sampling_max_frames, "ball_threshold": 0.0})
    reconstructed = {**identified, "alignment": "camera_alignment"}
    add("player_triangulation", PlayerTriangulationModule(ids, reprojection_px=cfg.player_reprojection_px,
        joint_confidence=cfg.joint_confidence, enabled=cfg.enabled["player_reconstruction"]),
        PlayerTriangulationInputAssembler(cfg.human_vis_threshold), reconstructed, lambda: cfg.processing_settings["player_reconstruction"])
    add("ball_triangulation", BallTriangulationModule(ids, reprojection_px=cfg.ball_reprojection_px,
        min_frames=cfg.ball_min_frames, enabled=cfg.enabled["ball_reconstruction"]),
        BallTriangulationInputAssembler(0.0), {"alignment": "camera_alignment", "calibration": "court_calibration", **balls},
        lambda: {"config": cfg.processing_settings["ball_reconstruction"], "threshold": 0.0})
    add("body_view_selection", BodyViewSelectionModule(ids, max_frames=cfg.sampling_max_frames, enabled=body_enabled),
        BodyViewSelectionInputAssembler(cfg.human_vis_threshold), reconstructed,
        lambda: {"policy": "coverage_confidence_camera_id", "visibility": cfg.human_vis_threshold, "max_frames": cfg.sampling_max_frames, "enabled": body_enabled})
    add("gvhmr", GVHMRModule(cfg.people, enabled=body_enabled), GVHMRInputAssembler(), {"selection": "body_view_selection"},
        lambda: {"assets": body_assets(), "runtime": cfg.people.runtime.hmr2, "enabled": body_enabled})
    add("body_placement", BodyPlacementModule(ids, cfg.people, cfg.player_placement,
        reprojection_px=cfg.player_reprojection_px, enabled=body_enabled), BodyPlacementInputAssembler(cfg.human_vis_threshold),
        {**reconstructed, "skeleton": "player_triangulation", "recovered": "gvhmr"},
        lambda: {"config": cfg.player_placement, "assets": body_assets()})
    add("scene_assembly", SceneAssemblyModule(ids, require_people=cfg.enabled["player_reconstruction"],
        require_ball=cfg.enabled["ball_reconstruction"], require_body=body_enabled), SceneAssemblyInputAssembler(cfg.human_vis_threshold),
        {**reconstructed, "skeleton": "player_triangulation", "ball": "ball_triangulation", "placement": "body_placement"}, lambda: {"enabled": cfg.enabled})
    return tuple(nodes)


def enabled_model_assets(cfg: PipelineRuntimeConfig) -> dict[str, Path]:
    """Every model asset the standard recipe reads for the enabled features."""
    people, body = cfg.enabled["person_observations"], cfg.enabled["gvhmr"]
    return {
        "court": cfg.court_kp.checkpoint,
        "ball": cfg.ball_detection.checkpoint,
        "ball_refiner_manifest": cfg.ball_refiner.bundle / "manifest.json",
        "ball_refiner_weights": cfg.ball_refiner.bundle / "weights.pt",
        **({"ball_refiner_calibration": cfg.ball_refiner.calibration_artifact}
           if cfg.ball_refiner.calibration_artifact is not None else {}),
        **({"detector": cfg.people.detector_checkpoint, "vitpose": cfg.people.vitpose_checkpoint} if people else {}),
        **({"association_encoder": cfg.association_encoder_weights} if people and cfg.association_encoder_weights is not None else {}),
        **({"tracking_encoder": cfg.tracking_encoder_weights} if people and cfg.tracking.method != 'all_person_botsort' else {}),
        **({"aflink": cfg.aflink_checkpoint} if people and cfg.tracking.offline else {}),
        **(cfg.people.body_assets() if body else {}),
    }
