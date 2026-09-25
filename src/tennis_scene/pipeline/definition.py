"""The standard recipe: explicit bindings, replaceable components and input builders."""

from __future__ import annotations

from dataclasses import fields
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from src.tennis_scene.pipeline.artifacts import json_value
from src.tennis_scene.pipeline.components.ball_detection import BallDetectionModule
from src.tennis_scene.pipeline.components.body_placement import BodyPlacementModule
from src.tennis_scene.pipeline.components.body_view_selection import (
    BodyViewSelectionModule,
)
from src.tennis_scene.pipeline.components.camera_alignment import CameraAlignmentModule
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationModule,
)
from src.tennis_scene.pipeline.components.court_kp import CourtKPModule
from src.tennis_scene.pipeline.components.gvhmr import GVHMRModule
from src.tennis_scene.pipeline.components.person_association import (
    CourtSideModule,
    PlayerReIDModule,
)
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionModule
from src.tennis_scene.pipeline.components.person_tracking import PersonTrackingModule
from src.tennis_scene.pipeline.components.pose_estimation import PoseEstimationModule
from src.tennis_scene.pipeline.components.scene_assembly import SceneAssemblyModule
from src.tennis_scene.pipeline.components.tracking_identity import (
    MAX_APPEARANCE_LAB_DISTANCE,
    MAX_CENTER_DISTANCE_DIAGONALS,
    MAX_DUPLICATE_SIZE_RATIO,
    MAX_GAP_FRAMES,
    MAX_OVERLAP_SPAN_FRAMES,
    MAX_SIZE_RATIO,
    MIN_DUPLICATE_CONTAINMENT,
)
from src.tennis_scene.pipeline.components.triangulation import (
    BallTriangulationModule,
    PlayerTriangulationModule,
)
from src.tennis_scene.pipeline.contracts import AssemblyContext, ClipSource
from src.tennis_scene.pipeline.input_assembly.body import (
    BodyPlacementInputAssembler,
    BodyViewSelectionInputAssembler,
    GVHMRInputAssembler,
)
from src.tennis_scene.pipeline.input_assembly.observations import (
    PersonAssociationInputAssembler,
)
from src.tennis_scene.pipeline.input_assembly.preprocessing import (
    BallDetectionInputAssembler,
    CourtCalibrationInputAssembler,
    CourtDetectionInputAssembler,
    PersonDetectionInputAssembler,
    PersonTrackingInputAssembler,
    PoseEstimationInputAssembler,
)
from src.tennis_scene.pipeline.input_assembly.reconstruction import (
    BallTriangulationInputAssembler,
    CameraAlignmentInputAssembler,
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
    digest = None
    if path.is_file():
        stat = path.stat()
        digest = _asset_digest(path, (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
    return {"path": str(path), "sha256": digest}


def standard_definition(cfg: PipelineRuntimeConfig, source: ClipSource, *, code_identity: str,
                        overrides: dict[str, Any] | None = None) -> tuple[ComponentNode, ...]:
    """Overrides select implementations through the same declarations, including tests."""
    ids = source.camera_ids
    nodes: list[ComponentNode] = []
    overrides = {} if overrides is None else overrides

    def add(name: str, component: Any, assembler: Any, bindings: dict[str, str], settings: Any,
            *, camera: str | None = None) -> None:
        component = overrides.get(name, component)
        stage = name.split("/")[0]
        mode = "load" if cfg.cache_source == "load" else cfg.component_sources[stage]
        nodes.append(ComponentNode(name, component, component.io, assembler, bindings,
            AssemblyContext(source, camera), json_value(settings), code_identity, mode))

    for camera in ids:
        add(f"court_detection/{camera}", CourtKPModule(cfg.court_kp), CourtDetectionInputAssembler(), {},
            {"config": cfg.processing_settings["court_kp"], "checkpoint": file_identity(cfg.court_kp.checkpoint)}, camera=camera)
        add(f"ball_detection/{camera}", BallDetectionModule(cfg.ball_detection, enabled=cfg.enabled["ball_detection"]), BallDetectionInputAssembler(), {},
            {"config": cfg.processing_settings["ball_detection"], "checkpoint": file_identity(cfg.ball_detection.checkpoint)}, camera=camera)
    add("court_calibration", CourtCalibrationModule(ids, cfg.camera_geometry, roi_margins=cfg.person_roi_margins), CourtCalibrationInputAssembler(),
        {c: f"court_detection/{c}" for c in ids}, {"geometry": cfg.camera_geometry, "roi": cfg.person_roi_margins, "enabled": cfg.enabled["person_observations"]})
    for camera in ids:
        add(f"person_detection/{camera}", PersonDetectionModule(cfg.people, enabled=cfg.enabled["person_observations"]), PersonDetectionInputAssembler(),
            {"calibration": "court_calibration"}, {"detector": cfg.people.detector,
             "checkpoint": file_identity(cfg.people.dino_checkpoint if cfg.people.detector == "dino" else cfg.people.yolo_checkpoint),
             "runtime": cfg.people.runtime.dino_detector, "yolo_confidence": cfg.people.runtime.tracking.yolo_confidence, "roi": cfg.person_roi_margins, "enabled": cfg.enabled["person_observations"]}, camera=camera)
        add(f"person_tracking/{camera}", PersonTrackingModule(), PersonTrackingInputAssembler(),
            {"detections": f"person_detection/{camera}"}, {"algorithm": "botsort_then_unique_tracklet_links",
                "max_link_gap_frames": MAX_GAP_FRAMES, "max_center_distance_box_diagonals": MAX_CENTER_DISTANCE_DIAGONALS,
                "max_size_ratio": MAX_SIZE_RATIO, "max_appearance_lab_distance": MAX_APPEARANCE_LAB_DISTANCE,
                "max_duplicate_overlap_span_frames": MAX_OVERLAP_SPAN_FRAMES,
                "min_duplicate_containment": MIN_DUPLICATE_CONTAINMENT,
                "max_duplicate_size_ratio": MAX_DUPLICATE_SIZE_RATIO,
                "cumulative_capacity": 4}, camera=camera)
        add(f"pose_estimation/{camera}", PoseEstimationModule(cfg.people), PoseEstimationInputAssembler(),
            {"tracks": f"person_tracking/{camera}"}, {"checkpoint": file_identity(cfg.people.vitpose_checkpoint),
             "runtime": cfg.people.runtime.vitpose, "bbox_enlarge": cfg.people.runtime.tracking.bbox_enlarge}, camera=camera)
    poses = {f"pose_{c}": f"pose_estimation/{c}" for c in ids}
    balls = {f"ball_{c}": f"ball_detection/{c}" for c in ids}
    observations = {"calibration": "court_calibration", **poses}
    person_settings = {"policy": cfg.inference_policy, "visibility": cfg.human_vis_threshold}
    add("person_reid", PlayerReIDModule(cfg.plcs_reid_checkpoint, device=cfg.device, camera_ids=ids,
        policy=cfg.inference_policy, visibility=cfg.human_vis_threshold, enabled=cfg.enabled["plcs_reid"]),
        PersonAssociationInputAssembler(cfg.ball_detection.score_threshold), observations,
        {**person_settings, "checkpoint": file_identity(cfg.plcs_reid_checkpoint), "enabled": cfg.enabled["plcs_reid"]})
    add("court_side", CourtSideModule(cfg.court_side_checkpoint, device=cfg.device, camera_ids=ids,
        policy=cfg.inference_policy, visibility=cfg.human_vis_threshold),
        PersonAssociationInputAssembler(cfg.ball_detection.score_threshold, True), {**observations, **balls},
        {**person_settings, "checkpoint": file_identity(cfg.court_side_checkpoint)})
    identified = {**observations, "identities": "person_reid"}
    add("camera_alignment", CameraAlignmentModule(ids, cfg.camera_geometry, player_reprojection_px=cfg.player_reprojection_px,
        ball_reprojection_px=cfg.ball_reprojection_px, joint_confidence=cfg.joint_confidence, max_frames=cfg.inference_policy.max_frames),
        CameraAlignmentInputAssembler(cfg.human_vis_threshold, cfg.ball_detection.score_threshold), {**identified, **balls, "side": "court_side"},
        {"config": cfg.camera_geometry, "player_reprojection": cfg.player_reprojection_px, "ball_reprojection": cfg.ball_reprojection_px,
         "joint_confidence": cfg.joint_confidence, **person_settings, "ball_threshold": cfg.ball_detection.score_threshold})
    reconstructed = {**identified, "alignment": "camera_alignment"}
    add("player_triangulation", PlayerTriangulationModule(ids, reprojection_px=cfg.player_reprojection_px,
        joint_confidence=cfg.joint_confidence, enabled=cfg.enabled["player_reconstruction"]),
        PlayerTriangulationInputAssembler(cfg.human_vis_threshold), reconstructed, cfg.processing_settings["player_reconstruction"])
    add("ball_triangulation", BallTriangulationModule(ids, reprojection_px=cfg.ball_reprojection_px,
        min_frames=cfg.ball_min_frames, enabled=cfg.enabled["ball_reconstruction"]),
        BallTriangulationInputAssembler(cfg.ball_detection.score_threshold), {"alignment": "camera_alignment", "calibration": "court_calibration", **balls},
        {"config": cfg.processing_settings["ball_reconstruction"], "threshold": cfg.ball_detection.score_threshold})
    add("body_view_selection", BodyViewSelectionModule(ids, max_frames=cfg.inference_policy.max_frames, enabled=cfg.enabled["gvhmr"]),
        BodyViewSelectionInputAssembler(cfg.human_vis_threshold), reconstructed,
        {"policy": "coverage_confidence_camera_id", **person_settings, "enabled": cfg.enabled["gvhmr"]})
    body_settings = {"hmr2": file_identity(cfg.people.hmr2_checkpoint), "gvhmr": file_identity(cfg.people.gvhmr_checkpoint),
        "body_model": file_identity(cfg.people.body_models_dir / "smplx" / "SMPLX_NEUTRAL.npz"),
        "bundled_assets": {field.name: file_identity(getattr(cfg.people.bundled_assets, field.name)) for field in fields(cfg.people.bundled_assets)},
        "runtime": cfg.people.runtime.hmr2,
        "enabled": cfg.enabled["gvhmr"]}
    add("gvhmr", GVHMRModule(cfg.people, enabled=cfg.enabled["gvhmr"]), GVHMRInputAssembler(), {"selection": "body_view_selection"}, body_settings)
    add("body_placement", BodyPlacementModule(ids, cfg.people, cfg.player_placement,
        reprojection_px=cfg.player_reprojection_px, enabled=cfg.enabled["gvhmr"]), BodyPlacementInputAssembler(cfg.human_vis_threshold),
        {**reconstructed, "skeleton": "player_triangulation", "recovered": "gvhmr"},
        {"config": cfg.player_placement, "models": body_settings, "root": file_identity(cfg.people.bundled_assets.smpl_neutral_joint_regressor)})
    add("scene_assembly", SceneAssemblyModule(ids, require_people=cfg.enabled["player_reconstruction"],
        require_ball=cfg.enabled["ball_reconstruction"], require_body=cfg.enabled["gvhmr"]), SceneAssemblyInputAssembler(cfg.human_vis_threshold),
        {**reconstructed, "skeleton": "player_triangulation", "ball": "ball_triangulation", "placement": "body_placement"}, {"enabled": cfg.enabled})
    return tuple(nodes)
