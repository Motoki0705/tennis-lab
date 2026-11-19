"""The standard recipe: explicit bindings, replaceable components and input builders."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from typing import TYPE_CHECKING, Any

from src.submodules.smpl_assets import require_smpl_asset
from src.tasks.person_tracking.court_linking import LinkingConfig
from src.tasks.player_association.appearance.encoders import build_encoder
from src.tasks.player_association.appearance.sampling import CropSamplingConfig
from src.tennis_scene.pipeline.artifacts import json_value
from src.tennis_scene.pipeline.components.ball_annotations import (
    ImportedBallAnnotationModule,
)
from src.tennis_scene.pipeline.components.ball_detection import BallDetectionModule
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
from src.tennis_scene.pipeline.components.hmr2_features import HMR2FeaturesModule
from src.tennis_scene.pipeline.components.identity import (
    PLAYER_ASSOCIATION,
    PlayerAssociationModule,
)
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionModule
from src.tennis_scene.pipeline.components.person_features import (
    PersonAppearanceFeaturesModule,
    PersonPoseFeaturesModule,
)
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
from src.tennis_scene.pipeline.contracts import (
    STANDARD_COMPONENTS,
    AssemblyContext,
    ClipSource,
)
from src.tennis_scene.pipeline.inference_runtime import InferenceComponent
from src.tennis_scene.pipeline.input_assembly.ball_points import (
    BallPointsInputAssembler,
)
from src.tennis_scene.pipeline.input_assembly.body import (
    BodyPlacementInputAssembler,
    BodyViewSelectionInputAssembler,
    GVHMRInputAssembler,
    HMR2FeaturesInputAssembler,
)
from src.tennis_scene.pipeline.input_assembly.preprocessing import (
    BallDetectionInputAssembler,
    CourtCalibrationInputAssembler,
    CourtDetectionInputAssembler,
    PersonAppearanceFeaturesInputAssembler,
    PersonDetectionInputAssembler,
    PersonPoseFeaturesInputAssembler,
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
from src.tennis_scene.pipeline.model_assets import asset_identities as asset_identities
from src.tennis_scene.pipeline.model_assets import file_identity as file_identity
from src.tennis_scene.pipeline.model_resources import appearance_model, model_identities
from src.tennis_scene.pipeline.runner import ComponentNode
from src.tennis_scene.pipeline.video_court.component import (
    SharedCourtAssembler,
    SharedCourtModule,
)
from src.tennis_scene.pipeline.video_court.diagnostics import drift_identity
from src.tennis_scene.pipeline.video_court.store import VideoCourtArtifact
from src.tennis_scene.pipeline.video_court_side.artifact import VideoCourtSideArtifact
from src.tennis_scene.pipeline.video_court_side.component import SharedCourtSideModule
from src.tennis_scene.pipeline.video_court_side.prepare import (
    verify_prepared_court_side,
)

if TYPE_CHECKING:
    from pathlib import Path

    from src.submodules.configuration import (
        DinoDetectorRuntimeConfig,
        ViTPoseRuntimeConfig,
    )
    from src.tennis_scene.configuration import PipelineRuntimeConfig


def standard_model_components(cfg: PipelineRuntimeConfig, ids: tuple[str, ...]) -> dict[str, Any]:
    """One set of model constructors for node identities and publication markers.

    Construction is CPU-only. Only the selected, non-overridden node's settings
    bind required assets; no model is loaded while planning or publishing.
    """
    people, body = cfg.enabled["person_observations"], cfg.enabled["gvhmr"]
    association = cfg.player_association
    name = None if association.appearance is None else association.appearance.encoder

    def appearance(profile: str, checkpoint: Path) -> Any:
        return appearance_model(profile, checkpoint, cfg.device,
            partial(build_encoder, profile, checkpoint=checkpoint, checkpoint_root=cfg.roots.checkpoint_root,
                external_root=cfg.roots.external_asset_root, device=cfg.device),
            repository=cfg.roots.external_asset_root / "dinov3" if profile.startswith("dinov3_") else None)

    encoder = None if not people or name is None or cfg.association_encoder_weights is None else appearance(name, cfg.association_encoder_weights)
    tracking = None if not people else appearance(cfg.tracking.encoder, cfg.tracking_encoder_weights)
    pose = PersonPoseFeaturesModule(cfg.people, cfg.tracking.features, enabled=people)
    selection, sampling = LinkingConfig(max_candidates=cfg.max_tracks_per_camera), CropSamplingConfig()
    return {
        "court_detection": CourtKPModule(cfg.court_kp),
        "ball_detection": ImportedBallAnnotationModule() if cfg.ball_detection is None else BallDetectionModule(cfg.ball_detection),
        "person_detection": PersonDetectionModule(cfg.people, enabled=people, merge_duplicates=cfg.merge_duplicate_person_boxes),
        "person_pose_features": pose,
        "person_appearance_features": PersonAppearanceFeaturesModule(tracking, cfg.tracking.features, enabled=people),
        "person_tracking": PersonTrackingModule(cfg.tracking, aflink_checkpoint=cfg.aflink_checkpoint, enabled=people,
            expected_pose=pose.model_specs().get('pose'), expected_encoder=tracking),
        "player_selection": PlayerSelectionModule(selection, footpoints=association.footpoints, sampling=sampling,
            encoder=encoder, encoder_name=name if encoder is not None else None, device=cfg.device, enabled=people),
        "pose_estimation": PoseEstimationModule(cfg.people, require_evidence=people, enabled=people),
        PLAYER_ASSOCIATION: PlayerAssociationModule(ids, association, sampling=sampling, device=cfg.device, enabled=people, encoder=encoder,
            reuse_clip_embeddings=cfg.reuse_clip_embeddings, feature_config=cfg.tracking.features),
        "hmr2_features": HMR2FeaturesModule(cfg.people, enabled=body),
        "gvhmr": GVHMRModule(cfg.people, enabled=body),
        "body_placement": BodyPlacementModule(ids, cfg.people, cfg.player_placement, reprojection_px=cfg.player_reprojection_px, enabled=body),
    }


def declared_model_identities(cfg: PipelineRuntimeConfig, stage: str, component: Any) -> dict[str, Any]:
    # Shared/imported Court and disabled pose consume persisted observations.
    # Standalone PoseEstimationModule keeps its optional inference API, but the
    # disabled standard recipe has no person tracks and no model demand.
    if (stage == "court_detection" and (cfg.direct_annotation is not None or cfg.video_court is not None)
            or stage == "pose_estimation" and not cfg.enabled["person_observations"]
            or not isinstance(component, InferenceComponent)):
        return {}
    return dict(model_identities(component.model_specs(), cfg.court_kp.asset_catalog.file_identity))


def publication_model_identities(cfg: PipelineRuntimeConfig) -> dict[str, Any]:
    return {stage: declared_model_identities(cfg, stage, component)
        for stage, component in standard_model_components(cfg, ()).items()}


def component_override_identity(cfg: PipelineRuntimeConfig, component: Any) -> dict[str, Any]:
    """An injected implementation must declare its models or explicit settings.

    ModelSpec covers effective model settings; additional processing knobs can
    be bound with cache_identity(). Stateless non-inference implementations use
    that method too. A class name alone never authorizes artifact reuse.
    """
    specs = component.model_specs() if isinstance(component, InferenceComponent) else {}
    identity = getattr(component, 'cache_identity', None)
    settings = identity() if callable(identity) else None
    if (not specs and settings is None) or (settings is not None and not isinstance(settings, Mapping)):
        raise ValueError('Injected component needs ModelSpec declarations or an explicit mapping from cache_identity()')
    return {'override': f'{type(component).__module__}.{type(component).__qualname__}',
        'models': model_identities(specs, cfg.court_kp.asset_catalog.file_identity),
        'settings': {} if settings is None else json_value(settings)}


def validate_override_names(overrides: Mapping[str, Any]) -> None:
    unknown = {name for name in overrides if name.split('/')[0] not in STANDARD_COMPONENTS}
    if unknown:
        raise ValueError(f'Unknown component override: {sorted(unknown)}')


def _person_runtime_identity(runtime: DinoDetectorRuntimeConfig | ViTPoseRuntimeConfig,
                             device: str, *, key: str = "runtime") -> dict[str, Any]:
    options = json_value(runtime)
    if runtime.precision == "fp32":
        del options["precision"]
        return {key: options}
    return {key: options, "device": device}


def check_shared_side_execution(cfg: PipelineRuntimeConfig, overrides: Mapping[str, Any] | None) -> None:
    if cfg.video_court_side is None:
        return
    if cfg.direct_annotation is not None:
        raise ValueError("court_side.mode=video_shared is incompatible with direct_annotation and publication-only resume")
    if overrides is not None and "court_side" in overrides:
        raise ValueError("A court_side override is incompatible with court_side.mode=video_shared")


def standard_definition(cfg: PipelineRuntimeConfig, source: ClipSource, *, code_identity: str,
                        overrides: dict[str, Any] | None = None, video_court: VideoCourtArtifact | None = None,
                        video_court_side: VideoCourtSideArtifact | None = None) -> tuple[ComponentNode, ...]:
    """Overrides select implementations through the same declarations, including tests.

    ``settings`` of a standard component are built lazily so that asset
    identities (which require every enabled asset to exist) are computed only
    for components that are not overridden.
    """
    ids = source.camera_ids
    overrides = {} if overrides is None else overrides
    validate_override_names(overrides)
    check_shared_side_execution(cfg, overrides)
    if (cfg.video_court is None) != (video_court is None):
        raise ValueError('Shared court must be prepared explicitly before building the clip plan')
    if video_court is not None:
        video_court.bind(source)
        requested_drift = None if cfg.video_court_drift is None else cfg.video_court_drift.digest
        if drift_identity(video_court).get('artifact_digest') != requested_drift:
            raise ValueError('Execution plan drift artifact disagrees with explicit configuration')
    if (cfg.video_court_side is None) != (video_court_side is None):
        raise ValueError("Shared court side must be prepared before building the clip plan")
    if video_court_side is not None:
        if cfg.video_court_side is None or cfg.video_court is None or video_court is None:
            raise ValueError("Shared court side requires shared calibration")
        if ((cfg.video_court_side.dataset_id, cfg.video_court_side.video_id)
                != (video_court.manifest["dataset_id"], video_court.manifest["video_id"])
                or cfg.video_court.action == "load" and cfg.video_court.artifact_digest != video_court.digest):
            raise ValueError("Execution plan court disagrees with the configured shared side scope/geometry")
        verify_prepared_court_side(video_court_side, cfg.video_court_side, cfg.resolver, code_identity=code_identity)
        video_court_side.check_settings(video_court, cfg.court_side, cfg.sampling_max_frames, court_policy=cfg.video_court_unobservable_policy)
        if source.clip_id not in video_court_side.clips:
            raise ValueError("Clip is missing from the shared court side artifact")
    nodes: list[ComponentNode] = []
    components = standard_model_components(cfg, ids)

    def bound_assets(enabled: bool, paths: dict[str, Any]) -> dict[str, Any]:
        return {name: cfg.court_kp.asset_catalog.file_identity(path) for name, path in paths.items()} if enabled else {"enabled": False}

    def add(name: str, component: Any, assembler: Any, bindings: dict[str, str], settings: Callable[[], Any],
            *, camera: str | None = None, require_component: Callable[[], None] | None = None,
            after_components: tuple[str, ...] = ()) -> None:
        stage = name.split("/")[0]
        resolved: dict[str, Any]
        if name in overrides:
            component = overrides[name]
            resolved = component_override_identity(cfg, component)
        elif cfg.direct_annotation is not None and stage in {"court_detection", "ball_detection"}:
            resolved = {"input": "fixed_geometry" if stage == "court_detection" else "saved_video_ball_annotation_v3"}
        else:
            resolved = settings()
            if isinstance(component, InferenceComponent):
                resolved = {**resolved, "models": declared_model_identities(cfg, stage, component)}
        mode = "load" if cfg.cache_source == "load" else cfg.component_sources[stage]
        nodes.append(ComponentNode(name, component, component.io, assembler, bindings,
            AssemblyContext(source, camera), json_value(resolved), code_identity, mode, require_component=require_component,
            after_components=after_components))

    people_enabled = cfg.enabled["person_observations"]
    selection = LinkingConfig(max_candidates=cfg.max_tracks_per_camera)
    association = cfg.player_association
    weights = cfg.association_encoder_weights
    encoder_name = None if association.appearance is None else association.appearance.encoder
    sampling = CropSamplingConfig()
    encoder = components["player_association"].encoder
    body_enabled = cfg.enabled["gvhmr"]
    tracking_encoder = components["person_appearance_features"].encoder
    encoder_assets = {} if encoder is None else {"encoder" if name == "checkpoint" else name: path for name, path in encoder.weights.items()}

    runtime = cfg.people.runtime
    detector_runtime = _person_runtime_identity(runtime.dino_detector, runtime.device)
    pose_runtime = _person_runtime_identity(runtime.vitpose, runtime.device)

    def body_assets() -> dict[str, Any]:
        if body_enabled:
            cfg.people.smpl_assets.require_files()
        return bound_assets(body_enabled, cfg.people.body_assets())

    def hmr2_assets() -> dict[str, Any]:
        if body_enabled:
            require_smpl_asset(cfg.people.smpl_assets.hmr2_mean_params, "hmr2_mean_params")
        return bound_assets(body_enabled, {
            "checkpoint": cfg.people.hmr2_checkpoint, "mean_params": cfg.people.smpl_assets.hmr2_mean_params})

    def shared_settings(camera_id: str) -> dict[str, Any]:
        assert video_court is not None
        return {'mode': 'video_shared', 'camera_artifact_digest': video_court.camera_digest(camera_id),
                'unobservable_policy': cfg.video_court_unobservable_policy,
                'drift_artifact_digest': drift_identity(video_court).get('camera_digests', {}).get(camera_id)}

    def court_settings() -> dict[str, Any]:
        court = CourtKPModule(cfg.court_kp)
        assets = court.asset_identities()
        checkpoint = assets.pop("checkpoint")
        return {"config": cfg.processing_settings["court_kp"], "model": court.model_identity(), "checkpoint": checkpoint, "external_assets": assets}

    def tracking_settings(module: PersonTrackingModule) -> dict[str, Any]:
        specs = {}
        if people_enabled:
            assert module.expected_pose is not None and module.expected_encoder is not None
            specs = {'pose': module.expected_pose, 'encoder': module.expected_encoder}
        return {'profile': cfg.tracking.identity(), 'expected_features': {
            'models': model_identities(specs, cfg.court_kp.asset_catalog.file_identity),
            'pose_preprocessing': module.pose_preprocessing, 'appearance_preprocessing': module.appearance_preprocessing},
            'assets': bound_assets(people_enabled, {'aflink': cfg.aflink_checkpoint}),
            'boxes': 'raw_detection_rows', 'enabled': people_enabled}

    for camera in ids:
        if video_court is None:
            add(f"court_detection/{camera}", components["court_detection"], CourtDetectionInputAssembler(), {},
                court_settings, camera=camera)
        else:
            add(f"court_detection/{camera}", SharedCourtModule(cfg.video_court_unobservable_policy), SharedCourtAssembler(video_court), {},
                partial(shared_settings, camera), camera=camera)
    if not cfg.enabled["ball_detection"] and cfg.cache_source != "load":
        if cfg.component_sources["court_side"] != "load" and "court_side" not in overrides:
            raise ValueError("court_side decides sides from the ball alone; it requires ball_detection.enabled or execution.court_side=load")
        if any(cfg.component_sources[key] != "load" for key in ("ball_detection", "ball_points")):
            raise ValueError("Disabled ball detection requires explicit load for the detector/point chain")
    ball_config = cfg.ball_detection
    for camera in ids:
        if ball_config is None and cfg.ball_method != "gpt_annotation":
            raise ValueError("Model ball detection requires its model configuration")
        detector = components["ball_detection"]
        add(f"ball_detection/{camera}", detector,
            BallDetectionInputAssembler(), {},
            lambda: {"config": cfg.processing_settings["ball_detection"],
                     "checkpoint": cfg.court_kp.asset_catalog.file_identity(ball_config.checkpoint) if ball_config is not None else None}, camera=camera)
        add(f"ball_points/{camera}", BallPointsModule(cfg.ball_smoothing), BallPointsInputAssembler(),
            {"detections": f"ball_detection/{camera}"},
            lambda: {"point": "motion_consensus_loess_observed_only", "smoothing": cfg.ball_smoothing}, camera=camera)
    add("court_calibration", CourtCalibrationModule(ids, cfg.camera_geometry, roi_margins=cfg.person_roi_margins,
        court_schema_version=3 if video_court is not None else 2), CourtCalibrationInputAssembler(),
        {c: f"court_detection/{c}" for c in ids}, lambda: {"geometry": cfg.camera_geometry, "roi": cfg.person_roi_margins, "enabled": people_enabled})
    for camera in ids:
        add(f"person_detection/{camera}", components["person_detection"], PersonDetectionInputAssembler(),
            {}, lambda: {"detector": "dino",
             "assets": bound_assets(people_enabled, {"checkpoint": cfg.people.dino_checkpoint}),
             **detector_runtime,
             "scope": "full_frame", "enabled": people_enabled, "merge_duplicates": cfg.merge_duplicate_person_boxes,
             "merge_rule": "greedy_iou_ge_0.8_score_desc_source_row_asc"}, camera=camera)
        add(f"person_pose_features/{camera}", components["person_pose_features"],
            PersonPoseFeaturesInputAssembler(), {"detections": f"person_detection/{camera}"},
            lambda: {**components['person_pose_features'].cache_identity(), **pose_runtime,
                "assets": bound_assets(people_enabled, {"pose": cfg.people.vitpose_checkpoint})}, camera=camera,
            after_components=('person_detection',))
        add(f"person_appearance_features/{camera}", components["person_appearance_features"],
            PersonAppearanceFeaturesInputAssembler(), {"detections": f"person_detection/{camera}"},
            lambda: {**components['person_appearance_features'].cache_identity(), "encoder": cfg.tracking.encoder,
                "assets": bound_assets(people_enabled, {} if tracking_encoder is None else dict(tracking_encoder.weights)),
                "enabled": people_enabled}, camera=camera, after_components=('person_detection', 'person_pose_features'))
        pose_component = overrides.get(f'person_pose_features/{camera}', components['person_pose_features'])
        appearance_component = overrides.get(f'person_appearance_features/{camera}', components['person_appearance_features'])
        tracker = components['person_tracking']
        if isinstance(pose_component, PersonPoseFeaturesModule) and isinstance(appearance_component, PersonAppearanceFeaturesModule):
            tracker = PersonTrackingModule(cfg.tracking, aflink_checkpoint=cfg.aflink_checkpoint, enabled=people_enabled,
                expected_pose=pose_component.model_specs().get('pose'), expected_encoder=appearance_component.model_specs().get('encoder'),
                pose_preprocessing=pose_component.cache_identity(), appearance_preprocessing=appearance_component.cache_identity())
        add(f"person_tracking/{camera}", tracker, PersonTrackingInputAssembler(),
            {"detections": f"person_detection/{camera}", "poses": f"person_pose_features/{camera}",
             "appearance": f"person_appearance_features/{camera}"}, partial(tracking_settings, tracker), camera=camera)
        add(f"player_selection/{camera}", components["player_selection"],
            PlayerSelectionInputAssembler(), {"tracks": f"person_tracking/{camera}", "calibration": "court_calibration"},
            lambda: {"rule": selection, "footpoints": association.footpoints, "sampling": sampling, "encoder": encoder_name,
                "enabled": people_enabled, "assets": bound_assets(people_enabled and weights is not None,
                    encoder_assets)}, camera=camera)
        add(f"pose_estimation/{camera}", components["pose_estimation"], PoseEstimationInputAssembler(),
            {"selection": f"player_selection/{camera}"}, lambda: components['pose_estimation'].cache_identity(), camera=camera)
    poses = {f"pose_{c}": f"pose_estimation/{c}" for c in ids}
    balls = {f"ball_{c}": f"ball_points/{c}" for c in ids}
    observations = {"calibration": "court_calibration", **poses}
    side_module: CourtSideModule | SharedCourtSideModule
    require_side = None
    if video_court_side is None:
        side_module = CourtSideModule(ids, cfg.court_side, max_frames=cfg.sampling_max_frames)
        side_settings: dict[str, Any] = {"config": cfg.court_side, "max_frames": cfg.sampling_max_frames, "ball_threshold": 0.0}
    else:
        assert video_court is not None
        side_module = SharedCourtSideModule(ids, video_court_side, video_court, cfg.court_side_unobservable_policy,
                                           court_policy=cfg.video_court_unobservable_policy)
        require_side = partial(side_module.require_clip, source.clip_id)
        side_settings = video_court_side.node_settings(source.clip_id, cfg.court_side_unobservable_policy, cfg.video_court_unobservable_policy)
    add("court_side", side_module, CourtSideInputAssembler(0.0), {"calibration": "court_calibration", **balls},
        lambda: side_settings, require_component=require_side)
    association_component = overrides.get(PLAYER_ASSOCIATION, components[PLAYER_ASSOCIATION])
    reuse_clip_embeddings = isinstance(association_component, PlayerAssociationModule) and association_component.reuse_clip_embeddings
    association_bindings = {"calibration": "court_calibration", "side": "court_side", **{f"tracks_{c}": f"player_selection/{c}" for c in ids}}
    if reuse_clip_embeddings:
        association_bindings.update({f"detections_{c}": f"person_detection/{c}" for c in ids})
        association_bindings.update({f"appearance_{c}": f"person_appearance_features/{c}" for c in ids})
    add(PLAYER_ASSOCIATION, components[PLAYER_ASSOCIATION], PlayerAssociationInputAssembler(reuse_clip_embeddings),
        association_bindings, lambda: {**components[PLAYER_ASSOCIATION].cache_identity(),
                                       "assets": bound_assets(people_enabled and weights is not None, encoder_assets)})
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
    add("hmr2_features", components["hmr2_features"], HMR2FeaturesInputAssembler(),
        {"selection": "body_view_selection"}, lambda: {
            "assets": hmr2_assets(),
            "runtime": cfg.people.runtime.hmr2, "enabled": body_enabled})
    add("gvhmr", components["gvhmr"], GVHMRInputAssembler(),
        {"selection": "body_view_selection", "features": "hmr2_features"},
        lambda: {"assets": body_assets(), "static_cam": cfg.people.runtime.static_cam, "enabled": body_enabled})
    add("body_placement", components["body_placement"], BodyPlacementInputAssembler(cfg.human_vis_threshold),
        {**reconstructed, "skeleton": "player_triangulation", "recovered": "gvhmr"},
        lambda: {"config": cfg.player_placement, "assets": body_assets()})
    add("scene_assembly", SceneAssemblyModule(ids, require_people=cfg.enabled["player_reconstruction"],
        require_ball=cfg.enabled["ball_reconstruction"], require_body=body_enabled), SceneAssemblyInputAssembler(cfg.human_vis_threshold),
        {**reconstructed, "skeleton": "player_triangulation", "ball": "ball_triangulation", "placement": "body_placement"}, lambda: {"enabled": cfg.enabled})
    unknown = set(overrides) - {node.name for node in nodes}
    if unknown:
        raise ValueError(f'Unknown component override: {sorted(unknown)}')
    return tuple(nodes)


def enabled_model_assets(cfg: PipelineRuntimeConfig) -> dict[str, Path]:
    """Checkpoint inputs for preflight (including non-model AFLink).

    Complete model/implementation inventories are bound by the ModelSpec-based
    ``publication_model_identities`` and ``declared_model_identities`` helpers.
    """
    people, body = cfg.enabled["person_observations"], cfg.enabled["gvhmr"]
    return {
        **({"court": cfg.court_kp.checkpoint} if cfg.direct_annotation is None and cfg.video_court is None else {}),
        **({"ball": cfg.ball_detection.checkpoint} if cfg.ball_detection is not None else {}),
        **({"detector": cfg.people.dino_checkpoint, "vitpose": cfg.people.vitpose_checkpoint} if people else {}),
        **({"association_encoder": cfg.association_encoder_weights} if people and cfg.association_encoder_weights is not None else {}),
        **({"tracking_encoder": cfg.tracking_encoder_weights, "aflink": cfg.aflink_checkpoint} if people else {}),
        **(cfg.people.body_assets() if body else {}),
    }
