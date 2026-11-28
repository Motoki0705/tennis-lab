"""The standard recipe: explicit bindings, replaceable components and input builders."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.submodules.models.ecpose.inference import require_weight
from src.submodules.models.fasthmr.assets import require_checkpoint_sha256
from src.submodules.smpl_assets import require_smpl_asset
from src.tasks.court_side.hypothesis import PlayerCourtSideConfig
from src.tasks.person_tracking.court_linking import LinkingConfig
from src.tasks.player_association.appearance.encoders import build_encoder
from src.tasks.player_association.appearance.sampling import CropSamplingConfig
from src.tennis_scene.pipeline.appearance_media import (
    APPEARANCE_MEDIA,
    appearance_media_identity,
)
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
from src.tennis_scene.pipeline.components.court_side import (
    CourtSideModule,
    PlayerCourtSideModule,
)
from src.tennis_scene.pipeline.components.ecpose import (
    ECPoseDetectionsModule,
    ECPoseFeaturesModule,
    ECPoseModule,
)
from src.tennis_scene.pipeline.components.event_detection import (
    EVENT_DETECTION,
    EventDetectionInputAssembler,
    EventDetectionModule,
)
from src.tennis_scene.pipeline.components.fasthmr import FastHMRModule
from src.tennis_scene.pipeline.components.gvhmr import GVHMRModule
from src.tennis_scene.pipeline.components.hmr2_features import HMR2FeaturesModule
from src.tennis_scene.pipeline.components.identity import (
    PLAYER_ASSOCIATION,
    PlayerAssociationModule,
)
from src.tennis_scene.pipeline.components.monocular_plcs import (
    MonocularPLCSInputAssembler,
    MonocularPLCSReconstructionModule,
    SingleViewCameraAlignmentInputAssembler,
    SingleViewCameraAlignmentModule,
    SingleViewIdentityInputAssembler,
    SingleViewIdentityModule,
    SkeletonOnlyPlacementInputAssembler,
    SkeletonOnlyPlacementModule,
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
    FastHMRInputAssembler,
    GVHMRInputAssembler,
    HMR2FeaturesInputAssembler,
)
from src.tennis_scene.pipeline.input_assembly.ecpose import (
    ECPoseDetectionsAssembler,
    ECPoseFeaturesAssembler,
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
    PlayerCourtSideInputAssembler,
    PlayerTriangulationInputAssembler,
)
from src.tennis_scene.pipeline.input_assembly.scene import SceneAssemblyInputAssembler
from src.tennis_scene.pipeline.model_assets import asset_identities as asset_identities
from src.tennis_scene.pipeline.model_assets import file_identity as file_identity
from src.tennis_scene.pipeline.model_resources import appearance_model, model_identities
from src.tennis_scene.pipeline.recipes import (
    ECPOSE_RECIPE,
    FASTHMR_RECIPE,
    STANDARD_RECIPE,
    RecipeName,
)
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
        RFDetrRuntimeConfig,
        RTMPoseRuntimeConfig,
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
            repository=cfg.roots.external_asset_root / "dinov3" if profile.startswith("dinov3_") else None,
            clip=cfg.people.runtime.clip_reid)

    encoder = None if not people or name is None or cfg.association_encoder_weights is None else appearance(name, cfg.association_encoder_weights)
    tracking = None if not people else appearance(cfg.tracking.encoder, cfg.tracking_encoder_weights)
    if cfg.ecpose is None:
        pose = PersonPoseFeaturesModule(cfg.people, cfg.tracking.features, enabled=people)
        pose_spec, pose_preprocessing = pose.model_specs().get("pose"), None
        people_inference: dict[str, Any] = {
            "person_detection": PersonDetectionModule(cfg.people, enabled=people, merge_duplicates=cfg.merge_duplicate_person_boxes),
            "person_pose_features": pose,
        }
    else:
        ecpose = ECPoseModule(cfg.ecpose)
        pose_spec, pose_preprocessing = ecpose.spec, ecpose.cache_identity()
        people_inference = {
            ECPOSE_RECIPE: ecpose,
            "person_detection": ECPoseDetectionsModule(),
            "person_pose_features": ECPoseFeaturesModule(),
        }
    selection, sampling = LinkingConfig(max_candidates=cfg.max_tracks_per_camera), CropSamplingConfig()
    body_inference = ({
        "hmr2_features": HMR2FeaturesModule(cfg.people, enabled=body),
        "gvhmr": GVHMRModule(cfg.people, enabled=body),
    } if cfg.fasthmr is None else {FASTHMR_RECIPE: FastHMRModule(cfg.fasthmr)})
    return {
        "court_detection": CourtKPModule(cfg.court_kp),
        **({"ball_detection": ImportedBallAnnotationModule() if cfg.ball_detection is None else BallDetectionModule(cfg.ball_detection)}
           if not cfg.omit_ball_components else {}),
        **people_inference,
        "person_appearance_features": PersonAppearanceFeaturesModule(tracking, cfg.tracking.features, enabled=people,
            capture_frame_sha256=cfg.reuse_clip_embeddings),
        "person_tracking": PersonTrackingModule(cfg.tracking, aflink_checkpoint=cfg.aflink_checkpoint, enabled=people,
            expected_pose=pose_spec, expected_encoder=tracking, pose_preprocessing=pose_preprocessing),
        "player_selection": PlayerSelectionModule(selection, footpoints=association.footpoints, sampling=sampling,
            encoder=encoder, encoder_name=name if encoder is not None else None, device=cfg.device, enabled=people),
        "pose_estimation": PoseEstimationModule(cfg.people, require_evidence=people, enabled=people),
        PLAYER_ASSOCIATION: PlayerAssociationModule(ids, association, sampling=sampling, device=cfg.device, enabled=people, encoder=encoder,
            reuse_clip_embeddings=cfg.reuse_clip_embeddings, feature_config=cfg.tracking.features),
        **body_inference,
        "body_placement": BodyPlacementModule(ids, cfg.people, cfg.player_placement,
            reprojection_px=cfg.player_reprojection_px, enabled=body, fasthmr=cfg.fasthmr),
    }


def declared_model_identities(cfg: PipelineRuntimeConfig, stage: str, component: Any) -> dict[str, Any]:
    # Shared/imported Court and disabled pose consume persisted observations.
    # Standalone PoseEstimationModule keeps its optional inference API, but the
    # disabled standard recipe has no person tracks and no model demand.
    if (stage == "court_detection" and (cfg.direct_annotation is not None or cfg.video_court is not None)
            or stage == "pose_estimation" and not cfg.enabled["person_observations"]
            or not isinstance(component, InferenceComponent)):
        return {}
    specs = component.model_specs()
    file_identity = cfg.court_kp.asset_catalog.file_identity
    if cfg.fasthmr is None or stage not in {FASTHMR_RECIPE, "body_placement"}:
        identities = dict(model_identities(specs, file_identity))
        if cfg.ecpose is not None and stage == ECPOSE_RECIPE:
            checkpoint_identity = identities[ECPOSE_RECIPE]["assets"]["checkpoint"]
            require_weight(cfg.ecpose.checkpoint, cfg.ecpose.release, sha256=checkpoint_identity["sha256"])
        return identities
    pins = {path: name for name, path in cfg.fasthmr.pinned_weights.items()}

    def bind_pinned_asset(path: Path) -> dict[str, Any]:
        identity: dict[str, Any] = file_identity(path)
        name = pins.get(path)
        if name is not None:
            require_checkpoint_sha256(name, path, identity["sha256"])
        return identity

    return dict(model_identities(specs, bind_pinned_asset))


def publication_model_identities(cfg: PipelineRuntimeConfig) -> dict[str, Any]:
    identities = {stage: declared_model_identities(cfg, stage, component)
                  for stage, component in standard_model_components(cfg, ()).items()}
    identities[EVENT_DETECTION] = model_identities(
        cfg.require_event_detection().model_specs(), cfg.court_kp.asset_catalog.file_identity,
    )
    return identities


def binds_appearance_media(cfg: PipelineRuntimeConfig, overrides: Mapping[str, Any]) -> bool:
    if PLAYER_ASSOCIATION in overrides:
        module = overrides[PLAYER_ASSOCIATION]
        return (isinstance(module, PlayerAssociationModule) and module.enabled
                and module.reuse_clip_embeddings and module.encoder is not None)
    return (cfg.enabled["person_observations"] and cfg.reuse_clip_embeddings
            and cfg.player_association.appearance is not None)


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


def validate_override_names(overrides: Mapping[str, Any], recipe: RecipeName = STANDARD_RECIPE) -> None:
    stages = set(STANDARD_COMPONENTS)
    if recipe == FASTHMR_RECIPE:
        stages.difference_update(("hmr2_features", "gvhmr"))
        stages.add(FASTHMR_RECIPE)
    if recipe == ECPOSE_RECIPE:
        stages.add(ECPOSE_RECIPE)
    unknown = {name for name in overrides if name.split('/')[0] not in stages}
    if unknown:
        raise ValueError(f'Unknown component override: {sorted(unknown)}')


def _person_runtime_identity(
                             runtime: DinoDetectorRuntimeConfig | ViTPoseRuntimeConfig | RFDetrRuntimeConfig | RTMPoseRuntimeConfig,
                             device: str, *, key: str = "runtime") -> dict[str, Any]:
    options = json_value(runtime)
    del options["engines"]
    if options.get("batch_size") == 1 and "confidence" in options:
        del options["batch_size"]
    if runtime.precision == "fp32":
        del options["precision"]
        return {key: options}
    return {key: options, "device": device}


def _bound_assets(cfg: PipelineRuntimeConfig, enabled: bool, paths: Mapping[str, Path]) -> dict[str, Any]:
    return {name: cfg.court_kp.asset_catalog.file_identity(path) for name, path in paths.items()} if enabled else {"enabled": False}


def person_detection_settings(cfg: PipelineRuntimeConfig) -> dict[str, Any]:
    enabled = cfg.enabled["person_observations"]
    return {
        "detector": cfg.people.detector,
        "assets": _bound_assets(cfg, enabled, cfg.people.detection_assets()),
        **_person_runtime_identity(cfg.people.detector_runtime, cfg.people.runtime.device),
        "scope": "full_frame", "enabled": enabled, "merge_duplicates": cfg.merge_duplicate_person_boxes,
        "merge_rule": "greedy_iou_ge_0.8_score_desc_source_row_asc",
    }


def person_pose_feature_settings(cfg: PipelineRuntimeConfig, component: PersonPoseFeaturesModule) -> dict[str, Any]:
    return {
        **component.cache_identity(),
        **_person_runtime_identity(cfg.people.pose_runtime, cfg.people.runtime.device),
        "assets": _bound_assets(cfg, cfg.enabled["person_observations"], cfg.people.pose_assets()),
    }


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
    Stage-only callers can leave event detection unbound.
    """
    ids = source.camera_ids
    overrides = {} if overrides is None else overrides
    validate_override_names(overrides, cfg.recipe)
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
    appearance_media = appearance_media_identity(ids, source.paths) if binds_appearance_media(cfg, overrides) else None
    decoders = {} if appearance_media is None else appearance_media["decoders"]

    bound_assets = partial(_bound_assets, cfg)

    def add(name: str, component: Any, assembler: Any, bindings: dict[str, str], settings: Callable[[], Any],
            *, camera: str | None = None, require_component: Callable[[], None] | None = None,
            after_components: tuple[str, ...] = (), component_origin_only: bool = False) -> None:
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
        if appearance_media is not None and stage in {"person_appearance_features", PLAYER_ASSOCIATION}:
            resolved[APPEARANCE_MEDIA] = {**appearance_media, "decoders": decoders if camera is None else {camera: decoders[camera]}}
        mode = "load" if cfg.cache_source == "load" else cfg.component_sources[stage]
        nodes.append(ComponentNode(name, component, component.io, assembler, bindings,
            AssemblyContext(source, camera), json_value(resolved), code_identity, mode, require_component=require_component,
            after_components=after_components, component_origin_only=component_origin_only))

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

    def body_assets() -> dict[str, Any]:
        if body_enabled:
            cfg.people.smpl_assets.require_files()
        return bound_assets(body_enabled, cfg.people.body_assets())

    def hmr2_assets() -> dict[str, Any]:
        if body_enabled:
            require_smpl_asset(cfg.people.smpl_assets.hmr2_mean_params, "hmr2_mean_params")
        return bound_assets(body_enabled, {
            "checkpoint": cfg.people.hmr2_checkpoint, "mean_params": cfg.people.smpl_assets.hmr2_mean_params})

    def placement_settings() -> dict[str, Any]:
        body = {"assets": body_assets()} if cfg.fasthmr is None else {"recipe": cfg.recipe}
        return {"config": cfg.player_placement, **body}

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
    include_ball = not cfg.omit_ball_components
    if include_ball and not cfg.enabled["ball_detection"] and cfg.cache_source != "load":
        if cfg.component_sources["court_side"] != "load" and "court_side" not in overrides:
            raise ValueError("court_side decides sides from the ball alone; it requires ball_detection.enabled or execution.court_side=load")
        if any(cfg.component_sources[key] != "load" for key in ("ball_detection", "ball_points")):
            raise ValueError("Disabled ball detection requires explicit load for the detector/point chain")
    ball_config = cfg.ball_detection
    for camera in ids if include_ball else ():
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
        detection_node = f"person_detection/{camera}"
        if cfg.ecpose is None:
            add(detection_node, components["person_detection"], PersonDetectionInputAssembler(),
                {}, partial(person_detection_settings, cfg), camera=camera,
                component_origin_only=cfg.people.uses_alternative_models)
            add(f"person_pose_features/{camera}", components["person_pose_features"],
                PersonPoseFeaturesInputAssembler(), {"detections": detection_node},
                partial(person_pose_feature_settings, cfg, components['person_pose_features']), camera=camera,
                after_components=('person_detection',), component_origin_only=cfg.people.uses_alternative_models)
        else:
            joint_node = f"{ECPOSE_RECIPE}/{camera}"
            ecpose_module = components[ECPOSE_RECIPE]
            add(joint_node, ecpose_module, PersonDetectionInputAssembler(),
                {}, lambda: {"recipe": cfg.recipe}, camera=camera, component_origin_only=True)
            add(detection_node, components["person_detection"], ECPoseDetectionsAssembler(),
                {"joint": joint_node}, lambda: {"recipe": cfg.recipe}, camera=camera, component_origin_only=True)
            add(f"person_pose_features/{camera}", components["person_pose_features"], ECPoseFeaturesAssembler(),
                {"joint": joint_node, "detections": detection_node}, lambda: {"recipe": cfg.recipe},
                camera=camera, component_origin_only=True)
        add(f"person_appearance_features/{camera}", components["person_appearance_features"],
            PersonAppearanceFeaturesInputAssembler(expected_decoder=decoders.get(camera)), {"detections": f"person_detection/{camera}"},
            lambda: {**components['person_appearance_features'].cache_identity(), "encoder": cfg.tracking.encoder,
                "assets": bound_assets(people_enabled, {} if tracking_encoder is None else dict(tracking_encoder.weights)),
                "enabled": people_enabled}, camera=camera, after_components=('person_detection', 'person_pose_features'))
        pose_component = overrides.get(f'person_pose_features/{camera}', components['person_pose_features'])
        appearance_component = overrides.get(f'person_appearance_features/{camera}', components['person_appearance_features'])
        tracker = components['person_tracking']
        if isinstance(pose_component, PersonPoseFeaturesModule) and isinstance(appearance_component, PersonAppearanceFeaturesModule):
            tracker = PersonTrackingModule(cfg.tracking, aflink_checkpoint=cfg.aflink_checkpoint, enabled=people_enabled,
                expected_pose=pose_component.model_specs().get('pose'), expected_encoder=appearance_component.model_specs().get('encoder'),
                pose_preprocessing=pose_component.cache_identity(), appearance_preprocessing=appearance_component.preprocessing_settings())
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
    balls = {f"ball_{c}": f"ball_points/{c}" for c in ids} if include_ball else {}
    tracks = {f"tracks_{c}": f"player_selection/{c}" for c in ids}
    observations = {"calibration": "court_calibration", **poses}
    side_module: CourtSideModule | PlayerCourtSideModule | SharedCourtSideModule
    side_assembler: CourtSideInputAssembler | PlayerCourtSideInputAssembler
    if isinstance(cfg.court_side, PlayerCourtSideConfig):
        side_assembler, side_observations = PlayerCourtSideInputAssembler(), tracks
    else:
        side_assembler, side_observations = CourtSideInputAssembler(0.0), balls
    require_side = None
    side_settings: dict[str, Any]
    if video_court_side is not None:
        assert video_court is not None
        side_module = SharedCourtSideModule(ids, video_court_side, video_court, cfg.court_side_unobservable_policy,
                                           court_policy=cfg.video_court_unobservable_policy)
        require_side = partial(side_module.require_clip, source.clip_id)
        side_settings = video_court_side.node_settings(source.clip_id, cfg.court_side_unobservable_policy, cfg.video_court_unobservable_policy)
    elif isinstance(cfg.court_side, PlayerCourtSideConfig):
        side_module = PlayerCourtSideModule(ids, cfg.court_side, max_frames=cfg.sampling_max_frames)
        side_settings = {"evidence": "player", "config": cfg.court_side, "max_frames": cfg.sampling_max_frames}
    else:
        side_module = CourtSideModule(ids, cfg.court_side, max_frames=cfg.sampling_max_frames)
        side_settings = {"config": cfg.court_side, "max_frames": cfg.sampling_max_frames, "ball_threshold": 0.0}
    add("court_side", side_module, side_assembler, {"calibration": "court_calibration", **side_observations},
        lambda: side_settings, require_component=require_side)
    association_component = overrides.get(PLAYER_ASSOCIATION, components[PLAYER_ASSOCIATION])
    if isinstance(association_component, SingleViewIdentityModule):
        association_assembler: Any = SingleViewIdentityInputAssembler(
            association_component.camera_id
        )
        association_bindings = {
            "calibration": "court_calibration",
            f"pose_{association_component.camera_id}": f"pose_estimation/{association_component.camera_id}",
        }
    else:
        reuse_clip_embeddings = isinstance(association_component, PlayerAssociationModule) and association_component.reuse_clip_embeddings
        association_assembler = PlayerAssociationInputAssembler(reuse_clip_embeddings, expected_decoders=decoders)
        association_bindings = {"calibration": "court_calibration", "side": "court_side", **tracks}
        if reuse_clip_embeddings:
            association_bindings.update({f"detections_{c}": f"person_detection/{c}" for c in ids})
            association_bindings.update({f"appearance_{c}": f"person_appearance_features/{c}" for c in ids})
    add(PLAYER_ASSOCIATION, components[PLAYER_ASSOCIATION], association_assembler,
        association_bindings, lambda: {**components[PLAYER_ASSOCIATION].cache_identity(),
                                       "assets": bound_assets(people_enabled and weights is not None, encoder_assets)})
    identified = {**observations, "identities": "player_association"}
    ball_threshold = 0.0 if include_ball else None
    alignment_override = overrides.get("camera_alignment")
    if isinstance(alignment_override, SingleViewCameraAlignmentModule):
        alignment_assembler: Any = SingleViewCameraAlignmentInputAssembler()
        alignment_bindings: dict[str, str] = {"calibration": "court_calibration"}
    else:
        alignment_assembler = CameraAlignmentInputAssembler(cfg.human_vis_threshold, ball_threshold)
        alignment_bindings = {**identified, **balls, "side": "court_side"}
    add("camera_alignment", CameraAlignmentModule(ids, cfg.camera_geometry, player_reprojection_px=cfg.player_reprojection_px,
        ball_reprojection_px=cfg.ball_reprojection_px, joint_confidence=cfg.joint_confidence, max_frames=cfg.sampling_max_frames,
        include_ball=include_ball),
        alignment_assembler, alignment_bindings,
        lambda: {"config": cfg.camera_geometry, "player_reprojection": cfg.player_reprojection_px, "ball_reprojection": cfg.ball_reprojection_px,
         "joint_confidence": cfg.joint_confidence, "visibility": cfg.human_vis_threshold, "max_frames": cfg.sampling_max_frames, "ball_threshold": ball_threshold})
    reconstructed = {**identified, "alignment": "camera_alignment"}
    player_tri_assembler: Any = (
        MonocularPLCSInputAssembler(cfg.human_vis_threshold)
        if isinstance(overrides.get("player_triangulation"), MonocularPLCSReconstructionModule)
        else PlayerTriangulationInputAssembler(cfg.human_vis_threshold)
    )
    add("player_triangulation", PlayerTriangulationModule(ids, reprojection_px=cfg.player_reprojection_px,
        joint_confidence=cfg.joint_confidence, enabled=cfg.enabled["player_reconstruction"]),
        player_tri_assembler, reconstructed, lambda: cfg.processing_settings["player_reconstruction"])
    if include_ball:
        add("ball_triangulation", BallTriangulationModule(ids, reprojection_px=cfg.ball_reprojection_px,
            min_frames=cfg.ball_min_frames, enabled=cfg.enabled["ball_reconstruction"]),
            BallTriangulationInputAssembler(0.0), {"alignment": "camera_alignment", "calibration": "court_calibration", **balls},
            lambda: {"config": cfg.processing_settings["ball_reconstruction"], "threshold": 0.0})
    add("body_view_selection", BodyViewSelectionModule(ids, max_frames=cfg.sampling_max_frames, enabled=body_enabled),
        BodyViewSelectionInputAssembler(cfg.human_vis_threshold), reconstructed,
        lambda: {"policy": "coverage_confidence_camera_id", "visibility": cfg.human_vis_threshold, "max_frames": cfg.sampling_max_frames, "enabled": body_enabled})
    if cfg.fasthmr is None:
        add("hmr2_features", components["hmr2_features"], HMR2FeaturesInputAssembler(),
            {"selection": "body_view_selection"}, lambda: {
                "assets": hmr2_assets(),
                "runtime": {"batch_size": cfg.people.runtime.hmr2.batch_size}, "enabled": body_enabled})
        add("gvhmr", components["gvhmr"], GVHMRInputAssembler(),
            {"selection": "body_view_selection", "features": "hmr2_features"},
            lambda: {"assets": body_assets(), "static_cam": cfg.people.runtime.static_cam, "enabled": body_enabled})
        recovery = "gvhmr"
    else:
        add(FASTHMR_RECIPE, components[FASTHMR_RECIPE], FastHMRInputAssembler(),
            {"selection": "body_view_selection"}, lambda: {"recipe": cfg.recipe}, component_origin_only=True)
        recovery = FASTHMR_RECIPE
    if isinstance(overrides.get("body_placement"), SkeletonOnlyPlacementModule):
        placement_assembler: Any = SkeletonOnlyPlacementInputAssembler()
        placement_bindings: dict[str, str] = {"skeleton": "player_triangulation"}
    else:
        placement_assembler = BodyPlacementInputAssembler(cfg.human_vis_threshold)
        placement_bindings = {**reconstructed, "skeleton": "player_triangulation", "recovered": recovery}
    add("body_placement", components["body_placement"], placement_assembler, placement_bindings, placement_settings)
    add("scene_assembly", SceneAssemblyModule(ids, require_people=cfg.enabled["player_reconstruction"],
        require_ball=cfg.enabled["ball_reconstruction"], require_body=body_enabled, include_ball=include_ball),
        SceneAssemblyInputAssembler(cfg.human_vis_threshold, include_ball=include_ball),
        {**reconstructed, "skeleton": "player_triangulation", "placement": "body_placement",
         **({"ball": "ball_triangulation"} if include_ball else {})}, lambda: {"enabled": cfg.enabled})
    if cfg.event_detection is not None:
        events = EventDetectionModule(cfg.event_detection, source.clip_id)
        add(EVENT_DETECTION, events, EventDetectionInputAssembler(), {"scene": "scene_assembly"},
            events.cache_identity, require_component=None if EVENT_DETECTION in overrides else events.require_inputs,
            component_origin_only=True)
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
    body_assets = (cfg.people.body_assets() if cfg.fasthmr is None else
                   {f"fasthmr_{name}": path for name, path in
                    {**cfg.fasthmr.inference_weights, **cfg.fasthmr.geometry_weights}.items()})
    person_assets = ({
        "detector" if cfg.people.detector == "dino" else cfg.people.detector: cfg.people.detection_assets()["checkpoint"],
        "vitpose" if cfg.people.pose == "vitpose-h" else cfg.people.pose: cfg.people.pose_assets()["pose"],
    } if cfg.ecpose is None else {"ecpose": cfg.ecpose.checkpoint})
    event_checkpoint = None if cfg.event_detection is None else cfg.event_detection.checkpoint
    return {
        **({"court": cfg.court_kp.checkpoint} if cfg.direct_annotation is None and cfg.video_court is None else {}),
        **({"ball": cfg.ball_detection.checkpoint} if cfg.ball_detection is not None and not cfg.omit_ball_components else {}),
        **({"ball_event_refiner": event_checkpoint} if event_checkpoint is not None else {}),
        **(person_assets if people else {}),
        **({"association_encoder": cfg.association_encoder_weights} if people and cfg.association_encoder_weights is not None else {}),
        **({"tracking_encoder": cfg.tracking_encoder_weights, "aflink": cfg.aflink_checkpoint} if people else {}),
        **(body_assets if body else {}),
    }
