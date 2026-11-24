"""Re-select players and prepare body requests from archived CPU evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from frozen_clip import FrozenClip, read_json
from omegaconf import OmegaConf

from src.tasks.person_tracking.court_linking import LinkingConfig
from src.tasks.person_tracking.feature_tracks import evidence_appearance
from src.tasks.player_association.appearance.sampling import (
    CropSamplingConfig,
    TrackAppearance,
)
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.generate_dataset.manifest import DatasetClipRecord
from src.tennis_scene.pipeline.artifacts import (
    document_digest,
    json_value,
    write_json_atomic,
)
from src.tennis_scene.pipeline.components.association_appearance import (
    SavedAssociationAppearance,
)
from src.tennis_scene.pipeline.components.body_view_selection import (
    BodyViewSelectionModule,
    BodyViewSelectionOutput,
)
from src.tennis_scene.pipeline.components.camera_alignment import CameraAlignmentOutput
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.components.court_side import CourtSideOutput
from src.tennis_scene.pipeline.components.identity import (
    PlayerAssociationModule,
    PlayerIdentitiesOutput,
)
from src.tennis_scene.pipeline.components.person_tracking import PersonTrackingOutput
from src.tennis_scene.pipeline.components.player_selection import (
    PlayerSelectionInput,
    PlayerSelectionModule,
)
from src.tennis_scene.pipeline.components.pose_estimation import (
    PoseEstimationInput,
    PoseEstimationModule,
)
from src.tennis_scene.pipeline.components.triangulation import PlayerTriangulationModule
from src.tennis_scene.pipeline.contracts import AssemblyContext, ClipSource
from src.tennis_scene.pipeline.errors import ReconstructionUnavailable
from src.tennis_scene.pipeline.imports.external_ball_geometry import (
    SHARED_GEOMETRY,
    require_matching_alignment,
)
from src.tennis_scene.pipeline.imports.external_ball_publication import ScenePointers
from src.tennis_scene.pipeline.inference_runtime import ModelBorrower, ModelSpec
from src.tennis_scene.pipeline.input_assembly.body import (
    BodyViewSelectionInputAssembler,
)
from src.tennis_scene.pipeline.input_assembly.reconstruction import (
    PlayerAssociationInputAssembler,
    PlayerTriangulationInputAssembler,
)
from src.tennis_scene.pipeline.observation_types import (
    GroupedObservations,
    ObjectObservations,
)
from src.tennis_scene.pipeline.source_identity import source_digests
from src.tennis_scene.pipeline.storage.clip_store import ArtifactRef
from src.tennis_scene.pipeline.video_court_side.artifact import VideoCourtSideArtifact
from src.utils.checksum import dual_sha256
from src.utils.paths import PROJECT_ROOT
from src.utils.schema.court import HALF_DOUBLES_WIDTH
from src.utils.video.reader import OpenCVDecodeIdentity

SIDE_STORE = Path("knowledge/runs/run-court-side-video-shared-20261118/side-store")
REPLAY_PREFIX = "doubles_players"


@dataclass(frozen=True)
class ReplayRun:
    dataset: Path
    output: Path
    comparison: Path
    config_yaml: str
    recipe: dict[str, Any]

    @classmethod
    def bind(cls, dataset: Path, output: Path, comparison: Path) -> ReplayRun:
        (progress_path,) = (dataset.parents[1] / "all/progress").glob("*/progress.json")
        progress = read_json(progress_path)
        if (progress["dataset_id"], progress["video_id"]) != (
            "meiji_4cam",
            "video_000",
        ):
            raise ValueError("Expected the archived Meiji 4cam video configuration")
        config_yaml = progress["pipeline_config_yaml"]
        recipe = {
            "schema": "meiji_doubles_player_replay_v1",
            "pipeline_config_sha256": document_digest(config_yaml),
            "dataset_sha256": dual_sha256(dataset / "dataset.json"),
            "src_sha256": document_digest(source_digests(PROJECT_ROOT / "src")),
            "cpu_script_sha256": {
                name: dual_sha256(Path(__file__).with_name(name))
                for name in ("prepare.py", "frozen_clip.py")
            },
        }
        return cls(dataset, output, comparison, config_yaml, recipe)

    def config(self) -> PipelineRuntimeConfig:
        config = OmegaConf.create(self.config_yaml)
        config.paths.project_root = str(PROJECT_ROOT)
        cfg = PipelineRuntimeConfig.from_config(config, bind_inputs=False)
        if cfg.player_association.players_per_side != 2 or not all(
            cfg.enabled[name]
            for name in ("person_observations", "player_reconstruction", "gvhmr")
        ):
            raise ValueError(
                "Expected the archived doubles player and body configuration"
            )
        return cfg

    def preparation_path(self, record: DatasetClipRecord) -> Path:
        return self.output / record.clip_name / "cpu.json"


class SavedModels(ModelBorrower):
    def __init__(self, encoder_identity: str | None = None) -> None:
        self.encoder_identity = encoder_identity

    def identity(self, alias: str) -> str:
        if alias != "encoder" or self.encoder_identity is None:
            raise ValueError(f"Missing saved model identity for {alias}")
        return self.encoder_identity

    def borrow(self, alias: str) -> Any:
        raise ValueError(f"The CPU replay requires archived evidence for {alias}")


def fail_model_creation() -> Any:
    raise ValueError("Cannot create an inference model during CPU replay")


class SavedAppearanceAssociation(PlayerAssociationModule):
    def _appearances(
        self,
        source: ClipSource,
        tracks: list[PersonTrackingOutput],
        sampling: CropSamplingConfig,
        saved: Mapping[str, SavedAssociationAppearance],
        expected_decoders: Mapping[str, OpenCVDecodeIdentity],
        models: ModelBorrower,
    ) -> tuple[list[tuple[TrackAppearance, ...]], dict[str, Any]]:
        appearance = self.config.appearance
        if appearance is None:
            raise ValueError("The archived association requires appearance evidence")
        samples = []
        for track in tracks:
            evidence = track.evidence
            if (
                evidence is None
                or evidence.encoder != appearance.encoder
                or (track.observed & ~evidence.appearance_valid).any()
            ):
                raise ValueError(
                    f"{source.clip_id}/{track.camera_id}: missing archived appearance"
                )
            samples.append(
                tuple(
                    evidence_appearance(
                        track.boxes_xyxy,
                        track.observed,
                        evidence,
                        source.size,
                        sampling,
                    )
                )
            )
        return samples, {
            "source": "archived_detection_features",
            "samples": sum(len(item.frames) for camera in samples for item in camera),
        }


def replay_node(component: str, camera: str | None = None) -> str:
    prefix = f"{REPLAY_PREFIX}_{component}"
    return prefix if camera is None else f"{prefix}/{camera}"


def fixed_alignment(
    clip: FrozenClip,
    calibration: CourtCalibrationOutput,
) -> tuple[CameraAlignmentOutput, CourtSideOutput, str]:
    node = clip.pinned["alignment_node"]
    alignment = clip.load(node, CameraAlignmentOutput, "aligned_cameras")
    geometry = alignment.geometry
    if geometry is None or calibration.calibration.camera_ids != clip.source.camera_ids:
        raise ValueError(
            "The external ball revision requires all four calibrated views"
        )
    desc = clip.descriptor(node, "aligned_cameras", 1)
    if desc["dependencies"]["calibration"] != json_value(clip.ref("court_calibration")):
        raise ValueError("Fixed alignment was built with a different calibration")
    if node == "camera_alignment":
        side_node = "court_side"
        side = clip.load(side_node, CourtSideOutput, "court_side", 3)
        if desc["dependencies"]["side"] != json_value(clip.ref(side_node)):
            raise ValueError("Fixed alignment was built with a different court side")
    else:
        side_node = node
        fixed = desc["provenance"]["geometry"]
        if fixed["origin"] != SHARED_GEOMETRY:
            raise ValueError("Expected #2458's shared-court geometry")
        shared = VideoCourtSideArtifact.load(SIDE_STORE, fixed["shared_side_digest"])
        if (
            shared.decision.evidence != "ball"
            or clip.source.clip_id not in shared.clips
        ):
            raise ValueError("Shared side provenance does not cover the clip")
        side = shared.output()
    cameras = tuple(
        view.camera.half_turned(turn)
        for view, turn in zip(
            calibration.calibration.views, side.view_half_turns, strict=True
        )
    )
    require_matching_alignment(
        geometry, side.reference_camera, side.view_half_turns, cameras
    )
    if side.camera_ids != geometry.camera_ids:
        raise ValueError("Fixed side and alignment disagree on camera order")
    return alignment, side, side_node


def equal_body_requests(
    old: BodyViewSelectionOutput, new: BodyViewSelectionOutput
) -> bool:
    if len(old.selections) != len(new.selections):
        return False
    for a, b in zip(old.selections, new.selections, strict=True):
        if (a.person_id, a.camera_id, a.observed_samples, len(a.requests)) != (
            b.person_id,
            b.camera_id,
            b.observed_samples,
            len(b.requests),
        ):
            return False
        for x, y in zip(a.requests, b.requests, strict=True):
            if (x.video_path, x.size) != (y.video_path, y.size) or any(
                not np.array_equal(getattr(x, field), getattr(y, field))
                for field in ("source_frames", "keypoints", "boxes_xys", "intrinsic")
            ):
                return False
    return True


def equal_grouped_people(old: GroupedObservations, new: GroupedObservations) -> bool:
    # Row numbers identify pose carriers. The rejoined observations are what
    # triangulation, placement and the final scene use.
    return all(
        np.array_equal(getattr(old, field), getattr(new, field))
        for field in ("identities", "uv_px", "confidence", "visibility")
    )


def read_preparation(
    run: ReplayRun, record: DatasetClipRecord, clip: FrozenClip
) -> dict[str, Any]:
    preparation = read_json(run.preparation_path(record))
    if preparation["clip_id"] != record.clip_id or preparation["pinned"] != clip.pinned:
        raise ValueError(
            f"{record.clip_id}: CPU evidence no longer matches the frozen inputs"
        )
    clip.staged = {
        name: ArtifactRef(**ref) for name, ref in preparation["staged"].items()
    }
    clip.check_staged(clip.staged)
    return preparation


def prepare_clip(run: ReplayRun, record: DatasetClipRecord) -> dict[str, Any]:
    started = perf_counter()
    clip = FrozenClip.bind(run.dataset, record, run.recipe)
    target = run.preparation_path(record)
    comparison_path = run.comparison / record.clip_id / "selection.npz"
    comparison_sha = dual_sha256(comparison_path)
    if target.exists():
        saved = read_preparation(run, record, clip)
        if saved["selection_comparison_sha256"] != comparison_sha:
            raise ValueError(f"{record.clip_id}: #2440 selection evidence changed")
        return saved

    cfg = run.config()
    source = clip.source
    cameras = source.camera_ids
    context = AssemblyContext(source)
    appearance = cfg.player_association.appearance
    if appearance is None or cfg.tracking.encoder != appearance.encoder:
        raise ValueError("Expected the same archived tracking and association encoder")
    encoder = ModelSpec(
        "saved_appearance",
        {},
        appearance.encoder,
        "cpu",
        "float32",
        {},
        fail_model_creation,
    )
    selection = PlayerSelectionModule(
        LinkingConfig(max_candidates=cfg.max_tracks_per_camera),
        footpoints=cfg.player_association.footpoints,
        sampling=CropSamplingConfig(),
        encoder=encoder,
        encoder_name=appearance.encoder,
        device="cpu",
        enabled=True,
    )
    pose = PoseEstimationModule(cfg.people, require_evidence=True)
    if (selection.config.core_half_width_m, selection.config.min_presence_fraction) != (
        HALF_DOUBLES_WIDTH,
        0.25,
    ):
        raise ValueError("Expected the standard doubles width and 25% dwell")

    calibration = clip.load(
        "court_calibration", CourtCalibrationOutput, "local_court_calibration"
    )
    alignment, side, side_node = fixed_alignment(clip, calibration)
    selection_bindings = {}
    outputs: dict[str, Any] = {
        "calibration": calibration,
        "alignment": alignment,
        "side": side,
    }
    bindings = {
        "calibration": "court_calibration",
        "alignment": clip.pinned["alignment_node"],
    }
    same_pose = True
    masks: dict[str, Any] = {}
    with np.load(comparison_path, allow_pickle=False) as expected:
        for camera in cameras:
            raw_node = f"person_tracking/{camera}"
            raw = clip.load(raw_node, PersonTrackingOutput, "person_tracks", 7)
            if (
                raw.evidence is None
                or (raw.observed & ~raw.evidence.appearance_valid).any()
            ):
                raise ValueError(
                    f"{source.clip_id}/{camera}: tracking lost appearance evidence"
                )
            old_node = f"player_selection/{camera}"
            old = clip.load_output(old_node, selection.io)
            if clip.descriptor(
                old_node, selection.io.output_schema, selection.io.version
            )["dependencies"] != {
                "calibration": json_value(clip.ref("court_calibration")),
                "tracks": json_value(clip.ref(raw_node)),
            }:
                raise ValueError(
                    f"{source.clip_id}/{camera}: selection was built with different tracking or calibration"
                )
            new = selection.process_with_models(
                PlayerSelectionInput(source.video(camera), calibration, raw),
                SavedModels(raw.evidence.encoder_identity),
            )
            if (
                not np.array_equal(new.raw_track_ids, old.raw_track_ids)
                or not np.array_equal(old.selected, expected[f"baseline.{camera}"])
                or not np.array_equal(new.selected, expected[f"wide.{camera}"])
            ):
                raise ValueError(
                    f"{source.clip_id}/{camera}: selection mask differs from #2440"
                )
            mask_equal = np.array_equal(old.selected, new.selected)
            carriers_equal = all(
                np.array_equal(a, b)
                for a, b in (
                    (old.origin_rows, new.origin_rows),
                    (old.tracks.track_ids, new.tracks.track_ids),
                    (old.tracks.boxes_xyxy, new.tracks.boxes_xyxy),
                )
            )
            selected_node = old_node
            if not mask_equal or not carriers_equal:
                selected_node = clip.stage(
                    replay_node("player_selection", camera),
                    new,
                    selection.io,
                    {"tracks": raw_node, "calibration": "court_calibration"},
                    settings=selection.cache_identity(),
                )
            select_port, pose_port = f"tracks_{camera}", f"pose_{camera}"
            selection_bindings[select_port] = selected_node
            outputs[select_port] = new
            pose_node = f"pose_estimation/{camera}"
            if not carriers_equal:
                projected = pose.process_with_models(
                    PoseEstimationInput(source.video(camera), new.tracks), SavedModels()
                )
                pose_node = clip.stage(
                    replay_node("pose_estimation", camera),
                    projected,
                    pose.io,
                    {"selection": selected_node},
                )
                outputs[pose_port] = projected
            else:
                if clip.descriptor(pose_node, pose.io.output_schema, pose.io.version)[
                    "dependencies"
                ] != {
                    "selection": json_value(clip.ref(old_node)),
                }:
                    raise ValueError(
                        f"{source.clip_id}/{camera}: saved pose uses different carriers"
                    )
                outputs[pose_port] = clip.load_output(pose_node, pose.io)
            bindings[pose_port] = pose_node
            same_pose &= carriers_equal
            masks[camera] = {
                "same_mask_as_before": bool(mask_equal),
                "same_pose_carriers_as_before": bool(carriers_equal),
                "before_observations": int(old.selected.sum()),
                "after_observations": int(new.selected.sum()),
                "selection_artifact_id": clip.ref(selected_node).artifact_id,
                "pose_artifact_id": clip.ref(pose_node).artifact_id,
            }
    selection_seconds = perf_counter() - started
    cpu_before = clip.before_counts()
    mode, stop, body, reused_body = "reuse_scene_players", None, None, False
    association_seconds = triangulation_seconds = body_seconds = 0.0
    player_frames: int = cpu_before["joint_player_frames"]
    player_count: int = cpu_before["players"]
    if clip.pinned["alignment_node"] != "camera_alignment" or not same_pose:
        stage_start = perf_counter()
        try:
            association = SavedAppearanceAssociation(
                cameras,
                cfg.player_association,
                sampling=selection.sampling,
                encoder=selection.encoder,
                device=cfg.device,
                enabled=True,
            )
            identities = association.process_with_models(
                PlayerAssociationInputAssembler().assemble(context, outputs),
                SavedModels(),
            )
            bindings["identities"] = clip.stage(
                replay_node("player_association"),
                identities,
                association.io,
                {
                    "calibration": "court_calibration",
                    "side": side_node,
                    **selection_bindings,
                },
                settings=association.cache_identity(),
            )
            outputs["identities"] = identities
            association_seconds = perf_counter() - stage_start

            stage_start = perf_counter()
            triangulation = PlayerTriangulationModule(
                cameras,
                reprojection_px=cfg.player_reprojection_px,
                joint_confidence=cfg.joint_confidence,
            )
            assemble_people = PlayerTriangulationInputAssembler(cfg.human_vis_threshold)
            triangulation_input = assemble_people.assemble(context, outputs)
            reusable_skeleton = False
            if clip.pinned["alignment_node"] == "camera_alignment":
                previous = {
                    "calibration": calibration,
                    "alignment": alignment,
                    "identities": clip.load(
                        "player_association",
                        PlayerIdentitiesOutput,
                        "person_identities",
                        3,
                    ),
                    **{
                        f"pose_{camera}": clip.load(
                            f"pose_estimation/{camera}",
                            ObjectObservations,
                            "person_poses",
                            2,
                        )
                        for camera in cameras
                    },
                }
                previous_people = assemble_people.assemble(
                    context, previous
                ).observations
                reusable_skeleton = equal_grouped_people(
                    previous_people, triangulation_input.observations
                )
            if reusable_skeleton:
                bindings["skeleton"] = "player_triangulation"
                skeleton = clip.load_output("player_triangulation", triangulation.io)
            else:
                skeleton = triangulation.process(triangulation_input)
                bindings["skeleton"] = clip.stage(
                    replay_node("player_triangulation"),
                    skeleton,
                    triangulation.io,
                    bindings,
                )
            triangulation_seconds = perf_counter() - stage_start
            if skeleton.skeleton is None or not skeleton.skeleton.valid.any():
                raise ReconstructionUnavailable(
                    "no_valid_skeleton", "No player frame with a supported 3D joint"
                )
            player_count = len(skeleton.skeleton.valid)
            player_frames = int(skeleton.skeleton.valid.any(axis=-1).sum())

            stage_start = perf_counter()
            view = BodyViewSelectionModule(cameras, max_frames=cfg.sampling_max_frames)
            views = view.process(
                BodyViewSelectionInputAssembler(cfg.human_vis_threshold).assemble(
                    context, outputs
                )
            )
            body = clip.stage(
                replay_node("body_view_selection"),
                views,
                view.io,
                {port: bindings[port] for port in view.io.inputs},
            )
            if "body_view_selection" in clip.refs:
                reused_body = equal_body_requests(
                    clip.load_output("body_view_selection", view.io), views
                )
            body_seconds = perf_counter() - stage_start
            if reused_body:
                mode = (
                    "reuse_scene_players" if reusable_skeleton else "place_saved_bodies"
                )
            else:
                mode = "recover_bodies"
        except ReconstructionUnavailable as error:
            if association_seconds == 0.0:
                association_seconds = perf_counter() - stage_start
            mode = "stopped"
            stop = {
                "reason_code": error.reason,
                "error": str(error),
                "diagnostics": error.diagnostics,
            }
            player_count = player_frames = 0

    result = {
        "clip_id": source.clip_id,
        "pinned": clip.pinned,
        "selection_rule": asdict(selection.config),
        "selection_comparison_sha256": comparison_sha,
        "staged": json_value(clip.staged),
        "selection": selection_bindings,
        "reconstruction": bindings,
        "body_view_selection": body,
        "mode": mode,
        "stop": stop,
        "masks": masks,
        "before": cpu_before,
        "before_stop": clip.descriptor("scene_assembly", "scene_result", 2)["payload"][
            "metadata"
        ]["player_reconstruction"],
        "cpu_player_count": player_count,
        "cpu_joint_player_frames": player_frames,
        "seconds": {
            "selection_and_input": selection_seconds,
            "association": association_seconds,
            "triangulation": triangulation_seconds,
            "body_view_selection": body_seconds,
            "cpu": perf_counter() - started,
        },
    }
    clip.check_staged(clip.staged)
    clip.clip.snapshot.assert_unchanged()
    if clip.current != ScenePointers.read(clip.store.root):
        raise ValueError(
            f"{source.clip_id}: published pointers changed during CPU reconstruction"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(target, result)
    return result
