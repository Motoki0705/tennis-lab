"""Replay selection through the standard player triangulation on saved i2363 evidence."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from inputs import (
    SELECTION,
    ClipRecord,
    SavedRun,
    read_artifact,
    read_store,
    source_from_index,
    write_json,
)
from numpy.typing import NDArray
from omegaconf import OmegaConf

from src.tasks.court_side.hypothesis import PlayerCourtSideConfig
from src.tasks.person_tracking.feature_tracks import evidence_appearance
from src.tasks.player_association.appearance.sampling import (
    CropSamplingConfig,
    TrackAppearance,
)
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.pipeline.components.association_appearance import (
    SavedAssociationAppearance,
)
from src.tennis_scene.pipeline.components.camera_alignment import (
    CameraAlignmentModule,
    CameraAlignmentOutput,
)
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.components.court_side import PlayerCourtSideModule
from src.tennis_scene.pipeline.components.identity import (
    PlayerAssociationModule,
    PlayerIdentitiesOutput,
)
from src.tennis_scene.pipeline.components.person_tracking import PersonTrackingOutput
from src.tennis_scene.pipeline.components.player_reconstruction import (
    triangulate_players,
)
from src.tennis_scene.pipeline.components.player_selection import (
    PlayerSelectionInput,
    PlayerSelectionModule,
    PlayerSelectionOutput,
)
from src.tennis_scene.pipeline.components.pose_estimation import (
    PoseEstimationInput,
    PoseEstimationModule,
)
from src.tennis_scene.pipeline.components.triangulation import PlayerTriangulationOutput
from src.tennis_scene.pipeline.contracts import AssemblyContext, ClipSource
from src.tennis_scene.pipeline.definition import standard_model_components
from src.tennis_scene.pipeline.errors import ReconstructionUnavailable
from src.tennis_scene.pipeline.inference_runtime import ModelBorrower
from src.tennis_scene.pipeline.input_assembly.reconstruction import (
    CameraAlignmentInputAssembler,
    PlayerAssociationInputAssembler,
    PlayerCourtSideInputAssembler,
    PlayerTriangulationInputAssembler,
)
from src.tennis_scene.pipeline.observation_types import (
    GroupedObservations,
    ObjectObservations,
)
from src.utils.checksum import dual_sha256
from src.utils.geometry.triangulation import PinholeCamera
from src.utils.paths import PROJECT_ROOT
from src.utils.video.reader import OpenCVDecodeIdentity


class ArchivedModels(ModelBorrower):
    """Use the model identity attested by the adopted selection, with no inference."""

    def __init__(self, encoder_identity: str | None = None) -> None:
        self.encoder_identity = encoder_identity

    def identity(self, alias: str) -> str:
        if alias != "encoder" or self.encoder_identity is None:
            raise ValueError(f"No archived model identity for {alias}")
        return self.encoder_identity

    def borrow(self, alias: str) -> Any:
        raise ValueError(f"Replay requires archived evidence for {alias}")


class ArchivedAppearanceAssociation(PlayerAssociationModule):
    """Evaluate with per-detection embeddings sampled on each candidate group timeline."""

    def _appearances(
        self,
        source: ClipSource,
        tracks: list[PersonTrackingOutput],
        sampling: CropSamplingConfig,
        saved: Mapping[str, SavedAssociationAppearance],
        expected_decoders: Mapping[str, OpenCVDecodeIdentity],
        models: ModelBorrower,
    ) -> tuple[list[tuple[TrackAppearance, ...]] | None, dict[str, Any]]:
        if self.config.appearance is None:
            raise ValueError("Replay requires the archived appearance model")
        appearances = []
        for track in tracks:
            evidence = track.evidence
            if evidence is None or evidence.encoder != self.config.appearance.encoder:
                raise ValueError(f"{source.clip_id}/{track.camera_id}: missing appearance evidence")
            if (track.observed & ~evidence.appearance_valid).any():
                raise ValueError(f"{source.clip_id}/{track.camera_id}: selected observation lacks archived appearance")
            appearances.append(
                tuple(evidence_appearance(track.boxes_xyxy, track.observed, evidence, source.size, sampling))
            )
        return appearances, {
            "source": "archived_detection_features",
            "samples": sum(len(item.frames) for camera in appearances for item in camera),
        }


def select_camera(
    inputs: PlayerSelectionInput,
    archived: PlayerSelectionOutput,
    template: PlayerSelectionModule,
) -> dict[str, PlayerSelectionOutput]:
    evidence = inputs.tracks.evidence
    if evidence is None:
        raise ValueError("Replay requires person_tracks v7 detection evidence")
    if (
        "linking" in archived.diagnostics
        and inputs.tracks.observed.any()
        and archived.diagnostics["appearance"]["source"] != "tracked_detection_features"
    ):
        raise ValueError("Archived selection recomputed appearances")
    result = {}
    for name, config in SELECTION.items():
        module = PlayerSelectionModule(
            config, footpoints=template.footpoints, sampling=template.sampling,
            encoder=template.encoder, encoder_name=template.encoder_name, device=template.device, enabled=True,
        )
        result[name] = module.process_with_models(inputs, ArchivedModels(evidence.encoder_identity))
    baseline = result["baseline"]
    if any(
        not np.array_equal(getattr(baseline, field), getattr(archived, field))
        for field in ("raw_track_ids", "selected", "origin_rows")
    ):
        raise ValueError(f"{inputs.video.camera_id}: baseline selection differs from the adopted artifact")
    diagnostics = dict(baseline.diagnostics)
    if "rule" in diagnostics:
        # The i2363 descriptor predates explicit width and region fields.
        diagnostics["rule"] = dict(diagnostics["rule"])
        diagnostics["linking"] = dict(diagnostics["linking"])
        diagnostics["rule"].pop("core_half_width_m")
        diagnostics["linking"].pop("region")
    if diagnostics != archived.diagnostics:
        raise ValueError(f"{inputs.video.camera_id}: baseline diagnostics differ from the adopted artifact")
    return result


def group(
    cfg: PipelineRuntimeConfig,
    context: AssemblyContext,
    calibration: CourtCalibrationOutput,
    selection: dict[str, PlayerSelectionOutput],
    association: PlayerAssociationModule,
) -> tuple[dict[str, Any], PlayerIdentitiesOutput]:
    cameras = context.source.camera_ids
    if not isinstance(cfg.court_side, PlayerCourtSideConfig):
        raise ValueError("Replay requires player-based court side")
    artifacts: dict[str, Any] = {
        "calibration": calibration,
        **{f"tracks_{camera}": value for camera, value in selection.items()},
    }
    side = PlayerCourtSideModule(cameras, cfg.court_side, max_frames=cfg.sampling_max_frames)
    artifacts["side"] = side.process(PlayerCourtSideInputAssembler().assemble(context, artifacts))
    inputs = PlayerAssociationInputAssembler().assemble(context, artifacts)
    identities = association.process_with_models(inputs, ArchivedModels())
    artifacts["identities"] = identities
    pose = PoseEstimationModule(cfg.people, require_evidence=True)
    for camera in cameras:
        pose_input = PoseEstimationInput(context.source.video(camera), selection[camera].tracks)
        artifacts[f"pose_{camera}"] = pose.process_with_models(pose_input, ArchivedModels())
    return artifacts, identities


@dataclass(frozen=True)
class TriangulatedFrames:
    cameras: tuple[PinholeCamera, ...]
    grouped: GroupedObservations
    valid: NDArray[np.bool_]


def triangulated_frames(
    cfg: PipelineRuntimeConfig,
    source: ClipSource,
    current: GroupedObservations,
    cameras: tuple[PinholeCamera, ...],
    previous: list[TriangulatedFrames],
) -> NDArray[np.bool_]:
    """Triangulation is frame-local. Reuse results with identical camera and joint inputs."""
    result = np.zeros((len(current.identities), source.num_frames), bool)
    for player in range(len(current.identities)):
        seen = np.zeros(source.num_frames, bool)
        for saved in previous:
            if len(saved.cameras) != len(cameras) or any(
                now.camera_id != old.camera_id or not all(
                    np.array_equal(getattr(now, field), getattr(old, field))
                    for field in ("intrinsic", "rotation", "translation")
                )
                for now, old in zip(cameras, saved.cameras, strict=True)
            ):
                continue
            for other in range(len(saved.grouped.identities)):
                same = (
                    (current.uv_px[player] == saved.grouped.uv_px[other]).all(axis=(0, 2, 3))
                    & (current.confidence[player] == saved.grouped.confidence[other]).all(axis=(0, 2))
                    & (current.visibility[player] == saved.grouped.visibility[other]).all(axis=(0, 2))
                )
                result[player, same] = saved.valid[other, same]
                seen |= same
        sufficient = current.visibility[player] & (current.confidence[player] >= cfg.joint_confidence)
        pending = np.flatnonzero(~seen & (sufficient.sum(axis=0) >= 2).any(axis=-1))
        if len(pending):
            rows = slice(player, player + 1)
            observations = GroupedObservations(
                current.identities[rows], current.uv_px[rows][:, :, pending],
                current.confidence[rows][:, :, pending], current.visibility[rows][:, :, pending],
                current.raw_indices[rows][:, :, pending],
            )
            skeleton = triangulate_players(
                observations, cameras, reprojection_px=cfg.player_reprojection_px * source.pixel_threshold_scale,
                joint_confidence=cfg.joint_confidence,
            )
            result[player, pending] = skeleton.valid[0].any(axis=-1)
    return result


def compare_counts(current: NDArray[np.int64], reference: NDArray[np.int64]) -> dict[str, int]:
    delta = current - reference
    return {
        "player_frames_3d": int(current.sum()),
        "gained_player_frames": int(np.maximum(delta, 0).sum()),
        "lost_player_frames": int(np.maximum(-delta, 0).sum()),
    }


def replay_clip(
    record: ClipRecord, *, config_yaml: str, output: Path, reuse_triangulation: bool
) -> dict[str, Any]:
    archived_config = OmegaConf.create(config_yaml)
    archived_config.paths.project_root = str(PROJECT_ROOT)
    cfg = PipelineRuntimeConfig.from_config(archived_config, bind_inputs=False)
    if cfg.max_tracks_per_camera != SELECTION["baseline"].max_candidates or cfg.player_association.players_per_side != 2:
        raise ValueError("Expected i2363's archived candidate cap and doubles association")
    store = read_store(record)
    source = source_from_index(store)
    context = AssemblyContext(source)
    cameras = source.camera_ids
    components = standard_model_components(cfg, cameras)
    template = components["player_selection"]
    association = ArchivedAppearanceAssociation(
        cameras, cfg.player_association, sampling=template.sampling,
        encoder=template.encoder, device=cfg.device, enabled=True,
    )
    calibration = read_artifact(store, "court_calibration", "local_court_calibration", 1, CourtCalibrationOutput)
    selections = {}
    for camera in cameras:
        raw = read_artifact(store, f"person_tracking/{camera}", "person_tracks", 7, PersonTrackingOutput)
        archived = read_artifact(store, f"player_selection/{camera}", "selected_player_tracks", 3, PlayerSelectionOutput)
        inputs = PlayerSelectionInput(source.video(camera), calibration, raw)
        selections[camera] = select_camera(inputs, archived, template)
    index_sha = dual_sha256(store.index_path)
    clip_out = output / record.clip_id
    clip_out.mkdir(parents=True)
    np.savez_compressed(
        clip_out / "selection.npz",
        **{f"{name}.{camera}": selections[camera][name].selected for name in SELECTION for camera in cameras},
    )
    write_json(clip_out / "selection.json", {
        "clip_id": record.clip_id,
        "scene_index_sha256": index_sha,
        "masks_sha256": dual_sha256(clip_out / "selection.npz"),
        "conditions": {name: asdict(config) for name, config in SELECTION.items()},
        "diagnostics": {
            camera: {name: value.diagnostics for name, value in conditions.items()}
            for camera, conditions in selections.items()
        },
    })

    assemble_triangulation = PlayerTriangulationInputAssembler(cfg.human_vis_threshold)
    previous: list[TriangulatedFrames] = []
    archived_ids: PlayerIdentitiesOutput | None = None
    archived_counts = np.zeros(source.num_frames, np.int64)
    if record.status == "generated":
        saved = read_artifact(store, "player_triangulation", "player_skeletons", 1, PlayerTriangulationOutput)
        if saved.skeleton is None:
            raise ValueError(f"{record.clip_id}: generated clip has no archived skeleton")
        archived_valid = saved.skeleton.valid.any(axis=-1)
        archived_counts = archived_valid.sum(axis=0)
        archived_ids = read_artifact(store, "player_association", "person_identities", 3, PlayerIdentitiesOutput)
        if reuse_triangulation:
            artifacts = {
                "calibration": calibration, "identities": archived_ids,
                "alignment": read_artifact(store, "camera_alignment", "aligned_cameras", 1, CameraAlignmentOutput),
                **{f"pose_{camera}": read_artifact(store, f"pose_estimation/{camera}", "person_poses", 2, ObjectObservations)
                   for camera in cameras},
            }
            saved_inputs = assemble_triangulation.assemble(context, artifacts)
            if saved_inputs.alignment.geometry is None:
                raise ValueError(f"{record.clip_id}: generated clip has no archived alignment")
            previous.append(TriangulatedFrames(saved_inputs.alignment.geometry.cameras, saved_inputs.observations, archived_valid))
    baseline_counts = np.zeros(source.num_frames, np.int64)
    result: dict[str, Any] = {}
    alignment = CameraAlignmentModule(
        cameras, cfg.camera_geometry, player_reprojection_px=cfg.player_reprojection_px,
        ball_reprojection_px=cfg.ball_reprojection_px, joint_confidence=cfg.joint_confidence,
        max_frames=cfg.sampling_max_frames, include_ball=False,
    )
    assemble_alignment = CameraAlignmentInputAssembler(cfg.human_vis_threshold, None)
    evaluated: dict[tuple[bytes, ...], tuple[dict[str, Any], NDArray[np.int64]]] = {}
    for name in SELECTION:
        selection = {camera: values[name] for camera, values in selections.items()}
        key = tuple(value.origin_rows.tobytes() for value in selection.values())
        if key not in evaluated:
            counts = np.zeros(source.num_frames, np.int64)
            detail: dict[str, Any] = {"status": "failed", "reason": None}
            try:
                artifacts, identities = group(cfg, context, calibration, selection, association)
                detail["appearance_samples"] = identities.diagnostics["appearance"]["samples"]
                detail["same_archived_assignments"] = archived_ids is not None and (
                    identities.camera_ids == archived_ids.camera_ids
                    and np.array_equal(identities.local_track_ids, archived_ids.local_track_ids)
                    and np.array_equal(identities.player_ids, archived_ids.player_ids)
                )
                artifacts["alignment"] = alignment.process(assemble_alignment.assemble(context, artifacts))
                tri = assemble_triangulation.assemble(context, artifacts)
                if tri.alignment.geometry is None:
                    raise ReconstructionUnavailable("no_aligned_cameras", "No geometry from player observations")
                geometry = tri.alignment.geometry.cameras
                valid = triangulated_frames(cfg, source, tri.observations, geometry, previous)
                if reuse_triangulation:
                    previous.append(TriangulatedFrames(geometry, tri.observations, valid))
                counts = valid.sum(axis=0)
                if not counts.any():
                    raise ReconstructionUnavailable("no_valid_skeleton", "No frames with a valid joint")
                detail["status"] = "triangulated"
            except ReconstructionUnavailable as error:
                detail.update(reason=error.reason, error=str(error))
            evaluated[key] = detail, counts
        detail, counts = evaluated[key]
        if name == "baseline":
            baseline_counts = counts
        result[name] = {
            **detail,
            **compare_counts(counts, baseline_counts),
            "vs_archived": compare_counts(counts, archived_counts),
            "raw_observations_selected": sum(int(value.selected.sum()) for value in selection.values()),
        }
    result = {
        "clip_id": record.clip_id, "scene_index_sha256": index_sha, "frames": source.num_frames,
        "archived": {"status": record.status, "reason": record.reason, "player_frames_3d": int(archived_counts.sum())},
        "conditions": result,
    }
    write_json(clip_out / "comparison.json", result)
    return result


def summarize(clips: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    baseline = "baseline"
    for name in SELECTION:
        rows = [clip["conditions"][name] for clip in clips]
        success = [clip for clip in clips if clip["conditions"][name]["status"] == "triangulated"]
        output[name] = {
            **{key: sum(row[key] for row in rows) for key in (
                "player_frames_3d", "gained_player_frames", "lost_player_frames", "raw_observations_selected",
            )},
            "triangulated_clips": len(success),
            "stopped_clips": len(clips) - len(success),
            "stop_reasons": dict(Counter(row["reason"] for row in rows if row["reason"])),
            "recovered_clips": [
                clip["clip_id"] for clip in success if clip["conditions"][baseline]["status"] != "triangulated"
            ],
            "new_stops": [
                clip["clip_id"] for clip in clips
                if clip["conditions"][baseline]["status"] == "triangulated"
                and clip["conditions"][name]["status"] != "triangulated"
            ],
            "gained_player_frames_on_baseline_success": sum(
                clip["conditions"][name]["gained_player_frames"]
                for clip in clips if clip["conditions"][baseline]["status"] == "triangulated"
            ),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("i2363_run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--clip", action="append", help="Optional full clip ID for a probe")
    parser.add_argument("--no-triangulation-reuse", action="store_true")
    args = parser.parse_args()
    run = SavedRun(args.i2363_run)
    records = list(run.select(args.clip))
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    args.output.mkdir(parents=True)
    evaluate = partial(
        replay_clip, config_yaml=run.progress["pipeline_config_yaml"], output=args.output,
        reuse_triangulation=not args.no_triangulation_reuse,
    )
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        clips = list(pool.map(evaluate, records))
    write_json(args.output / "comparison.json", {
        "source": run.receipt(), "archived": dict(Counter(clip["archived"]["status"] for clip in clips)),
        "summary": summarize(clips), "clips": clips,
    })


if __name__ == "__main__":
    main()
