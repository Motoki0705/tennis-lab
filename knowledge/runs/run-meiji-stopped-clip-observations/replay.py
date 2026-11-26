"""Replay #2363 stop decisions with CPU stages and reviewed observation edits."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from collections.abc import Iterator, Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np
import torch
from inputs import Archive, ArchivedClip, read_artifact, read_json, write_json
from omegaconf import OmegaConf
from review import candidate

from src.tasks.court_side.hypothesis import PlayerCourtSideConfig
from src.tasks.person_tracking.contracts import DetectionFeatures
from src.tasks.person_tracking.court_linking import LinkingConfig
from src.tasks.person_tracking.feature_tracks import evidence_appearance
from src.tasks.person_tracking.sequence import track_sequence
from src.tasks.person_tracking.strongsort_offline import AFLink
from src.tasks.player_association.appearance.encoders import AppearanceEncoder
from src.tasks.player_association.appearance.sampling import (
    CropSamplingConfig,
    TrackAppearance,
)
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.pipeline.components.association_appearance import (
    SavedAssociationAppearance,
)
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.components.court_side import PlayerCourtSideModule
from src.tennis_scene.pipeline.components.identity import (
    PlayerAssociationModule,
    PlayerIdentitiesOutput,
)
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionOutput
from src.tennis_scene.pipeline.components.person_features import (
    PersonAppearanceFeaturesOutput,
    PersonPoseFeaturesOutput,
    require_alignment,
)
from src.tennis_scene.pipeline.components.person_tracking import PersonTrackingOutput
from src.tennis_scene.pipeline.components.player_selection import (
    PlayerSelectionInput,
    PlayerSelectionModule,
    PlayerSelectionOutput,
)
from src.tennis_scene.pipeline.contracts import AssemblyContext, ClipSource, SourceVideo
from src.tennis_scene.pipeline.errors import ReconstructionUnavailable
from src.tennis_scene.pipeline.inference_runtime import ModelBorrower, ModelSpec
from src.tennis_scene.pipeline.input_assembly.reconstruction import (
    PlayerAssociationInputAssembler,
    PlayerCourtSideInputAssembler,
)
from src.tennis_scene.visualization.review.store import StoreReader
from src.utils.paths import PROJECT_ROOT
from src.utils.video.reader import OpenCVDecodeIdentity

BUNDLE = Path(__file__).resolve().parent
ROLES = ("T", "W", "C", "D")


def reject_inference() -> AppearanceEncoder:
    raise RuntimeError("This replay only accepts archived appearance features")


class SavedModelIdentity(ModelBorrower):
    def __init__(self, encoder_identity: str | None = None) -> None:
        self.encoder_identity = encoder_identity

    def identity(self, alias: str) -> str:
        if alias != "encoder" or self.encoder_identity is None:
            raise ValueError(f"Missing archived identity for {alias}")
        return self.encoder_identity

    def borrow(self, alias: str) -> Any:
        raise ValueError(f"CPU replay cannot borrow {alias}")


class ArchivedAppearanceAssociation(PlayerAssociationModule):
    """Use the #2440 CPU appearance proxy with unchanged association thresholds."""

    def _appearances(
        self,
        source: ClipSource,
        tracks: list[PersonTrackingOutput],
        sampling: CropSamplingConfig,
        saved: Mapping[str, SavedAssociationAppearance],
        expected_decoders: Mapping[str, OpenCVDecodeIdentity],
        models: ModelBorrower,
    ) -> tuple[list[tuple[TrackAppearance, ...]], dict[str, Any]]:
        if self.config.appearance is None:
            raise ValueError("Expected the archived appearance configuration")
        appearances = []
        for track in tracks:
            evidence = track.evidence
            if (
                evidence is None
                or evidence.encoder != self.config.appearance.encoder
                or (track.observed & ~evidence.appearance_valid).any()
            ):
                raise ValueError(
                    f"{source.clip_id}/{track.camera_id}: missing archived appearance"
                )
            appearances.append(
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
        return appearances, {"source": "archived_detection_features"}


def edit_detection_input(
    cfg: PipelineRuntimeConfig,
    video: SourceVideo,
    store: StoreReader,
    calibration: CourtCalibrationOutput,
    original: PlayerSelectionOutput,
    selector: PlayerSelectionModule,
    edits: list[int],
) -> PlayerSelectionOutput:
    camera = video.camera_id
    detection = read_artifact(
        store, f"person_detection/{camera}", PersonDetectionOutput
    )
    pose = read_artifact(
        store, f"person_pose_features/{camera}", PersonPoseFeaturesOutput
    )
    appearance = read_artifact(
        store, f"person_appearance_features/{camera}", PersonAppearanceFeaturesOutput
    )
    archived = read_artifact(store, f"person_tracking/{camera}", PersonTrackingOutput)
    rows = require_alignment(video, detection, pose, appearance)
    if (
        detection.duplicate_merges
        or len(edits) != len(set(edits))
        or not set(edits).issubset(rows)
        or archived.evidence is None
        or archived.evidence.encoder_identity != appearance.encoder_identity
    ):
        raise ValueError(f"{video.path}: inconsistent review edits or saved features")
    selection_models = SavedModelIdentity(appearance.encoder_identity)
    aflink = AFLink(cfg.aflink_checkpoint)

    def retrack(keep: np.ndarray) -> PersonTrackingOutput:
        def detection_frames() -> Iterator[DetectionFeatures]:
            for frame in range(video.num_frames):
                start, end = detection.frame_offsets[frame : frame + 2]
                indices = np.flatnonzero(keep[start:end]) + start
                yield DetectionFeatures(
                    frame,
                    rows[indices],
                    detection.boxes_xyxy[indices],
                    detection.confidence[indices],
                    pose.keypoints[indices],
                    appearance.embeddings[indices],
                    appearance.appearance_valid[indices],
                )

        result = track_sequence(
            detection_frames(), fps=video.fps, config=cfg.tracking, aflink=aflink
        )
        return PersonTrackingOutput(
            camera,
            result.track_ids,
            result.boxes,
            result.observed,
            result.source_track_ids,
            (),
            replace(result.evidence, encoder_identity=appearance.encoder_identity),
            result.reconstruction,
            result.link_candidates,
        )

    control = retrack(np.ones(len(rows), dtype=bool))
    if (
        control.evidence is None
        or not np.array_equal(
            control.evidence.detection_rows, archived.evidence.detection_rows
        )
        or control.source_track_ids != archived.source_track_ids
        or not np.array_equal(control.boxes_xyxy, archived.boxes_xyxy)
    ):
        raise ValueError(
            f"{video.path}: unchanged CPU tracker did not reproduce the stored rows"
        )
    control_selection = selector.process_with_models(
        PlayerSelectionInput(video, calibration, control), selection_models
    )
    if any(
        not np.array_equal(getattr(control_selection, key), getattr(original, key))
        for key in ("raw_track_ids", "selected", "origin_rows")
    ):
        raise ValueError(
            f"{video.path}: unchanged CPU selection did not reproduce the stored rows"
        )
    edited = retrack(~np.isin(rows, edits))
    return selector.process_with_models(
        PlayerSelectionInput(video, calibration, edited), selection_models
    )


def edit_groups(
    selected: PlayerSelectionOutput, *, remove: int, join_into: int | None = None
) -> PlayerSelectionOutput:
    """Optional reviewer-observed merge of two carriers. Preserve saved raw origins."""
    tracks = selected.tracks
    evidence = tracks.evidence
    if evidence is None or remove == join_into or remove not in tracks.track_ids:
        raise ValueError(
            "Group correction requires known group IDs with saved observations"
        )
    keep = tracks.track_ids != remove
    group_ids = tracks.track_ids[keep]
    mapping = np.broadcast_to(
        np.flatnonzero(keep)[:, None], (len(group_ids), tracks.observed.shape[1])
    ).copy()
    mapping[~tracks.observed[keep]] = -1
    if join_into is not None:
        (primary,) = np.flatnonzero(group_ids == join_into)
        (secondary,) = np.flatnonzero(~keep)
        missing = (mapping[primary] < 0) & tracks.observed[secondary]
        mapping[primary, missing] = secondary
    groups, frames = np.nonzero(mapping >= 0)
    origins = np.full_like(mapping, -1)
    origins[groups, frames] = selected.origin_rows[mapping[groups, frames], frames]
    boxes = np.zeros((*mapping.shape, 4), dtype=np.float32)
    boxes[groups, frames] = tracks.boxes_xyxy[mapping[groups, frames], frames]
    result = PersonTrackingOutput(
        tracks.camera_id,
        group_ids,
        boxes,
        mapping >= 0,
        tuple((int(i),) for i in group_ids),
        (),
        evidence.regroup(mapping),
    )
    return replace(selected, tracks=result, origin_rows=origins)


def score_assignments(
    output: PlayerIdentitiesOutput,
    selections: dict[str, PlayerSelectionOutput],
    labels: list[dict[str, str]],
) -> dict[str, Any]:
    mapped: dict[str, Counter[int]] = {role: Counter() for role in ROLES}
    views: dict[str, set[str]] = {role: set() for role in ROLES}
    known = 0
    for label in labels:
        camera = label["camera"]
        view = output.camera_ids.index(camera)
        frame = int(label["frame"])
        carrier = selections[camera].tracks
        if carrier.evidence is None or not np.array_equal(
            carrier.track_ids, output.local_track_ids[view, : len(carrier.track_ids)]
        ):
            raise ValueError("Assignment cannot be matched to the archived DINO rows")
        rows = carrier.evidence.detection_rows[:, frame]
        for role in ROLES:
            if not label[role].isdigit():
                continue
            known += 1
            hit = np.flatnonzero(rows == int(label[role]))
            if len(hit) > 1:
                raise ValueError(
                    "A DINO row has multiple carriers in the same camera/frame"
                )
            if (
                len(hit)
                and (identity := int(output.player_ids[view, hit[0], frame])) >= 0
            ):
                mapped[role][identity] += 1
                views[role].add(camera)
    count_by_role = {key: dict(value) for key, value in mapped.items()}
    side_counts = {
        side: len(result["players"])
        for side, result in output.diagnostics["association"]["selection"].items()
    }
    distinct_ids = {identity for counts in mapped.values() for identity in counts}
    return {
        "known_player_rows": known,
        "assigned_player_rows": sum(count.total() for count in mapped.values()),
        "ids_by_role": count_by_role,
        "camera_counts_by_role": {
            role: len(cameras) for role, cameras in views.items()
        },
        "players_per_side": side_counts,
        "consistent_four_on_reviewed_rows": (
            side_counts == {"-1": 2, "1": 2}
            and len(distinct_ids) == 4
            and all(len(ids) == 1 for ids in mapped.values())
            and all(len(cameras) >= 2 for cameras in views.values())
        ),
    }


def diagnostic_summary(details: dict[str, Any]) -> dict[str, Any]:
    if "association" in details:
        association = details["association"]
        return {
            key: association[key]
            for key in (
                "selection",
                "identities",
                "min_player_margin",
                "ambiguous_pairs",
            )
            if key in association
        }
    return {
        key: details[key]
        for key in ("hypotheses", "frames", "margin")
        if key in details
    }


def replay_clip(
    clip: ArchivedClip,
    *,
    config_yaml: str,
    edits: dict[str, Any],
    labels: list[dict[str, str]],
    output: Path,
) -> dict[str, Any]:
    torch.set_num_threads(1)
    started = monotonic()
    config = OmegaConf.create(config_yaml)
    config.paths.project_root = str(PROJECT_ROOT)
    cfg = PipelineRuntimeConfig.from_config(
        config, bind_inputs=False, bind_event_detection=False
    )
    if (
        not isinstance(cfg.court_side, PlayerCourtSideConfig)
        or cfg.reuse_clip_embeddings
    ):
        raise ValueError("Expected the #2363 court-side and appearance profiles")
    store = clip.store()
    source = clip.source(store, verify_media=False)
    context = AssemblyContext(source)
    cameras = source.camera_ids
    calibration = read_artifact(store, "court_calibration", CourtCalibrationOutput)
    archive = {}
    for camera in cameras:
        node = f"player_selection/{camera}"
        deps = store.descriptor(node)["dependencies"]
        if (
            deps["tracks"] != store.references[f"person_tracking/{camera}"]
            or deps["calibration"] != store.references["court_calibration"]
        ):
            raise ValueError(
                f"{clip.name}/{camera}: selected observations use different tracking or calibration"
            )
        archive[camera] = read_artifact(store, node, PlayerSelectionOutput)

    appearance = cfg.player_association.appearance
    if appearance is None or appearance.encoder != cfg.tracking.encoder:
        raise ValueError(
            f"{clip.name}: expected the same archived CLIP profile for tracking and association"
        )
    # Use the archived encoder identity at the component boundary. No TensorRT
    # engine is bound even when the historical config selected one.
    encoder = ModelSpec(
        "archived_appearance",
        {},
        appearance.encoder,
        "cpu",
        "float32",
        {},
        reject_inference,
    )
    selector = PlayerSelectionModule(
        LinkingConfig(max_candidates=cfg.max_tracks_per_camera),
        footpoints=cfg.player_association.footpoints,
        sampling=CropSamplingConfig(),
        encoder=encoder,
        encoder_name=appearance.encoder,
        device="cpu",
        enabled=True,
    )
    if any(
        archived.diagnostics["rule"] != asdict(selector.config)
        for archived in archive.values()
    ):
        raise ValueError(f"{clip.name}: new selection rule differs from #2363")
    association = ArchivedAppearanceAssociation(
        cameras,
        cfg.player_association,
        sampling=selector.sampling,
        encoder=selector.encoder,
        device="cpu",
        enabled=True,
    )
    side = PlayerCourtSideModule(
        cameras, cfg.court_side, max_frames=cfg.sampling_max_frames
    )

    def decide(selections: dict[str, PlayerSelectionOutput]) -> dict[str, Any]:
        artifacts = {
            "calibration": calibration,
            **{f"tracks_{camera}": value for camera, value in selections.items()},
        }
        stage = "court_side"
        try:
            side_result = side.process(
                PlayerCourtSideInputAssembler().assemble(context, artifacts)
            )
            artifacts["side"] = side_result
            stage = "player_association"
            inputs = PlayerAssociationInputAssembler().assemble(context, artifacts)
            identities = association.process_with_models(inputs, SavedModelIdentity())
        except ReconstructionUnavailable as error:
            return {
                "status": "stopped",
                "stage": stage,
                "reason": error.reason,
                "error": str(error),
                **diagnostic_summary(error.diagnostics),
            }
        return {
            "status": "passed",
            "stage": stage,
            **diagnostic_summary(identities.diagnostics),
            **score_assignments(identities, selections, labels),
        }

    original = clip.stop
    outcomes = {"unchanged": decide(archive)}
    if outcomes["unchanged"].get("reason") != original["reason"]:
        print(f"{clip.name}: original stop differs from CPU proxy", flush=True)
    if remove := edits.get("remove_detection_rows"):
        selections = dict(archive)
        for camera, rows in remove.items():
            selections[camera] = edit_detection_input(
                cfg,
                source.video(camera),
                store,
                calibration,
                archive[camera],
                selector,
                rows,
            )
        outcomes["delete_reviewed_dino_rows"] = {
            **decide(selections),
            "removed": remove,
            "unchanged_tracking_and_selection_match": list(remove),
        }
    if joined := edits.get("join_groups"):
        camera = joined["camera"]
        preferred, absorbed = joined["groups"]
        merged = edit_groups(archive[camera], remove=absorbed, join_into=preferred)
        outcomes["join_reviewed_groups"] = {
            **decide({**archive, camera: merged}),
            "groups": joined,
        }
    if original["reason"] == "player_association_ambiguous_players":
        detail = original["diagnostics"]["association"]
        (runner_id,) = [
            item["runner_up"]
            for item in detail["selection"].values()
            if item["runner_up"] is not None
        ]
        (runner,) = [
            item for item in detail["identities"] if item["identity"] == runner_id
        ]
        (focus,) = [candidate(detail, i) for i in runner["candidates"]]
        removed = edit_groups(archive[focus.camera], remove=focus.identity)
        outcomes["omit_runner_up_group"] = {
            **decide({**archive, focus.camera: removed}),
            "camera": focus.camera,
            "group": focus.identity,
        }
    result = {
        "clip": clip.name,
        "original_reason": original["reason"],
        "original": diagnostic_summary(original["diagnostics"]),
        "conditions": outcomes,
        "elapsed_seconds": round(monotonic() - started, 3),
    }
    write_json(output / f"{clip.name}.json", result)
    print(
        f"{clip.name}: "
        + ", ".join(
            f"{key}={val.get('reason', val['status'])}" for key, val in outcomes.items()
        ),
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("i2363_run", type=Path)
    parser.add_argument("output", type=Path, help="New output directory")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--clip", action="append", help="Optional stopped clip name such as clip_002"
    )
    args = parser.parse_args()
    if not 1 <= args.jobs <= 4:
        parser.error("--jobs must be between 1 and 4")
    started = monotonic()
    archive = Archive(args.i2363_run.resolve())
    if args.clip and not set(args.clip) <= {
        clip.name for clip in archive.clips.values()
    }:
        parser.error("--clip must name a stopped #2363 clip")
    clips = [
        clip
        for clip in archive.clips.values()
        if not args.clip or clip.name in args.clip
    ]
    with (BUNDLE / "anchor-labels.csv").open(newline="") as handle:
        anchors = list(csv.DictReader(handle))
    edits = read_json(BUNDLE / "replay-edits.json")
    args.output.mkdir(parents=True, exist_ok=False)
    replay = partial(
        replay_clip,
        config_yaml=archive.progress["pipeline_config_yaml"],
        output=args.output,
    )
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        jobs = [
            pool.submit(
                replay,
                clip,
                edits=edits.get(clip.name, {}),
                labels=[row for row in anchors if row["clip"] == clip.name],
            )
            for clip in clips
        ]
        results = [job.result() for job in jobs]
    write_json(
        args.output / "replay.json", {"source": archive.receipt, "clips": results}
    )
    matched = sum(
        clip["conditions"]["unchanged"].get("reason") == clip["original_reason"]
        for clip in results
    )
    print(
        f"Reproduced {matched}/{len(results)} stop codes on CPU in {monotonic() - started:.2f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
