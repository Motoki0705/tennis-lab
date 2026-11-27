"""Replay selection on fixed Meiji 3cam and 4cam evidence with CPU-only downstream code."""

from __future__ import annotations

import argparse
import hashlib
import platform
import subprocess
import time
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, NoReturn, cast

import numpy as np
import scipy
import torch
from archived_inputs import (
    CLIP_ENCODER,
    DEV,
    REPAIRED_CLIP,
    RESERVED,
    Receipt,
    labelled_inputs,
    read_component,
    source_from_index,
)
from numpy.typing import NDArray
from omegaconf import OmegaConf
from scores import (
    box_counts,
    box_rates,
    pool_pairs,
    pool_tracking,
    prediction,
    project_ids,
    rate,
    tracking_scores,
)
from selection import (
    CONDITIONS,
    IMAGE_HEIGHT_SCALE,
    LINKING,
    scaled_sampling,
    select,
    track_appearance,
)

from src.tasks.court_side.hypothesis import PlayerCourtSideConfig
from src.tasks.player_association.appearance.sampling import (
    CropSamplingConfig,
    TrackAppearance,
)
from src.tasks.player_association.association.associate import (
    AssociationUndecided,
    CameraTracks,
    associate,
)
from src.tasks.player_association.association.config import (
    DEFAULT_CONFIG,
    load_association_config,
)
from src.tasks.player_association.evaluation.metrics import evaluate, match_to_labels
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.pipeline.artifacts import document_digest
from src.tennis_scene.pipeline.components.association_appearance import (
    SavedAssociationAppearance,
)
from src.tennis_scene.pipeline.components.camera_alignment import CameraAlignmentModule
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.components.court_side import PlayerCourtSideModule
from src.tennis_scene.pipeline.components.identity import PlayerAssociationModule
from src.tennis_scene.pipeline.components.person_tracking import PersonTrackingOutput
from src.tennis_scene.pipeline.components.player_selection import PlayerSelectionOutput
from src.tennis_scene.pipeline.components.pose_estimation import (
    PoseEstimationInput,
    PoseEstimationModule,
)
from src.tennis_scene.pipeline.components.triangulation import (
    PlayerTriangulationModule,
    TriangulationInput,
)
from src.tennis_scene.pipeline.contracts import AssemblyContext, ClipSource
from src.tennis_scene.pipeline.errors import ReconstructionUnavailable
from src.tennis_scene.pipeline.inference_runtime import InferenceRuntime, ModelBorrower
from src.tennis_scene.pipeline.input_assembly.reconstruction import (
    CameraAlignmentInputAssembler,
    PlayerAssociationInputAssembler,
    PlayerCourtSideInputAssembler,
    PlayerTriangulationInputAssembler,
)
from src.tennis_scene.pipeline.model_resources import appearance_model
from src.tennis_scene.pipeline.storage.codec import restore_type
from src.tennis_scene.visualization.review.store import StoreReader
from src.utils.checksum import dual_sha256
from src.utils.io import write_json_atomic
from src.utils.paths import PROJECT_ROOT
from src.utils.video.reader import OpenCVDecodeIdentity


def replay_labelled(
    clip: str, dataset: Path, archive: Path
) -> tuple[dict[str, Any], Receipt]:
    data = labelled_inputs(dataset, archive, clip)
    source, raw, labels, cameras = data.source, data.raw, data.labels, data.cameras
    config = load_association_config(players_per_side=1)
    selected = {
        camera.camera_id: select(
            raw[camera.camera_id], source, camera, config.footpoints, CLIP_ENCODER
        )
        for camera in cameras
    }
    scale = source.pixel_threshold_scale
    config = replace(
        config,
        footpoints=replace(
            config.footpoints,
            bottom_border_px=config.footpoints.bottom_border_px * scale,
        ),
    )
    raw_predictions = {camera: prediction(tracks) for camera, tracks in raw.items()}
    matches = match_to_labels(labels, raw_predictions, min_iou=0.5)

    conditions = {}
    sampling = scaled_sampling(source)
    for condition in CONDITIONS:
        choices = {camera: values[condition] for camera, values in selected.items()}
        boxes = box_counts(labels, raw_predictions, matches, choices)
        tracks = []
        for camera in cameras:
            view = choices[camera.camera_id].tracks
            tracks.append(
                CameraTracks(
                    camera,
                    source.size,
                    view.track_ids,
                    view.boxes_xyxy,
                    view.observed,
                    track_appearance(view, source.size, sampling, CLIP_ENCODER),
                )
            )
        try:
            association = associate(tracks, source.fps, config)
        except AssociationUndecided as error:
            predictions = raw_predictions
            status = {
                "status": "stopped",
                "reason": error.reason,
                "message": str(error),
            }
        else:
            predictions = {
                camera: project_ids(raw_predictions[camera], choices[camera], ids)
                for camera, ids in zip(
                    association.camera_ids, association.player_ids, strict=True
                )
            }
            status = {"status": "associated"}
        pair_counts = evaluate(labels, predictions, min_iou=0.5)["pairs"]
        conditions[condition] = {
            "boxes": {**boxes, **box_rates(boxes)},
            "raw_id": tracking_scores(
                labels, source.size, raw, cameras, choices, groups=False
            ),
            "group_id": tracking_scores(
                labels, source.size, raw, cameras, choices, groups=True
            ),
            "association": {**status, "pairs": pool_pairs([pair_counts])},
            "groups_by_camera": {
                camera: len(choice.tracks.track_ids)
                for camera, choice in choices.items()
            },
        }
    return {
        "clip": clip,
        "frames": source.num_frames,
        "conditions": conditions,
    }, data.pins


def refuse_inference() -> NoReturn:
    raise ValueError("Replay requires the saved model evidence; inference is disabled")


class ArchivedAssociation(PlayerAssociationModule):
    """Use the fixed per-detection CLIP features with standard crop sampling."""

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
            raise ValueError("Expected the standard association scale with CLIP")
        appearances = [
            track_appearance(
                track, source.size, sampling, self.config.appearance.encoder
            )
            for track in tracks
        ]
        return appearances, {
            "source": "archived_detection_features",
            "samples": sum(len(item.frames) for view in appearances for item in view),
        }


@dataclass(frozen=True)
class DoublesRecord:
    clip: str
    index: Path
    frames: int
    status: str
    reason: str | None


def doubles_inventory(root: Path) -> tuple[list[DoublesRecord], str, Receipt]:
    pins = Receipt()
    report = pins.read("input.json", root / "input.json")
    (progress_path,) = (root / "all/progress").glob("*/progress.json")
    progress = pins.read("all/progress/progress.json", progress_path)
    outcome = pins.read("all/outcome.json", root / "all/outcome.json")
    expected = {f"video_000/clip_{index:03d}" for index in range(69)}
    progress_status = {key: value["status"] for key, value in progress["clips"].items()}
    if (
        outcome["statuses"] != progress_status
        or set(progress_status) != expected
        or len(report["clips"]) != len(expected)
        or {row["clip_id"] for row in report["clips"]} != expected
        or (progress["dataset_id"], progress["video_id"]) != ("meiji_4cam", "video_000")
        or Counter(progress_status.values()) != {"generated": 55, "failed": 14}
    ):
        raise ValueError(
            "Expected the full 69-clip #2363 benchmark and its original outcomes"
        )
    records = [
        DoublesRecord(
            row["clip_id"],
            root
            / "inputs/dataset"
            / row["path"]
            / "annotations/tennis_scene/scene.json",
            row["num_frames"],
            progress_status[row["clip_id"]],
            progress["clips"][row["clip_id"]].get("reason"),
        )
        for row in report["clips"]
    ]
    return records, progress["pipeline_config_yaml"], pins


def triangulation_key(inputs: TriangulationInput) -> str:
    geometry = inputs.alignment.geometry
    if geometry is None:
        raise ValueError("Missing aligned geometry")
    observations = inputs.observations
    digest: str = document_digest(
        {
            "cameras": geometry.cameras,
            "arrays": [
                [
                    value.shape,
                    str(value.dtype),
                    hashlib.sha256(value.tobytes()).hexdigest(),
                ]
                for value in (
                    observations.uv_px,
                    observations.confidence,
                    observations.visibility,
                )
            ],
        }
    )
    return digest


def replay_doubles_condition(
    cfg: PipelineRuntimeConfig,
    source: ClipSource,
    calibration: CourtCalibrationOutput,
    choices: dict[str, PlayerSelectionOutput],
    association: ArchivedAssociation,
    cache: dict[str, NDArray[np.bool_]],
) -> tuple[dict[str, Any], NDArray[np.bool_]]:
    context = AssemblyContext(source)
    camera_ids = source.camera_ids
    if not isinstance(cfg.court_side, PlayerCourtSideConfig):
        raise ValueError("Expected #2363's player-based court-side setting")
    artifacts: dict[str, Any] = {
        "calibration": calibration,
        **{f"tracks_{camera}": choice for camera, choice in choices.items()},
    }
    detail: dict[str, Any] = {
        "selected_observations": sum(
            int(choice.selected.sum()) for choice in choices.values()
        ),
        "candidate_groups": sum(
            len(choice.tracks.track_ids) for choice in choices.values()
        ),
    }
    stage = "court_side"
    models = ModelBorrower(InferenceRuntime({}), (source.clip_id, "archive_replay"))
    try:
        side = PlayerCourtSideModule(
            camera_ids, cfg.court_side, max_frames=cfg.sampling_max_frames
        )
        artifacts["side"] = side.process(
            PlayerCourtSideInputAssembler().assemble(context, artifacts)
        )
        detail["half_turns"] = list(artifacts["side"].view_half_turns)
        stage = "player_association"
        identities = association.process_with_models(
            PlayerAssociationInputAssembler().assemble(context, artifacts), models
        )
        artifacts["identities"] = identities
        detail["appearance_samples"] = identities.diagnostics["appearance"]["samples"]

        stage = "camera_alignment"
        pose = PoseEstimationModule(cfg.people, require_evidence=True)
        for camera, choice in choices.items():
            inputs = PoseEstimationInput(source.video(camera), choice.tracks)
            artifacts[f"pose_{camera}"] = pose.process_with_models(inputs, models)
        alignment = CameraAlignmentModule(
            camera_ids,
            cfg.camera_geometry,
            player_reprojection_px=cfg.player_reprojection_px,
            ball_reprojection_px=cfg.ball_reprojection_px,
            joint_confidence=cfg.joint_confidence,
            max_frames=cfg.sampling_max_frames,
            include_ball=False,
        )
        artifacts["alignment"] = alignment.process(
            CameraAlignmentInputAssembler(cfg.human_vis_threshold, None).assemble(
                context, artifacts
            )
        )
        stage = "player_triangulation"
        inputs = PlayerTriangulationInputAssembler(cfg.human_vis_threshold).assemble(
            context, artifacts
        )
        if inputs.alignment.geometry is None:
            raise ReconstructionUnavailable(
                "no_aligned_cameras", "No alignment from player observations"
            )
        key = triangulation_key(inputs)
        if key not in cache:
            module = PlayerTriangulationModule(
                camera_ids,
                reprojection_px=cfg.player_reprojection_px,
                joint_confidence=cfg.joint_confidence,
            )
            result = module.process(inputs)
            if result.skeleton is None:
                raise ValueError("Triangulation must produce a skeleton mask")
            cache[key] = result.skeleton.valid
        valid = cache[key]
        if valid.shape != (4, source.num_frames, 17):
            raise ValueError(
                f"Expected four COCO17 tracks on the source grid, got {valid.shape}"
            )
        if not valid.any():
            raise ReconstructionUnavailable(
                "no_valid_skeleton", "No frames with a valid joint"
            )
    except ReconstructionUnavailable as error:
        return {
            **detail,
            "status": "stopped",
            "stage": stage,
            "reason": error.reason,
            "message": str(error),
        }, np.zeros((4, source.num_frames, 17), bool)
    return {**detail, "status": "triangulated"}, valid


def replay_doubles(
    record: DoublesRecord, config_yaml: str
) -> tuple[dict[str, Any], Receipt]:
    pins = Receipt()
    store = StoreReader(record.index)
    pins.record("scene_index", store.index_path)
    source = source_from_index(store, record.clip)
    if len(source.camera_ids) != 4 or source.num_frames != record.frames:
        raise ValueError(f"{record.clip}: source grid differs from the #2363 manifest")
    document = OmegaConf.create(config_yaml)
    document.paths.project_root = str(PROJECT_ROOT)
    # Bind downstream settings from the saved profile. The replay has no
    # runtime model declarations and does not bind any TensorRT engines.
    cfg = PipelineRuntimeConfig.from_config(
        document, bind_inputs=False, bind_event_detection=False
    )
    if (
        cfg.max_tracks_per_camera != LINKING.max_candidates
        or cfg.player_association.players_per_side != 2
    ):
        raise ValueError(
            "Expected #2363's adopted candidate cap and doubles association"
        )
    if cfg.association_encoder_weights is None:
        raise ValueError("Expected the archived CLIP model declaration")
    association = ArchivedAssociation(
        source.camera_ids,
        cfg.player_association,
        sampling=CropSamplingConfig(),
        encoder=appearance_model(
            CLIP_ENCODER, cfg.association_encoder_weights, "cpu", refuse_inference
        ),
        device="cpu",
        enabled=True,
    )

    calibration = cast(
        CourtCalibrationOutput,
        restore_type(
            read_component(store, "court_calibration", "local_court_calibration", 1),
            CourtCalibrationOutput,
        ),
    )
    local = {
        view.camera.camera_id: view.camera for view in calibration.calibration.views
    }
    selected = {}
    parity = {}
    for camera in source.camera_ids:
        node = f"person_tracking/{camera}"
        stale = store.stale_dependencies(node)
        if stale:
            raise ValueError(
                f"{record.clip}/{camera}: noncurrent track lineage {stale}"
            )
        raw = cast(
            PersonTrackingOutput,
            restore_type(
                read_component(store, node, "person_tracks", 7), PersonTrackingOutput
            ),
        )
        choices = select(
            raw,
            source,
            local.get(camera),
            cfg.player_association.footpoints,
            CLIP_ENCODER,
        )
        archived = read_component(
            store,
            f"player_selection/{camera}",
            "selected_player_tracks",
            3,
            "raw_track_ids",
            "selected",
            "origin_rows",
        )
        parity[camera] = all(
            np.array_equal(getattr(choices["court_standard"], field), value)
            for field, value in archived.items()
        )
        selected[camera] = choices

    saved_valid = np.zeros((4, source.num_frames, 17), bool)
    if record.status == "generated":
        saved_valid = read_component(
            store, "player_triangulation", "player_skeletons", 1, "skeleton.valid"
        )["skeleton.valid"]
        if saved_valid.shape != (4, source.num_frames, 17) or not saved_valid.any():
            raise ValueError(
                f"{record.clip}: archived skeleton does not match the source/players"
            )

    cache: dict[str, NDArray[np.bool_]] = {}
    outcomes = {}
    for condition in CONDITIONS:
        choices = {camera: views[condition] for camera, views in selected.items()}
        metrics, valid = replay_doubles_condition(
            cfg, source, calibration, choices, association, cache
        )
        person_frame = valid.any(axis=-1)
        outcomes[condition] = {
            **metrics,
            "player_frames_3d": int(person_frame.sum()),
            "joints_3d": int(valid.sum()),
            "all_four_frames_3d": int(person_frame.all(axis=0).sum()),
        }
        if condition == "court_standard":
            status_equal = (
                metrics["status"] == "triangulated"
                if record.status == "generated"
                else metrics.get("reason") == record.reason
            )
            outcomes[condition].update(
                archive_outcome_equal=status_equal,
                archive_frame_counts_equal=bool(
                    np.array_equal(
                        person_frame.sum(axis=0), saved_valid.any(axis=-1).sum(axis=0)
                    )
                ),
                archive_joint_mask_equal=bool(np.array_equal(valid, saved_valid)),
            )
    return {
        "clip": record.clip,
        "frames": source.num_frames,
        "conditions": outcomes,
        "archive_selection_equal": parity,
        "archive": {
            "status": record.status,
            "reason": record.reason,
            "player_frames_3d": int(saved_valid.any(axis=-1).sum()),
        },
    }, pins


def labelled_summary(rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    scores = [row["conditions"][condition] for row in rows]
    boxes = {
        key: sum(row["boxes"][key] for row in scores)
        for key, value in scores[0]["boxes"].items()
        if type(value) is int
    }
    return {
        "clips": len(rows),
        "frames": sum(row["frames"] for row in rows),
        "boxes": {**boxes, **box_rates(boxes)},
        "raw_id": pool_tracking([row["raw_id"] for row in scores]),
        "group_id": pool_tracking([row["group_id"] for row in scores]),
        "pair": pool_pairs([row["association"]["pairs"] for row in scores]),
        "associated_clips": sum(
            row["association"]["status"] == "associated" for row in scores
        ),
        "association_stops": dict(
            Counter(
                row["association"]["reason"]
                for row in scores
                if row["association"]["status"] == "stopped"
            )
        ),
    }


def doubles_summary(rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    scores = [row["conditions"][condition] for row in rows]
    frames = sum(row["frames"] for row in rows)
    return {
        "clips": len(rows),
        "frames": frames,
        "max_player_frames": 4 * frames,
        "max_joints": 4 * 17 * frames,
        "triangulated_clips": sum(
            score["status"] == "triangulated" for score in scores
        ),
        "stop_reasons": dict(
            Counter(score["reason"] for score in scores if score["status"] == "stopped")
        ),
        **{
            key: sum(score[key] for score in scores)
            for key in (
                "player_frames_3d",
                "joints_3d",
                "all_four_frames_3d",
                "selected_observations",
                "candidate_groups",
            )
        },
        "player_frame_coverage": rate(
            sum(score["player_frames_3d"] for score in scores), 4 * frames
        ),
        "joint_coverage": rate(
            sum(score["joints_3d"] for score in scores), 4 * 17 * frames
        ),
        "all_four_coverage": rate(
            sum(score["all_four_frames_3d"] for score in scores), frames
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset3", type=Path, required=True)
    parser.add_argument("--archive3", type=Path, required=True)
    parser.add_argument("--archive4", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--clip3", action="append", default=[], help="Limit 3cam to the listed clips"
    )
    parser.add_argument(
        "--clip4", action="append", default=[], help="Limit 4cam to the listed clips"
    )
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    records, config_yaml, receipts4 = doubles_inventory(args.archive4)
    if args.clip3 or args.clip4:
        clips3 = tuple(args.clip3)
        records = [record for record in records if record.clip in args.clip4]
        if (
            len(records) != len(args.clip4)
            or len(set(clips3)) != len(clips3)
            or set(clips3) - {*DEV, *RESERVED}
        ):
            parser.error("Unknown or duplicate probe clip")
    else:
        clips3 = (*DEV, *RESERVED)
    records.sort(key=lambda record: (-record.frames, record.clip))
    args.output.mkdir(parents=True)
    start = time.perf_counter()
    results3: dict[str, dict[str, Any]] = {}
    results4: dict[str, dict[str, Any]] = {}
    pins3: dict[str, Receipt] = {}
    pins4: dict[str, Receipt] = {}
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        jobs = {
            **{
                pool.submit(replay_labelled, clip, args.dataset3, args.archive3): (
                    "3cam",
                    clip,
                )
                for clip in clips3
            },
            **{
                pool.submit(replay_doubles, record, config_yaml): ("4cam", record.clip)
                for record in records
            },
        }
        for job in as_completed(jobs):
            kind, clip = jobs[job]
            try:
                result, pins = job.result()
            except Exception as error:
                for pending in jobs:
                    pending.cancel()
                raise RuntimeError(f"{kind}/{clip}: replay failed") from error
            results, receipts = (
                (results3, pins3) if kind == "3cam" else (results4, pins4)
            )
            results[clip], receipts[clip] = result, pins
            clip_output = args.output / kind / clip
            write_json_atomic(clip_output / "inputs.json", pins)
            write_json_atomic(clip_output / "result.json", result)
            print(f"{kind}/{clip}: replayed {len(CONDITIONS)} conditions", flush=True)

    splits = {
        "dev": DEV,
        "reserved": RESERVED,
        "reserved_without_recovered_cam0": tuple(
            clip for clip in RESERVED if clip != REPAIRED_CLIP
        ),
    }
    summary3 = {}
    for split, clips in splits.items():
        available = [results3[clip] for clip in clips if clip in results3]
        if available:
            summary3[split] = {
                condition: labelled_summary(available, condition)
                for condition in CONDITIONS
            }
    summary4 = (
        {
            condition: doubles_summary(list(results4.values()), condition)
            for condition in CONDITIONS
        }
        if results4
        else {}
    )
    write_json_atomic(
        args.output / "meiji-3cam.json", {"summary": summary3, "clips": results3}
    )
    write_json_atomic(
        args.output / "meiji-4cam.json", {"summary": summary4, "clips": results4}
    )
    write_json_atomic(
        args.output / "inputs.json",
        {
            "meiji_3cam": pins3,
            "meiji_4cam": {"receipts": receipts4, "clips": pins4},
            "association_config": dual_sha256(DEFAULT_CONFIG),
            "source_tree": subprocess.check_output(
                ["git", "rev-parse", "HEAD:src"], text=True
            ).strip(),
            "scripts": {
                name: dual_sha256(Path(__file__).with_name(name))
                for name in (
                    "archived_inputs.py",
                    "replay.py",
                    "scores.py",
                    "selection.py",
                )
            },
            "linking": asdict(LINKING),
            "image_height_scale": IMAGE_HEIGHT_SCALE,
            "conditions": CONDITIONS,
        },
    )
    write_json_atomic(
        args.output / "execution.json",
        {
            "status": "ok",
            "wall_seconds": time.perf_counter() - start,
            "command_args": vars(args),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": str(torch.__version__),
            "main_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
        },
    )


if __name__ == "__main__":
    main()
