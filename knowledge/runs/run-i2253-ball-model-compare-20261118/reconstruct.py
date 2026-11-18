"""Replay downstream ball components against the published #1752 inputs."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import cast

import numpy as np
import torch
from infer import HERE, read, save_npz, scene_configs, write
from numpy.typing import NDArray

from src.tennis_scene.archive import load_scene_result
from src.tennis_scene.pipeline.artifacts import document_digest, json_value
from src.tennis_scene.pipeline.components.ball_detection import (
    BallDetectionModule,
    BallDetectionOutput,
)
from src.tennis_scene.pipeline.components.ball_points import (
    BallPointsInput,
    BallPointsModule,
    BallPointsOutput,
)
from src.tennis_scene.pipeline.components.body_placement import BodyPlacementOutput
from src.tennis_scene.pipeline.components.body_view_selection import (
    BodyViewSelectionOutput,
)
from src.tennis_scene.pipeline.components.camera_alignment import CameraAlignmentOutput
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.components.court_kp import CourtKPResult
from src.tennis_scene.pipeline.components.court_side import CourtSideOutput
from src.tennis_scene.pipeline.components.gvhmr import GVHMROutput
from src.tennis_scene.pipeline.components.identity import PlayerIdentitiesOutput
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionOutput
from src.tennis_scene.pipeline.components.person_tracking import PersonTrackingOutput
from src.tennis_scene.pipeline.components.player_selection import PlayerSelectionOutput
from src.tennis_scene.pipeline.components.scene_assembly import SceneAssemblyModule
from src.tennis_scene.pipeline.components.triangulation import (
    BallTriangulationModule,
    BallTriangulationOutput,
    PlayerTriangulationOutput,
)
from src.tennis_scene.pipeline.contracts import AssemblyContext, ClipSource, SourceVideo
from src.tennis_scene.pipeline.input_assembly.reconstruction import (
    BallTriangulationInputAssembler,
)
from src.tennis_scene.pipeline.input_assembly.scene import SceneAssemblyInputAssembler
from src.tennis_scene.pipeline.observation_types import ObjectObservations
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.tennis_scene.pipeline.storage.scene_index import (
    indexed_scene_path,
    read_component_descriptor,
)
from src.utils.checksum import dual_sha256
from src.utils.data.heatmaps import heatmaps_to_argmax, refine_peaks_log_parabolic
from src.utils.geometry.keypoints import denormalize_grid_keypoints
from src.utils.geometry.triangulation import PointRejection

EXECUTED = {"ball_detection", "ball_points", "ball_triangulation", "scene_assembly"}
FROZEN_TYPES = {
    "court_detection": CourtKPResult,
    "court_calibration": CourtCalibrationOutput,
    "court_side": CourtSideOutput,
    "camera_alignment": CameraAlignmentOutput,
    "person_detection": PersonDetectionOutput,
    "person_tracking": PersonTrackingOutput,
    "player_selection": PlayerSelectionOutput,
    "pose_estimation": ObjectObservations,
    "player_association": PlayerIdentitiesOutput,
    "player_triangulation": PlayerTriangulationOutput,
    "body_view_selection": BodyViewSelectionOutput,
    "gvhmr": GVHMROutput,
    "body_placement": BodyPlacementOutput,
}
SCENE_BALL_FIELDS = (
    "ball_uv",
    "ball_vis",
    "ball_3d",
    "ball_3d_valid",
    "ball_rejection_code",
)


@dataclass(frozen=True)
class DetectorPoints:
    """The point-only input used by BallPointsModule in this replay."""

    camera_id: str
    frame_indices: NDArray[np.int64]
    uv_px: NDArray[np.float32]
    confidence: NDArray[np.float32]
    observed: NDArray[np.bool_]


class SavedClip:
    def __init__(self, clip):
        self.index_path = Path(clip["store"]) / "scene.json"
        self.digest = dual_sha256(self.index_path)
        self.index = read(self.index_path)
        self.root = self.index_path.parent.resolve()
        self.source = ClipSource(
            clip["source_clip_id"],
            tuple(
                SourceVideo(**dict(row, path=Path(row["path"])))
                for row in clip["inference_videos"]
            ),
        )
        if (
            json_value(self.source) != self.index["source"]
            or document_digest(self.index["source"]) != self.index["source_sha256"]
        ):
            raise ValueError("Scene source identity differs from the comparison inputs")
        self.scene = load_scene_result(indexed_scene_path(self.index_path))
        self.fixed = {}
        self.records = {}
        overrides = read(HERE.parent / "run-i1752-shared-fp16-20261118/settings.json")[
            "overrides"
        ]
        side_override = "court_side.video_shared.artifact_digest="
        (shared_side_digest,) = (
            item.removeprefix(side_override)
            for item in overrides
            if item.startswith(side_override)
        )
        for node in self.index["artifacts"]:
            stage = node.split("/")[0]
            if stage in EXECUTED:
                continue
            output_type = FROZEN_TYPES[stage]
            value, descriptor = self.load(node, output_type)
            self.records[node] = {"mode": "load", **self.index["artifacts"][node]}
            if stage == "court_side":
                if (
                    descriptor["identity"]["settings"]["artifact_digest"]
                    != shared_side_digest
                ):
                    raise ValueError(
                        "Saved court_side does not use the pinned video-shared artifact"
                    )
                if (
                    value.camera_ids != self.source.camera_ids
                    or value.view_half_turns != (False, False, True, True)
                ):
                    raise ValueError(
                        "Saved court_side camera order or half turns changed"
                    )
                self.records[node]["shared_artifact_digest"] = shared_side_digest
                self.records[node]["view_half_turns"] = list(value.view_half_turns)
            if stage in {
                "court_calibration",
                "camera_alignment",
                "player_association",
                "player_triangulation",
                "body_placement",
                "pose_estimation",
            }:
                self.fixed[node] = value
        if {name.split("/")[0] for name in self.records} != set(FROZEN_TYPES):
            raise ValueError("Saved scene is missing a frozen stage")
        geometry = self.fixed["camera_alignment"].geometry
        if (
            geometry is None
            or geometry.camera_ids != self.source.camera_ids
            or geometry.view_half_turns != (False, False, True, True)
        ):
            raise ValueError(
                "Saved camera alignment differs from the 4-camera court side"
            )
        self.verify_index()

    def verify_index(self):
        if dual_sha256(self.index_path) != self.digest:
            raise ValueError("The published source store changed during replay")

    def load(self, node, output_type):
        ref = self.index["artifacts"][node]
        descriptor = read_component_descriptor(
            self.root, ref, node=node, source_sha256=self.index["source_sha256"]
        )
        value = ArtifactCodec(output_type).load(
            descriptor["payload"],
            (self.root / ref["path"]).parent,
            descriptor["arrays"],
        )
        return value, descriptor


def delta(actual, expected):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError("Reference array shape/dtype changed")
    unchanged = np.array_equal(actual, expected, equal_nan=True)
    return {
        "differing_values": 0
        if unchanged
        else int(np.count_nonzero(actual != expected)),
        "max_abs": 0.0
        if unchanged
        else float(np.max(np.abs(actual.astype(np.float64) - expected))),
    }


def compare_ball_points(current, old):
    result = {
        key: delta(getattr(current, key), getattr(old, key))
        for key in ("uv_px", "confidence", "observed")
    }
    for field in fields(current.smoothing):
        actual, expected = (
            getattr(current.smoothing, field.name),
            getattr(old.smoothing, field.name),
        )
        if isinstance(actual, np.ndarray):
            result[f"smoothing.{field.name}"] = delta(actual, expected)
        elif actual != expected:
            raise ValueError(f"Smoothing scalar {field.name} differs")
    return result


def baseline_detections(saved, camera):
    detector, _ = saved.load(f"ball_detection/{camera}", BallDetectionOutput)
    evidence = detector.evidence
    if evidence is None:
        raise ValueError("The reference detector must have native heatmap evidence")
    coords, scores = [], []
    for begin in range(0, len(evidence.heatmaps), 32):
        heatmaps = torch.from_numpy(evidence.heatmaps[None, begin : begin + 32].copy())
        cells, score = heatmaps_to_argmax(heatmaps)
        coords.append(refine_peaks_log_parabolic(heatmaps, cells)[0].numpy())
        scores.append(score[0].numpy())
    uv = denormalize_grid_keypoints(np.concatenate(coords), *saved.source.size)
    return detector, uv, np.concatenate(scores)


def replay(
    saved, runtime, model, raw, *, compare_reference=False, use_saved_gate=False
):
    source = saved.source
    cfg = runtime.ball_detection
    assert cfg is not None
    threshold = model["threshold"]
    gate = BallDetectionModule(
        replace(cfg, score_threshold=0.0 if threshold is None else threshold)
    )
    points = BallPointsModule(runtime.ball_smoothing)
    fixed = saved.fixed
    dependencies = {
        "calibration": fixed["court_calibration"],
        "alignment": fixed["camera_alignment"],
    }
    reference = {}
    accepted = []
    count = source.num_frames
    if (
        raw["raw_uv_px"].shape != (4, count, 2)
        or raw["raw_uv_px"].dtype != np.float32
        or raw["raw_score"].shape != (4, count)
        or raw["raw_score"].dtype != np.float32
    ):
        raise ValueError("Detector arrays do not cover the full camera/frame grid")
    if threshold is None and not np.isnan(raw["raw_score"]).all():
        raise ValueError("Direct coordinates have no raw detector confidence")
    if threshold is not None and (
        not np.isfinite(raw["raw_score"]).all()
        or not np.all((raw["raw_score"] >= 0) & (raw["raw_score"] <= 1))
    ):
        raise ValueError("Heatmap scores must be finite probabilities in [0,1]")
    if (
        not np.isfinite(raw["raw_uv_px"]).all()
        or (raw["raw_uv_px"] < 0).any()
        or (raw["raw_uv_px"] > np.asarray(source.size) - 1).any()
    ):
        raise ValueError("Raw model coordinates must be in source pixel bounds")
    for i, camera in enumerate(source.camera_ids):
        raw_uv, raw_score = raw["raw_uv_px"][i], raw["raw_score"][i]
        # Direct coordinates provide one proposal per frame. Give the geometric
        # gate and smoother a unit availability weight for each proposal.
        input_score = (
            np.ones(count, dtype=np.float32) if threshold is None else raw_score
        )
        uv, score, visible = gate._accept_detections(raw_uv, input_score)
        if compare_reference:
            old_detector, old_uv, old_score = baseline_detections(saved, camera)
        if use_saved_gate:
            uv, score, visible = (
                old_detector.uv_px,
                old_detector.confidence,
                old_detector.observed,
            )
        observation = DetectorPoints(
            camera, np.arange(count, dtype=np.int64), uv, score, visible
        )
        # BallPointsModule reads these five point fields. The experimental input
        # never enters the persisted BallDetectionOutput heatmap schema.
        inputs = BallPointsInput(
            cast(BallDetectionOutput, observation), source.size, source.fps
        )
        point = points.process(inputs)
        dependencies[f"ball_{camera}"] = point
        accepted.append((uv, score, visible))
        if compare_reference:
            old_points, _ = saved.load(f"ball_points/{camera}", BallPointsOutput)
            reference[camera] = {
                "raw_uv_px": delta(raw_uv, old_uv),
                "raw_score": delta(raw_score, old_score),
                "gate_uv_px": delta(uv, old_detector.uv_px),
                "gate_score": delta(score, old_detector.confidence),
                "gate_observed": delta(visible, old_detector.observed),
                "ball_points": compare_ball_points(point, old_points),
            }

    context = AssemblyContext(source)
    ball_input = BallTriangulationInputAssembler(0.0).assemble(context, dependencies)
    ball_output = BallTriangulationModule(
        source.camera_ids,
        reprojection_px=runtime.ball_reprojection_px,
        min_frames=runtime.ball_min_frames,
    ).process(ball_input)
    ball = ball_output.ball
    if ball is None:
        raise ValueError(
            "The fixed camera geometry must produce a ball result with validity"
        )
    scene_input = SceneAssemblyInputAssembler(runtime.human_vis_threshold).assemble(
        context,
        {
            **dependencies,
            "identities": fixed["player_association"],
            "skeleton": fixed["player_triangulation"],
            "placement": fixed["body_placement"],
            "ball": ball_output,
            **{
                f"pose_{camera}": fixed[f"pose_estimation/{camera}"]
                for camera in source.camera_ids
            },
        },
    )
    scene = SceneAssemblyModule(
        source.camera_ids,
        require_people=runtime.enabled["player_reconstruction"],
        require_ball=runtime.enabled["ball_reconstruction"],
        require_body=runtime.enabled["gvhmr"],
    ).process(scene_input)
    for field in fields(scene):
        if field.name not in (*SCENE_BALL_FIELDS, "metadata"):
            np.testing.assert_array_equal(
                getattr(scene, field.name),
                getattr(saved.scene, field.name),
                err_msg=field.name,
            )
    for key in ("camera_ids", "court_reference", "track_ids"):
        if scene.metadata[key] != saved.scene.metadata[key]:
            raise ValueError(f"Fixed scene metadata changed: {key}")
    if compare_reference:
        old, _ = saved.load("ball_triangulation", BallTriangulationOutput)
        assert old.ball is not None
        reference["ball_triangulation"] = {
            field.name: delta(
                getattr(ball.trajectory, field.name),
                getattr(old.ball.trajectory, field.name),
            )
            for field in fields(ball.trajectory)
        }
        reference["scene_assembly"] = {
            key: delta(getattr(scene, key), getattr(saved.scene, key))
            for key in SCENE_BALL_FIELDS
        }
    trajectory = ball.trajectory
    valid = trajectory.valid
    reprojection = np.full((4, count), np.nan, dtype=np.float32)
    for i, camera in enumerate(fixed["camera_alignment"].geometry.cameras):
        projected, front = camera.project(trajectory.positions)
        use = valid & ball.visibility[i] & front
        residual = np.linalg.norm(projected[use] - ball.uv_px[i, use], axis=-1)
        np.testing.assert_allclose(
            residual, trajectory.reprojection_px[i, use], atol=0.005, rtol=1e-5
        )
        reprojection[i, use] = residual
    uv, score, visible = (np.stack(values) for values in zip(*accepted, strict=True))
    arrays = dict(
        **raw,
        uv_px=uv,
        score=score,
        visible=visible,
        ball_3d=trajectory.positions,
        ball_3d_valid=valid,
        ball_rejection_code=trajectory.reasons.astype(np.float32),
        reprojection_px=reprojection,
    )
    validate_arrays(arrays, count)
    saved.verify_index()
    return arrays, reference


def validate_arrays(arrays, count):
    shapes = {
        "uv_px": (4, count, 2),
        "raw_uv_px": (4, count, 2),
        "score": (4, count),
        "raw_score": (4, count),
        "visible": (4, count),
        "ball_3d": (count, 3),
        "ball_3d_valid": (count,),
        "ball_rejection_code": (count,),
        "reprojection_px": (4, count),
    }
    if set(arrays) != set(shapes):
        raise ValueError("Unexpected NPZ fields")
    for name, shape in shapes.items():
        dtype = np.bool_ if name in {"visible", "ball_3d_valid"} else np.float32
        if arrays[name].shape != shape or arrays[name].dtype != dtype:
            raise ValueError(f"Wrong NPZ shape/dtype for {name}")
    rejected = ~arrays["visible"]
    absent = ~arrays["ball_3d_valid"]
    codes = arrays["ball_rejection_code"]
    if (
        np.any(arrays["uv_px"][rejected])
        or np.any(arrays["score"][rejected])
        or np.any(arrays["ball_3d"][absent])
        or not np.isnan(arrays["reprojection_px"][:, absent]).all()
        or not np.array_equal(codes == PointRejection.VALID, ~absent)
        or not np.isin(codes, [int(code) for code in PointRejection]).all()
        or not np.isfinite(arrays["ball_3d"]).all()
    ):
        raise ValueError("Invalid masking in comparison NPZ")


def summary(model, clip, arrays):
    valid, pts = arrays["ball_3d_valid"], arrays["ball_3d"]
    bad = (np.abs(pts[:, 0]) > 9) | (np.abs(pts[:, 1]) > 18)
    reproj = arrays["reprojection_px"][np.isfinite(arrays["reprojection_px"])]
    return {
        "model": model["name"],
        "clip_id": clip["clip_id"],
        "num_frames": clip["num_frames"],
        **{
            f"{camera}_visible_rate": float(rate)
            for camera, rate in zip(
                clip["camera_ids"], arrays["visible"].mean(axis=1), strict=True
            )
        },
        "valid_3d_frames": int(valid.sum()),
        "valid_3d_rate": float(valid.mean()),
        **{
            f"rejection_{code.name.lower()}": int(
                np.count_nonzero(arrays["ball_rejection_code"] == code)
            )
            for code in PointRejection
            if code != PointRejection.VALID
        },
        "outside_court_count": int(np.count_nonzero(valid & bad)),
        "median_reprojection_px": float(np.median(reproj)) if reproj.size else None,
    }


def replay_model(root, plan, model, *, preflight=False):
    name = model["name"]
    baseline = name == "pipeline_x3d_dpt"
    runtime = scene_configs()["pipeline_x3d_dpt"]
    saved_settings = read(HERE.parent / "run-i1752-shared-fp16-20261118/entry.json")[
        "processing_settings"
    ]
    for key in (
        "ball_detection",
        "ball_smoothing",
        "ball_reconstruction",
        "person_observations",
        "gvhmr",
    ):
        if json_value(runtime.processing_settings[key]) != saved_settings[key]:
            raise ValueError(f"Replay config differs from the source run at {key}")
    rows, references = [], {}
    for clip in plan["clips"]:
        clip_id = clip["clip_id"]
        saved = SavedClip(clip)
        if saved.digest != clip["store_index_sha256"]:
            raise ValueError("The source index was modified after preflight")
        if preflight:
            raw_uv_score = [
                baseline_detections(saved, camera)[1:]
                for camera in saved.source.camera_ids
            ]
            uv, score = (np.stack(items) for items in zip(*raw_uv_score, strict=True))
            raw = {"raw_uv_px": uv, "raw_score": score}
        else:
            with np.load(
                root / "work" / name / f"{clip_id}.npz", allow_pickle=False
            ) as data:
                raw = {key: data[key] for key in data.files}
        arrays, reference = replay(
            saved,
            runtime,
            model,
            raw,
            compare_reference=baseline,
            use_saved_gate=preflight,
        )
        if preflight:
            for block in ("scene_assembly", "ball_triangulation"):
                if any(item["differing_values"] for item in reference[block].values()):
                    raise ValueError(
                        f"Saved detector CPU replay does not reproduce {block}: {reference[block]}"
                    )
        else:
            output = root / name / f"{clip_id}.npz"
            output.parent.mkdir(exist_ok=True)
            save_npz(output, arrays)
            with np.load(output, allow_pickle=False) as published:
                validate_arrays(
                    {key: published[key] for key in published.files}, clip["num_frames"]
                )
        rows.append(summary(model, clip, arrays))
        references[clip_id] = {
            "load": saved.records,
            "non_ball_scene_equal": True,
            "parity": reference,
        }
        print(name, clip_id, rows[-1], flush=True)
    write(
        root
        / "work"
        / ("preflight.json" if preflight else f"{name}/reconstruction.json"),
        {"summary": rows, "clips": references},
    )
    return rows, references
