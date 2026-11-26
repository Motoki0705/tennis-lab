"""Build and publish the read-only Meiji player evaluation benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from audit import (
    MARGINS,
    compare_mesh_boxes,
    diagnostic_arrays,
    overlap_and_size,
    projection_check,
    quality_counts,
    render_overlay,
    smoke_predictions,
)
from inputs import (
    CAMERA_IDS,
    CLIPS,
    SOURCE_BENCHMARK,
    SOURCE_SHA256,
    ClipInputs,
    load_inputs,
    source_records,
)

from src.submodules.smpl_assets import SMPL_ASSET_SOURCES
from src.tennis_scene.evaluation.player_pose.bundle import (
    BODY_INPUT_SCHEMA,
    MANIFEST_SCHEMA,
    load_body_input_manifest,
    load_reference_frames,
)
from src.tennis_scene.evaluation.player_pose.reference import (
    BoxQuality,
    JointQuality,
    ReferencePolicy,
    load_regressors,
    project_placed_smpl,
    qualify_reference,
)
from src.tennis_scene.evaluation.player_pose.scoring import (
    FrameKey,
    Predictions,
    describe,
    score_players,
)
from src.tennis_scene.pipeline.frame_sampling import sampled_frame_indices
from src.utils.checksum import dual_sha256
from src.utils.schema.player import COCO_KP_NAMES

BENCHMARK = "meiji_4cam/multiview-player-evaluation-v1"
BODY_ID = "meiji_4cam/multiview-player-evaluation-v1/body-recovery-inputs"
POLICY = ReferencePolicy(
    box_margin=0.10,
    min_box_joints=12,
    min_supporting_views=2,
    occlusion_depth_gap_m=0.25,
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _file_record(root: Path, path: Path) -> dict[str, str]:
    return {"path": path.relative_to(root).as_posix(), "sha256": dual_sha256(path)}


def _git_revision() -> str:
    root = Path(__file__).resolve().parents[3]
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    status = subprocess.check_output(
        [
            "git",
            "status",
            "--porcelain",
            "--",
            "src",
            str(Path(__file__).resolve().parent),
        ],
        cwd=root,
        text=True,
    )
    if status:
        raise ValueError(
            f"Commit the generator and source before assigning a revision:\n{status}"
        )
    return revision


def _axes(inputs: ClipInputs) -> dict[str, np.ndarray]:
    return {
        "player_ids": inputs.player_ids,
        "camera_ids": np.asarray(CAMERA_IDS),
        "evaluation_frames": inputs.evaluation_frames,
    }


def _fixed_body_file(root: Path, path: Path, inputs: ClipInputs) -> dict[str, Any]:
    times = np.arange(inputs.scene.num_frames) / inputs.scene.fps
    np.savez_compressed(
        path,
        **_axes(inputs),
        **inputs.request_data,
        source_frame=np.arange(inputs.scene.num_frames, dtype=np.int64),
        clip_time_sec=times,
        video_time_sec=inputs.start_seconds + times,
        K=np.stack([cam.intrinsic for cam in inputs.cameras]),
        R=np.stack([cam.rotation for cam in inputs.cameras]),
        t=np.stack([cam.translation for cam in inputs.cameras]),
    )
    segment = inputs.request_data["segment_index"]
    observed = inputs.request_data["selected_view_observed"]
    return {
        "clip_id": inputs.clip_id,
        "player_ids": inputs.player_ids.tolist(),
        "camera_ids": CAMERA_IDS,
        "evaluation_frames": inputs.evaluation_frames.tolist(),
        "source_frames": inputs.scene.num_frames,
        "width": inputs.scene.width,
        "height": inputs.scene.height,
        "fps": inputs.scene.fps,
        "clip_start_video_seconds": inputs.start_seconds,
        "videos": inputs.record["cameras"],
        "scene_exports": inputs.record["exports"],
        "source_components": inputs.source_components,
        "fixed_inputs": _file_record(root, path),
        "players": [
            {
                "player_id": int(player),
                "selected_camera": CAMERA_IDS[
                    inputs.request_data["selected_view_index"][p]
                ],
                "request_frames": int(inputs.inferred[p].sum()),
                "segments": int(segment[p].max()) + 1,
                "missing_request_frames": int((segment[p] < 0).sum()),
                "request_without_observed_box_frames": int(
                    (inputs.inferred[p] & ~observed[p]).sum()
                ),
            }
            for p, player in enumerate(inputs.player_ids)
        ],
    }


def build(stage: Path, source: Path, assets: Path, revision: str) -> str:
    cv2.setNumThreads(1)
    regressors = load_regressors(assets)
    cohort, body_clips, counts = [], [], []
    per_clip: dict[str, Any] = {}
    residuals: dict[str, list[np.ndarray]] = {}
    smoke: dict[FrameKey, Predictions] = {}
    fixed_frame_check = None
    for record in source_records(source):
        inputs = load_inputs(source, record)
        print(
            f"{inputs.clip_id}: verified source and production body segments",
            flush=True,
        )
        scene = inputs.scene
        clip_id = inputs.clip_id
        name = clip_id.split("/")[-1]
        directory = stage / "clips" / name
        directory.mkdir(parents=True)
        inferred = inputs.inferred
        frames = inputs.evaluation_frames
        grid = sampled_frame_indices(
            scene.num_frames, scene.fps, max_frames=scene.num_frames
        )
        if not len(frames) or not np.isin(frames, grid).all():
            raise ValueError(
                f"Saved GVHMR indices are outside the production grid for {clip_id}"
            )

        observation = inputs.observed_keypoints
        support = observation[..., 2] > 0
        assert (
            scene.player_smpl_valid is not None
            and scene.player_rejection_code is not None
        )
        geom = project_placed_smpl(scene, inputs.cameras, inferred, regressors)
        quality = qualify_reference(
            geom, inferred, scene.player_smpl_valid, support, inputs.size, POLICY
        )
        diagnostics = diagnostic_arrays(inputs, geom, observation)
        if name == "clip_003":
            fixed_frame_check = projection_check(inputs, geom, observation, regressors)
            print(
                json.dumps({"projection_check": fixed_frame_check}, allow_nan=False),
                flush=True,
            )
        mesh_iou = compare_mesh_boxes(geom, inputs.size, POLICY.min_box_joints)
        distributions = {
            **diagnostics,
            **{f"mesh_bbox_iou_margin_{m}": v for m, v in mesh_iou.items()},
        }
        per_clip[clip_id] = {key: describe(val) for key, val in distributions.items()}
        per_clip[clip_id]["by_camera"] = {
            camera: {
                key: describe(diagnostics[key][:, view])
                for key in ("observation_error_px", "triangulation_error_px")
            }
            for view, camera in enumerate(CAMERA_IDS)
        }
        for key, val in distributions.items():
            residuals.setdefault(key, []).append(val.ravel())
        counts.extend(quality_counts(inputs, quality))
        stats = overlap_and_size(inputs, observation)
        reference_file = directory / "pseudo_gt.npz"
        np.savez_compressed(
            reference_file,
            **_axes(inputs),
            **diagnostics,
            joints_court_m=geom.joints_m,
            keypoints_px=geom.pixels,
            cheirality=geom.front,
            in_image=geom.inside,
            depth_m=geom.depth_m,
            mesh_boxes_xyxy=geom.mesh_boxes,
            gvhmr_inferred=inferred,
            scene_smpl_valid=scene.player_smpl_valid,
            root_rejection_code=scene.player_rejection_code,
            boxes_xyxy=quality.boxes,
            box_reasons=quality.box_reasons,
            joint_reasons=quality.joint_reasons,
            likely_inter_player_occlusion=quality.likely_occluded,
            same_view_observed=support,
            joint_support_views=quality.joint_view_support,
            person_support_views=quality.person_view_support,
            vitpose_keypoints_xyc=observation,
        )
        body_clips.append(
            _fixed_body_file(stage, directory / "body_recovery_inputs.npz", inputs)
        )
        smoke.update(smoke_predictions(inputs, observation, POLICY.box_margin))

        overlay = []
        for view, camera in enumerate(CAMERA_IDS):
            video = directory / f"{camera}.mp4"
            audit = render_overlay(inputs, view, geom, quality, observation, video)
            overlay.append(
                {
                    "camera_id": camera,
                    **audit,
                    "video": _file_record(stage, video),
                    "representative_frame": _file_record(
                        stage, video.with_suffix(".png")
                    ),
                }
            )
            print(
                f"{clip_id}/{camera}: wrote {audit['video_frames']} audited frames",
                flush=True,
            )
        cohort.append(
            {
                "clip_id": clip_id,
                "source_frames": scene.num_frames,
                "fps": scene.fps,
                "width": scene.width,
                "height": scene.height,
                "camera_ids": CAMERA_IDS,
                "evaluation_frames": frames.tolist(),
                "player_ids": inputs.player_ids.tolist(),
                "gvhmr_frames_by_player": dict(
                    zip(
                        inputs.player_ids.astype(str).tolist(),
                        inferred.sum(1).tolist(),
                        strict=True,
                    )
                ),
                "all_players_inferred_frames": int(inferred.all(0).sum()),
                "production_grid_without_any_prediction": np.setdiff1d(
                    grid, frames
                ).tolist(),
                "selection": {"reason": CLIPS[name], **stats},
                "source_videos": record["cameras"],
                "source_scene_index": record["scene"],
                "pseudo_ground_truth": _file_record(stage, reference_file),
                "overlays": overlay,
            }
        )
    if (
        sum(clip["source_frames"] for clip in cohort) != 3414
        or fixed_frame_check is None
    ):
        raise ValueError("Common cohort differs from the six agreed source clips")

    count_file = stage / "quality_counts.csv"
    with count_file.open("w", encoding="utf-8", newline="") as handle:
        fields = list(dict.fromkeys(key for row in counts for key in row))
        writer = csv.DictWriter(handle, fieldnames=fields, restval=0)
        writer.writeheader()
        writer.writerows(counts)
    audit_file = stage / "projection_audit.json"
    _write_json(
        audit_file,
        {
            "fixed_frame_check": fixed_frame_check,
            "overall": {
                key: describe(np.concatenate(val)) for key, val in residuals.items()
            },
            "by_clip": per_clip,
            "limitation": "Existing ViTPose and triangulation contribute to scene placement and observation-based quality. "
            "The residuals and smoke scores do not measure independent ground truth.",
        },
    )
    body_manifest = stage / "body_recovery_manifest.json"
    source_pin = {"benchmark_id": SOURCE_BENCHMARK, "manifest_sha256": SOURCE_SHA256}
    _write_json(
        body_manifest,
        {
            "schema": BODY_INPUT_SCHEMA,
            "manifest_id": BODY_ID,
            "cohort_id": BENCHMARK,
            "source_benchmark": source_pin,
            "generator_commit": revision,
            "joint_order": COCO_KP_NAMES,
            "clips": body_clips,
        },
    )
    manifest = stage / "manifest.json"
    _write_json(
        manifest,
        {
            "schema": MANIFEST_SCHEMA,
            "benchmark_id": BENCHMARK,
            "source_benchmark": source_pin,
            "generator_commit": revision,
            "joint_order": COCO_KP_NAMES,
            "assets": {
                name: {
                    "source": SMPL_ASSET_SOURCES[name].source_url,
                    "sha256": SMPL_ASSET_SOURCES[name].sha256,
                }
                for name in ("smpl_coco17_regressor", "smpl_neutral_joint_regressor")
            },
            "reference_policy": asdict(POLICY),
            "box_quality_codes": {code.name.lower(): int(code) for code in BoxQuality},
            "joint_quality_codes": {
                code.name.lower(): int(code) for code in JointQuality
            },
            "bbox_margin_audit": MARGINS,
            "source_frames": sum(clip["source_frames"] for clip in cohort),
            "evaluation_frames": sum(len(clip["evaluation_frames"]) for clip in cohort),
            "evaluation_images": sum(
                len(clip["evaluation_frames"]) * len(CAMERA_IDS) for clip in cohort
            ),
            "body_recovery_manifest": _file_record(stage, body_manifest),
            "clips": cohort,
            "quality_counts": _file_record(stage, count_file),
            "projection_audit": _file_record(stage, audit_file),
            "vitpose_smoke_report": "vitpose_smoke.json",
        },
    )
    sha256 = dual_sha256(manifest)
    refs = load_reference_frames(stage, expected_sha256=sha256)
    fixed = load_body_input_manifest(stage, expected_sha256=sha256)
    if (
        len(fixed["clips"]) != len(cohort)
        or len(refs) != len(smoke)
        or refs.keys() != smoke.keys()
    ):
        raise ValueError("Scoring and body inputs do not cover the same fixed cohort")
    report = score_players(refs, smoke)
    detection = report["overall"]["detection"]
    pose = report["overall"]["pose"]["with_detection_coverage"]
    if (
        detection["references"] != sum(row["box_eligible"] for row in counts)
        or pose["eligible_joints"] != sum(row["joint_eligible"] for row in counts)
        or not (
            0 < detection["recall_by_iou"]["0.50"] <= 1 and 0 < pose["oks"]["mean"] <= 1
        )
    ):
        raise ValueError("ViTPose smoke counts or scores are inconsistent")
    _write_json(
        stage / "vitpose_smoke.json",
        {
            "manifest_sha256": sha256,
            "inputs": "Stored ViTPose poses paired with their expanded observed joint envelopes as candidate boxes. "
            "Detection scores are max observed joint confidence. The smoke metrics do not evaluate DINO detection.",
            **report,
        },
    )
    (stage / "README.md").write_text(
        (Path(__file__).resolve().parent / "bundle-layout.md").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    return sha256


def _publish(source: Path, stage: Path, output: Path) -> None:
    files = sorted(path for path in stage.rglob("*") if path.is_file())
    with (stage / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for path in files:
            handle.write(f"{dual_sha256(path)}  {path.relative_to(stage).as_posix()}\n")
    subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=stage, check=True)
    for path in stage.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    stage.chmod(0o555)
    if output.exists() or output.is_symlink() or output.parent != source.parent:
        raise FileExistsError(
            f"Output must be a new sibling of the source benchmark: {output}"
        )
    stage.rename(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    output = args.output.parent.resolve(strict=True) / args.output.name
    if output.exists() or output.is_symlink() or output.parent != source.parent:
        raise FileExistsError(
            f"Output must be a new sibling of the source benchmark: {output}"
        )
    revision = _git_revision()
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        digest = build(stage, source, args.assets.resolve(strict=True), revision)
        _publish(source, stage, output)
    finally:
        if stage.exists():
            stage.chmod(0o755)
            for path in stage.rglob("*"):
                if path.is_dir():
                    path.chmod(0o755)
            shutil.rmtree(stage)
    print(
        json.dumps(
            {
                "benchmark_id": BENCHMARK,
                "manifest_sha256": digest,
                "directory": str(output),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
