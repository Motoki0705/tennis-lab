"""Read-only input preflight for the recorded clip000 strategy experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from src.tasks.ball_detection.annotation.video.agent.contracts import AnnotationPolicy
from src.tasks.ball_detection.annotation.video.contracts import (
    BallTarget,
    FrameIdentity,
    VideoAnnotationInput,
    VideoIdentity,
)
from src.tasks.base.annotation.contracts import FrameRange
from src.tasks.base.annotation.media import probe_video
from src.tennis_scene.ground_truth import (
    ReferenceIdentity,
    load_user_reference,
    sha256_file,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    expected = json.loads(
        (root / "knowledge/runs/run-ball3d-evaluation-foundation-20261103/inputs.json")
        .read_text()
    )
    policies = tuple(
        AnnotationPolicy.model_validate(row)
        for row in json.loads(Path(__file__).with_name("policies.json").read_text())
    )
    report = {
        "schema_version": 1,
        "checked_at": datetime.now(UTC).isoformat(),
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "policy_count": len(policies),
        "comparison_runs_started": 0,
        "acceptance_started": False,
        "videos": [],
        "reference": {},
        "fixed_bundle": {},
        "comparison_inputs_verified": False,
        "acceptance_inputs_verified": False,
    }
    errors = []
    inputs = []
    for camera, identity in expected["videos"].items():
        path = args.video_dir / (camera + ".mp4")
        result = {"camera_id": camera, "file_name": path.name, "status": "missing"}
        if path.is_file():
            try:
                digest = sha256_file(path)
                if digest != identity["sha256"]:
                    raise ValueError("video bytes differ from the pinned identity")
                timeline = probe_video(path)
                if (timeline.width, timeline.height, len(timeline.pts)) != (
                    identity["width"], identity["height"], identity["num_frames"]
                ) or abs(float(timeline.rate) - identity["fps"]) > 1e-6:
                    raise ValueError("video metadata differs from the pinned identity")
                task_input = VideoAnnotationInput(
                    source=VideoIdentity(
                        recording_id="meiji_3cam/video_000",
                        camera_id=camera,
                        file_name=path.name,
                        media_sha256=digest,
                        size_bytes=path.stat().st_size,
                        width=timeline.width,
                        height=timeline.height,
                        frame_count=len(timeline.pts),
                        time_base=str(timeline.time_base),
                        nominal_fps=str(timeline.rate),
                    ),
                    target=BallTarget(target_semantics="active_ball", track_id="active_ball"),
                    frame_range=FrameRange(start=0, stop=64),
                    frames=tuple(
                        FrameIdentity(frame_index=i, pts=timeline.pts[i])
                        for i in range(64)
                    ),
                )
                inputs.append(task_input)
                result.update(status="verified", input=task_input.model_dump(mode="json"))
            except (OSError, ValueError) as error:
                result.update(status="invalid", error=str(error))
        if result["status"] != "verified":
            errors.append(camera + ": " + result["status"])
        report["videos"].append(result)

    base = args.asset_root
    reference_path = base / "human_annotations/ground_truth_v1/ground_truth.json"
    sources = [
        {
            "relative_path": row["path"],
            "present": (reference_path.parent / row["path"]).is_file(),
        }
        for row in expected["reference_source_files"]
    ]
    report["reference"] = {
        "json_present": reference_path.is_file(), "source_files": sources,
        "status": "missing", "sha256": None,
    }
    if reference_path.is_file() and len(inputs) == 3:
        try:
            identity = ReferenceIdentity(
                clip_id="meiji_3cam/video_000/clip_000",
                camera_ids=tuple(expected["videos"]),
                num_frames=1010,
                fps=59.94006,
                source_size_wh=(1920, 1080),
                video_sha256={c: v["sha256"] for c, v in expected["videos"].items()},
            )
            truth = load_user_reference(
                reference_path, expected=identity,
                expected_sha256=expected["annotations_sha256"],
            )
            report["reference"].update(status="verified", sha256=truth.sha256)
        except (OSError, ValueError) as error:
            report["reference"].update(status="invalid", error=str(error))
    if report["reference"]["status"] != "verified":
        errors.append("reference JSON and source CSVs are not verified")

    # Fixed geometry is needed by acceptance, not by the 2D strategy comparison.
    comparison_errors = list(errors)

    bundle_root = base / "fixed_input_bundle_v1"
    report["fixed_bundle"] = {
        "json_present": (bundle_root / "bundle.json").is_file(),
        "scene_present": (bundle_root / "scene.json").is_file(),
        "status": "missing", "sha256": None,
    }
    if (bundle_root / "bundle.json").is_file():
        try:
            from src.tennis_scene.evaluation.ball3d.bundle import load_bundle

            bundle = load_bundle(bundle_root, expected_sha256=expected["bundle_sha256"])
            if bundle.manifest["clip_id"] != "meiji_3cam/video_000/clip_000":
                raise ValueError("fixed bundle belongs to another clip")
            if tuple(bundle.source.camera_ids) != tuple(expected["videos"]):
                raise ValueError("fixed bundle has different camera axes")
            for video in bundle.source.videos:
                wanted = expected["videos"][video.camera_id]
                if (video.sha256, video.width, video.height, video.num_frames) != (
                    wanted["sha256"], wanted["width"], wanted["height"], wanted["num_frames"]
                ) or abs(video.fps - wanted["fps"]) > 1e-6:
                    raise ValueError("fixed bundle has another video identity")
            report["fixed_bundle"].update(
                status="verified_evaluation_bundle", sha256=bundle.manifest_sha256
            )
        except (OSError, ValueError) as error:
            report["fixed_bundle"].update(status="invalid", error=str(error))
    if report["fixed_bundle"]["status"] != "verified_evaluation_bundle":
        errors.append("original fixed bundle is not verified")

    report["comparison_input_errors"] = comparison_errors
    report["acceptance_input_errors"] = errors
    report["comparison_inputs_verified"] = not comparison_errors
    report["acceptance_inputs_verified"] = not errors
    # This script never creates resource scopes, runtimes, tasks or workers.
    # Passing the input gate still requires an explicitly approved runtime.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "comparison_input_errors": comparison_errors,
        "acceptance_input_errors": errors,
        "comparison_inputs_verified": not comparison_errors,
    }))
    return 2 if comparison_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
