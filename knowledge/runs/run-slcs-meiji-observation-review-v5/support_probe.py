"""CPU-only audit of production singles association; never loads a model."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.tennis_scene.reference_pipeline import reconstruction
from src.utils.checksum import dual_sha256


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def counts(keypoints: np.ndarray) -> list[int]:
    gates = (keypoints[..., [5, 6, 11, 12], 2] >= 0.3).all(axis=-1)
    return [int(x) for x in (gates.sum(axis=1) >= 2).sum(axis=1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observation-report", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be empty")
    output: Path = args.output
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    root = Path.cwd().resolve()
    source = Path(reconstruction.__file__).resolve()
    report: dict[str, Any] = {
        "status": "running",
        "cwd": str(root),
        "git_root": git("rev-parse", "--show-toplevel"),
        "commit": git("rev-parse", "HEAD"),
        "production_source": str(source),
        "command": sys.argv,
        "device": "cpu",
        "clips": [],
        "definition": "After production fixed singles association, cameras with all shoulders 5,6 and hips 11,12 >= 0.3; at least two cameras per frame. Not final teacher coverage: excludes geometry, reprojection, speed and quality weights.",
    }
    files: set[Path] = {
        source,
        Path(__file__).resolve(),
        root / "src/utils/checksum.py",
        root / "src/tennis_scene/reference_pipeline/observations.py",
    }
    before: dict[str, str] = {}
    try:
        if source != root / "src/tennis_scene/reference_pipeline/reconstruction.py":
            raise RuntimeError("Wrong production source")
        config_path = (
            args.project_root
            / "outputs/tennis_scene/generate/meiji_rgb_v8/s42-001/config.yaml"
        )
        files.add(config_path)
        config = yaml.safe_load(config_path.read_text())
        if (
            config["view_half_turns"] != [False, False, True]
            or config["reference_camera"] != "cam0"
        ):
            raise ValueError("Unexpected reference declarations")
        report["view_half_turns"] = config["view_half_turns"]
        report["reference_camera"] = config["reference_camera"]
        input_report = args.observation_report.resolve()
        files.add(input_report)
        observation_report = json.loads(input_report.read_text())
        if observation_report["status"] != "done":
            raise ValueError("Incomplete observation report")
        clip_ids = [c["clip_id"] for c in observation_report["clips"]]
        if clip_ids != [f"video_002/clip_{n:03d}" for n in (7, 8, 9)]:
            raise ValueError("Expected three selected video_002 clips")
        for reviewed_clip in observation_report["clips"]:
            for camera_report in reviewed_clip["cameras"]:
                files.add(root / camera_report["contact_sheet"])
                files.update(Path(p) for p in camera_report["input_sha256"])
        observed = args.project_root / "outputs" / config["observation_directory"]
        dataset = args.project_root / "data" / config["dataset_directory"] / "videos"
        manifests: dict[str, Path] = {}
        for clip_id in clip_ids:
            video, clip = clip_id.split("/")
            manifests[clip_id] = dataset / video / "clips" / clip / "clip.json"
            files.add(manifests[clip_id])
            files.add(observed / clip_id / "court.npz")
            files.update(observed / clip_id / f"cam{i}_people.npz" for i in range(3))
        before = {str(p): dual_sha256(p) for p in sorted(files)}
        for clip_id in clip_ids:
            clip = json.loads(manifests[clip_id].read_text())
            if clip["camera_ids"] != ["cam0", "cam1", "cam2"]:
                raise ValueError(f"{clip_id}: camera order mismatch")
            directory = observed / clip_id
            with np.load(directory / "court.npz") as court:
                court_shapes = {k: list(court[k].shape) for k in court.files}
                hs = court["homographies"]
                if hs.shape != (3, 3, 3) or not all(
                    np.isfinite(court[k]).all() for k in court.files
                ):
                    raise ValueError("Invalid court arrays")
            canonical, assignments = reconstruction.associate_people(
                clip, directory, hs, config["view_half_turns"]
            )
            expected, raw, orders, ids_by_camera = [], [], {}, {}
            local_diagnostics = []
            for camera in clip["camera_ids"]:
                with np.load(directory / f"{camera}_people.npz") as data:
                    kp, ids, supported = (
                        data["keypoints"],
                        data["track_ids"],
                        data["pose_supported_mask"],
                    )
                if (
                    kp.shape != (2, clip["num_frames"], 17, 3)
                    or not np.isfinite(kp).all()
                ):
                    raise ValueError("Invalid keypoint shape/values")
                if (
                    supported.shape != kp.shape[:2]
                    or supported.dtype != np.bool_
                    or ids.shape != (2,)
                    or len(set(ids.tolist())) != 2
                ):
                    raise ValueError("Invalid support mask/track IDs")
                kp[..., 2] = np.where(supported[..., None], kp[..., 2], 0)
                selected_ids = assignments[camera]["track_ids_near_far_cam0"]
                order = [ids.tolist().index(track_id) for track_id in selected_ids]
                wanted = [1, 0] if camera == "cam2" else [0, 1]
                if order != wanted:
                    raise ValueError(
                        f"{clip_id}/{camera}: unexpected assignment {order}, expected {wanted}"
                    )
                for local in range(2):
                    missing = np.flatnonzero(~supported[local])
                    intervals = []
                    if len(missing):
                        groups = np.split(
                            missing, np.flatnonzero(np.diff(missing) > 1) + 1
                        )
                        intervals = [[int(g[0]), int(g[-1])] for g in groups]
                    torso = kp[local, :, [5, 6, 11, 12], 2]
                    local_diagnostics.append(
                        {
                            "camera": camera,
                            "raw_local_player": local,
                            "raw_track_id": int(ids[local]),
                            "canonical_player": order.index(local),
                            "pose_supported_false_intervals_inclusive": intervals,
                            "pose_supported_false_frames": int(len(missing)),
                            "pose_supported_fraction": float(supported[local].mean()),
                            "torso_confidence_min_after_mask": float(torso.min()),
                            "torso_supported_fraction": float(
                                (torso >= 0.3).all(axis=0).mean()
                            ),
                        }
                    )
                orders[camera], ids_by_camera[camera] = order, ids.tolist()
                raw.append(kp)
                expected.append(kp[order])
            np.testing.assert_array_equal(canonical, np.stack(expected, axis=1))
            raw_counts, canonical_counts = (
                counts(np.stack(raw, axis=1)),
                counts(canonical),
            )
            report["clips"].append(
                {
                    "clip_id": clip_id,
                    "camera_ids": clip["camera_ids"],
                    "frames": clip["num_frames"],
                    "court_keys_shapes": court_shapes,
                    "raw_track_ids": ids_by_camera,
                    "canonical_raw_indices": orders,
                    "assignments": assignments,
                    "array_equal_after_support_mask_and_reorder": True,
                    "shape_finite_checks_passed": True,
                    "raw_two_view_support_frames": raw_counts,
                    "canonical_two_view_support_frames": canonical_counts,
                    "canonical_two_view_support_fraction": [
                        n / clip["num_frames"] for n in canonical_counts
                    ],
                    "delta_frames_canonical_minus_raw": [
                        c - r for c, r in zip(canonical_counts, raw_counts, strict=True)
                    ],
                    "changed_players": [
                        i
                        for i, (c, r) in enumerate(
                            zip(canonical_counts, raw_counts, strict=True)
                        )
                        if c != r
                    ],
                    "local_diagnostics": local_diagnostics,
                }
            )
        for reviewed_clip in observation_report["clips"]:
            if (
                before[str(manifests[reviewed_clip["clip_id"]])]
                != reviewed_clip["manifest_sha256"]
            ):
                raise ValueError("Observation manifest SHA mismatch")
            for camera_report in reviewed_clip["cameras"]:
                for path, digest in camera_report["input_sha256"].items():
                    if before[path] != digest:
                        raise ValueError(f"Observation input SHA mismatch: {path}")
                audited = report["clips"][clip_ids.index(reviewed_clip["clip_id"])][
                    "local_diagnostics"
                ]
                for player in camera_report["players"]:
                    local = next(
                        d
                        for d in audited
                        if d["camera"] == camera_report["camera"]
                        and d["raw_local_player"] == player["player"]
                    )
                    if (
                        local["pose_supported_fraction"]
                        != player["pose_supported_fraction"]
                        or local["torso_supported_fraction"]
                        != player["hips_and_shoulders_confident_fraction"]
                    ):
                        raise ValueError(
                            "Per-view support differs from observation report"
                        )
        report["observation_report_hashes_and_support_match"] = True
        report["status"] = "done"
    except Exception as exc:
        report["status"], report["error"] = "failed", repr(exc)
        raise
    finally:
        after = {str(p): dual_sha256(p) for p in sorted(files)}
        report["pre_dual_sha256"], report["post_dual_sha256"] = before, after
        report["input_and_source_hashes_unchanged"] = before == after
        if before != after:
            report["status"] = "failed"
            report["hash_error"] = (
                "Input/source changed or incomplete pre-hash inventory"
            )
        with output.open("x") as handle:
            handle.write(json.dumps(report, indent=2) + "\n")
    if report["status"] != "done":
        raise RuntimeError("Audit failed")


if __name__ == "__main__":
    main()
