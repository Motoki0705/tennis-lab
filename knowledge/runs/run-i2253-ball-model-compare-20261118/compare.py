"""Run the fixed-geometry #2253 comparison through the shared GPU queue."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

from infer import HERE, REPO, read, write

sys.path.append(str(HERE.parent / "run-i1751-runtime-gpu-a5"))
from health_guard import capture_diagnostics, run_variant, utc  # noqa: E402

BASE = "d392dd375"
CLIPS = ("clip_000", "clip_027", "clip_037")
CAMERAS = ("cam0", "cam1", "cam2", "cam3")
SOURCE_RUN = "video-batch-shared-fp16-20261118"


def git(*args):
    return subprocess.check_output(["git", "-C", str(REPO), *args]).decode().strip()


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require_hash(path, expected):
    if digest(path) != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")


def label_audit(dataset, clips):
    migration = HERE.parent / "run-ball-v3-migration-20261113"
    with (migration / "correspondence.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    seen = {
        clip["clip_id"]: sorted(
            path.relative_to(dataset).as_posix()
            for path in (dataset / f"videos/video_000/clips/{clip['clip_id']}").rglob(
                "*.json"
            )
        )
        for clip in clips
    }
    if any(
        len(paths) != 1 or not paths[0].endswith("/clip.json")
        for paths in seen.values()
    ):
        raise ValueError(
            "The clip has annotation files requiring identity and score verification"
        )
    matches = [
        row
        for row in rows
        if "meiji_4cam" in row["recording_id"] or "meiji_4cam" in row["original_path"]
    ]
    if matches:
        raise ValueError(
            "Found four-camera records to verify as potential ground truth"
        )
    return {
        "available": False,
        "matched_camera_clips": 0,
        "local_clip_json_files": seen,
        "v3_correspondence_rows_checked": len(rows),
        "v3_correspondence_sha256": digest(migration / "correspondence.csv"),
        "v3_recordings": sorted({row["recording_id"] for row in rows}),
        "note": "Four-camera clips have only source metadata. The migrated labels are bound to meiji_3cam media.",
    }


def prepare_clips(prior, dataset):
    source_inputs = read(
        HERE.parent / "run-i1752-shared-fp16-20261118/input-manifest.json"
    )
    media_hashes = {entry["path"]: entry["sha256"] for entry in source_inputs["files"]}
    clips = []
    for name in CLIPS:
        prefix = f"videos/video_000/clips/{name}"
        store = prior / "datasets/batch" / prefix / "annotations/tennis_scene"
        index_path = store / "scene.json"
        index = read(index_path)
        videos = index["source"]["videos"]
        clip_meta = read(dataset / prefix / "clip.json")
        actual = tuple(row["camera_id"] for row in videos)
        if (
            actual != CAMERAS
            or tuple(clip_meta["camera_ids"]) != CAMERAS
            or clip_meta["dataset_id"] != "meiji_4cam"
            or index["source"]["clip_id"] != clip_meta["clip_id"]
            or clip_meta["clip_id"] != f"video_000/{name}"
        ):
            raise ValueError("Unexpected source dataset or camera order")
        originals = []
        for video, original in zip(videos, clip_meta["video_paths"], strict=True):
            original_path = dataset / prefix / original
            source_path = Path(video["path"])
            relative = source_path.relative_to(prior / "input").as_posix()
            expected = media_hashes[relative]
            if expected != video["sha256"] or any(
                video[k] != clip_meta[k]
                for k in ("num_frames", "width", "height", "fps")
            ):
                raise ValueError(
                    "Source metadata disagrees with the #1752 pinned media"
                )
            require_hash(source_path, expected)
            require_hash(original_path, expected)
            originals.append(str(original_path.resolve()))
        clips.append(
            {
                "clip_id": name,
                "source_clip_id": clip_meta["clip_id"],
                "camera_ids": list(CAMERAS),
                "num_frames": clip_meta["num_frames"],
                "fps": clip_meta["fps"],
                "image_size_wh": [clip_meta["width"], clip_meta["height"]],
                "video_paths": originals,
                "inference_videos": videos,
                "store": str(store),
                "store_index_sha256": digest(index_path),
            }
        )
    return clips


def model_command(root, model, *, preflight=False):
    cwd = Path(model["source_directory"])
    paths = [str(cwd)]
    if "training_site_packages" in model:
        paths.insert(0, model["training_site_packages"])
    command = [
        "env",
        f"PYTHONPATH={os.pathsep.join(paths)}",
        sys.executable,
        "-B",
        "-u",
        str(HERE / "infer.py"),
        str(root),
        model["name"],
    ]
    if preflight:
        command[1:1] = ["CUDA_VISIBLE_DEVICES="]
        command.append("--preflight")
    return command, cwd


def prepare_model(root, model, historical_site):
    name = model["name"]
    if "profile" in model:
        model["checkpoint"] = str((REPO / model["checkpoint"]).resolve())
        model["source_directory"] = str(REPO)
        model["model_source_commit"] = git("rev-parse", BASE)
        model["score_semantics"] = "heatmap_peak"
        model["window_stride"] = model["frames"] // 2
        model["batch_size"] = 1 if model["frames"] == 32 else 4
    else:
        model["checkpoint"] = str(root / model["checkpoint"])
        training = root / model["training_record"]
        require_hash(training, model["training_record_sha256"])
        model["training_record"] = str(training)
        commit = model["model_source_commit"]
        snapshot = root / "model-source" / commit
        if not snapshot.is_dir():
            contents = subprocess.check_output(
                ["git", "-C", str(REPO), "archive", commit, "src"]
            )
            snapshot.mkdir(parents=True)
            with tarfile.open(fileobj=io.BytesIO(contents)) as archive:
                archive.extractall(snapshot, filter="data")
        source_hashes = read(training)["code"]["source_sha256"]
        for relative, expected in source_hashes.items():
            require_hash(snapshot / relative, expected)
        model.update(
            source_directory=str(snapshot),
            training_site_packages=str(historical_site),
            verified_training_source_files=len(source_hashes),
            window_stride=model["frames"] // 2,
            batch_size=1,
            score_semantics="coordinate_availability"
            if model["threshold"] is None
            else "heatmap_peak",
        )
    require_hash(Path(model["checkpoint"]), model["checkpoint_sha256"])
    (root / "work" / name).mkdir(parents=True, exist_ok=True)
    return model


def prepare(root, prior, dataset, historical_site):
    from reconstruct import replay_model

    if (root / "attempt.json").exists():
        raise ValueError("This output has already been used by the queue")
    models = [
        prepare_model(root, model, historical_site)
        for model in read(HERE / "models.json")
    ]
    clips = prepare_clips(prior, dataset)
    plan = {
        "schema": "meiji_ball_model_comparison.v1",
        "base_commit": git("rev-parse", BASE),
        "source_run": str(prior),
        "models": models,
        "clips": clips,
        "ground_truth_2d": label_audit(dataset, clips),
        "settings": {
            "frozen_side": [False, False, True, True],
            "window_ownership": "nearest centre, earlier start on ties, backfilled last full window",
            "uv_px": "detector observation after the profile threshold and trajectory gate, before ball_points",
            "raw_uv_px": "highest heatmap peak with subpixel refinement, or direct coordinate output",
            "raw_score": "heatmap peak probability; NaN for direct coordinates",
            "score": "heatmap peak score or unit geometric availability weight for direct coordinates; zero when rejected",
            "reprojection_px": "distance to the accepted smoothed observation in each front-facing camera; NaN with no comparison",
            "outside_court": "accepted 3D with |x| > 9 m or |y| > 18 m",
        },
    }
    write(root / "plan.json", plan)
    for model in models:
        command, cwd = model_command(root, model, preflight=True)
        subprocess.run(command, cwd=cwd, check=True)
    replay_model(root, plan, models[0], preflight=True)


def run(root, expected_commit):
    from types import SimpleNamespace

    from reconstruct import replay_model

    if not os.environ.get("TENNIS_RUN_ID"):
        raise RuntimeError("Use the shared training queue")
    if git("rev-parse", "HEAD") != expected_commit:
        raise ValueError("Worker HEAD differs from the registered commit")
    subprocess.run(
        ["git", "diff", "--exit-code", BASE, "--", "src"], cwd=REPO, check=True
    )
    subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--", str(HERE)], cwd=REPO, check=True
    )
    plan = read(root / "plan.json")
    if not (root / "work/preflight.json").is_file():
        raise RuntimeError("Complete the CPU source and replay preflight")
    start = {
        "captured_at_utc": utc(),
        "queue_job": os.environ["TENNIS_RUN_ID"],
        "commit": expected_commit,
    }
    with (root / "attempt.json").open("x") as stream:
        json.dump(start, stream)
    status = {**start, "status": "running", "models": {}}
    all_rows = []
    try:
        for model in plan["models"]:
            name = model["name"]
            require_hash(Path(model["checkpoint"]), model["checkpoint_sha256"])
            work = root / "work" / name
            command, cwd = model_command(root, model)
            print(f"Starting {name}", flush=True)
            health = run_variant(work, "gpu", command, cwd, heartbeat_timeout=300)
            status["models"][name] = health
            if health["status"] != "complete":
                if health.get("pid") is not None and "diagnostics" not in health:
                    exited = SimpleNamespace(
                        pid=health["pid"], poll=lambda code=health["returncode"]: code
                    )
                    health["diagnostics"] = capture_diagnostics(work, "gpu", exited)
                raise RuntimeError(f"{name} detector stopped; see {work}")
            rows, references = replay_model(root, plan, model)
            model["inference"] = read(work / "inference.json")
            all_rows.extend(rows)
            if name == "pipeline_x3d_dpt":
                write(
                    root / "baseline-parity.json",
                    {clip: entry["parity"] for clip, entry in references.items()},
                )
                write(
                    root / "loaded-stages.json",
                    {
                        clip: {
                            "non_ball_scene_equal": entry["non_ball_scene_equal"],
                            "nodes": entry["load"],
                        }
                        for clip, entry in references.items()
                    },
                )
            print(f"Published {name}", flush=True)
            write(root / "progress.json", status)
        with (root / "summary.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
        write(root / "manifest.json", {**plan, "queue": start})
        status["status"] = "complete"
    except BaseException as exc:
        status.update(status="failed", error=str(exc))
        raise
    finally:
        write(root / "terminal.json", {**status, "finished_at_utc": utc()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--historical-site", type=Path)
    parser.add_argument("--expect-commit")
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to((REPO / "outputs").resolve()):
        raise ValueError("Output must stay under this worktree's OUTPUT root")
    if args.action == "run":
        run(root, args.expect_commit)
    else:
        if args.historical_site is None or not args.historical_site.is_dir():
            parser.error("prepare requires the saved Torch 2.14.1 site-packages path")
        main_root = Path(
            git("rev-parse", "--path-format=absolute", "--git-common-dir")
        ).parent
        prior = (
            main_root
            / ".worktree-outputs/video-batch-cli/tennis_scene/evaluate"
            / SOURCE_RUN
        )
        dataset = REPO / "data/tennis_multiview/processed/meiji_4cam/dataset"
        prepare(root, prior, dataset, args.historical_site.resolve())


if __name__ == "__main__":
    main()
