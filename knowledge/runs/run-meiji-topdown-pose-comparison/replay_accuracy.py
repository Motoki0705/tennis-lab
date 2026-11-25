"""Re-score retained fixed-box predictions with the current common pose scorer."""

from __future__ import annotations

import argparse
import inspect
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from evaluate import MODEL_IDS, score_fixed
from fixed_inputs import RowKey, row_keys
from numpy.typing import NDArray
from pose_contract import PoseBatch
from rtmpose_accuracy import paired_delta, read_json, score_changes

from src.tennis_scene.evaluation.player_pose.bundle import (
    load_manifest,
    load_reference_frames,
    pinned_file,
)
from src.tennis_scene.evaluation.player_pose.scoring import (
    FrameKey,
    ReferenceFrame,
    score_players,
)
from src.utils.checksum import dual_sha256
from src.utils.inference_benchmark import Backend
from src.utils.io import json_value, write_json_atomic
from src.utils.schema.player import COCO_KP_NAMES


def row_columns(
    references: Mapping[FrameKey, ReferenceFrame], keys: tuple[RowKey, ...]
) -> dict[str, NDArray[np.generic]]:
    return {
        "clip_id": np.array([key.frame.clip_id for key in keys]),
        "camera_id": np.array([key.frame.camera_id for key in keys]),
        "source_frame": np.array([key.frame.source_frame for key in keys]),
        "player_id": np.array([key.player_id for key in keys]),
        "bbox_xyxy": np.concatenate(
            [references[key].boxes_xyxy for key in sorted(references)]
        ).astype(np.float32),
    }


def load_predictions(
    path: Path, expected: Mapping[str, NDArray[np.generic]]
) -> PoseBatch:
    with np.load(path, allow_pickle=False) as arrays:
        for name, values in expected.items():
            if not np.array_equal(arrays[name], values):
                raise ValueError(
                    f"Prediction {name} differs from the fixed rows: {path}"
                )
        keypoints = arrays["keypoints_xyc"]
    if len(keypoints) != len(expected["player_id"]):
        raise ValueError(f"Prediction count differs from the fixed rows: {path}")
    return PoseBatch(keypoints)


def _stats(report: dict[str, Any]) -> dict[str, Any]:
    def without_joints(pose: dict[str, Any]) -> dict[str, Any]:
        return {name: value for name, value in pose.items() if name != "by_joint"}

    return {
        "box_recall": report["overall"]["detection"]["recall_by_iou"],
        "fixed": without_joints(report["overall"]["pose"]["with_detection_coverage"]),
        "finite_pairs": without_joints(report["finite_pairs"]["overall"]),
    }


def _copy_queue_log(source: Path, target: Path) -> None:
    shutil.copyfile(source, target)
    if dual_sha256(target) != dual_sha256(source):
        raise ValueError("Queue log changed during retention")


def _retained_copy(
    repro: Path, files: tuple[Path, ...], receipt: dict[str, Any], log: Path
) -> Path:
    """Publish an intact copy of the original result into the retained queue repro."""
    target = repro / "predictions"
    with TemporaryDirectory(prefix=".retaining-", dir=repro) as scratch:
        staging = Path(scratch) / "predictions"
        staging.mkdir()
        for file in files:
            shutil.copyfile(file, staging / file.name)
        if read_json(staging / "receipt.json") != receipt:
            raise ValueError("Accuracy receipt changed during retention")
        for kind in ("predictions", "scores"):
            pinned_file(staging, receipt[kind])
        _copy_queue_log(log, staging / "queue.log")
        staging.rename(target)
    return target


def read_condition(
    queue: Path,
    job: str,
    cohort: tuple[str, str],
    expected: Mapping[str, NDArray[np.generic]],
    *,
    retain: bool = False,
) -> tuple[tuple[str, Backend], PoseBatch, dict[str, Any], dict[str, Any]]:
    stem = job.removesuffix(".job")
    if Path(stem).name != stem or not job.endswith(".job"):
        raise ValueError(f"Expected a queue .job ID, got {job}")
    repro = queue / "repro" / stem
    run = read_json(repro / "run.json")
    retained = repro / "predictions"
    root = retained
    if not (retained / "receipt.json").exists():
        pointer = repro / "output_dir.txt"
        root = Path(pointer.read_text(encoding="utf-8").strip())
    receipt_file = root / "receipt.json"
    receipt = read_json(receipt_file)
    model, backend = receipt["model"], Backend(receipt["backend"])
    if (
        run["run_id"] != stem
        or run["commit"] != receipt["commit"]
        or receipt["status"] != "ok"
        or receipt["rows"] != len(expected["player_id"])
        or model not in MODEL_IDS
        or (receipt["bundle_id"], receipt["bundle_sha256"]) != cohort
        or receipt["joint_order"] != list(COCO_KP_NAMES)
    ):
        raise ValueError(
            f"Accuracy receipt does not match its queue job and cohort: {job}"
        )
    predictions_file = pinned_file(root, receipt["predictions"])
    scores_file = pinned_file(root, receipt["scores"])
    poses = load_predictions(predictions_file, expected)
    previous = read_json(scores_file)
    live_log = queue / "logs" / f"{stem}.log"
    retained_log = retained / "queue.log"
    if retain:
        if root != retained:
            root = _retained_copy(
                repro, (receipt_file, predictions_file, scores_file), receipt, live_log
            )
        elif not retained_log.is_file():
            with TemporaryDirectory(prefix=".retaining-log-", dir=retained) as scratch:
                staged = Path(scratch) / "queue.log"
                _copy_queue_log(live_log, staged)
                retained_log.hardlink_to(staged)
    log = retained_log if root == retained else live_log
    return (
        (model, backend),
        poses,
        previous,
        {
            "job_id": job,
            "execution_commit": receipt["commit"],
            "environment": receipt["environment"],
            "batch_size": receipt["batch_size"],
            "weights": {
                name: {key: item[key] for key in ("sha256", "bytes")}
                for name, item in receipt["weights"].items()
            },
            "engines": {
                name: {key: item[key] for key in ("sha256", "bytes")}
                for name, item in receipt["engines"].items()
            },
            "original_output": str(
                root.relative_to(queue) if root == retained else root
            ),
            "prediction_sha256": receipt["predictions"]["sha256"],
            "original_score_sha256": receipt["scores"]["sha256"],
            "receipt_sha256": dual_sha256(root / "receipt.json"),
            "queue_run_sha256": dual_sha256(repro / "run.json"),
            "command_sha256": dual_sha256(repro / "repro.sh"),
            "log": {
                "path": log.relative_to(queue).as_posix(),
                "sha256": dual_sha256(log),
            },
        },
    )


def compare_reports(
    previous: dict[str, Any], current: dict[str, Any]
) -> dict[str, Any]:
    previous_stats, current_stats = _stats(previous), _stats(current)
    return {
        "rescore_diff": score_changes(previous, current),
        "metrics": current_stats,
        **(
            {"previous_metrics": previous_stats}
            if previous_stats != current_stats
            else {}
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retain", action="store_true")
    parser.add_argument("jobs", nargs="+")
    args = parser.parse_args()

    manifest = load_manifest(args.bundle, expected_sha256=args.bundle_sha256)
    cohort = (manifest["benchmark_id"], args.bundle_sha256)
    references = load_reference_frames(args.bundle, expected_sha256=args.bundle_sha256)
    keys = row_keys(references)
    columns = row_columns(references, keys)
    args.output.mkdir(parents=True, exist_ok=False)
    cases = {}
    poses_by_case = {}
    for job in args.jobs:
        case, poses, previous, record = read_condition(
            args.queue, job, cohort, columns, retain=args.retain
        )
        if case in cases:
            raise ValueError(f"Duplicate result for {case}")
        model, backend = case
        report = json_value(score_fixed(references, keys, poses))
        score_file = args.output / f"{model}-{backend.lower()}.json"
        write_json_atomic(score_file, report)
        cases[case] = {
            **record,
            **compare_reports(previous, report),
            "score": {"path": score_file.name, "sha256": dual_sha256(score_file)},
        }
        poses_by_case[case] = poses
        print(
            model,
            backend,
            cases[case]["rescore_diff"]["changed_metric_sections"],
            flush=True,
        )

    conditions: dict[str, dict[str, Any]] = {}
    for (model, backend), record in cases.items():
        if backend is not Backend.PT32 and (model, Backend.PT32) in poses_by_case:
            difference = paired_delta(
                poses_by_case[model, Backend.PT32].keypoints_xyc,
                poses_by_case[model, backend].keypoints_xyc,
            )
            if difference["worst"]:
                key = keys[difference["worst"]["row_index"]]
                difference["worst"].update(key.frame._asdict(), player_id=key.player_id)
            record["vs_pt32"] = difference
        conditions.setdefault(model, {})[backend.value] = record
    write_json_atomic(
        args.output / "summary.json",
        {
            "bundle_id": cohort[0],
            "bundle_sha256": cohort[1],
            "images": len(references),
            "rows": len(keys),
            "replay_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "score_players_sha256": dual_sha256(Path(inspect.getfile(score_players))),
            "conditions": conditions,
        },
    )


if __name__ == "__main__":
    main()
