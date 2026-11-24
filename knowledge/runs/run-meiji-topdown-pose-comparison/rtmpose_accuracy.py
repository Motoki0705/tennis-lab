"""Verify the completed RTMPose accuracy runs and report changes from PT32."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from evaluate import score_fixed
from fixed_inputs import RowKey, row_keys
from pose_contract import PoseBatch
from rtmpose import MODELS, FloatArray

from src.tennis_scene.evaluation.player_pose.bundle import (
    load_manifest,
    load_reference_frames,
)
from src.utils.checksum import dual_sha256
from src.utils.inference_benchmark import Backend
from src.utils.io import json_value, write_json_atomic
from src.utils.schema.player import COCO_KP_NAMES


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def paired_delta(reference: FloatArray, candidate: FloatArray) -> dict[str, Any]:
    ref_valid = np.isfinite(reference).all(axis=-1)
    candidate_valid = np.isfinite(candidate).all(axis=-1)
    paired = ref_valid & candidate_valid
    delta = reference[paired].astype(np.float64) - candidate[paired].astype(np.float64)
    xy = np.linalg.norm(delta[:, :2], axis=-1)
    scores = np.abs(delta[:, 2])
    result = {
        "joints": paired.size,
        "paired_joints": int(paired.sum()),
        "missing_in_candidate_only": int((ref_valid & ~candidate_valid).sum()),
        "missing_in_reference_only": int((~ref_valid & candidate_valid).sum()),
        "missing_in_both": int((~ref_valid & ~candidate_valid).sum()),
        "xy_above_1px": int((xy > 1).sum()),
        "xy_above_5px": int((xy > 5).sum()),
    }
    if xy.size == 0:
        return {**result, "xy_px": None, "score_abs_p95": None, "worst": None}
    quantiles = np.asarray(np.percentile(xy, [50, 95, 99]))
    worst = int(xy.argmax())
    row, joint = np.argwhere(paired)[worst]
    return {
        **result,
        "xy_px": {
            "mean": float(xy.mean()),
            "p50": float(quantiles[0]),
            "p95": float(quantiles[1]),
            "p99": float(quantiles[2]),
            "max": float(xy[worst]),
        },
        "score_abs_p95": float(np.percentile(scores, 95)),
        "worst": {"row_index": int(row), "joint": COCO_KP_NAMES[joint]},
    }


def _artifact(root: Path, path: Path) -> dict[str, str]:
    return {"path": path.relative_to(root).as_posix(), "sha256": dual_sha256(path)}


def _require_file(root: Path, receipt: dict[str, Any], name: str) -> Path:
    path = root / name
    if receipt["path"] != name or dual_sha256(path) != receipt["sha256"]:
        raise ValueError(f"Artifact differs from its accuracy receipt: {path}")
    return path


def _pose_summary(scores: dict[str, Any]) -> dict[str, Any]:
    return {
        "gt_fixed_denominator": {
            key: value
            for key, value in scores["overall"]["pose"][
                "with_detection_coverage"
            ].items()
            if key != "by_joint"
        },
        "gt_finite_pairs": {
            key: value
            for key, value in scores["finite_pairs"]["overall"].items()
            if key != "by_joint"
        },
    }


def score_changes(previous: dict[str, Any], rescored: dict[str, Any]) -> dict[str, Any]:
    missing = object()
    previous_policy, current_policy = previous["policy"], rescored["policy"]
    previous_detection = previous["overall"]["detection"]
    current_detection = rescored["overall"]["detection"]
    return {
        "changed_metric_sections": sorted(
            name
            for name in (previous.keys() | rescored.keys()) - {"policy"}
            if previous.get(name, missing) != rescored.get(name, missing)
        ),
        "changed_policy_fields": sorted(
            name
            for name in previous_policy.keys() | current_policy.keys()
            if previous_policy.get(name, missing) != current_policy.get(name, missing)
        ),
        "recall_by_iou_before": previous_detection["recall_by_iou"],
        "recall_by_iou_after": current_detection["recall_by_iou"],
        "matched_at_iou_0_50_before": previous_detection["matched_at_iou_0_50"],
        "matched_at_iou_0_50_after": current_detection["matched_at_iou_0_50"],
        "reference_boxes": current_detection["references"],
    }


def read_condition(
    queue: Path,
    job: str,
    cohort: tuple[str, str],
    expected_rows: dict[str, Any],
) -> tuple[tuple[str, Backend], FloatArray, dict[str, Any]]:
    stem = job.removesuffix(".job")
    if Path(stem).name != stem:
        raise ValueError(f"Expected a queue job ID: {job}")
    repro = queue / "repro" / stem
    predictions = repro / "predictions"
    run = read_json(repro / "run.json")
    receipt = read_json(predictions / "receipt.json")
    row_count = len(expected_rows["player_id"])
    model, backend = receipt["model"], Backend(receipt["backend"])
    if (
        run["run_id"] != stem
        or receipt["commit"] != run["commit"]
        or receipt["status"] != "ok"
        or receipt["rows"] != row_count
        or (receipt["bundle_id"], receipt["bundle_sha256"]) != cohort
        or receipt["joint_order"] != list(COCO_KP_NAMES)
        or receipt["weights"]["checkpoint"]["sha256"] != MODELS[model].sha256
    ):
        raise ValueError(f"Accuracy receipt differs from the frozen condition: {repro}")

    poses_path = _require_file(predictions, receipt["predictions"], "predictions.npz")
    scores_path = _require_file(predictions, receipt["scores"], "scores.json")
    with np.load(poses_path, allow_pickle=False) as saved:
        for field, expected in expected_rows.items():
            np.testing.assert_array_equal(
                saved[field], expected, err_msg=f"Different {field} in {poses_path}"
            )
        poses = saved["keypoints_xyc"]
    if poses.shape != (row_count, len(COCO_KP_NAMES), 3) or poses.dtype != np.float32:
        raise ValueError(f"Invalid COCO17 pose array: {poses_path}")
    record = {
        "job_id": job,
        "execution_commit": receipt["commit"],
        "batch_size": receipt["batch_size"],
        "environment": receipt["environment"],
        "checkpoint": receipt["weights"]["checkpoint"],
        "files": {
            "predictions": {
                "path": poses_path.relative_to(queue).as_posix(),
                "sha256": receipt["predictions"]["sha256"],
            },
            "scores": {
                "path": scores_path.relative_to(queue).as_posix(),
                "sha256": receipt["scores"]["sha256"],
            },
            "receipt": _artifact(queue, predictions / "receipt.json"),
            "repro": _artifact(queue, repro / "run.json"),
            "command": _artifact(queue, repro / "repro.sh"),
            "log": _artifact(queue, queue / "logs" / f"{stem}.log"),
        },
        **_pose_summary(read_json(scores_path)),
    }
    if backend is Backend.TRT16:
        plan = receipt["engines"]["plan"]
        engine_path = Path(plan["path"])
        engine = read_json(engine_path.with_name("engine.json"))
        build_path, layers_path = repro / "build.json", repro / "layers-b8.json"
        build = read_json(build_path)
        if (
            build["failure"] is not None
            or dual_sha256(engine_path) != plan["sha256"]
            or engine["plan_sha256"] != plan["sha256"]
        ):
            raise ValueError(
                f"TensorRT artifacts differ from the completed build: {repro}"
            )
        layers = read_json(layers_path)["Layers"]
        formats = Counter(
            tensor["Format/Datatype"]
            for layer in layers
            for field in ("Inputs", "Outputs")
            for tensor in layer[field]
        )
        record["tensorrt"] = {
            "plan": plan,
            "engine": engine,
            "build_wall_ms": build["wall_ms"],
            "build_receipt": _artifact(queue, build_path),
            "layer_inspector": _artifact(queue, layers_path),
            "layer_tensor_formats": dict(formats),
            "layers_reporting_f32_tactics": [
                {"name": layer["Name"], "tactic": layer["TacticName"]}
                for layer in layers
                if "_f32_" in layer["TacticName"]
            ],
        }
    return (model, backend), poses, record


def _address(row: RowKey) -> dict[str, Any]:
    return {**row.frame._asdict(), "player_id": row.player_id}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--rescore-dir",
        type=Path,
        help="Write full current-scorer reports in a new directory",
    )
    parser.add_argument("jobs", nargs="+")
    args = parser.parse_args()
    manifest = load_manifest(args.bundle, expected_sha256=args.bundle_sha256)
    cohort = (manifest["benchmark_id"], args.bundle_sha256)
    references = load_reference_frames(args.bundle, expected_sha256=args.bundle_sha256)
    keys = row_keys(references)
    expected_rows = {
        "clip_id": [row.frame.clip_id for row in keys],
        "camera_id": [row.frame.camera_id for row in keys],
        "source_frame": [row.frame.source_frame for row in keys],
        "player_id": [row.player_id for row in keys],
        "bbox_xyxy": np.concatenate(
            [references[key].boxes_xyxy for key in sorted(references)]
        ).astype(np.float32),
    }
    poses_by_case = {}
    records = {}
    for job in args.jobs:
        case, poses, record = read_condition(args.queue, job, cohort, expected_rows)
        if case in records:
            raise ValueError(f"Duplicate result for {case}")
        poses_by_case[case] = poses
        records[case] = record
    if records.keys() != {(model, backend) for model in MODELS for backend in Backend}:
        raise ValueError("Supply one completed accuracy job for each RTMPose condition")

    rescoring = None
    if args.rescore_dir is not None:
        args.rescore_dir.mkdir(parents=True, exist_ok=False)
        rescoring = {
            "execution_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "reports_directory": str(args.rescore_dir.resolve()),
            "scoring_source": _artifact(
                Path.cwd(),
                Path("src/tennis_scene/evaluation/player_pose/scoring.py").resolve(),
            ),
        }

    conditions: dict[str, dict[str, dict[str, Any]]] = {}
    for model in MODELS:
        conditions[model] = {}
        pt32 = poses_by_case[model, Backend.PT32]
        for backend in Backend:
            record = records[model, backend]
            if args.rescore_dir is not None:
                previous = read_json(args.queue / record["files"]["scores"]["path"])
                rescored = json_value(
                    score_fixed(
                        references, keys, PoseBatch(poses_by_case[model, backend])
                    )
                )
                path = args.rescore_dir / f"{model}-{backend.value}.json"
                write_json_atomic(path, rescored)
                record.update(_pose_summary(rescored))
                record["rescore"] = {
                    **score_changes(previous, rescored),
                    "scores": _artifact(args.rescore_dir, path),
                }
            if backend is not Backend.PT32:
                difference = paired_delta(pt32, poses_by_case[model, backend])
                if difference["worst"]:
                    difference["worst"].update(
                        _address(keys[difference["worst"]["row_index"]])
                    )
                record["vs_pt32"] = difference
            conditions[model][backend.value] = record
    payload = {
        "bundle_id": cohort[0],
        "bundle_sha256": cohort[1],
        "rows": len(keys),
        "conditions": conditions,
    }
    if rescoring is not None:
        payload["rescoring"] = rescoring
    write_json_atomic(args.output, payload)


if __name__ == "__main__":
    main()
