"""Rescore saved PMPose predictions with the canonical fixed-box matcher."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from evaluate import score_fixed
from fixed_inputs import RowKey, row_keys
from numpy.typing import NDArray
from pmpose import EXTRA_FIELDS, NATIVE_JOINTS
from pmpose_assets import WEIGHTS
from pose_contract import PoseBatch

from src.tennis_scene.evaluation.player_pose.bundle import (
    load_manifest,
    load_reference_frames,
    pinned_file,
)
from src.tennis_scene.evaluation.player_pose.scoring import (
    FrameKey,
    ReferenceFrame,
    describe,
)
from src.utils.checksum import dual_sha256
from src.utils.inference_benchmark import Backend
from src.utils.io import write_json_atomic
from src.utils.models.tensorrt_artifacts import PLAN_FILE, EngineBundle
from src.utils.schema.player import COCO_KP_NAMES, NUM_HUMAN_KP


def _artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": dual_sha256(path)}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return value


def fixed_prediction_rows(
    references: Mapping[FrameKey, ReferenceFrame], keys: tuple[RowKey, ...]
) -> dict[str, NDArray[np.generic]]:
    return {
        "clip_id": np.array([row.frame.clip_id for row in keys]),
        "camera_id": np.array([row.frame.camera_id for row in keys]),
        "source_frame": np.array([row.frame.source_frame for row in keys]),
        "player_id": np.array([row.player_id for row in keys]),
        "bbox_xyxy": np.concatenate(
            [references[key].boxes_xyxy for key in sorted(references)]
        ).astype(np.float32),
    }


def validate_predictions(
    arrays: Mapping[str, NDArray[Any]], expected_rows: Mapping[str, NDArray[Any]]
) -> PoseBatch:
    for field, expected in expected_rows.items():
        np.testing.assert_array_equal(
            arrays[field], expected, err_msg=f"Changed {field}"
        )
    points = arrays["keypoints_xyc"]
    if len(points) != len(expected_rows["player_id"]):
        raise ValueError("Prediction array changed the number of fixed player rows")
    extras = {
        name: arrays[f"extra__{name}"]
        for name in (*EXTRA_FIELDS, "heatmap_peak", "native_keypoints_xy")
    }
    for name, values in extras.items():
        shape: tuple[int, ...] = (len(points), NATIVE_JOINTS)
        if name == "native_keypoints_xy":
            shape = (*shape, 2)
        if values.shape != shape or values.dtype != np.float32:
            raise ValueError(f"Invalid PMPose auxiliary output {name}: {values.shape}")
    result = PoseBatch(points, extras)
    np.testing.assert_array_equal(
        points[..., :2], extras["native_keypoints_xy"][:, :NUM_HUMAN_KP]
    )
    np.testing.assert_array_equal(
        points[..., 2], extras["expected_oks"][:, :NUM_HUMAN_KP]
    )
    return result


def _engine_evidence(
    assets: Path, model: str, receipt: dict[str, Any]
) -> dict[str, Any]:
    bundle = EngineBundle.read(assets / f"{model}-engine")
    plan = receipt["engines"]["plan"]
    if (
        plan["sha256"] != bundle.engine.plan_sha256
        or plan["bytes"] != (bundle.root / PLAN_FILE).stat().st_size
        or bundle.export.model != model
        or bundle.export.sources["checkpoint"] != WEIGHTS[model].sha256
    ):
        raise ValueError(
            f"Engine differs from the completed accuracy run: {bundle.root}"
        )
    build_file = assets / f"{model}-trt16-build.json"
    build = _json(build_file)
    if build["phase"] != "engine_build" or build["failure"] is not None:
        raise ValueError(f"Invalid engine build receipt: {build_file}")
    layer_file = assets / f"{model}-trt16-b8-layers.json"
    formats: Counter[str] = Counter()
    fp32_layers = []
    for layer in _json(layer_file)["Layers"]:
        bindings = [
            tensor["Format/Datatype"]
            for tensor in (*layer["Inputs"], *layer["Outputs"])
        ]
        formats.update(bindings)
        if "Float" in bindings:
            fp32_layers.append(layer["Name"])
    return {
        "plan": plan,
        "engine": bundle.engine.model_dump(mode="json"),
        "onnx_files": bundle.export.onnx_files,
        "build_wall_ms": build["wall_ms"],
        "build_receipt": _artifact(build_file),
        "layer_inspector": _artifact(layer_file),
        "layer_tensor_formats": dict(formats),
        "layers_with_float32_bindings": fp32_layers,
    }


def _read_case(
    queue: Path,
    job_id: str,
    assets: Path,
    cohort: tuple[str, str],
    expected_rows: Mapping[str, NDArray[Any]],
) -> tuple[tuple[str, Backend], PoseBatch, Path, dict[str, Any]]:
    stem = job_id.removesuffix(".job")
    if stem == job_id or Path(stem).name != stem:
        raise ValueError(f"Expected a .job filename: {job_id}")
    repro = queue / "repro" / stem
    run_path = repro / "run.json"
    run = _json(run_path)
    pointer = repro / "output_dir.txt"
    directory = Path(pointer.read_text().strip()).resolve()
    receipt_path = directory / "receipt.json"
    receipt = _json(receipt_path)
    model, backend = receipt["model"], Backend(receipt["backend"])
    if (
        run["run_id"] != stem
        or run["commit"] != receipt["commit"]
        or receipt["status"] != "ok"
        or (receipt["bundle_id"], receipt["bundle_sha256"]) != cohort
        or receipt["rows"] != len(expected_rows["player_id"])
        or receipt["joint_order"] != list(COCO_KP_NAMES)
        or receipt["weights"]["checkpoint"]["sha256"] != WEIGHTS[model].sha256
        or directory != (assets / f"{model}-{backend.value.lower()}-accuracy").resolve()
    ):
        raise ValueError(f"Accuracy receipt differs from the pinned condition: {repro}")
    predictions = pinned_file(directory, receipt["predictions"])
    scores = pinned_file(directory, receipt["scores"])
    with np.load(predictions, allow_pickle=False) as saved:
        array = validate_predictions(dict(saved.items()), expected_rows)
    record = {
        "job_id": job_id,
        "execution_commit": run["commit"],
        "batch_size": receipt["batch_size"],
        "environment": receipt["environment"],
        "checkpoint": receipt["weights"]["checkpoint"],
        "files": {
            "predictions": {
                "path": str(predictions),
                "sha256": receipt["predictions"]["sha256"],
            },
            "original_scores": {
                "path": str(scores),
                "sha256": receipt["scores"]["sha256"],
            },
            "accuracy_receipt": _artifact(receipt_path),
            "repro": _artifact(run_path),
            "command": _artifact(repro / "repro.sh"),
            "log": _artifact(repro / "predictions" / "queue.log"),
        },
    }
    if backend is Backend.TRT16:
        record["tensorrt"] = _engine_evidence(assets, model, receipt)
    return (model, backend), array, scores, record


def _paired_delta(
    reference: NDArray[np.float32],
    candidate: NDArray[np.float32],
    keys: tuple[RowKey, ...],
) -> dict[str, Any]:
    a, b = np.isfinite(reference).all(axis=-1), np.isfinite(candidate).all(axis=-1)
    both = a & b
    deltas = candidate[both].astype(np.float64) - reference[both]
    xy = np.linalg.norm(deltas[:, :2], axis=-1)
    result = {
        "eligible": int(both.size),
        "reference_only": int(np.count_nonzero(a & ~b)),
        "candidate_only": int(np.count_nonzero(~a & b)),
        "missing_in_both": int(np.count_nonzero(~a & ~b)),
        "coordinate_l2_px": describe(xy),
        "confidence_abs": describe(np.abs(deltas[:, 2])),
    }
    if xy.size:
        worst = xy.argmax()
        row, joint = np.unravel_index(np.flatnonzero(both)[worst], both.shape)
        key = keys[row]
        result["worst_joint"] = {
            **key.frame._asdict(),
            "player_id": key.player_id,
            "joint": COCO_KP_NAMES[joint],
            "coordinate_l2_px": float(xy[worst]),
        }
    return result


def _overall(scores: dict[str, Any]) -> dict[str, Any]:
    return {
        "gt_fixed_denominator": scores["overall"]["pose"]["with_detection_coverage"],
        "gt_finite_pairs": scores["finite_pairs"]["overall"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("jobs", nargs="+")
    args = parser.parse_args()
    manifest = load_manifest(args.bundle, expected_sha256=args.bundle_sha256)
    cohort = (manifest["benchmark_id"], args.bundle_sha256)
    references = load_reference_frames(args.bundle, expected_sha256=args.bundle_sha256)
    keys = row_keys(references)
    expected_rows = fixed_prediction_rows(references, keys)
    cases = {}
    for job in args.jobs:
        key, poses, scores, record = _read_case(
            args.queue, job, args.assets, cohort, expected_rows
        )
        if key in cases:
            raise ValueError(f"Repeated accuracy condition: {key}")
        cases[key] = (poses, scores, record)
    if cases.keys() != {(model, backend) for model in WEIGHTS for backend in Backend}:
        raise ValueError("Supply the six completed PMPose accuracy runs")

    args.output.mkdir(parents=True, exist_ok=False)
    models = {}
    for model in WEIGHTS:
        conditions = {}
        pt32 = cases[(model, Backend.PT32)][0].keypoints_xyc
        for backend in Backend:
            poses, original_scores, record = cases[(model, backend)]
            rescored = score_fixed(references, keys, poses)
            original = _json(original_scores)
            overall = _overall(rescored)
            record.update(
                {
                    name: {
                        key: value for key, value in view.items() if key != "by_joint"
                    }
                    for name, view in overall.items()
                }
            )
            target = args.output / f"{model}-{backend.value.lower()}-scores.json"
            write_json_atomic(target, rescored)
            record["rescoring"] = {
                "entire_score_report_unchanged": _json(target) == original,
                "recall_unchanged": rescored["overall"]["detection"]
                == original["overall"]["detection"],
                "oks_pck_unchanged": all(
                    rescored_view[metric] == original_view[metric]
                    for rescored_view, original_view in zip(
                        overall.values(), _overall(original).values(), strict=True
                    )
                    for metric in ("oks", "pck")
                ),
            }
            record["files"]["rescored"] = _artifact(target)
            if backend is not Backend.PT32:
                record["vs_pt32"] = _paired_delta(pt32, poses.keypoints_xyc, keys)
            conditions[backend.value] = record
            print(model, backend, record["rescoring"], flush=True)
        models[model] = conditions

    write_json_atomic(
        args.output / "summary.json",
        {
            "bundle_id": cohort[0],
            "bundle_sha256": cohort[1],
            "rows": len(keys),
            "rescore_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "conditions": models,
        },
    )


if __name__ == "__main__":
    main()
