"""Validate a complete multi-court dataset release."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, cast

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[5]

from src.synthetic_data_generation.dataset.blcs.rendering.nht import (  # noqa: E402
    _canonical_sha256,
    _sha256_file,
)
from src.synthetic_data_generation.dataset.court.components.labels import (  # noqa: E402
    SYMMETRIC_KEYPOINT_CLASS_NAMES,
    decode_heatmap_atlas_u16,
)
from src.synthetic_data_generation.dataset.court.rendering.nht import (  # noqa: E402
    DATASET_SCHEMA,
    FRAME_SCHEMA,
    HEATMAP_ENCODING,
)

REPORT_SCHEMA = "tennis_multicourt_p7_acceptance_v1"
DIAGNOSTIC_SCHEMA = "tennis_multicourt_dataset_diagnostic_v1"
_SPLITS = ("train", "validation", "test")
_BUCKETS = ("full", "near_full", "partial", "sparse")


def _load_dataset(root: Path) -> dict[str, Any]:
    payload = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if payload.get("schema") != DATASET_SCHEMA:
        raise RuntimeError("Unsupported P7 dataset schema.")
    declared = payload.get("dataset_fingerprint")
    unsigned = dict(payload)
    unsigned.pop("dataset_fingerprint", None)
    if declared != _canonical_sha256(unsigned):
        raise RuntimeError("P7 dataset fingerprint mismatch.")
    return cast(dict[str, Any], payload)


def _verify_ref(root: Path, record: dict[str, Any]) -> Path:
    relative = PurePosixPath(record["relative_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"Unsafe P7 dataset path: {relative}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise RuntimeError(f"Missing P7 dataset file: {relative}")
    if (
        path.stat().st_size != record["size_bytes"]
        or _sha256_file(path) != record["sha256"]
    ):
        raise RuntimeError(f"P7 dataset file integrity mismatch: {relative}")
    return path


def _tree_inventory(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256_file(path)
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }


def _load_diagnostic(path: Path, dataset: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != DIAGNOSTIC_SCHEMA:
        raise RuntimeError("Unsupported P7 diagnostic schema.")
    declared = payload.get("diagnostic_fingerprint")
    unsigned = dict(payload)
    unsigned.pop("diagnostic_fingerprint", None)
    if declared != _canonical_sha256(unsigned):
        raise RuntimeError("P7 diagnostic fingerprint mismatch.")
    if payload["dataset"]["dataset_fingerprint"] != dataset["dataset_fingerprint"]:
        raise RuntimeError("P7 diagnostic references a different dataset.")
    root = path.parent
    for record in payload["files"].values():
        _verify_ref(root, record)
    return cast(dict[str, Any], payload)


def _scaled_camera_matrices(
    labels: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    camera = labels["camera"]
    width, height = labels["resolution"]
    intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64).reshape(3, 3)
    intrinsics[0] *= width / camera["width"]
    intrinsics[1] *= height / camera["height"]
    camera_to_scene = np.asarray(
        camera["camera_to_scene"],
        dtype=np.float64,
    ).reshape(4, 4)
    return intrinsics, np.linalg.inv(camera_to_scene)


def _verify_frame(
    root: Path,
    frame: dict[str, Any],
    *,
    expected_split: str,
) -> dict[str, Any]:
    rgb_path = _verify_ref(root, frame["rgb"])
    atlas_path = _verify_ref(root, frame["heatmap_atlas"])
    labels_path = _verify_ref(root, frame["labels"])
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    if labels.get("schema") != FRAME_SCHEMA:
        raise RuntimeError("Unsupported P7 frame-label schema.")
    if (
        frame["split"] != expected_split
        or labels["split"] != expected_split
        or labels["family_id"] != frame["family_id"]
        or labels["camera"]["camera_id"] != frame["camera_id"]
    ):
        raise RuntimeError("P7 frame split/family/camera identity mismatch.")
    if labels["rgb_overlay_used"] or not labels["all_finite"]:
        raise RuntimeError("P7 frame used RGB overlay or contains non-finite data.")
    target = labels["training_target"]
    if target != {
        "heatmap_channels": 7,
        "heatmap_encoding": HEATMAP_ENCODING,
        "heatmap_atlas_shape": [180, 2240],
        "heatmap_channel_names": list(SYMMETRIC_KEYPOINT_CLASS_NAMES),
        "court_instance_grouping": False,
        "court_instance_grouping_stage": "postprocess",
        "multi_peak_composition": "pixelwise-maximum",
    }:
        raise RuntimeError("P7 frame training target differs from approved semantics.")

    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    atlas = np.asarray(Image.open(atlas_path))
    if rgb.shape != (180, 320, 3) or rgb.dtype != np.uint8:
        raise RuntimeError("P7 RGB shape or dtype differs.")
    if atlas.shape != (180, 2240) or atlas.dtype != np.uint16:
        raise RuntimeError("P7 heatmap atlas shape or dtype differs.")
    heatmaps = decode_heatmap_atlas_u16(atlas)
    intrinsics, scene_to_camera = _scaled_camera_matrices(labels)

    courts = labels["projection"]["courts"]
    if [court["court_instance_id"] for court in courts] != [
        "court_0",
        "court_1",
    ]:
        raise RuntimeError("P7 frame court instance order/identity differs.")
    uv_errors = []
    depth_errors = []
    visible_peak_values = []
    visible_by_class: Counter[int] = Counter()
    visible_count = 0
    occluded_count = 0
    point_count = 0
    width, height = labels["resolution"]
    for court in courts:
        classes = court["classes"]
        if [value["class_id"] for value in classes] != list(range(7)):
            raise RuntimeError("P7 frame classes are not ordered 0..6.")
        for class_record in classes:
            class_id = class_record["class_id"]
            if (
                class_record["class_name"] != SYMMETRIC_KEYPOINT_CLASS_NAMES[class_id]
                or len(class_record["points"]) != 2
            ):
                raise RuntimeError("P7 frame symmetric class contract differs.")
            for point in class_record["points"]:
                point_count += 1
                if not isinstance(point["visible"], bool) or not isinstance(
                    point["occluded"],
                    bool,
                ):
                    raise RuntimeError("P7 point visibility/occlusion must be boolean.")
                expected_occluded = bool(point["in_frame"] and not point["visible"])
                if point["occluded"] != expected_occluded:
                    raise RuntimeError("P7 point occlusion semantics differ.")
                if point["visible"] and not point["in_frame"]:
                    raise RuntimeError("Out-of-frame P7 point cannot be visible.")
                xyz = np.asarray((*point["xyz_scene"], 1.0), dtype=np.float64)
                camera_xyz = (scene_to_camera @ xyz)[:3]
                projected = intrinsics @ camera_xyz
                uv = projected[:2] / projected[2]
                stored_uv = np.asarray(point["uv"], dtype=np.float64)
                uv_errors.append(float(np.linalg.norm(uv - stored_uv)))
                depth_errors.append(
                    abs(float(camera_xyz[2]) - float(point["depth_scene"]))
                )
                if point["in_frame"] and not (
                    0.0 <= stored_uv[0] < width
                    and 0.0 <= stored_uv[1] < height
                    and camera_xyz[2] > 0.0
                ):
                    raise RuntimeError("P7 in-frame point violates image bounds.")
                if point["occluded"]:
                    occluded_count += 1
                if point["visible"]:
                    visible_count += 1
                    visible_by_class[class_id] += 1
                    x = int(round(stored_uv[0]))
                    y = int(round(stored_uv[1]))
                    x = min(max(x, 0), width - 1)
                    y = min(max(y, 0), height - 1)
                    visible_peak_values.append(float(heatmaps[class_id, y, x]))
    if point_count != 28:
        raise RuntimeError("Every P7 frame must contain two courts x fourteen points.")
    if visible_count != labels["renderer_visible_points"]:
        raise RuntimeError("P7 renderer-visible point count differs.")
    if [visible_by_class[index] for index in range(7)] != labels[
        "renderer_visible_peaks_by_class"
    ]:
        raise RuntimeError("P7 visible per-class counts differ.")
    return {
        "uv_error_max": max(uv_errors),
        "depth_error_max": max(depth_errors),
        "visible_peak_min": (min(visible_peak_values) if visible_peak_values else 1.0),
        "visible_count": visible_count,
        "occluded_count": occluded_count,
        "point_count": point_count,
        "coverage": labels["geometric_coverage"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--repeat-dir", type=Path, required=True)
    parser.add_argument("--diagnostic-manifest", type=Path, required=True)
    parser.add_argument("--visual-review", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    dataset_root = args.dataset_dir.resolve()
    repeat_root = args.repeat_dir.resolve()
    diagnostic_path = args.diagnostic_manifest.resolve()
    visual_review_path = args.visual_review.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"Refusing to overwrite output directory: {output_dir}")
    if not visual_review_path.is_file():
        raise FileNotFoundError(visual_review_path)

    dataset = _load_dataset(dataset_root)
    repeated = _load_dataset(repeat_root)
    if dataset != repeated:
        raise RuntimeError("Canonical and repeated P7 manifests differ.")
    canonical_inventory = _tree_inventory(dataset_root)
    repeat_inventory = _tree_inventory(repeat_root)
    if canonical_inventory != repeat_inventory:
        raise RuntimeError("Canonical and repeated P7 trees are not byte-identical.")
    diagnostic = _load_diagnostic(diagnostic_path, dataset)

    split_records = dataset["split_assignment"]["records"]
    family_to_split = {record["family_id"]: record["split"] for record in split_records}
    if len(family_to_split) != 18:
        raise RuntimeError("P7 split assignment must contain eighteen families.")
    family_sets = {
        split: {
            family_id
            for family_id, assigned in family_to_split.items()
            if assigned == split
        }
        for split in _SPLITS
    }
    if (
        not family_sets["train"].isdisjoint(family_sets["validation"])
        or not family_sets["train"].isdisjoint(family_sets["test"])
        or not family_sets["validation"].isdisjoint(family_sets["test"])
    ):
        raise RuntimeError("P7 orbit families leak between dataset splits.")
    for split in ("validation", "test"):
        records = [record for record in split_records if record["split"] == split]
        if (
            {record["shape"] for record in records} != {"circle", "ellipse"}
            or {record["scale_label"] for record in records} != {"0.75", "1.00", "1.30"}
            or {record["target_label"] for record in records}
            != {"complex", "court_0", "court_1"}
        ):
            raise RuntimeError(f"P7 {split} semantic family coverage differs.")

    frame_split_counts: Counter[str] = Counter()
    coverage_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    uv_error_max = 0.0
    depth_error_max = 0.0
    visible_peak_min = 1.0
    visible_count = 0
    occluded_count = 0
    point_count = 0
    for index, frame in enumerate(dataset["frames"]):
        if frame["sample_index"] != index:
            raise RuntimeError("P7 sample indices are not contiguous.")
        expected_split = family_to_split[frame["family_id"]]
        metrics = _verify_frame(
            dataset_root,
            frame,
            expected_split=expected_split,
        )
        frame_split_counts[expected_split] += 1
        for bucket in metrics["coverage"]:
            coverage_by_split[expected_split][bucket] += 1
        uv_error_max = max(uv_error_max, metrics["uv_error_max"])
        depth_error_max = max(depth_error_max, metrics["depth_error_max"])
        visible_peak_min = min(visible_peak_min, metrics["visible_peak_min"])
        visible_count += metrics["visible_count"]
        occluded_count += metrics["occluded_count"]
        point_count += metrics["point_count"]

    expected_frame_counts = {"train": 284, "validation": 72, "test": 72}
    gates = {
        "dataset_status_mechanics_only": (
            dataset["status"] == "passed-mechanics-only"
            and dataset["automatic_gate_passed"]
        ),
        "export_first_491_cameras": (
            dataset["export_first_source"]["camera_count"] == 491
            and dataset["export_first_source"]["scene_fingerprint"]
            == dataset["scene_contract"]["scene_fingerprint"]
        ),
        "frame_and_family_counts": (
            len(dataset["frames"]) == 428
            and len(family_to_split) == 18
            and dict(frame_split_counts) == expected_frame_counts
        ),
        "family_disjoint_semantic_splits": (
            {split: len(values) for split, values in family_sets.items()}
            == {"train": 12, "validation": 3, "test": 3}
        ),
        "all_coverage_buckets_in_every_split": all(
            all(coverage_by_split[split][bucket] > 0 for bucket in _BUCKETS)
            for split in _SPLITS
        ),
        "two_courts_seven_channels_postprocess_grouping": (
            dataset["metrics"]["court_instance_count"] == 2
            and dataset["training_target"]["heatmap_channels"] == 7
            and not dataset["training_target"]["court_instance_grouping"]
            and dataset["training_target"]["court_instance_grouping_stage"]
            == "postprocess"
            and dataset["metrics"]["maximum_visible_peaks_in_one_channel"] == 4
        ),
        "projection_round_trip": (
            uv_error_max <= 1.0e-6 and depth_error_max <= 1.0e-10
        ),
        "heatmap_visible_peak_alignment": visible_peak_min >= 0.85,
        "visibility_complete": (
            point_count == 428 * 2 * 14
            and visible_count == dataset["metrics"]["renderer_visible_point_count"]
            and occluded_count > 0
        ),
        "byte_identical_repeat": (
            canonical_inventory == repeat_inventory and len(canonical_inventory) == 1285
        ),
        "diagnostic_and_visual_review": (
            diagnostic["rgb_overlay_scope"]
            == "diagnostic-only; dataset RGB is unchanged"
            and "passed-mechanics-only"
            in visual_review_path.read_text(encoding="utf-8")
        ),
        "no_rgb_overlay": not dataset["rgb_overlay_used"],
        "appearance_limitation_explicit": (
            "not production photorealism" in dataset["appearance_scope"]
        ),
    }
    passed = all(gates.values())
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "passed" if passed else "failed",
        "p7_complete": passed,
        "dataset": {
            "manifest_sha256": _sha256_file(dataset_root / "manifest.json"),
            "dataset_fingerprint": dataset["dataset_fingerprint"],
        },
        "repeat": {
            "manifest_sha256": _sha256_file(repeat_root / "manifest.json"),
            "dataset_fingerprint": repeated["dataset_fingerprint"],
        },
        "diagnostic": {
            "manifest_sha256": _sha256_file(diagnostic_path),
            "diagnostic_fingerprint": diagnostic["diagnostic_fingerprint"],
        },
        "visual_review": {
            "sha256": _sha256_file(visual_review_path),
            "decision": "passed-mechanics-only",
        },
        "gates": gates,
        "metrics": {
            "byte_identical_file_count": len(canonical_inventory),
            "frame_count": len(dataset["frames"]),
            "point_record_count": point_count,
            "renderer_visible_point_count": visible_count,
            "renderer_occluded_point_count": occluded_count,
            "projection_uv_max_error_px": uv_error_max,
            "projection_uv_threshold_px": 1.0e-6,
            "projection_depth_max_error_scene": depth_error_max,
            "projection_depth_threshold_scene": 1.0e-10,
            "minimum_visible_point_heatmap_value": visible_peak_min,
            "split_frame_counts": dict(frame_split_counts),
            "split_family_counts": {
                split: len(family_sets[split]) for split in _SPLITS
            },
            "coverage_by_split": {
                split: dict(sorted(coverage_by_split[split].items()))
                for split in _SPLITS
            },
        },
        "appearance_scope": dataset["appearance_scope"],
        "rgb_overlay_used": False,
    }
    report["content_fingerprint"] = _canonical_sha256(report)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.",
            suffix=".tmp",
            dir=output_dir.parent,
        )
    )
    try:
        (temporary / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.rename(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit("P7 acceptance gates failed.")


if __name__ == "__main__":
    main()
