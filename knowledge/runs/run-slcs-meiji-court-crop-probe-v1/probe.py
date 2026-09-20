"""One queued diagnostic; no cache writes and no automatic variant selection."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from src.tasks.court_detection.inference.predictor import CourtKeypointPredictor
from src.tennis_scene.dataset_pipeline.court import (
    StaticCourtSettings,
    ball_guided_roi,
    fit_static_court,
)
from src.tennis_scene.generate_dataset.manifest import ClipManifest
from src.utils.checksum import dual_sha256
from src.utils.configuration import PathResolver, RuntimePathRoots
from src.utils.schema.court import COURT_SKELETON, CourtConfig, court_keypoints_3d

MAIN = Path("/home/kamimura/projects/tennis-lab")
REPO = Path(__file__).resolve().parents[3]
DATA = MAIN / "data/tennis_multivew/processed/meiji_3cam/dataset"
CACHE = MAIN / "outputs/tennis_scene/precompute/meiji_dino_vitpose"
CLIPS = ("video_000/clip_000", "video_001/clip_003", "video_002/clip_017")
CHECKPOINT = "court_detection/multiscale-depth3-local-rtx/logs/version_4/checkpoints/court-detection-epoch=17.ckpt"
EXPECTED_SHA = "b863df1f01f00d2ff00a21d56a461879e3c5af193a9f32cec47a88d2842f8383"
NEAR = {2, 3, 5, 7}


def validate_roi(
    roi: tuple[int, int, int, int], size: tuple[int, int]
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = roi
    if not (0 <= x0 < x1 <= size[0] and 0 <= y0 < y1 <= size[1]):
        raise ValueError(f"Invalid ROI {roi} for image {size}")
    return roi


def union_roi(
    base: tuple[int, int, int, int], projected: np.ndarray, size: tuple[int, int]
) -> tuple[int, int, int, int]:
    validate_roi(base, size)
    if projected.shape != (14, 2) or not np.isfinite(projected).all():
        raise ValueError("Saved H must project 14 finite points")
    low = np.minimum(base[:2], projected.min(0)) - 20
    high = np.maximum(base[2:], projected.max(0)) + 20
    bounds = np.concatenate(
        [np.clip(np.floor(low), 0, size), np.clip(np.ceil(high), 0, size)]
    ).astype(int)
    return validate_roi(
        (int(bounds[0]), int(bounds[1]), int(bounds[2]), int(bounds[3])), size
    )


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def snapshot(paths: set[Path]) -> dict[str, Any]:
    result = {}
    for path in sorted(paths):
        before = path.stat()
        digest = dual_sha256(path)
        after = path.stat()

        def identity(st: os.stat_result) -> list[int]:
            return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]

        if identity(before) != identity(after):
            raise ValueError(f"Input changed while hashing {path}")
        result[str(path)] = {"dual_sha256": digest, "stat": identity(after)}
    return result


def difference(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    if actual.shape != expected.shape:
        return {
            "exact": False,
            "actual_shape": list(actual.shape),
            "expected_shape": list(expected.shape),
        }
    return {
        "exact": bool(np.array_equal(actual, expected)),
        "max_abs": float(np.max(np.abs(actual - expected))),
        "different_elements": int(np.count_nonzero(actual != expected)),
    }


def manual_error(
    path: Path,
    camera_index: int,
    indices: np.ndarray,
    prediction: np.ndarray,
    size: tuple[int, int],
) -> dict[str, Any]:
    manual = (
        np.asarray(json.loads(path.read_text())["keypoints"])[camera_index, indices]
        * size
    )
    if manual.shape != (len(indices), 14, 2) or not np.isfinite(manual).all():
        raise ValueError("Invalid manual evaluation input")
    errors = np.linalg.norm(manual - prediction[None], axis=-1)
    return {
        "source": str(path),
        "per_frame_per_channel_px": errors.tolist(),
        "median_px": float(np.median(errors)),
        "p95_px": float(np.percentile(errors, 95)),
    }


def contact_sheet(
    path: Path,
    frames: list[np.ndarray],
    indices: np.ndarray,
    variants: list[dict[str, Any]],
) -> None:
    rows = []
    for variant in variants:
        cells = []
        for index in (0, len(indices) // 2, len(indices) - 1):
            raw = frames[index]
            overlay = raw.copy()
            x0, y0, x1, y1 = variant["roi_xyxy"]
            cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), (255, 255, 0), 2)
            if "predicted_points_px" in variant:
                points = np.rint(variant["predicted_points_px"]).astype(int)
                for a, b in COURT_SKELETON:
                    if max(a, b) < 14:
                        cv2.line(
                            overlay, tuple(points[a]), tuple(points[b]), (0, 255, 0), 2
                        )
                for channel, point in enumerate(points):
                    cv2.circle(overlay, tuple(point), 4, (0, 0, 255), -1)
                    cv2.putText(
                        overlay,
                        str(channel),
                        tuple(point + 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )
            pair = np.concatenate(
                [cv2.resize(raw, (768, 432)), cv2.resize(overlay, (768, 432))], axis=0
            )
            cv2.putText(
                pair,
                f"{variant['variant']} frame={indices[index]} {variant['status']} | raw above / fitted H below",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )
            cells.append(pair)
        rows.append(np.concatenate(cells, axis=1))
    if not cv2.imwrite(str(path), np.concatenate(rows, axis=0)):
        raise OSError(f"Cannot write {path}")


def run(output: Path) -> None:
    if not os.environ.get("TENNIS_RUN_ID") or os.environ.get(
        "TENNIS_GPU_RESOURCE"
    ) not in {"half", "all"}:
        raise RuntimeError("A shared training-queue reservation is required")
    if (
        Path(os.environ.get("TRAINING_QUEUE_DIR", "")).resolve()
        != MAIN / ".training_queue"
    ):
        raise RuntimeError("TRAINING_QUEUE_DIR must be the shared main-repo queue")
    if output.exists() or output.is_relative_to(DATA) or output.is_relative_to(CACHE):
        raise ValueError("Output must be new and outside source dataset/cache")
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = MAIN / "outputs" / CHECKPOINT
    settings = StaticCourtSettings(
        Path(CHECKPOINT),
        9,
        0.15,
        10,
        20.0,
        15.0,
        {"cam0": 0.25, "cam1": 0.25, "cam2": None},
    )
    physical = court_keypoints_3d(CourtConfig(0.914, None)).numpy()[:14, :2]
    clips = [
        ClipManifest.load(DATA / "videos" / c.split("/")[0] / "clips" / c.split("/")[1])
        for c in CLIPS
    ]
    manual = clips[0].clip_dir / "annotations/manual_court_kp_result.json"
    paths = {
        checkpoint,
        manual,
        Path(__file__).resolve(),
        *REPO.glob("src/tennis_scene/configs/build_slcs_dataset*.yaml"),
        *REPO.glob("src/**/*.py"),
    }
    for clip in clips:
        paths.add(clip.manifest_path)
        for version in ("s42-001", "s42-004"):
            paths.update(
                CACHE / version / clip.clip_id / f for f in ("court.json", "court.npz")
            )
        for camera in ("cam0", "cam1"):
            paths.update(
                (
                    clip.media_path(camera),
                    clip.clip_dir / "outsource" / f"{camera}_annotations.json",
                    CACHE / "s42-001" / clip.clip_id / f"{camera}_court_samples.npz",
                )
            )
    before = snapshot(paths)
    write_json(output / "inputs_before.json", before)
    if before[str(checkpoint)]["dual_sha256"] != EXPECTED_SHA:
        raise ValueError("Checkpoint identity mismatch")
    provenance = {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "command": sys.argv,
        "cwd": str(Path.cwd()),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "environment": {
            k: os.environ.get(k)
            for k in (
                "TENNIS_RUN_ID",
                "TENNIS_REPRO_DIR",
                "TRAINING_QUEUE_DIR",
                "TENNIS_GPU_RESOURCE",
                "TENNIS_GPU_SLOT",
                "CUDA_VISIBLE_DEVICES",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
        "settings": {**settings.__dict__, "checkpoint": CHECKPOINT},
        "device": "cuda",
        "subpixel_refine": True,
        "peak_threshold": 0.15,
        "status": "running",
    }
    write_json(output / "runtime.json", provenance)
    roots = RuntimePathRoots(
        REPO,
        MAIN / "data",
        MAIN / "outputs",
        MAIN / "artifacts",
        MAIN / "outputs",
        MAIN / ".cache",
        MAIN / "third_party",
    )
    all_results = []
    try:
        predictor = CourtKeypointPredictor.load_from_checkpoint(
            settings.checkpoint,
            resolver=PathResolver(roots),
            device="cuda",
            subpixel_refine=True,
            peak_threshold=0.15,
        )
        provenance["gpu_name"] = torch.cuda.get_device_name()
        for clip in clips:
            current = CACHE / "s42-004" / clip.clip_id
            original = CACHE / "s42-001" / clip.clip_id
            for name in ("court.json", "court.npz"):
                if (
                    before[str(current / name)]["dual_sha256"]
                    != before[str(original / name)]["dual_sha256"]
                ):
                    raise ValueError(f"Cache versions disagree: {clip.clip_id}/{name}")
            metadata = json.loads((current / "court.json").read_text())
            identity = metadata["identity"]
            if (
                identity["checkpoint_sha256"] != EXPECTED_SHA
                or identity["settings"] != provenance["settings"]
                or identity["clip_sha256"]
                != before[str(clip.manifest_path)]["dual_sha256"]
            ):
                raise ValueError("Cached court identity/settings mismatch")
            indices = np.unique(
                np.linspace(0, clip.num_frames - 1, 9).round().astype(int)
            )
            if len(indices) != 9:
                raise ValueError("Exactly nine frames required")
            for camera in ("cam0", "cam1"):
                for path, digest in (
                    (clip.media_path(camera), identity["video_sha256"][camera]),
                    (
                        clip.clip_dir / "outsource" / f"{camera}_annotations.json",
                        identity["ball_annotation_sha256"][camera],
                    ),
                ):
                    if before[str(path)]["dual_sha256"] != digest:
                        raise ValueError(f"Cached source identity mismatch: {path}")
                ci = clip.camera_index(camera)
                with np.load(current / "court.npz") as saved:
                    saved_h = saved["homographies"][ci].copy()
                with np.load(original / f"{camera}_court_samples.npz") as cached:
                    cached_raw, cached_scores = (
                        cached["keypoints_px"].copy(),
                        cached["scores"].copy(),
                    )
                    if not np.array_equal(indices, cached["frame_indices"]):
                        raise ValueError("Cached sample indices mismatch")
                base = validate_roi(
                    ball_guided_roi(clip, camera, 0.25), (clip.width, clip.height)
                )
                if list(base) != metadata["diagnostics"][ci]["source_roi_xyxy"]:
                    raise ValueError("Cached ROI mismatch")
                rois = (
                    base,
                    validate_roi(
                        ball_guided_roi(clip, camera, 0.5), (clip.width, clip.height)
                    ),
                    union_roi(
                        base,
                        cv2.perspectiveTransform(physical[None], saved_h)[0],
                        (clip.width, clip.height),
                    ),
                )
                capture = cv2.VideoCapture(str(clip.media_path(camera)))
                frames = []
                try:
                    for index in indices:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                        ok, frame = capture.read()
                        if not ok or frame.shape[:2] != (clip.height, clip.width):
                            raise ValueError(
                                f"Invalid frame {clip.clip_id}/{camera}/{index}"
                            )
                        frames.append(frame)
                finally:
                    capture.release()
                view = output / clip.clip_id / camera
                view.mkdir(parents=True)
                variants = []
                for label, roi in zip(
                    ("A_margin025", "B_margin050", "C_savedH_union20"),
                    rois,
                    strict=True,
                ):
                    x0, y0, x1, y1 = roi
                    raw_list, score_list = [], []
                    for frame in frames:
                        pred = predictor.predict(
                            cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
                        )
                        raw_list.append(pred.keypoints[:, 0].cpu().numpy() + [x0, y0])
                        score_list.append(pred.scores[:, 0].cpu().numpy())
                    raw, scores = np.stack(raw_list), np.stack(score_list)
                    np.savez_compressed(
                        view / f"{label}_raw.npz",
                        keypoints_px=raw,
                        scores=scores,
                        frame_indices=indices,
                    )
                    counts = ((scores >= 0.15) & np.isfinite(raw).all(-1)).sum(0)
                    result: dict[str, Any] = {
                        "clip": clip.clip_id,
                        "camera": camera,
                        "variant": label,
                        "roi_xyxy": list(roi),
                        "support_frames_per_channel": counts.tolist(),
                        "supported_channels": np.flatnonzero(counts >= 5).tolist(),
                        "status": "rejected",
                        "inlier_channels": None,
                        "near_baseline_inlier_count": None,
                        "fit_error_px_median": None,
                        "fit_error_px_p95": None,
                        "predicted_points_px": None,
                    }
                    del result["predicted_points_px"]
                    if label.startswith("A_"):
                        result["repeatability"] = {
                            "raw": difference(raw, cached_raw),
                            "scores": difference(scores, cached_scores),
                        }
                    try:
                        kp, h, diagnostic = fit_static_court(
                            raw,
                            scores,
                            size=(clip.width, clip.height),
                            settings=settings,
                        )
                        points = kp * np.asarray((clip.width, clip.height), np.float32)
                        result.update(diagnostic)
                        result.update(
                            status="fit_returned",
                            predicted_points_px=points.tolist(),
                            homography=h.tolist(),
                            near_baseline_inlier_count=len(
                                NEAR & set(diagnostic["inlier_channels"])
                            ),
                        )
                        if label.startswith("A_"):
                            result["repeatability"]["homography"] = difference(
                                h, saved_h
                            )
                    except ValueError as error:
                        result["rejection"] = str(error)
                    if clip.clip_id == CLIPS[0] and "predicted_points_px" in result:
                        result["manual_evaluation_only"] = manual_error(
                            manual, ci, indices, points, (clip.width, clip.height)
                        )
                    if label.startswith("A_"):
                        result["repeatability_established"] = "homography" in result[
                            "repeatability"
                        ] and all(
                            item["exact"] for item in result["repeatability"].values()
                        )
                    write_json(view / f"{label}.json", result)
                    variants.append(result)
                contact_sheet(view / "contact_sheet.jpg", frames, indices, variants)
                all_results.extend(variants)
        repeated = all(
            r["repeatability_established"]
            for r in all_results
            if r["variant"].startswith("A_")
        )
        write_json(
            output / "summary.json",
            {
                "status": "diagnostic_complete"
                if repeated
                else "repeatability_not_established",
                "automatic_selection": None,
                "residuals_are_not_ground_truth_error": True,
                "views": all_results,
            },
        )
        if not repeated:
            raise RuntimeError(
                "Baseline cache repeatability not established; comparison is not a successful repeatability result"
            )
        provenance["status"] = "diagnostic_complete_pending_image_review"
    except BaseException as error:
        provenance.update(status="failed", error=repr(error))
        raise
    finally:
        after = snapshot(paths)
        write_json(output / "inputs_after.json", after)
        if before != after:
            provenance.update(
                status="failed", error="Source/checkpoint changed during execution"
            )
        write_json(output / "runtime.json", provenance)
        if before != after:
            raise RuntimeError("Source/checkpoint changed during execution")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New absolute output directory; must not exist",
    )
    args = parser.parse_args()
    if not args.output.is_absolute():
        parser.error("--output must be absolute")
    run(args.output.resolve())


if __name__ == "__main__":
    main()
