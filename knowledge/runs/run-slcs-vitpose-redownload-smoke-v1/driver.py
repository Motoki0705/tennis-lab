"""One queued real-RGB ViTPose smoke; never hashes/replaces the checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest.mock import patch

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

from src.submodules.configuration import ViTPoseHeadConfig
from src.submodules.models import Pose2DRequest, TrackResult, ViTPosePose2D
from src.submodules.vendor.gvhmr.vitpose import VitPoseModel
from src.utils.video.reader import read_video_rgb


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def stat_record(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "inode": stat.st_ino,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "device": stat.st_dev,
    }


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def overlay(
    rgb: NDArray[np.uint8],
    boxes: NDArray[np.float32],
    keypoints: NDArray[np.float32],
    output: Path,
) -> None:
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    edges = [
        (5, 6),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
    ]
    colors = [(50, 230, 50), (40, 150, 255)]
    for person, (box, pose) in enumerate(zip(boxes, keypoints[:, 0], strict=True)):
        color = colors[person]
        x1, y1, x2, y2 = (round(float(v)) for v in box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            canvas,
            f"person {person}",
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )
        for a, b in edges:
            if min(pose[a, 2], pose[b, 2]) >= 0.3:
                cv2.line(
                    canvas,
                    tuple(int(v) for v in np.rint(pose[a, :2])),
                    tuple(int(v) for v in np.rint(pose[b, :2])),
                    color,
                    1,
                )
        for x, y, score in pose:
            if score >= 0.3:
                cv2.circle(canvas, (round(float(x)), round(float(y))), 2, color, -1)
    cv2.putText(
        canvas,
        "ViTPose smoke - predictions, not ground truth (confidence >= 0.3)",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    if not cv2.imwrite(str(output / "overlay.png"), canvas):
        raise RuntimeError("Cannot save overlay")
    tiles = []
    for person, box in enumerate(boxes):
        cx1, cy1, cx2, cy2 = (float(v) for v in box)
        margin = (cy2 - cy1) * 0.5
        crop = canvas[
            max(40, int(cy1 - margin)) : min(canvas.shape[0], int(cy2 + margin)),
            max(0, int(cx1 - margin)) : min(canvas.shape[1], int(cx2 + margin)),
        ]
        scale = 480 / crop.shape[0]
        crop = cv2.resize(
            crop, (round(crop.shape[1] * scale), 480), interpolation=cv2.INTER_NEAREST
        )
        cv2.putText(
            crop,
            f"person {person} (zoom)",
            (8, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        tiles.append(crop)
    if not cv2.imwrite(
        str(output / "overlay_crops.png"), np.concatenate(tiles, axis=1)
    ):
        raise RuntimeError("Cannot save zoomed overlay")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--installation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("TENNIS_RUN_ID"):
        raise RuntimeError("Run this GPU experiment through training-queue")
    args.output.mkdir(parents=True, exist_ok=False)
    metrics: dict[str, Any] = {
        "status": "running",
        "queue_id": os.environ["TENNIS_RUN_ID"],
    }
    model: ViTPosePose2D | None = None
    try:
        checkpoint = args.checkpoint.resolve(strict=True)
        installation = json.loads(args.installation.read_text())
        before = stat_record(checkpoint)
        assert before["inode"] == installation["new_inode"]
        assert before["size"] == installation["size"] == 2549075546
        frame = read_video_rgb(str(args.video), max_frames=1)[0]
        h, w = frame.shape[:2]
        with np.load(args.observations, allow_pickle=False) as saved:
            boxes = saved["boxes"][:, 0:1].copy()
            masks = saved["observed_masks"][:, 0].copy()
            support = saved["pose_supported_mask"][:, 0].copy()
            track_ids = saved["track_ids"].copy()
            detection_ids = saved["source_detection_ids"][:, 0].copy()
        assert boxes.shape == (2, 1, 4)
        assert np.isfinite(boxes).all() and masks.all() and support.all()
        assert (boxes[..., 2:] > boxes[..., :2]).all()
        assert (
            (boxes >= 0).all()
            and (boxes[..., ::2] < w).all()
            and (boxes[..., 1::2] < h).all()
        )
        tracks = TrackResult({i: torch.from_numpy(b) for i, b in enumerate(boxes)}, 1)
        xys = [tracks.bbx_xys(i, base_enlarge=1.2) for i in range(2)]
        head = ViTPoseHeadConfig(1280, 17, 2, (256, 256), (4, 4), 1, 0, ())
        root = Path.cwd()
        source_files = [
            "src/submodules/models/vitpose/pose2d.py",
            "src/submodules/vendor/gvhmr/vitpose/__init__.py",
            "src/submodules/vendor/gvhmr/hmr2/preproc.py",
            "src/submodules/vendor/gvhmr/hmr2/vit.py",
            "src/submodules/vendor/gvhmr/vitpose/heatmap_head.py",
            "src/tennis_scene/configs/build_slcs_dataset_base.yaml",
        ]
        provenance = {
            "video": str(args.video.resolve()),
            "video_stat": stat_record(args.video),
            "frame_index": 0,
            "frame_rgb_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
            "frame_shape": list(frame.shape),
            "observations": str(args.observations.resolve()),
            "observations_sha256": digest(args.observations),
            "boxes_xyxy": boxes.tolist(),
            "boxes_xys_enlarged": [x.tolist() for x in xys],
            "track_ids": track_ids.tolist(),
            "detection_ids": detection_ids.tolist(),
            "observed_masks": masks.tolist(),
            "pose_supported_mask": support.tolist(),
            "checkpoint": str(checkpoint),
            "checkpoint_before": before,
            "installation_record": str(args.installation.resolve()),
            "installation_sha256": digest(args.installation),
            "checkpoint_expected_sha256": "50e33f4077ef2a6bcfd7110c58742b24c5859b7798fb0eedd6d2215e0a8980bc",
            "checkpoint_hash_policy": "reference completed redownload validation; no checkpoint reread hash",
            "head_config": asdict(head),
            "precision": "bfloat16",
            "flip_test": True,
            "batch_size": 1,
            "device": "cuda:0",
            "driver_sha256": digest(Path(__file__)),
            "driver_untracked_at_execution": True,
            "execution_base_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "source_sha256": {p: digest(root / p) for p in source_files},
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "queue_id": os.environ["TENNIS_RUN_ID"],
            "queue_repro_dir": os.environ.get("TENNIS_REPRO_DIR"),
        }
        write_json(args.output / "provenance.json", provenance)
        np.savez_compressed(
            args.output / "input.npz",
            boxes=boxes,
            bbx_xys=torch.stack(xys).numpy(),
            frame_index=np.array(0),
            track_ids=track_ids,
        )
        strict_results: list[dict[str, Any]] = []
        original = VitPoseModel.load_state_dict

        def checked_load(
            module: torch.nn.Module,
            state_dict: Mapping[str, Any],
            strict: bool = True,
            assign: bool = False,
        ) -> Any:
            assert strict is True
            result = original(module, state_dict, strict=strict, assign=assign)
            strict_results.append(
                {
                    "strict": strict,
                    "missing_keys": list(result.missing_keys),
                    "unexpected_keys": list(result.unexpected_keys),
                }
            )
            return result

        model = ViTPosePose2D(
            checkpoint,
            device="cuda:0",
            flip_test=True,
            batch_size=1,
            head_config=head,
            precision="bfloat16",
        )
        start = time.perf_counter()
        with patch.object(VitPoseModel, "load_state_dict", checked_load):
            model.load()
        torch.cuda.synchronize()
        metrics["load_seconds"] = time.perf_counter() - start
        assert len(strict_results) == 1
        assert strict_results[0] == {
            "strict": True,
            "missing_keys": [],
            "unexpected_keys": [],
        }
        start = time.perf_counter()
        predictions = np.stack(
            [model.predict(Pose2DRequest(args.video, x)).keypoints.numpy() for x in xys]
        )
        torch.cuda.synchronize()
        metrics["forward_seconds"] = time.perf_counter() - start
        assert predictions.shape == (2, 1, 17, 3)
        assert np.isfinite(predictions).all()
        inside = (
            (predictions[..., 0] >= 0)
            & (predictions[..., 0] < w)
            & (predictions[..., 1] >= 0)
            & (predictions[..., 1] < h)
        )
        confidence = predictions[..., 2]
        np.savez_compressed(args.output / "predictions.npz", keypoints=predictions)
        overlay(frame, boxes[:, 0], predictions, args.output)
        after = stat_record(checkpoint)
        assert before == after, "Checkpoint changed during smoke"
        metrics.update(
            status="passed",
            strict_load=strict_results[0],
            shape=list(predictions.shape),
            all_finite=True,
            confidence_min=float(confidence.min()),
            confidence_mean=float(confidence.mean()),
            confidence_max=float(confidence.max()),
            confidence_ge_03_fraction=float((confidence >= 0.3).mean()),
            keypoints_inside_image_fraction=float(inside.mean()),
            per_person_inside_image_fraction=inside.mean(axis=(1, 2)).tolist(),
            checkpoint_after=after,
            gpu_name=torch.cuda.get_device_name(0),
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
            precision="bfloat16",
        )
    except Exception as exc:
        metrics.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if model is not None:
            model.unload()
        write_json(args.output / "metrics.json", metrics)
        print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
