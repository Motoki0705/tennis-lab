"""CPU overlay of the full clip_010 source-video outputs, 3 cameras at 1/4 speed."""

from __future__ import annotations

import argparse
import json
import resource
import time
from contextlib import ExitStack
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
import torch

from src.tasks.ball_refiner.data.targets import TargetReason
from src.tasks.ball_refiner.visualization.overlay import (
    CYAN,
    GREEN,
    ORANGE,
    RED,
    draw_mixture,
    mark,
    text_line,
)
from src.tennis_scene.pipeline.components.ball_detection import BallDetectionOutput
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.utils.checksum import dual_sha256
from src.utils.resource_guard import available_ram_bytes

BUNDLE = Path(__file__).resolve().parent
SOURCE_RUN = BUNDLE.parent / "run-i935-source-check-retry-r25-20260930"


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def view(
    raw: np.ndarray, prediction: dict[str, np.ndarray], labels: dict[str, np.ndarray],
    candidate_px: np.ndarray, candidate_valid: bool, frame: int, crop: tuple[int, int, int, int],
) -> np.ndarray:
    x, y, width, height = crop
    output: np.ndarray = cv2.resize(raw[y:y + height, x:x + width], (640, 360), interpolation=cv2.INTER_AREA)
    factor = np.array([640 / width, 360 / height])
    source_scale = np.asarray(prediction["source_size_wh"], np.float64) - 1
    offset = np.array([x, y])
    # Same DΣDᵀ as the full view; only the origin changes for the magnified view.
    means = prediction["means"][frame].astype(np.float64) - offset / source_scale
    display_scale = source_scale * factor
    draw_mixture(output, means, prediction["scale_tril"][frame], prediction["mixture_logits"][frame],
                 (float(display_scale[0]), float(display_scale[1])), point_summary="top_component")
    if candidate_valid:
        mark(output, (candidate_px - offset) * factor, RED, cv2.MARKER_DIAMOND)
    reason = TargetReason(int(labels["target_reason"][frame]))
    if reason in (TargetReason.OBSERVED, TargetReason.INTERPOLATED, TargetReason.OCCLUSION_ESTIMATED):
        point = (labels["target_uv"][frame].astype(np.float64) * source_scale - offset) * factor
        mark(output, point, GREEN if reason == TargetReason.OBSERVED else ORANGE, cv2.MARKER_SQUARE)
    return output


def main(output: Path) -> None:
    if not output.is_absolute() or output.exists():
        raise ValueError("Use a new absolute output directory")
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    started = time.monotonic()
    plan = read(SOURCE_RUN / "plan.json")
    gate = read(BUNDLE / "results/gate.json")
    if plan["clip_id"] != "meiji/video_000/clip_010":
        raise ValueError("Only the requested validation clip is allowed")
    source_root = Path(plan["report"])
    hashes = read(BUNDLE / "results/input_sha256.json")
    hashes[str(BUNDLE / "results/gate.json")] = dual_sha256(BUNDLE / "results/gate.json")
    inputs = []
    for row in plan["cameras"]:
        camera = row["camera"]
        label_path = BUNDLE / f"results/{camera}-labels.npz"
        hashes[str(label_path)] = dual_sha256(label_path)
        with np.load(label_path, allow_pickle=False) as saved:
            labels = dict(saved)
        with np.load(source_root / camera / "execute.npz", allow_pickle=False) as saved:
            prediction = dict(saved)
        np.testing.assert_array_equal(labels["pts"], prediction["pts"])
        receipt = read(source_root / camera / "execute.json")
        store = ClipStore(source_root / camera / "store", receipt["source"], memory_entries=0)
        reference = store.active(f"ball_detection/{camera}")
        if reference is None:
            raise ValueError("Missing source-video detector artifact")
        detections = store.load(reference, ArtifactCodec(BallDetectionOutput))
        evidence = detections.evidence
        if evidence is None or detections.camera_id != camera:
            raise ValueError("Require source-video e9 candidate evidence")
        np.testing.assert_array_equal(detections.frame_indices, prediction["frame_indices"])
        if (evidence.candidate_scores[:, 0:1] < evidence.candidate_scores).any():
            raise ValueError("Decoder slot0 is not the highest-scored candidate")
        inputs.append((row, prediction, labels, evidence))
    for name, expected in hashes.items():
        if dual_sha256(Path(name)) != expected:
            raise ValueError(f"Video input changed: {name}")
    output.mkdir(parents=True)
    video = output / "clip_010-source-3cam-4x-slow.mp4"
    snapshots = (0, 67, 135, 202, 269)
    minimum_available = available_ram_bytes()
    frame_metadata = []
    with ExitStack() as stack:
        decoders = []
        rates = []
        for row, _, _, _ in inputs:
            container = stack.enter_context(av.open(row["video"]))
            source_stream = container.streams.video[0]
            source_stream.codec_context.thread_count = 1
            if source_stream.average_rate is None:
                raise ValueError("Source video has no declared frame rate")
            rates.append(Fraction(source_stream.average_rate))
            decoders.append(iter(container.decode(source_stream)))
        if len(set(rates)) != 1:
            raise ValueError("Camera rates are not synchronized")
        rate = rates[0] / 4
        destination = stack.enter_context(av.open(str(video), "w", options={"movflags": "+faststart"}))
        stream = destination.add_stream("libx264", rate=rate)
        stream.width, stream.height, stream.pix_fmt = 1920, 960, "yuv420p"
        stream.codec_context.thread_count = 1
        stream.options = {"crf": "19", "preset": "fast"}
        for frame_index in range(270):
            minimum_available = min(minimum_available, available_ram_bytes())
            if minimum_available < 6 * 1024**3:
                raise RuntimeError("Rendering requires at least 6 GiB MemAvailable")
            canvas: np.ndarray = np.zeros((960, 1920, 3), np.uint8)
            frame_rows = []
            for column, ((row, prediction, labels, evidence), decoder) in enumerate(zip(inputs, decoders, strict=True)):
                decoded = next(decoder)
                if decoded.pts != int(prediction["pts"][frame_index]) or str(decoded.time_base) != str(prediction["time_base"]):
                    raise ValueError(f"Source-video PTS mismatch: {row['camera']}/{frame_index}")
                raw = decoded.to_ndarray(format="bgr24")
                if raw.shape != (1080, 1920, 3):
                    raise ValueError("Source video dimensions changed")
                source_scale = np.array([1919., 1079.])
                reason = TargetReason(int(labels["target_reason"][frame_index]))
                located = reason in (TargetReason.OBSERVED, TargetReason.INTERPOLATED, TargetReason.OCCLUSION_ESTIMATED)
                top = int(prediction["mixture_logits"][frame_index].argmax())
                mean = prediction["means"][frame_index, top].astype(np.float64) * source_scale
                det = evidence.candidate_uv_px[frame_index, 0]
                valid = bool(evidence.candidate_valid[frame_index, 0])
                crop = (0, 0, 1920, 1080)
                zoom_text = "No position label: full view repeated"
                if located:
                    target = labels["target_uv"][frame_index].astype(np.float64) * source_scale
                    x = int(np.clip(round(float(target[0])) - 192, 0, 1920 - 384))
                    y = int(np.clip(round(float(target[1])) - 108, 0, 1080 - 216))
                    crop = (x, y, 384, 216)
                    zoom_text = "5x view magnification; crop follows label"
                panel = canvas[:, column * 640:(column + 1) * 640]
                text_line(panel, f"{row['camera']} | frame {frame_index:03d}/269 | {reason.name}", 10, 22, size=.55)
                weight = float(prediction["weights"][frame_index, top])
                text_line(panel, f"Source t={float(decoded.pts * decoded.time_base):.3f}s | heaviest K{top} w={weight:.3f}",
                          10, 44, CYAN, .5)
                if reason == TargetReason.OBSERVED:
                    error = float(np.linalg.norm(mean - target))
                    det_text = f"{np.linalg.norm(det - target):.1f}" if valid else "missing"
                    text_line(panel, f"Observed error: e9 {det_text}px | anchored {error:.1f}px", 10, 65, size=.5)
                else:
                    text_line(panel, "Not included in observed-GT accuracy", 10, 65, ORANGE, .5)
                panel[74:434] = view(raw, prediction, labels, det, valid, frame_index, (0, 0, 1920, 1080))
                text_line(panel, zoom_text, 10, 455, size=.5)
                in_crop = (crop[0] <= mean[0] < crop[0] + crop[2] and crop[1] <= mean[1] < crop[1] + crop[3])
                text_line(panel, "Cyan mean outside crop: see full view" if not in_crop else "2-sigma component ellipses; alpha = weight",
                          10, 475, CYAN, .47)
                panel[482:842] = view(raw, prediction, labels, det, valid, frame_index, crop)
                frame_rows.append({"camera": row["camera"], "frame": frame_index, "pts": decoded.pts,
                                   "crop_xywh": crop, "heaviest_component": top, "weight": weight,
                                   "top_component_mean_px": mean.tolist(), "detector_top1_valid": valid,
                                   "detector_top1_px": det.tolist(), "label_reason": reason.name})
            text_line(canvas, "clip_010 | SOURCE MP4 OUTPUTS | 4x slow (all 270 frames) | e9 + anchored_12k seed42 | covariance x1.812515", 12, 866, size=.64)
            text_line(canvas, "GREEN square: observed GT | ORANGE square: estimated label | RED diamond: e9 top-1 | CYAN +: heaviest-component mean", 12, 891, size=.59)
            text_line(canvas, "Ellipses are individual component 2-sigma, NOT mixture HDR95. Top: full view. Bottom: label-centered crop (unknown: full view).", 12, 917, size=.58)
            text_line(canvas, "Fixed B gate: FAIL (pooled p90 +46.34 px > +5 px); current pipeline default retained. No context or new GPU inference.", 12, 945, ORANGE, .6)
            encoded = av.VideoFrame.from_ndarray(canvas, format="bgr24")
            encoded.pts, encoded.time_base = frame_index, 1 / rate
            for packet in stream.encode(encoded):
                destination.mux(packet)
            if frame_index in snapshots and not cv2.imwrite(str(output / f"frame-{frame_index:03d}.jpg"), canvas):
                raise OSError("Could not save video snapshot")
            frame_metadata.extend(frame_rows)
        for decoder in decoders:
            if next(decoder, None) is not None:
                raise ValueError("Unexpected frames after 269")
        for packet in stream.encode():
            destination.mux(packet)
    with av.open(str(video)) as verification:
        frames = list((f.pts, f.time_base, f.width, f.height) for f in verification.decode(video=0))
        if len(frames) != 270 or any((w, h) != (1920, 960) for _, _, w, h in frames):
            raise ValueError("Rendered movie frame count or dimensions differ")
        duration = float(270 / rate)
        if abs(float(frames[-1][0] * frames[-1][1]) - float(269 / rate)) > 1e-4:
            raise ValueError("Rendered movie timing differs")
    for name, expected in hashes.items():
        if dual_sha256(Path(name)) != expected:
            raise ValueError(f"Video input changed during rendering: {name}")
    metadata = {"video": str(video), "sha256": dual_sha256(video), "bytes": video.stat().st_size,
                "source_fps": str(rates[0]), "playback_fps": str(rate), "frames": 270, "camera_frames": 810,
                "duration_seconds": duration, "size_wh": [1920, 960], "slow_factor": 4,
                "cpu_seconds_wall": time.monotonic() - started, "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                "minimum_host_available_bytes": minimum_available, "b_gate_passed": gate["b_gate_passed"],
                "input_sha256": hashes, "frame_metadata": frame_metadata,
                "snapshots": {p.name: dual_sha256(p) for p in sorted(output.glob("*.jpg"))}}
    (output / "video.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in metadata.items() if k not in ("input_sha256", "frame_metadata")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
