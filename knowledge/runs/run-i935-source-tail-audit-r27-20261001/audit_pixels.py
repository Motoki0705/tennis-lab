"""Stream the exact 810 frames through both decoders and the store encoding recipe."""

from __future__ import annotations

import hashlib
import json
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
import torch
from audit_frames import CACHE, PLAN, ROOT, STORE, Inputs, csv_rows, read, write

from src.tasks.ball_detection.data.store import BallFrameStore, shard_name
from src.utils.resource_guard import available_ram_bytes
from src.utils.video import BgrToTensorTransform, OpenCVVideoFrameReader


def encode_store_frame(frame: np.ndarray, size: tuple[int, int], quality: int) -> tuple[np.ndarray, np.ndarray]:
    """Replay the existing store builder exactly, without writing a new store."""
    resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("Re-encoded JPEG failed to decode")
    return encoded, decoded


def difference(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    if a.shape != b.shape:
        raise ValueError("Compare corresponding pixels at the same dimensions")
    delta = a.astype(np.float32) - b.astype(np.float32)
    return {"mae": float(np.abs(delta).mean()), "rmse": float(np.sqrt((delta * delta).mean())),
            "max_abs": float(np.abs(delta).max())}


def main() -> None:
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    output = ROOT / "pixels"
    output.mkdir(exist_ok=False)
    inputs = Inputs()
    plan = read(inputs.pin(PLAN))
    metadata = read(inputs.pin(STORE / "metadata.json"))
    inputs.pin(STORE / "index.npz")
    cache_manifest = read(inputs.pin(CACHE / "manifest.json"))
    store = BallFrameStore(STORE)
    transform = BgrToTensorTransform(image_size=(288, 512), normalize_imagenet=False)
    rows, cameras = [], []
    start = time.monotonic()
    minimum_available = available_ram_bytes()
    for rec in plan["cameras"]:
        cam = rec["camera"]
        clip = store.clip_by_id(f"{plan['clip_id']}/{cam}")
        inputs.pin(Path(rec["video"]), clip.media_sha256)
        history = next(r for r in cache_manifest["clips"] if r["clip"]["clip_id"] == clip.clip_id)
        inputs.pin(STORE / "shards" / shard_name(clip.index), history["jpeg_shard_sha256"])
        cv_frames = OpenCVVideoFrameReader(Path(rec["video"]))
        count = 0
        with av.open(rec["video"]) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 2
            for packet, frame in zip(cv_frames, container.decode(stream), strict=True):
                f = packet.index
                row = store.row_of(clip, f)
                if (frame.pts is None or frame.time_base is None
                        or frame.pts * Fraction(frame.time_base) != int(store.frames["pts"][row]) * Fraction(clip.time_base)):
                    raise ValueError("Frame/PTS mismatch across source and store")
                raw = frame.to_ndarray(format="bgr24")
                encoded, reencoded = encode_store_frame(raw, (clip.width, clip.height), metadata["jpeg_quality"])
                jpeg = store.read_bgr(row)
                resized = cv2.resize(raw, (clip.width, clip.height), interpolation=cv2.INTER_AREA)
                # Compare RGB at the actual detector dimensions, in 0..255 units.
                tensors = {"source": transform(packet.frame), "store": transform(jpeg),
                           "reencoded": transform(reencoded), "resize_only": transform(resized)}
                measured: dict[str, Any] = {"camera": cam, "frame": f, "pts": int(frame.pts),
                    "opencv_pyav_equal": bool(np.array_equal(raw, packet.frame)),
                    "reencoded_jpeg_bytes_equal": bool(np.array_equal(encoded.reshape(-1), store.read_jpeg(row))),
                    "reencoded_store_pixels_equal": bool(np.array_equal(reencoded, jpeg)),
                    "detector_reencoded_store_equal": bool(torch.equal(tensors["reencoded"], tensors["store"]))}
                for name, a, b in (("decoder", raw, packet.frame), ("jpeg_at_720p", resized, jpeg)):
                    measured.update({f"{name}_{k}": v for k, v in difference(a, b).items()})
                for name, a, b in (("total", "source", "store"), ("resize_only", "source", "resize_only"),
                                   ("jpeg_only", "resize_only", "store")):
                    delta = difference(tensors[a].numpy() * 255., tensors[b].numpy() * 255.)
                    measured.update({f"detector_{name}_{k}": v for k, v in delta.items()})
                for name in ("source", "store", "reencoded"):
                    measured[f"detector_{name}_sha256"] = hashlib.sha256(tensors[name].numpy().tobytes()).hexdigest()
                rows.append(measured)
                count += 1
                minimum_available = min(minimum_available, available_ram_bytes())
                if minimum_available < 6 * 1024**3:
                    raise MemoryError("Host available memory fell below 6 GiB")
        if count != 270:
            raise ValueError(f"Expected 270 frames, got {count}")
        selected = rows[-270:]
        cameras.append({"camera": cam, "frames": count,
            **{key: sum(r[key] for r in selected) for key in ("opencv_pyav_equal", "reencoded_jpeg_bytes_equal", "reencoded_store_pixels_equal", "detector_reencoded_store_equal")},
            **{key: {"mean": float(np.mean([r[key] for r in selected])), "min": min(r[key] for r in selected), "max": max(r[key] for r in selected)}
               for key in selected[0] if key.endswith(("mae", "rmse", "max_abs"))}})
        print(json.dumps(cameras[-1]), flush=True)
    csv_rows(output / "frames.csv", rows)
    inputs.finish(output / "input_sha256.json")
    write(output / "summary.json", {"status": "complete", "camera_frames": len(rows), "cameras": cameras,
          "seconds": time.monotonic() - start, "minimum_host_available_bytes": minimum_available,
          "recipe": "PyAV bgr24 1920x1080 -> INTER_AREA 1280x720 -> JPEG quality90 -> OpenCV BGR -> INTER_LINEAR 512x288 -> RGB float32 /255",
          "pixel_units": "8-bit 0..255; aggregate means over equally sized frames; max is max across all pixels",
          "versions": {"cv2": cv2.__version__, "av": av.__version__, "torch": str(torch.__version__)}})


if __name__ == "__main__":
    main()
