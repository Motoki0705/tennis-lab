"""Capture the production DINO batch at the network boundary without model weights."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from inputs import check_split, sha256_file

from src.submodules.models.dino.architecture import (
    DINO_IMAGE_MEAN,
    DINO_IMAGE_STD,
    DINO_NUM_CLASSES,
)
from src.submodules.models.dino.person_detector import (
    DinoPersonDetector,
    PersonDetectionRequest,
)
from src.utils.video import OpenCVVideoFrameReader

HERE = Path(__file__).resolve().parent
PINS = HERE.parent / "run-i2297-clipreid-tensorrt" / "inputs.json"
LOCK = HERE / "tensor-lock.json"
DATASET = Path("data/tennis_multiview/processed/meiji_4cam/dataset")
CHECKPOINT = Path("ckpt/dino/checkpoint0029_4scale_swin.pth")
REPOSITORY = Path("third_party/DINO")
SHORT_SIDE = 800
MAX_LONG_SIDE = 1333
CONFIDENCE = 0.3


class BatchRecorder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch = None

    def forward(self, batch):
        self.batch = batch
        return {
            "pred_logits": batch.new_empty((len(batch), 0, DINO_NUM_CLASSES)),
            "pred_boxes": batch.new_empty((len(batch), 0, 4)),
        }


def input_recorder():
    detector = DinoPersonDetector(
        CHECKPOINT.resolve(),
        REPOSITORY.resolve(),
        device="cpu",
        confidence=CONFIDENCE,
        short_side=SHORT_SIDE,
        max_long_side=MAX_LONG_SIDE,
    )
    recorder = BatchRecorder()
    detector._model = recorder
    return detector, recorder


def tensor_digest(directory, samples):
    digest_rows = []
    for row in samples:
        digest = sha256_file(directory / row["file"])
        if digest != row["sha256"]:
            raise ValueError(f"Fixed tensor digest changed: {row['file']}")
        digest_rows.append(f"{digest}  {row['file']}\n")
    return hashlib.sha256("".join(digest_rows).encode()).hexdigest()


def bucket_name(shape):
    return f"{shape[-2]}x{shape[-1]}"


def require_bucket(lock, bucket):
    if bucket not in lock["buckets"]:
        raise ValueError(f"Out-of-profile DINO bucket: {bucket}")
    return tuple(lock["buckets"][bucket]["profile"]["opt"])


@torch.inference_mode()
def capture(dataset, output, *, record_lock=False):
    torch.set_num_threads(4)
    if record_lock and LOCK.exists():
        raise ValueError("The tracked tensor lock already exists")
    if not record_lock and not LOCK.exists():
        raise ValueError("Capture a new lock with --record-lock before reproducing")
    pins = json.loads(PINS.read_text())
    split_ids = check_split(pins)
    output.mkdir(parents=True)
    detector, recorder = input_recorder()
    samples = []
    for clip in pins["clips"]:
        for stream in clip["streams"]:
            media = dataset / stream["media"]
            if sha256_file(media) != stream["sha256"]:
                raise ValueError(f"Video digest changed: {media}")
            frames = stream["sampled_frames"]
            if (
                frames != sorted(set(frames))
                or not 0 <= min(frames) <= max(frames) < clip["num_frames"]
            ):
                raise ValueError(f"Invalid sampled frames: {media}")
            pending = set(frames)
            for packet in OpenCVVideoFrameReader(media, max_frames=max(frames) + 1):
                if packet.index not in pending:
                    continue
                detector._predict_batch_impl(
                    [PersonDetectionRequest(packet.frame)], batch_size=1
                )
                batch = recorder.batch
                if (
                    batch is None
                    or batch.dtype != torch.float32
                    or not torch.isfinite(batch).all()
                ):
                    raise ValueError(f"Invalid captured batch: {media}/{packet.index}")
                file = f"{clip['clip']}/{stream['camera']}/{packet.index:04d}.npy"
                destination = output / file
                destination.parent.mkdir(parents=True, exist_ok=True)
                np.save(destination, batch.numpy(), allow_pickle=False)
                samples.append(
                    {
                        "clip": clip["clip"],
                        "split": clip["split"],
                        "camera": stream["camera"],
                        "frame": packet.index,
                        "original_hw": list(packet.frame.shape[:2]),
                        "shape": list(batch.shape),
                        "file": file,
                        "sha256": sha256_file(destination),
                    }
                )
                pending.remove(packet.index)
            if pending:
                raise ValueError(
                    f"Video ended before all fixed frames: {media}: {sorted(pending)}"
                )

    buckets = {}
    for name in sorted({bucket_name(row["shape"]) for row in samples}):
        rows = [row for row in samples if bucket_name(row["shape"]) == name]
        shape = rows[0]["shape"]
        counts = Counter(row["split"] for row in rows)
        if set(counts) != {"eval", "calibration"}:
            raise ValueError(f"Missing eval or calibration frames for {name}")
        representatives = [
            row["file"]
            for row in rows
            if (row["clip"], row["frame"])
            in {("video_000/clip_000", 178), ("video_000/clip_037", 146)}
        ]
        buckets[name] = {
            "profile": dict.fromkeys(("min", "opt", "max"), shape),
            "counts": dict(counts),
            "representative": representatives[0],
        }
    receipt = {
        "pins_sha256": sha256_file(PINS),
        "split": split_ids,
        "cross_split_shared_media": 0,
        "cross_split_shared_raw_frame_intervals": 0,
        "preprocessing": {
            "short_side": SHORT_SIDE,
            "max_long_side": MAX_LONG_SIDE,
            "channels": "RGB",
            "mean": list(DINO_IMAGE_MEAN),
            "std": list(DINO_IMAGE_STD),
            "dtype": "float32",
        },
        "tensor_set_sha256": tensor_digest(output, samples),
        "buckets": buckets,
        "samples": samples,
    }
    if not record_lock and receipt != json.loads(LOCK.read_text()):
        raise ValueError("Rebuilt tensors or pins differ from tensor-lock.json")
    (output / "tensors.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(receipt["tensor_set_sha256"], buckets, flush=True)


def rows_for(lock, bucket, split):
    shape = require_bucket(lock, bucket)
    return [
        row
        for row in lock["samples"]
        if row["split"] == split and tuple(row["shape"]) == shape
    ]


def load_sample(directory, row):
    sample = np.load(directory / row["file"], allow_pickle=False)
    if list(sample.shape) != row["shape"] or sample.dtype != np.float32:
        raise ValueError(f"Out-of-profile tensor: {row['file']}")
    return sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record-lock", action="store_true")
    args = parser.parse_args()
    capture(args.dataset, args.output, record_lock=args.record_lock)


if __name__ == "__main__":
    main()
