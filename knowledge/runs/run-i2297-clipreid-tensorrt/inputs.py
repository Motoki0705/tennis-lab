"""Validate the frozen clip split and extract the actual CLIP visual-tower inputs."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CAMERAS = ("cam0", "cam1", "cam2", "cam3")
SPLITS = ("eval", "calibration")
INPUT_SHAPE = (3, 256, 128)
MODEL = "clipreid_vitb16_market1501"
CHECKPOINT = Path("ckpt/player_association/person_vit_clip_reid.pth")
CHECKPOINT_SHA256 = "bedc8fcc37f296045df19d901b162dd032bfa85d9b4f2406eff8313536d444dc"


def sha256_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_split(book):
    """Reject shared clips, camera media or inclusive raw-frame intervals."""
    clips = {split: [] for split in SPLITS}
    seen = set()
    for clip in book["clips"]:
        clip_id = clip["clip"]
        if clip_id in seen:
            raise ValueError(f"Duplicate or shared clip: {clip_id}")
        seen.add(clip_id)
        if tuple(s["camera"] for s in clip["streams"]) != CAMERAS:
            raise ValueError(f"Camera set or order changed: {clip_id}")
        clips[clip["split"]].append(clip)
        for s in clip["streams"]:
            lo, hi = s["source_frames_inclusive"]
            if not 0 <= lo <= hi:
                raise ValueError(f"Invalid raw frame interval: {clip_id}/{s['camera']}")
    if not all(clips.values()):
        raise ValueError("Both evaluation and calibration clips are required")
    for evaluation in clips["eval"]:
        for calibration in clips["calibration"]:
            eval_video = evaluation["clip"].split("/")[0]
            cal_video = calibration["clip"].split("/")[0]
            for eval_stream in evaluation["streams"]:
                for cal_stream in calibration["streams"]:
                    if eval_stream["sha256"] == cal_stream["sha256"]:
                        raise ValueError(
                            "Shared evaluation and calibration camera media"
                        )
                    if (
                        eval_video == cal_video
                        and eval_stream["camera"] == cal_stream["camera"]
                    ):
                        lo, hi = eval_stream["source_frames_inclusive"]
                        start, end = cal_stream["source_frames_inclusive"]
                        if max(lo, start) <= min(hi, end):
                            raise ValueError(
                                f"Shared raw source frames: {eval_video}/{eval_stream['camera']}"
                            )
    return {split: [c["clip"] for c in clips[split]] for split in SPLITS}


def eval_addresses(book):
    return [
        (clip["clip"], stream["camera"], obs["frame"], obs["player"])
        for clip in book["clips"]
        if clip["split"] == "eval"
        for stream in clip["streams"]
        for obs in stream["crops"]
    ]


def prepare(dataset, pins, output):
    from src.tasks.player_association.appearance.sampling import crop
    from src.utils.video import OpenCVVideoFrameReader

    book = json.loads(pins.read_text())
    split_ids = check_split(book)
    result = {split: [] for split in SPLITS}
    counts = {}
    for clip in book["clips"]:
        for stream in clip["streams"]:
            media = dataset / stream["media"]
            if sha256_file(media) != stream["sha256"]:
                raise ValueError(f"Media SHA-256 mismatch: {media}")
            by_frame = {}
            seen_rows = set()
            for obs in stream["crops"]:
                frame = obs["frame"]
                row = obs["detection_row"]
                if (
                    frame not in stream["sampled_frames"]
                    or not 0 <= frame < clip["num_frames"]
                ):
                    raise ValueError(
                        f"Crop frame outside the selection: {media}/{frame}"
                    )
                if row < 0 or row in seen_rows:
                    raise ValueError(
                        f"Missing or duplicated detection row: {media}/{row}"
                    )
                seen_rows.add(row)
                by_frame.setdefault(frame, []).append(obs)
            if not by_frame or list(by_frame) != sorted(by_frame):
                raise ValueError(f"Missing or unsorted crops: {media}")
            counts[f"{clip['clip']}/{stream['camera']}"] = len(seen_rows)
            for packet in OpenCVVideoFrameReader(media, max_frames=max(by_frame) + 1):
                for obs in by_frame.pop(packet.index, []):
                    pixels = crop(
                        packet.frame,
                        np.asarray(obs["xyxy"], dtype=np.float32),
                        INPUT_SHAPE[1:],
                    )
                    # The CLIP input RGB normalizer operates on float32 with mean=std=0.5.
                    result[clip["split"]].append(
                        (pixels - np.float32(0.5)) / np.float32(0.5)
                    )
            if by_frame:
                raise ValueError(
                    f"Video ended before all selected crop frames: {media}"
                )
    output.mkdir(parents=True, exist_ok=False)
    receipt = {
        "split": split_ids,
        "cross_split_shared_raw_frame_intervals": 0,
        "cross_split_shared_media": 0,
        "crops_per_camera": counts,
        "pins_sha256": sha256_file(pins),
        "tensors": {},
    }
    for split, crops in result.items():
        data = np.stack(crops)
        if (
            data.shape[1:] != INPUT_SHAPE
            or not np.isfinite(data).all()
            or np.abs(data).max() > 1
        ):
            raise ValueError(f"Invalid normalized {split} tensor")
        destination = output / f"{split}.npy"
        np.save(destination, data, allow_pickle=False)
        receipt["tensors"][split] = {
            "shape": list(data.shape),
            "dtype": str(data.dtype),
            "sha256": sha256_file(destination),
        }
    receipt["eval_players"] = {
        clip: dict(
            sorted(
                Counter(
                    str(row[3]) for row in eval_addresses(book) if row[0] == clip
                ).items()
            )
        )
        for clip in split_ids["eval"]
    }
    actual = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    lock = HERE / "tensor-lock.json"
    if lock.exists() and json.loads(actual) != json.loads(lock.read_text()):
        raise ValueError("Rebuilt tensors or pin document differ from tensor-lock.json")
    (output / "tensors.json").write_text(actual)
    print(actual, end="")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/tennis_multiview/processed/meiji_4cam/dataset"),
    )
    parser.add_argument("--pins", type=Path, default=HERE / "inputs.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.dataset, args.pins, args.output)


if __name__ == "__main__":
    main()
