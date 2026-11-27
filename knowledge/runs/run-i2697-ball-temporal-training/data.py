"""Frozen inputs, shared training packs and validation ownership for the ablation."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from src.tasks.ball_detection.data.annotation_states import annotation_states
from src.tasks.ball_detection.data.store import BallFrameStore, ClipRecord
from src.tasks.ball_detection.data.windows.all_frames import (
    LABEL_KINDS,
    decode_label_runs,
    evaluation_windows,
)
from src.utils.checksum import dual_sha256
from src.utils.data.file_ranges import FileRangeReader
from src.utils.data.verification import SharedFileVerification

RUN = Path(__file__).resolve().parent
ANCHORS = RUN.parent / "run-i2637-ball-temporal-context/anchors.json"
PREVIOUS = ANCHORS.with_name("comparison.csv")
SEED = 42
PACK_SIZE = 128
PACK_COUNT = 30_478
MICROBATCH = 32
CONDITIONS = {"T1_rgb": 1, "T8_rgb_mdd": 8, "T32_rgb_mdd": 32}
SOURCE_LABELS = {
    "chat_annotation": "Chat",
    "meiji": "Meiji 3cam",
    "tracknet": "TrackNet game9",
}
VAL_COUNTS = dict(
    masked=11_310, observed=85_473, reviewed_empty=29_173, out_of_frame=2_712
)
ANCHOR_COUNTS = dict(observed=88, reviewed_empty=32, out_of_frame=24)
MANIFEST_SHA = {
    "train": "1a2e1d4f0bf4d64afd3658c6f2ecdbb4fe439e3ff7dfc48db2ebd57dccddca2d",
    "evaluation": "af1773174bdbeed86ee5339846bb6f30ac42d86e26abd1e60c0c685b9c8b6fae",
}
STORE_SHA = {
    "index.npz": "0bb00facd10f9961081f9c47af1b62f5c3ca4f5d349a2ac011573d9e510a9feb",
    "metadata.json": "7a804cc4e3b3006d0128fbd953b0696712d45acee9dcef8a96c02ec46379037c",
}
DISTRIBUTION_SHA = "f8425ec2cc6f4d0ae018a3c9387cb3d48b3bd8353794a82888936c3d0bb7477b"
ANCHOR_SHA = "fd35c124f2ac4f9a12b8d7617fe60a729d43dc1b3958703d95fff88998a034d5"
IDENTITY_FIELDS = (
    "target_identity_sha256",
    "teacher_identity_sha256",
    "scored_frame_sha256",
)


def require_sha(path: Path, expected: str) -> None:
    if dual_sha256(path) != expected:
        raise ValueError(f"Frozen input SHA-256 mismatch: {path}")


@dataclass(frozen=True)
class Window:
    start: int
    owner_left: int
    owner_right: int


@dataclass(frozen=True)
class Block:
    """Read one continuous RGB range, with model windows allowed to overlap."""

    clip_index: int
    start: int
    stop: int
    windows: tuple[Window, ...] = ()


@dataclass(frozen=True)
class Labels:
    codes: NDArray[np.uint8]
    uv: NDArray[np.float32]
    endpoint: NDArray[np.float32]


def eval_blocks(
    clip_index: int, interval: tuple[int, int], frames: int, capacity: int = MICROBATCH
) -> list[Block]:
    """Reuse the repo's nearest-center ownership (stride T/2), with stride 1 at T=1."""
    left, right = interval
    if frames == 1:
        windows = (Window(f, f, f + 1) for f in range(left, right))
    else:
        windows = (
            Window(row["start"], *row["owner_range"])
            for row in evaluation_windows([interval], frames)
        )
    result = []
    while group := tuple(islice(windows, capacity // frames)):
        start, stop = group[0].start, group[-1].start + frames
        # The shared nvJPEG boundary requires 2+ decoded frames. If the tail
        # is a singleton, read one preceding frame and keep it out of the model.
        if stop - start == 1:
            start = max(left, start - 1)
            stop = max(start + 2, stop)
        if not left <= start < stop <= right:
            raise ValueError("Decode block leaves its frozen interval")
        result.append(Block(clip_index, start, stop, group))
    return result


def anchor_block(anchor: dict[str, Any], frames: int) -> Block:
    # Same index as #2637: zero-based 15 at T32, 3 at T8, 0 at T1.
    frame = anchor["frame_index"]
    start = frame - (frames - 1) // 2
    return Block(
        anchor["clip_index"],
        start,
        start + max(frames, 2),
        (Window(start, frame, frame + 1),),
    )


class FrozenInputs:
    """Validate the original manifests and use the same physical JPEGs throughout."""

    def __init__(self, root: Path):
        dataset = root / "dataset"
        for name, digest in STORE_SHA.items():
            require_sha(dataset / name, digest)
        require_sha(dataset / "distribution.json", DISTRIBUTION_SHA)
        require_sha(ANCHORS, ANCHOR_SHA)

        manifests = {}
        for mode, digest in MANIFEST_SHA.items():
            path = root / "prepared" / f"{mode}_manifest.json"
            require_sha(path, digest)
            manifests[mode] = json.loads(path.read_bytes())
            if manifests[mode]["ball_store"]["hashes"] != STORE_SHA:
                raise ValueError(f"Store identity differs from {mode} manifest")
        train, evaluation = (manifests[k] for k in MANIFEST_SHA)
        self.manifest_val = evaluation["summaries"]["val"]
        distribution = json.loads((dataset / "distribution.json").read_bytes())
        payloads = {row["path"]: row for row in distribution["files"]}
        self.store = BallFrameStore(dataset)
        if self.store.video_frames:
            raise ValueError("Expected the frozen JPEG shard distribution")
        self.train: list[Block] = []
        self.intervals: list[tuple[int, tuple[int, int]]] = []
        self.labels: dict[int, Labels] = {}
        self.paths: dict[int, Path] = {}
        hashes: dict[Path, str] = {}
        self.valid_by_source: dict[str, Counter[str]] = {}
        for tr, ev in zip(train["clips"], evaluation["clips"], strict=True):
            clip = self.store.clips[tr["clip"]["index"]]
            if (
                clip.clip_id != tr["clip"]["clip_id"]
                or clip.clip_id != ev["clip"]["clip_id"]
            ):
                raise ValueError("Clip identity differs across inputs")
            if any(
                tr[k] != ev[k]
                for k in ("common_intervals", "label_runs", "rgb_shard_sha256")
            ):
                raise ValueError(
                    "Common intervals, teacher or JPEG identity differ across manifests"
                )
            path = self.store.image_path(clip)
            digest = tr["rgb_shard_sha256"]
            file = payloads[path.relative_to(dataset).as_posix()]
            if file["sha256"] != digest or file["size_bytes"] != path.stat().st_size:
                raise ValueError(
                    f"JPEG shard differs from distribution: {clip.clip_id}"
                )
            if (clip.width, clip.height) != (
                1280,
                720,
            ) or clip.source not in SOURCE_LABELS:
                raise ValueError(f"Unexpected image geometry or source: {clip.clip_id}")
            self.paths[clip.index] = path
            hashes[path] = digest
            self.labels[clip.index] = self.clip_labels(clip, tr["label_runs"])

            if clip.split == "train":
                for pack in tr["packs"]:
                    start = pack["start"]
                    left, right = tr["common_intervals"][pack["interval_index"]]
                    if not left <= start <= right - PACK_SIZE:
                        raise ValueError("Training pack leaves its frozen interval")
                    self.train.append(Block(clip.index, start, start + PACK_SIZE))
            elif clip.split == "val":
                source = self.valid_by_source.setdefault(clip.source, Counter())
                for a, b in ev["common_intervals"]:
                    self.intervals.append((clip.index, (a, b)))
                    counts = np.bincount(
                        self.labels[clip.index].codes[a:b], minlength=len(LABEL_KINDS)
                    )
                    source.update(dict(zip(LABEL_KINDS, map(int, counts), strict=True)))

        counts = sum(self.valid_by_source.values(), Counter())
        if (
            len(self.train) != PACK_COUNT
            or counts != VAL_COUNTS
            or len(hashes) != 1_537
        ):
            raise ValueError(
                f"Unexpected pack/shard count or validation support: {len(self.train)}, {len(hashes)}, {counts}"
            )
        self.verification = SharedFileVerification(hashes)
        self.anchors = self.check_anchors(hashes)

    def clip_labels(self, clip: ClipRecord, runs: list[list[int]]) -> Labels:
        codes = decode_label_runs(runs, clip.frame_count)
        states = annotation_states(self.store, clip)
        observed = codes == 1
        endpoint = np.asarray(
            (clip.source_width - 1, clip.source_height - 1), np.float32
        )
        uv = np.zeros((clip.frame_count, 2), np.float32)
        uv[observed] = states.xy[observed] / (clip.scale * endpoint)
        if (
            not states.supervision[observed].all()
            or not np.isfinite(uv).all()
            or ((uv < 0) | (uv > 1)).any()
        ):
            raise ValueError(f"Invalid observed targets: {clip.clip_id}")
        return Labels(codes, uv, endpoint)

    def check_anchors(self, hashes: dict[Path, str]) -> list[dict[str, Any]]:
        plan = json.loads(ANCHORS.read_bytes())
        if (plan["seed"], plan["manifest_sha256"], plan["store_sha256"]) != (
            SEED,
            MANIFEST_SHA["evaluation"],
            STORE_SHA,
        ):
            raise ValueError("Anchor plan does not belong to the frozen validation")
        anchors = [a for a in plan["anchors"] if a["split"] == "evaluation"]
        keys = {(a["clip_id"], a["frame_index"]) for a in anchors}
        if (
            len(keys) != len(anchors)
            or Counter(a["kind"] for a in anchors) != ANCHOR_COUNTS
        ):
            raise ValueError("Anchor membership is not the fixed 144-frame set")

        for a in anchors:
            index, frame = a["clip_index"], a["frame_index"]
            clip, labels = self.store.clips[index], self.labels[index]
            teacher = (
                (labels.uv[frame] * labels.endpoint).tolist()
                if a["kind"] == "observed"
                else None
            )
            start = a["window_start"]
            if (
                clip.split != "val"
                or clip.clip_id != a["clip_id"]
                or clip.source != a["source"]
                or str(Fraction(clip.fps)) != a["native_fps"]
                or list(labels.endpoint + 1) != a["source_size"]
                or LABEL_KINDS[labels.codes[frame]] != a["kind"]
                or teacher != a["teacher_xy"]
                or start != frame - 15
                or hashes[self.paths[index]] != a["rgb_shard_sha256"]
                or not any(
                    i == index and left <= start and start + 32 <= right
                    for i, (left, right) in self.intervals
                )
            ):
                raise ValueError(
                    f"Anchor teacher, context or identity changed: {a['clip_id']}:{frame}"
                )
        return anchors

    def validation(self, frames: int) -> list[Block]:
        return [
            block
            for i, span in self.intervals
            for block in eval_blocks(i, span, frames)
        ]

    def audit(self) -> dict[str, Any]:
        return dict(
            dataset="ball-mix1800-720p-all-frame-train-val-v1",
            manifest_sha256=MANIFEST_SHA,
            store_sha256=STORE_SHA,
            distribution_sha256=DISTRIBUTION_SHA,
            anchor_sha256=ANCHOR_SHA,
            shards=len(self.paths),
            pack_count=len(self.train),
            pack_size=PACK_SIZE,
            validation_counts=VAL_COUNTS,
            validation_by_source=self.valid_by_source,
            validation_identity={k: self.manifest_val[k] for k in IDENTITY_FIELDS},
            anchor_counts=ANCHOR_COUNTS,
        )


class RGBBlocks(Dataset[dict[str, Any]]):
    """Worker-side bounded file range reads; nvJPEG runs in the main process."""

    def __init__(self, inputs: FrozenInputs, blocks: list[Block]):
        self.inputs, self.blocks = inputs, blocks
        self.reader = FileRangeReader()

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, index: int) -> dict[str, Any]:
        block = self.blocks[index]
        clip = self.inputs.store.clips[block.clip_index]
        rows = self.inputs.store.frames
        first = int(self.inputs.store.clip_start[clip.index])
        section = slice(first + block.start, first + block.stop)
        ranges = list(
            zip(
                map(int, rows["offset"][section]),
                map(int, rows["length"][section]),
                strict=True,
            )
        )
        path = self.inputs.paths[clip.index]
        version = self.inputs.verification.verify(path)
        packed = self.reader.read(path, ranges, version)
        return dict(
            block=block,
            index=index,
            jpeg=torch.from_numpy(packed),
            jpeg_lengths=tuple(n for _, n in ranges),
            image_shape=(1, block.stop - block.start, 3, clip.height, clip.width),
        )
