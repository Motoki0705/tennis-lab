"""Small deterministic receipts for RGB, native PTS and complete teacher inputs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

import cv2
import numpy as np

from src.tasks.ball_detection.data.store import BallFrameStore
from src.utils.checksum import dual_sha256


def sample_windows(
    store: BallFrameStore, windows: Sequence[tuple[str, int, int]],
) -> dict[str, Any]:
    """Digest unaugmented contiguous native-frame windows, ignoring global indices.

    Teacher inputs include every review fact, instance, explicit mask and the
    original Chat notes. No new teacher policy is applied by the transfer layer.
    RGB is OpenCV-decoded so its version is part of the receipt.
    """
    receipts = []
    for clip_id, start, length in windows:
        clip = store.clip_by_id(clip_id)
        if type(start) is not int or type(length) is not int or length < 1 or start < 0 or start + length > clip.frame_count:
            raise ValueError("Sample windows must be nonempty and wholly inside a clip")
        rows = np.asarray([store.row_of(clip, i) for i in range(start, start + length)])
        teacher = hashlib.sha256()
        pts = hashlib.sha256(store.frames["pts"][rows].tobytes())
        for name in ("frame_index", "annotated", "is_target", "segment_break", "event", "inst_count"):
            teacher.update(name.encode())
            teacher.update(store.frames[name][rows].tobytes())
        first = int(store.frames["inst_start"][rows[0]])
        stop = int(store.frames["inst_start"][rows[-1]] + store.frames["inst_count"][rows[-1]])
        for name, values in sorted(store.instances.items()):
            teacher.update(name.encode())
            teacher.update(values[first:stop].tobytes())
        if store.teacher_masks is not None:
            teacher.update(store.teacher_masks.heatmap[rows].tobytes())
            teacher.update(store.teacher_masks.coordinate[rows].tobytes())
        annotation = None
        if clip.source == "chat_annotation":
            annotation = dual_sha256(store.annotation_path(clip))
            if annotation != clip.annotation_sha256:
                raise ValueError("Sample original annotation differs from the store")
        rgb, jpeg = hashlib.sha256(), hashlib.sha256()
        for row in rows:
            encoded = store.read_jpeg(int(row))
            jpeg.update(len(encoded).to_bytes(8, "little"))
            jpeg.update(encoded.tobytes())
            rgb.update(cv2.cvtColor(store.read_bgr(int(row)), cv2.COLOR_BGR2RGB).tobytes())
        descriptor = asdict(clip)
        descriptor.pop("index")
        receipts.append(dict(
            clip=descriptor, start=start, length=length, pts_sha256=pts.hexdigest(),
            teacher_inputs_sha256=teacher.hexdigest(), annotation_sha256=annotation,
            jpeg_sha256=jpeg.hexdigest(), rgb_sha256=rgb.hexdigest(),
        ))
    return dict(schema_version=1, opencv_version=cv2.__version__, windows=receipts,
                window_spec_sha256=hashlib.sha256(json.dumps(windows, separators=(",", ":")).encode()).hexdigest())
