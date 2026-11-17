"""Read content-pinned model homographies, retaining rejected initializers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CourtSeed:
    candidate_id: str
    clip_id: str
    frame_index: int
    matrix: NDArray[np.float64]
    model_accepted: bool


def read_frozen_candidates(path: Path, *, sha256: str, dataset_id: str, video_id: str,
                           camera_id: str, size_wh: tuple[int, int],
                           clip_ids: tuple[str, ...]) -> tuple[CourtSeed, ...]:
    """No directory discovery, alternate camera, neural inference, or null-H repair.

    The producer records source/region hashes and the coordinate convention.
    This reader checks the manifest bytes, scope, and numerical contract before
    applying a crop translation exactly once. Empty candidates are legitimate
    input to an explicit no-solution result, not permission to run a model.
    """
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError("Frozen court candidate digest changed")
    document = json.loads(raw)
    if set(document) != {"schema", "dataset_id", "video_id", "camera_id", "size_wh", "provenance", "candidates"}:
        raise ValueError("Invalid frozen court candidate fields")
    if (document["schema"] != "court_seed_manifest_v1" or document["dataset_id"] != dataset_id
            or document["video_id"] != video_id or document["camera_id"] != camera_id
            or document["size_wh"] != list(size_wh)):
        raise ValueError("Frozen court candidate scope mismatch")
    if len(set(clip_ids)) != len(clip_ids) or not clip_ids or any(c.split("/")[0] != video_id for c in clip_ids):
        raise ValueError("Invalid court seed clip selection")
    if not isinstance(document["provenance"], dict) or not document["provenance"]:
        raise ValueError("Frozen candidates require provenance")
    seeds = []
    seen: set[str] = set()
    found: set[str] = set()
    for row in document["candidates"]:
        if set(row) != {"candidate_id", "clip_id", "frame_index", "matrix", "model_accepted", "region_xyxy", "source_sha256"}:
            raise ValueError("Invalid frozen court candidate row")
        key, clip = row["candidate_id"], row["clip_id"]
        if not isinstance(key, str) or not key or key in seen or clip.split("/")[0] != video_id:
            raise ValueError("Duplicate candidate or foreign video")
        seen.add(key)
        if type(row["frame_index"]) is not int or row["frame_index"] < 0 or type(row["model_accepted"]) is not bool:
            raise ValueError("Invalid seed frame/acceptance")
        source_hash = row["source_sha256"]
        if not isinstance(source_hash, str) or len(source_hash) != 64 or any(c not in "0123456789abcdef" for c in source_hash):
            raise ValueError("Seed requires a source content digest")
        region = row["region_xyxy"]
        if (not isinstance(region, list) or len(region) != 4 or any(type(n) is not int for n in region)
                or not 0 <= region[0] < region[2] <= size_wh[0]
                or not 0 <= region[1] < region[3] <= size_wh[1]):
            raise ValueError("Invalid seed crop")
        found.add(clip)
        matrix = None if row["matrix"] is None else np.asarray(row["matrix"], dtype=np.float64)
        if matrix is not None and (matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12):
            raise ValueError("Invalid seed homography")
        if clip not in clip_ids or matrix is None:
            continue
        transform = np.eye(3)
        transform[:2, 2] = region[:2]
        seeds.append(CourtSeed(key, clip, row["frame_index"], transform @ matrix, row["model_accepted"]))
    if set(clip_ids) - found:
        raise ValueError("Requested seed clip is absent from manifest")
    return tuple(sorted(seeds, key=lambda seed: seed.candidate_id))
