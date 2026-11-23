"""Load the fixed Meiji capture and detection-aligned features for comparison."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.tasks.person_tracking.contracts import DetectionFeatures
from src.tasks.player_association.appearance.parts import NativeParts
from src.tasks.player_association.pseudo_annotation.frozen_run import (
    FrozenClip,
    FrozenRun,
)
from src.tennis_scene.visualization.review.store import StoreReader
from src.utils.checksum import dual_sha256

CLIP = "clipreid_vitb16_market1501"
SOLIDER = "solider_swin_base_msmt17"
KPR = "kpr_market_solider_parts"
ENCODERS = (CLIP, SOLIDER, KPR)
SCENE = Path("annotations/tennis_scene/scene.json")


def verify_digest(path: Path, expected: str) -> Path:
    if dual_sha256(path) != expected:
        raise ValueError(f"Checksum mismatch: {path}")
    return path


def adopted_clip(frozen: FrozenRun, clip: FrozenClip, dataset: Path) -> FrozenClip:
    """Use the original store to get features omitted from #2442's copy."""
    relative = Path(frozen.dataset.clips[clip.clip.clip_id].path) / SCENE
    digest = frozen.checksums[(Path("dataset") / relative).as_posix()]
    store = StoreReader(verify_digest(dataset / relative, digest))
    return FrozenClip(clip.clip, store, clip.outcome, clip.media_sha256)


@dataclass(frozen=True)
class Detections:
    offsets: np.ndarray
    rows: np.ndarray
    boxes: np.ndarray
    scores: np.ndarray
    keypoints: np.ndarray

    def frames(self, *, appearance: np.ndarray, valid: np.ndarray,
               visibility: np.ndarray | None = None) -> list[DetectionFeatures]:
        count = len(self.rows)
        if (appearance.shape[0] != count or valid.shape != (count,)
                or (visibility is not None and (appearance.shape != (count, 6, 512) or visibility.shape != (count, 6)))):
            raise ValueError("Appearance must preserve the complete detection row axis")
        return [
            DetectionFeatures(
                frame, self.rows[a:b], self.boxes[a:b], self.scores[a:b], self.keypoints[a:b],
                appearance[a:b] if visibility is None else np.zeros((b - a, 1), np.float32), valid[a:b],
                None if visibility is None else NativeParts(appearance[a:b], visibility[a:b]),
            )
            for frame, (a, b) in enumerate(zip(self.offsets[:-1], self.offsets[1:], strict=True))
        ]


def _require_aligned(output: dict[str, Any], camera: str, offsets: np.ndarray, rows: np.ndarray) -> None:
    if (output["camera_id"] != camera or not np.array_equal(output["frame_offsets"], offsets)
            or not np.array_equal(output["source_rows"], rows)):
        raise ValueError(f"Feature grid differs from the fixed detections for {camera}")


def load_detections(adopted: FrozenClip, camera: str) -> Detections:
    offsets, boxes, rows = adopted.detections(camera)
    scores = adopted.read(f"person_detection/{camera}", "person_detections", 2, "confidence")["confidence"]
    poses = adopted.read(
        f"person_pose_features/{camera}", "person_pose_features", 2,
        "camera_id", "frame_offsets", "source_rows", "keypoints",
    )
    _require_aligned(poses, camera, offsets, rows)
    return Detections(offsets, rows, boxes, scores, poses["keypoints"])


def load_saved_clip(adopted: FrozenClip, camera: str, data: Detections) -> list[DetectionFeatures]:
    saved = adopted.read(
        f"person_appearance_features/{camera}", "person_appearance_features", 4,
        "camera_id", "frame_offsets", "source_rows", "embeddings", "appearance_valid", "encoder",
    )
    _require_aligned(saved, camera, data.offsets, data.rows)
    if saved["encoder"] != CLIP:
        raise ValueError(f"Unexpected saved encoder {saved['encoder']}")
    return data.frames(appearance=saved["embeddings"], valid=saved["appearance_valid"])


def feature_path(report: Path, encoder: str, clip: str, camera: str) -> Path:
    return report / "features" / encoder / clip / f"{camera}.npz"


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    """Publish a complete archive atomically so interrupted work can be rerun."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)  # type: ignore[arg-type]
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def asset_identity(frozen: FrozenRun, clip: FrozenClip, camera: str, encoder: str, weight: str) -> dict[str, Any]:
    return {
        "snapshot_sha256": frozen.sha256, "camera": camera, "encoder": encoder,
        "pose": clip.store.references[f"person_pose_features/{camera}"],
        "detections": clip.store.references[f"person_detection/{camera}"],
        "media_sha256": clip.media_sha256[camera], "weight_sha256": weight,
    }
