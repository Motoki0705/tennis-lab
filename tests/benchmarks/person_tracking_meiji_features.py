"""Additional appearance features on the pinned detection timeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from person_kpr_features import encode_frame  # type: ignore[import-not-found]
from person_tracking_meiji_inputs import (  # type: ignore[import-not-found]
    CLIP,
    KPR,
    SOLIDER,
    Detections,
    asset_identity,
    feature_path,
    load_detections,
    load_saved_clip,
    verify_digest,
    write_npz,
)

from src.tasks.person_tracking.contracts import DetectionFeatures
from src.tasks.person_tracking.features import (
    FeatureConfig,
    UnpromptedEncoder,
    encode_appearance,
)
from src.tasks.player_association.appearance.encoders import (
    build_encoder,
    encoder_weights,
)
from src.tasks.player_association.appearance.kpr import WEIGHT_SHA256, KprEncoder
from src.tasks.player_association.pseudo_annotation.frozen_run import (
    FrozenClip,
    FrozenRun,
)
from src.utils.checksum import dual_sha256
from src.utils.io import write_json_atomic
from src.utils.video import OpenCVVideoFrameReader

CLIP_SHA256 = "bedc8fcc37f296045df19d901b162dd032bfa85d9b4f2406eff8313536d444dc"
SOLIDER_SHA256 = "81555144f412d46182d9cc8a0a01334f470a3484ce2fede88af9a5779d2a05a7"
DIMENSIONS = {CLIP: 1280, SOLIDER: 1024}
WEIGHTS = {CLIP: CLIP_SHA256, SOLIDER: SOLIDER_SHA256, KPR: WEIGHT_SHA256}
RECIPE = {"version": 1, "precision": "fp32", "unprompted_min_height_px": 1.0, "unprompted_batch": 8, "kpr_batch": 2}


def weight_path(root: Path, encoder: str) -> Path:
    path: Path = (root / "player_association/kpr/kpr_market_SOLIDER_93.25_96.59_41453430.pth.tar"
            if encoder == KPR else encoder_weights(encoder, checkpoint_root=root, external_root=root.parent / "third_party"))
    verify_digest(path, WEIGHTS[encoder])
    return path


def _identity(frozen: FrozenRun, adopted: FrozenClip, camera: str, encoder: str) -> dict[str, Any]:
    return {**asset_identity(frozen, adopted, camera, encoder, WEIGHTS[encoder]), "recipe": RECIPE}


def load_encoder_frames(
    frozen: FrozenRun, adopted: FrozenClip, camera: str, data: Detections, report: Path, encoder: str, *, saved_clip: bool,
) -> list[DetectionFeatures]:
    frames: list[DetectionFeatures]
    if encoder == CLIP and saved_clip:
        frames = load_saved_clip(adopted, camera, data)
        return frames
    path = feature_path(report, encoder, adopted.clip.clip_id, camera)
    receipt = json.loads(path.with_suffix(".json").read_text())
    if receipt["input"] != _identity(frozen, adopted, camera, encoder):
        raise ValueError(f"Appearance input changed: {path}")
    with np.load(verify_digest(path, receipt["sha256"]), allow_pickle=False) as arrays:
        visible = arrays["part_visibility"] if encoder == KPR else None
        if encoder != KPR and arrays["embedding"].shape != (len(data.rows), DIMENSIONS[encoder]):
            raise ValueError(f"Appearance dimension differs from the encoder profile: {path}")
        frames = data.frames(appearance=arrays["embedding"], valid=arrays["valid"], visibility=visible)
    return frames


class FeatureProducer:
    def __init__(self, name: str, checkpoint_root: Path, *, device: str = "cuda") -> None:
        self.name = name
        weight = weight_path(checkpoint_root, name)
        self.encoder = (KprEncoder(weight, device=device) if name == KPR else UnpromptedEncoder(
            build_encoder(name, checkpoint_root=checkpoint_root, external_root=checkpoint_root.parent / "third_party",
                          device=device, checkpoint=weight), DIMENSIONS[name]))

    def extract(self, frozen: FrozenRun, adopted: FrozenClip, camera: str, report: Path) -> None:
        path = feature_path(report, self.name, adopted.clip.clip_id, camera)
        clip = adopted.clip
        detections = load_detections(adopted, camera)
        if path.with_suffix(".json").exists():
            load_encoder_frames(frozen, adopted, camera, detections, report, self.name, saved_clip=False)
            return
        n = len(detections.rows)
        frames = detections.frames(appearance=np.zeros((n, 1), np.float32), valid=np.zeros(n, dtype=bool))
        media = frozen.verify(clip.media_path(camera))
        native = isinstance(self.encoder, KprEncoder)
        valid: np.ndarray = np.zeros(n, bool)
        embeddings: np.ndarray
        visibility: np.ndarray
        if native:
            embeddings = np.zeros((n, 6, 512), np.float32)
            visibility = np.zeros((n, 6), bool)
        else:
            embeddings = np.zeros((n, self.encoder.dimension), np.float32)
        frame_count = 0
        config = FeatureConfig(min_appearance_height_px=1., appearance_batch_size=8)
        for frame, packet in zip(frames, OpenCVVideoFrameReader(media), strict=True):
            if frame.frame != packet.index or packet.original_size != (clip.width, clip.height):
                raise ValueError(f"Unexpected RGB timeline for {clip.clip_id}/{camera}")
            a, b = detections.offsets[frame.frame:frame.frame + 2]
            if isinstance(self.encoder, KprEncoder):
                parts = encode_frame(self.encoder, packet.frame, frame, 2)
                embeddings[a:b], visibility[a:b] = parts.embeddings, parts.visible
            else:
                encoded = encode_appearance(frame.frame, packet.frame, frame.rows, frame.boxes, frame.scores,
                                            frame.poses, self.encoder, config)
                embeddings[a:b], valid[a:b] = encoded.embeddings, encoded.appearance_valid
            frame_count += 1
        arrays = {"embedding": embeddings, "valid": valid}
        if native:
            arrays["part_visibility"] = visibility
        write_npz(path, **arrays)
        with np.load(path, allow_pickle=False) as saved:
            if any(not np.array_equal(saved[name], array) for name, array in arrays.items()):
                raise ValueError(f"Feature read-back differs: {path}")
        write_json_atomic(path.with_suffix(".json"), {
            "input": _identity(frozen, adopted, camera, self.name),
            "sha256": dual_sha256(path), "rows": n, "frames": frame_count,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        })
        print(f"{self.name} {clip.clip_id}/{camera}: {frame_count} frames, {n} rows", flush=True)
