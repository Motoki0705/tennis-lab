"""Versioned detector evidence; partial builds never become training inputs."""

from __future__ import annotations

import json
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from src.tasks.ball_detection.data.store import (
    SOURCES,
    SPLIT_CODES,
    BallFrameStore,
    ClipRecord,
    shard_name,
)
from src.tasks.ball_detection.inference.checkpoint import load_ball_checkpoint
from src.tasks.ball_detection.inference.predictor import BallDetectionPredictor
from src.tasks.ball_detection.model_io.contracts import BallCandidateConfig
from src.tasks.ball_refiner.data.evidence import ClipEvidence
from src.tasks.ball_refiner.data.evidence_inference import (
    WINDOW_SELECTION,
    infer_clip_evidence,
)
from src.tennis_scene.pipeline.artifacts import write_json_atomic
from src.utils.checksum import dual_sha256

SCHEMA = "ball_refiner_detector_evidence.v1"


def _clip_record(clip: ClipRecord) -> dict[str, Any]:
    return {**asdict(clip), "track_ids": list(clip.track_ids)}


def _store_hashes(directory: Path) -> dict[str, str]:
    return {name: dual_sha256(directory / name) for name in ("metadata.json", "index.npz")}


def select_clips(
    store: BallFrameStore, *, splits: tuple[str, ...], sources: tuple[str, ...],
) -> tuple[ClipRecord, ...]:
    """Select only explicit source/split pairs; never select by label quality."""
    for name, requested, allowed in (("splits", splits, SPLIT_CODES), ("sources", sources, SOURCES)):
        if not requested or len(set(requested)) != len(requested) or not set(requested) <= set(allowed):
            raise ValueError(f"Invalid or repeated {name}: {requested}")
    groups: dict[tuple[str, str], str] = {}
    for clip in store.clips:
        group = (clip.source, clip.group_id)
        if group in groups and groups[group] != clip.split:
            raise ValueError(f"Source group crosses splits: {group}")
        groups[group] = clip.split
    clips = tuple(clip for clip in store.clips if clip.split in splits and clip.source in sources)
    if {(clip.source, clip.split) for clip in clips} != {(source, split) for source in sources for split in splits}:
        raise ValueError("Every requested source/split pair must contain at least one clip")
    return clips


def generate_evidence_cache(
    *, store_directory: Path, checkpoint: Path, output: Path, splits: tuple[str, ...],
    sources: tuple[str, ...], device: str, subpixel_refine: bool,
    stride: int, batch_size: int, candidates: BallCandidateConfig,
) -> Path:
    """Restore a frozen checkpoint and publish progress after every complete clip.

    Reusing/overwriting an existing output is forbidden. Failed/interrupted
    builds remain inspectable with status=building and cannot be consumed.
    Original media/annotation hashes are inherited from the store; the actual
    decoded JPEG shards and store tables are rehashed before/after inference.
    """
    if not all(path.is_absolute() for path in (store_directory, checkpoint, output)):
        raise ValueError("Cache input and output paths must be absolute")
    if output.exists():
        raise FileExistsError(f"Evidence cache output already exists: {output}")
    hashes = _store_hashes(store_directory)
    store = BallFrameStore(store_directory)
    clips = select_clips(store, splits=splits, sources=sources)
    checkpoint_hash = dual_sha256(checkpoint)
    loaded = load_ball_checkpoint(checkpoint, strict=True, weights_only=False)
    image_size = tuple(loaded.config.data.image_size)
    if len(image_size) != 2 or any(type(value) is not int or value <= 1 for value in image_size):
        raise ValueError("Checkpoint image_size must be explicit positive integer [H,W]")
    predictor = BallDetectionPredictor(
        loaded.model_io, torch.device(device), subpixel_refine=subpixel_refine,
        image_normalization=loaded.image_normalization,
    )
    predictor.model.requires_grad_(False)
    length = predictor.configured_frames
    if not 1 <= stride <= length or batch_size < 1:
        raise ValueError("Invalid detector stride or batch size")
    short = [clip.clip_id for clip in clips if clip.frame_count < length]
    if short:
        raise ValueError(f"Short clips require an explicit separate policy; padding is forbidden: {short}")
    manifest: dict[str, Any] = {
        "schema": SCHEMA, "status": "building",
        "generator_sha256": {
            name: dual_sha256(Path(__file__).with_name(name))
            for name in ("evidence.py", "evidence_inference.py", "evidence_cache.py")
        },
        "store": {"directory": str(store_directory.resolve()), "sha256": hashes},
        "detector": {
            "checkpoint": str(checkpoint.resolve()), "sha256": checkpoint_hash,
            "model_config": OmegaConf.to_container(loaded.config.model, resolve=True),
            "image_normalization": asdict(loaded.image_normalization),
            "image_size_hw": list(image_size), "rgb": "JPEG BGR -> INTER_LINEAR -> RGB float32 [0,1]",
            "device": str(predictor.device), "torch_version": str(torch.__version__),
            "subpixel_refine": subpixel_refine, "candidates": asdict(candidates),
            "window_length": length, "stride": stride, "batch_size": batch_size,
            "window_selection": WINDOW_SELECTION, "tail_policy": "backfill_real_frames_no_padding",
        },
        "coordinate_system": "source_xy_div_size_minus_one",
        "context": {"pose": "not_generated", "court": "not_generated"},
        "dense_heatmaps": "not_stored; cache contains native local probability patches",
        "selection": {"splits": list(splits), "sources": list(sources), "clip_ids": [c.clip_id for c in clips]},
        "clips": [],
    }
    (output / "clips").mkdir(parents=True, exist_ok=False)
    write_json_atomic(output / "manifest.json", manifest)
    for clip in clips:
        shard = store.directory / "shards" / shard_name(clip.index)
        shard_hash = dual_sha256(shard)
        evidence = infer_clip_evidence(
            store, clip, predictor, image_size_hw=(image_size[0], image_size[1]),
            stride=stride, batch_size=batch_size, config=candidates,
        )
        if dual_sha256(shard) != shard_hash:
            raise ValueError(f"JPEG shard changed during inference: {clip.clip_id}")
        relative = f"clips/clip-{clip.index:05d}.npz"
        with (output / relative).open("xb") as stream:
            np.savez_compressed(stream, **evidence.arrays())
        manifest["clips"].append({
            "clip": _clip_record(clip), "file": relative, "sha256": dual_sha256(output / relative),
            "jpeg_shard_sha256": shard_hash, "heatmap_size_hw": list(evidence.heatmap_size_hw),
        })
        write_json_atomic(output / "manifest.json", manifest)
        print(json.dumps({"completed_clips": len(manifest["clips"]), "total_clips": len(clips),
                          "clip_id": clip.clip_id, "frames": clip.frame_count}), flush=True)
    if _store_hashes(store_directory) != hashes or dual_sha256(checkpoint) != checkpoint_hash:
        raise ValueError("Store or checkpoint changed during cache generation")
    manifest["status"] = "complete"
    write_json_atomic(output / "manifest.json", manifest)
    return output


class EvidenceCache:
    """Bind a complete cache to the exact label store before exposing any clip."""

    def __init__(self, directory: Path, store: BallFrameStore) -> None:
        if not directory.is_absolute():
            raise ValueError("Evidence cache directory must be absolute")
        self.directory, self.store = directory, store
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest["schema"] != SCHEMA or manifest["status"] != "complete":
            raise ValueError("Evidence cache is incomplete or has an unsupported schema")
        if manifest["store"]["sha256"] != _store_hashes(store.directory):
            raise ValueError("Evidence cache belongs to different store metadata/labels")
        if manifest["coordinate_system"] != "source_xy_div_size_minus_one":
            raise ValueError("Evidence cache coordinates must use source endpoint normalization")
        detector = manifest["detector"]
        if detector["window_selection"] != WINDOW_SELECTION or detector["tail_policy"] != "backfill_real_frames_no_padding":
            raise ValueError("Incompatible detector window policy")
        self.candidate_config = BallCandidateConfig(**detector["candidates"])
        self.window_length = detector["window_length"]
        if type(self.window_length) is not int or self.window_length < 1:
            raise ValueError("Invalid detector window length")
        selection = manifest["selection"]
        clips = select_clips(store, splits=tuple(selection["splits"]), sources=tuple(selection["sources"]))
        self.clip_ids = tuple(clip.clip_id for clip in clips)
        records = manifest["clips"]
        if selection["clip_ids"] != list(self.clip_ids) or [r["clip"]["clip_id"] for r in records] != list(self.clip_ids):
            raise ValueError("Evidence cache does not cover every declared clip exactly once")
        for clip, record in zip(clips, records, strict=True):
            if record["clip"] != _clip_record(clip) or record["file"] != f"clips/clip-{clip.index:05d}.npz":
                raise ValueError(f"Cache clip metadata/path disagrees with store: {clip.clip_id}")
        self.manifest: dict[str, Any] = manifest
        self._records = {record["clip"]["clip_id"]: record for record in records}

    def load(self, clip_id: str) -> ClipEvidence:
        record = self._records[clip_id]  # absent/ungenerated clips are errors
        path = self.directory / record["file"]
        if not path.resolve().is_relative_to(self.directory.resolve()):
            raise ValueError("Evidence file escapes the cache directory")
        digest = dual_sha256(path)
        if digest != record["sha256"]:
            raise ValueError(f"Evidence checksum mismatch: {clip_id}")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        if dual_sha256(path) != digest:
            raise ValueError(f"Evidence changed while reading: {clip_id}")
        height, width = record["heatmap_size_hw"]
        result = ClipEvidence.from_arrays(
            arrays, config=self.candidate_config, heatmap_size_hw=(height, width),
            window_length=self.window_length,
        )
        clip = self.store.clip_by_id(clip_id)
        rows = self.store.clip_rows(clip)
        if not np.array_equal(result.frame_index, self.store.frames["frame_index"][rows]) or not np.array_equal(
            result.pts, self.store.frames["pts"][rows],
        ):
            raise ValueError(f"Evidence timeline differs from the source: {clip_id}")
        seconds = ((result.pts - result.pts[0]).astype(np.float64) * float(Fraction(clip.time_base))).astype(np.float32)
        if not np.array_equal(result.timestamps_seconds, seconds):
            raise ValueError(f"Evidence seconds do not match source PTS/time_base: {clip_id}")
        return result
