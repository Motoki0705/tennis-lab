from __future__ import annotations

import gc
import os
import traceback
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from src.tasks.ball_detection.data.store import SHARDS_DIR, shard_name

from .selection import load_campaign
from .storage import (
    clip_root,
    digest,
    lock,
    read_json,
    verify_record,
    write_json,
    write_npz,
)


def _cached_chunk(path: Path) -> bool:
    receipt = path.with_suffix(".json")
    if not receipt.exists():
        return False
    verify_record(receipt)
    return True


def _save_chunk(path: Path, **arrays: Any) -> None:
    write_npz(path, **arrays)
    write_json(path.with_suffix(".json"), {"files": {path.name: digest(path)}})


def generate_clip(campaign: Path, index: int) -> None:
    """Persist detector/features in frame chunks; never invoke court inference."""
    if (
        not os.environ.get("TENNIS_RUN_ID")
        or os.environ.get("TENNIS_GPU_RESOURCE") != "all"
    ):
        raise RuntimeError(
            "CUDA generation must run in an exclusive shared training-queue job"
        )
    import torch

    from src.submodules.configuration import ViTPoseHeadConfig
    from src.submodules.models.dino.extension import validate_dino_extension
    from src.submodules.models.dino.person_detector import (
        DinoPersonDetector,
        PersonDetectionRequest,
    )
    from src.submodules.models.vitpose.pose2d import ViTPosePose2D
    from src.tasks.person_tracking.contracts import DetectionFeatures
    from src.tasks.person_tracking.features import (
        FeatureConfig,
        FeatureExtractor,
        UnpromptedEncoder,
    )
    from src.tasks.person_tracking.sequence import TrackingConfig, track_sequence
    from src.tasks.person_tracking.strongsort_offline import AFLink
    from src.tasks.player_association.appearance.encoders import ClipReIDEncoder

    config, plan, store = load_campaign(campaign)
    entry = plan["clips"][index]
    if not entry["selected"]:
        raise ValueError("Cannot generate a skipped clip")
    root = clip_root(campaign, index)
    with lock(root / "generation.lock", blocking=False):
        if (root / "generation.json").exists():
            verify_record(root / "generation.json")
            return
        clip = store.clips[index]
        shard = store.directory / SHARDS_DIR / shard_name(index)
        stat = shard.stat()
        if [stat.st_size, stat.st_mtime_ns] != entry["shard_stat"]:
            raise ValueError("Planned JPEG shard changed")
        if (root / "input.json").exists():
            if read_json(root / "input.json")["shard_sha256"] != digest(shard):
                raise ValueError("Cached input JPEG bytes changed")
        else:
            write_json(
                root / "input.json",
                {"clip_id": clip.clip_id, "shard_sha256": digest(shard)},
            )
        for name, expected in config["asset_hashes"].items():
            # Hash once when planning; a stat binding prevents changes during this run.
            path = Path(config["assets"][name])
            asset_stat = path.stat()
            if [asset_stat.st_size, asset_stat.st_mtime_ns] != config["asset_stats"][
                name
            ]:
                raise ValueError(f"Asset changed: {name} ({expected})")
        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(
            float(config["cuda_memory_fraction"])
        )
        if str(validate_dino_extension()) != str(
            Path(config["assets"]["dino_extension"]).resolve()
        ):
            raise ValueError("Unexpected DINO extension")
        ranges = [
            (s, min(s + config["chunk_frames"], clip.frame_count))
            for s in range(0, clip.frame_count, config["chunk_frames"])
        ]
        start_row = store.row_of(clip, 0)
        detector = None
        detection_chunks = []
        for start, stop in ranges:
            path = root / "detections" / f"{start:06d}-{stop:06d}.npz"
            detection_chunks.append(path)
            if _cached_chunk(path):
                continue
            if detector is None:
                detector = DinoPersonDetector(
                    Path(config["assets"]["dino"]),
                    Path(config["assets"]["dino_repository"]),
                    device="cuda",
                    confidence=0.3,
                    short_side=800,
                    max_long_side=1333,
                )
            boxes, scores, offsets = [], [], [0]
            for frame in range(start, stop):
                write_json(
                    root / "progress.json",
                    {"stage": "detection", "frame": frame, "frames": clip.frame_count},
                )
                result = detector.predict(
                    PersonDetectionRequest(store.read_bgr(start_row + frame))
                )
                boxes.append(result.boxes_xyxy)
                scores.append(result.scores)
                offsets.append(offsets[-1] + len(result.scores))
            _save_chunk(
                path,
                offsets=np.asarray(offsets, np.int64),
                boxes=np.concatenate(boxes),
                scores=np.concatenate(scores),
            )
        # Cleanup only follows success; it must not replace the original CUDA traceback.
        if detector is not None:
            detector.unload()
            del detector
        gc.collect()
        torch.cuda.empty_cache()
        pose = encoder = extractor = None
        all_features = []
        row_offset = 0
        for (start, stop), det_path in zip(ranges, detection_chunks, strict=True):
            path = root / "features" / f"{start:06d}-{stop:06d}.npz"
            with np.load(det_path, allow_pickle=False) as data:
                detections = {name: data[name] for name in data.files}
            offsets = detections["offsets"]
            if _cached_chunk(path):
                with np.load(path, allow_pickle=False) as data:
                    features = {name: data[name] for name in data.files}
            else:
                if pose is None:
                    head = ViTPoseHeadConfig(
                        in_channels=1280,
                        out_channels=17,
                        num_deconv_layers=2,
                        num_deconv_filters=(256, 256),
                        num_deconv_kernels=(4, 4),
                        final_conv_kernel=1,
                        num_conv_layers=0,
                        num_conv_kernels=(),
                    )
                    pose = ViTPosePose2D(
                        Path(config["assets"]["vitpose"]),
                        device="cuda",
                        flip_test=True,
                        batch_size=4,
                        head_config=head,
                        precision="float32",
                    )
                    encoder = ClipReIDEncoder(
                        "clipreid_vitb16_market1501",
                        Path(config["assets"]["clip_reid"]),
                        "cuda",
                    )
                    extractor = FeatureExtractor(
                        pose, UnpromptedEncoder(encoder, 1280), FeatureConfig()
                    )
                frames = []
                for local, frame in enumerate(range(start, stop)):
                    write_json(
                        root / "progress.json",
                        {
                            "stage": "pose_clip_features",
                            "frame": frame,
                            "frames": clip.frame_count,
                        },
                    )
                    a, b = int(offsets[local]), int(offsets[local + 1])
                    assert extractor is not None
                    frames.append(
                        extractor.extract(
                            frame,
                            store.read_bgr(start_row + frame),
                            np.arange(row_offset + a, row_offset + b, dtype=np.int64),
                            detections["boxes"][a:b],
                            detections["scores"][a:b],
                        )
                    )
                fields = (
                    "rows",
                    "boxes",
                    "scores",
                    "poses",
                    "embeddings",
                    "appearance_valid",
                )
                features = {
                    name: np.concatenate([getattr(f, name) for f in frames])
                    for name in fields
                }
                features["offsets"] = offsets
                _save_chunk(path, **features)
            if not np.array_equal(features["offsets"], offsets):
                raise ValueError("Feature/detection offsets changed")
            for local, frame in enumerate(range(start, stop)):
                a, b = int(offsets[local]), int(offsets[local + 1])
                values = {
                    name: value[a:b]
                    for name, value in features.items()
                    if name != "offsets"
                }
                if not np.array_equal(
                    values["rows"], np.arange(row_offset + a, row_offset + b)
                ):
                    raise ValueError("Cached detection identity changed")
                all_features.append(DetectionFeatures(frame, **values))
            row_offset += int(offsets[-1])
        if pose is not None:
            pose.unload()
        del pose, encoder, extractor
        gc.collect()
        torch.cuda.empty_cache()
        write_json(
            root / "progress.json", {"stage": "tracking", "frames": clip.frame_count}
        )
        result = track_sequence(
            all_features,
            fps=float(Fraction(clip.fps)),
            config=TrackingConfig(),
            aflink=AFLink(Path(config["assets"]["aflink"])),
        )
        tracks = root / "tracks.npz"
        rows = store.clip_rows(clip)
        write_npz(
            tracks,
            frame_index=store.frames["frame_index"][rows],
            pts=store.frames["pts"][rows],
            track_ids=result.track_ids,
            boxes=result.boxes,
            detection_rows=result.evidence.detection_rows,
            keypoints=result.evidence.poses,
        )
        write_json(
            root / "generation.json",
            {
                "status": "complete",
                "clip_id": clip.clip_id,
                "frame_count": clip.frame_count,
                "raw_tracks": len(result.track_ids),
                "court_policy": "disabled",
                "tracking_profile": TrackingConfig().identity(),
                "source_track_ids": result.source_track_ids,
                "link_candidates": result.link_candidates,
                "synthetic_poses_used": False,
                "files": {
                    "tracks.npz": digest(tracks),
                    "input.json": digest(root / "input.json"),
                },
            },
        )


def record_failure(campaign: Path, index: int) -> None:
    write_json(
        clip_root(campaign, index) / "exception.json", {"error": traceback.format_exc()}
    )
