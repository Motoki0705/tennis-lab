from __future__ import annotations

import gc
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from .selection import load_campaign
from .storage import (
    clip_root,
    digest,
    lock,
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


def _save_chunk(path: Path, *, seconds: float = 0.0, **arrays: Any) -> None:
    write_npz(path, **arrays)
    write_json(
        path.with_suffix(".json"),
        {"seconds": seconds, "files": {path.name: digest(path)}},
    )


def generate_clip(campaign: Path, index: int) -> None:
    """Track pinned corrected boxes; detection is never repeated here."""
    if not os.environ.get("TENNIS_RUN_ID") or os.environ.get("TENNIS_GPU_RESOURCE") != "all":
        raise RuntimeError("CUDA generation must run in an exclusive shared training-queue job")
    import torch

    from src.tasks.base.annotation.artifacts import (
        file_reference,
        read_document,
        verify_reference,
    )
    from src.tasks.base.annotation.contracts import ArtifactRef
    from src.tasks.base.annotation.paths import AnnotationLayout
    from src.tasks.person_tracking.features import FeatureExtractor, UnpromptedEncoder
    from src.tasks.person_tracking.strongsort_offline import AFLink
    from src.tasks.player_association.appearance.encoders import ClipReIDEncoder

    from .corrected import TrackingModels, TrackingReceipt, load_input, track_corrected

    config, plan, store = load_campaign(campaign)
    entry = plan["clips"][index]
    if not entry["selected"]:
        raise ValueError("Cannot generate a skipped clip")
    if "corrected_inputs" not in config or entry["clip_id"] not in config["corrected_inputs"]:
        raise ValueError("Tracking requires a pinned corrected bbox input for this clip")
    layout = AnnotationLayout.production(Path(config["project_root"]))
    reference = ArtifactRef.model_validate(config["corrected_inputs"][entry["clip_id"]])
    inputs = load_input(layout, reference)
    if inputs.store.directory != store.directory or inputs.clip.index != index:
        raise ValueError("Corrected bbox input belongs to a different ball snapshot")
    root = clip_root(campaign, index)
    with lock(root / "generation.lock", blocking=False):
        if (root / "generation.json").exists():
            existing = verify_record(root / "generation.json")
            if existing["corrected_input"] != reference.model_dump(mode="json"):
                raise ValueError("Tracking input revision changed")
            return
        for name in ("clip_reid", "aflink"):
            if digest(Path(config["assets"][name])) != config["asset_hashes"][name]:
                raise ValueError(f"Planned tracking weight changed: {name}")
        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(float(config["cuda_memory_fraction"]))
        encoder = ClipReIDEncoder("clipreid_vitb16_market1501", Path(config["assets"]["clip_reid"]), "cuda")
        extractor = FeatureExtractor(None, UnpromptedEncoder(encoder, 1280))
        started = time.monotonic()
        models = TrackingModels(**{name: file_reference(layout, Path(config["assets"][name]), role="checkpoint") for name in ("clip_reid", "aflink")})
        tracking_ref = track_corrected(layout, reference, models=models, extractor=extractor, aflink=AFLink(Path(config["assets"]["aflink"])))
        receipt = read_document(layout, tracking_ref, TrackingReceipt)
        with np.load(verify_reference(layout, receipt.tracks), allow_pickle=False) as archive:
            raw = dict(archive)
        write_npz(root / "tracks.npz", **raw)
        write_json(root / "input.json", {"clip_id": entry["clip_id"], "shard_sha256": inputs.shard.digest,
                   "corrected_input": reference.model_dump(mode="json")})
        write_json(root / "generation.json", {
            "status": "complete", "stage": "tracking", "generation_mode": config["generation_mode"],
            "all_detections": receipt.observations, "detector_calls": 0, "pose_crops": 0,
            "seconds": time.monotonic() - started, "clip_id": entry["clip_id"],
            "frame_count": inputs.clip.frame_count, "raw_tracks": len(raw["track_ids"]),
            "corrected_input": reference.model_dump(mode="json"), "tracking": tracking_ref.model_dump(mode="json"),
            "tracking_profile": receipt.profile, "unassigned_rows": list(receipt.unassigned_rows),
            "synthetic_poses_used": False,
            "files": {name: digest(root / name) for name in ("tracks.npz", "input.json")},
        })
        del encoder, extractor
        gc.collect()
        torch.cuda.empty_cache()

def record_failure(campaign: Path, index: int) -> None:
    write_json(
        clip_root(campaign, index) / "exception.json", {"error": traceback.format_exc()}
    )
