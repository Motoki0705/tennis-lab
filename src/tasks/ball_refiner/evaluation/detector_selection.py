"""Compare frozen detectors on validation using the refiner cache's RGB contract."""

from __future__ import annotations

import gc
import inspect
import json
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from src.tasks.ball_detection.data.store import (
    SOURCES,
    BallFrameStore,
    ClipRecord,
    shard_name,
)
from src.tasks.ball_detection.evaluation.candidate_recall import (
    CandidateRecallCounts,
    candidate_recall_counts,
)
from src.tasks.ball_detection.inference.checkpoint import load_ball_checkpoint
from src.tasks.ball_detection.inference.predictor import BallDetectionPredictor
from src.tasks.ball_detection.model_io.contracts import BallCandidateConfig
from src.tasks.ball_refiner.data.cache_identity import (
    clip_record,
    select_clips,
    store_hashes,
)
from src.tasks.ball_refiner.data.evidence_cache import SCHEMA as EVIDENCE_SCHEMA
from src.tasks.ball_refiner.data.evidence_inference import (
    WINDOW_SELECTION,
    infer_clip_evidence,
)
from src.tasks.ball_refiner.data.targets import project_store_targets
from src.tennis_scene.pipeline.artifacts import write_json_atomic
from src.utils.checksum import dual_sha256

CHECKPOINT_NAMES = ("ft-e13", "mixed-e0", "mixed-e11")
SCHEMA = "ball_refiner_detector_selection.v1"


@dataclass(frozen=True)
class SelectionCheckpoint:
    name: str
    path: Path
    sha256: str

    def verify(self) -> None:
        if not self.path.is_absolute() or not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("Checkpoint requires an absolute path and explicit sha256")
        if dual_sha256(self.path) != self.sha256:
            raise ValueError(f"Checkpoint checksum changed: {self.name}")


def validation_clips(store: BallFrameStore) -> tuple[ClipRecord, ...]:
    """Fixed campaign selection: Meiji video_000, TrackNet game9 and chat val."""
    clips: tuple[ClipRecord, ...] = select_clips(store, splits=("val",), sources=SOURCES)
    for source, expected in (("meiji", "video_000"), ("tracknet", "game9")):
        if {c.group_id for c in clips if c.source == source} != {expected}:
            raise ValueError(f"Validation {source} must use only {expected}")
    if any(c.camera_id is None for c in clips if c.source == "meiji"):
        raise ValueError("Meiji validation requires camera identities")
    return clips


def prepare_comparison(
    store_directory: Path, cache_manifest: Path, checkpoints: tuple[SelectionCheckpoint, ...],
) -> tuple[BallFrameStore, tuple[ClipRecord, ...], dict[str, Any]]:
    """CPU-only provenance checks; never read test predictions or person outputs."""
    if not store_directory.is_absolute() or not cache_manifest.is_absolute():
        raise ValueError("Comparison paths must be absolute")
    if tuple(c.name for c in checkpoints) != CHECKPOINT_NAMES:
        raise ValueError(f"Checkpoint order must be {CHECKPOINT_NAMES}")
    for checkpoint in checkpoints:
        checkpoint.verify()
    store = BallFrameStore(store_directory)
    clips = validation_clips(store)
    manifest_hash = dual_sha256(cache_manifest)
    cache = json.loads(cache_manifest.read_text())
    hashes = store_hashes(store_directory)
    if cache["schema"] != EVIDENCE_SCHEMA or cache["status"] != "complete" or cache["store"]["sha256"] != hashes:
        raise ValueError("Reference evidence manifest must be complete and belong to this store")
    detector = cache["detector"]
    if (detector["candidates"] != asdict(BallCandidateConfig(max_candidates=8, nms_kernel=5, patch_size=5))
            or detector["subpixel_refine"] is not True or detector["window_selection"] != WINDOW_SELECTION
            or detector["tail_policy"] != "backfill_real_frames_no_padding"):
        raise ValueError("Reference cache must use K=8/NMS=5/patch=5/subpixel and the agreed window rule")
    selected = set(cache["selection"]["clip_ids"])
    if not {c.clip_id for c in clips} <= selected:
        raise ValueError("Reference cache does not cover the entire validation set")
    if dual_sha256(cache_manifest) != manifest_hash:
        raise ValueError("Reference manifest changed while reading")
    return store, clips, {
        "schema": SCHEMA, "status": "prepared", "radius_source_px": 20.0,
        "store": {"directory": str(store_directory), "sha256": hashes},
        "reference_cache": {"path": str(cache_manifest), "sha256": manifest_hash},
        "inference": {key: detector[key] for key in (
            "candidates", "subpixel_refine", "window_length", "stride", "window_selection", "tail_policy", "image_size_hw",
        )},
        "selection": {"split": "val", "primary": "meiji/video_000", "clips": [clip_record(c) for c in clips]},
        "checkpoints": [{"name": c.name, "path": str(c.path), "sha256": c.sha256} for c in checkpoints],
        "targets": "single observed ball only; unknown/estimated/multiple/absent excluded",
        "ranking": "score descending, decoder order for ties; all rates use observed frames unless explicitly conditional",
        "tie_break": "equal primary hit counts retain ft-e13, then mixed-e0, then mixed-e11",
        "code_sha256": {function.__module__: dual_sha256(Path(inspect.getfile(function))) for function in (
            prepare_comparison, infer_clip_evidence, candidate_recall_counts,
        )},
        "results": [],
    }


def summarize_clips(records: list[dict[str, Any]]) -> dict[str, dict[str, int | float | None]]:
    totals: dict[str, dict[str, int]] = defaultdict(lambda: dict.fromkeys(CandidateRecallCounts.__dataclass_fields__, 0))
    for record in records:
        clip = record["clip"]
        groups = [clip["source"]]
        if clip["camera_id"] is not None:
            groups.append(f"{clip['source']}/{clip['camera_id']}")
        for group in groups:
            for name, count in record["counts"].items():
                totals[group][name] += count
    return {group: CandidateRecallCounts(**counts).report() for group, counts in sorted(totals.items())}


def select_winner(results: list[dict[str, Any]]) -> dict[str, Any]:
    if tuple(r["name"] for r in results) != CHECKPOINT_NAMES:
        raise ValueError("Selection requires all three checkpoints in the declared order")
    primary = [r["groups"]["meiji"] for r in results]
    if len({m["observed"] for m in primary}) != 1 or primary[0]["observed"] <= 0:
        raise ValueError("Primary validation denominators must be equal and positive")
    hits = [m["recalled_at_k"] for m in primary]
    return {
        "winner": CHECKPOINT_NAMES[hits.index(max(hits))],
        "tied_best": [name for name, count in zip(CHECKPOINT_NAMES, hits, strict=True) if count == max(hits)],
        "mixed_e11_beats_e0": hits[2] > hits[1],
        "primary": "meiji/video_000 recall@8 <=20 source px",
        "evidence_cache_rebuild_required": CHECKPOINT_NAMES[hits.index(max(hits))] != "ft-e13",
    }


def _evaluate_checkpoint(
    checkpoint: SelectionCheckpoint, store: BallFrameStore, clips: tuple[ClipRecord, ...],
    output: Path, manifest: dict[str, Any], *, device: torch.device, batch_size: int,
) -> dict[str, Any]:
    started = time.monotonic()
    checkpoint.verify()
    loaded = load_ball_checkpoint(checkpoint.path, strict=True, weights_only=False)
    settings = manifest["inference"]
    if list(loaded.config.data.image_size) != settings["image_size_hw"]:
        raise ValueError(f"Checkpoint image size differs from the reference cache: {checkpoint.name}")
    predictor = BallDetectionPredictor(
        loaded.model_io, device, subpixel_refine=settings["subpixel_refine"],
        image_normalization=loaded.image_normalization,
    )
    predictor.model.requires_grad_(False)
    if predictor.configured_frames != settings["window_length"]:
        raise ValueError(f"Checkpoint window length differs from the reference cache: {checkpoint.name}")
    result: dict[str, Any] = {
        "name": checkpoint.name, "sha256": checkpoint.sha256,
        "model_config": OmegaConf.to_container(loaded.config.model, resolve=True),
        "image_normalization": asdict(loaded.image_normalization), "clips": [],
    }
    manifest["results"].append(result)
    directory = output / checkpoint.name
    directory.mkdir()
    for clip in clips:
        shard = store.directory / "shards" / shard_name(clip.index)
        shard_hash = dual_sha256(shard)
        evidence = infer_clip_evidence(
            store, clip, predictor, image_size_hw=tuple(settings["image_size_hw"]),
            stride=settings["stride"], batch_size=batch_size, config=BallCandidateConfig(**settings["candidates"]),
        )
        if dual_sha256(shard) != shard_hash:
            raise ValueError(f"JPEG shard changed during comparison: {clip.clip_id}")
        targets = project_store_targets(store, clip)
        scale = np.asarray((clip.source_width - 1, clip.source_height - 1), dtype=np.float32)
        xy = evidence.candidates.coords[0].numpy() * scale
        gt = targets.uv * scale
        scores, valid = evidence.candidates.scores[0].numpy(), evidence.candidates.valid[0].numpy()
        counts = candidate_recall_counts(xy, scores, valid, gt, targets.position_valid, radius_px=manifest["radius_source_px"])
        file = directory / f"clip-{clip.index:05d}.npz"
        with file.open("xb") as stream:
            np.savez_compressed(
                stream, frame_index=targets.frame_index, pts=targets.pts, candidate_xy_source_px=xy,
                candidate_scores=scores, candidate_valid=valid, target_xy_source_px=gt,
                observed=targets.position_valid, target_reason=targets.reason,
                window_start=evidence.window_start, time_index=evidence.time_index,
            )
        result["clips"].append({
            "clip": clip_record(clip), "file": str(file.relative_to(output)), "sha256": dual_sha256(file),
            "jpeg_shard_sha256": shard_hash, "counts": asdict(counts), "target_counts": targets.counts(),
        })
        write_json_atomic(output / "manifest.json", manifest)
        print(json.dumps({"checkpoint": checkpoint.name, "clips": len(result["clips"]), "total": len(clips),
                          "clip": clip.clip_id, "seconds": time.monotonic() - started}), flush=True)
    checkpoint.verify()
    result.update(groups=summarize_clips(result["clips"]), seconds=time.monotonic() - started)
    return result


def comparison_markdown(manifest: dict[str, Any]) -> str:
    rows = ["| checkpoint | source / camera | observed | recall@8 | recall@1 | 候補外 | 誤候補が上位 |",
            "|---|---|---:|---:|---:|---:|---:|"]
    for result in manifest["results"]:
        for group, metrics in result["groups"].items():
            rates = [metrics[key] for key in ("recall_at_k", "recall_at_1", "not_in_candidates_rate", "wrong_ranked_above_true_rate")]
            values = " | ".join("N/A" if rate is None else f"{100 * rate:.3f}%" for rate in rates)
            rows.append(f"| {result['name']} | {group} | {metrics['observed']} | {values} |")
    rows += ["", "距離≤20 source px、K=8/NMS=5/patch=5/subpixel。全率の分母はobserved frame。",
             "誤候補が上位 = 正解候補が存在しtop-1が誤り。同scoreはdecoder順を保持し、strict score版と同率件数をJSONに併記。",
             "camera IDのないsourceはcamera別値を作らない。", "", "| checkpoint | sha256 |", "|---|---|"]
    rows += [f"| {c['name']} | `{c['sha256']}` |" for c in manifest["checkpoints"]]
    rows += ["", f"選定: `{manifest['decision']['winner']}`。Meiji video_000だけで決定。",
             f"mixed e11 > e0: `{manifest['decision']['mixed_e11_beats_e0']}`。", ""]
    return "\n".join(rows)


def configure_comparison_device(device: str, cuda_allocator_limit_gib: float) -> torch.device:
    """Bind CUDA's current device to an explicit index before memory-limit APIs."""
    target = torch.device(device)
    if target.type not in ("cpu", "cuda") or not 0 < cuda_allocator_limit_gib <= 6:
        raise ValueError("Use cpu/cuda and a positive CUDA allocator limit of at most 6 GiB")
    if target.type == "cuda":
        index = torch.cuda.current_device() if target.index is None else target.index
        target = torch.device("cuda", index)
        total = torch.cuda.get_device_properties(target).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, cuda_allocator_limit_gib * 2**30 / total), target)
        torch.cuda.reset_peak_memory_stats(target)
    return target


def compare_detectors(
    *, store_directory: Path, cache_manifest: Path, checkpoints: tuple[SelectionCheckpoint, ...],
    output: Path, device: str, batch_size: int, cuda_allocator_limit_gib: float,
) -> Path:
    if not output.is_absolute() or batch_size < 1:
        raise ValueError("Output must be absolute and batch size positive")
    if output.exists():
        raise FileExistsError(f"Comparison output already exists: {output}")
    store, clips, manifest = prepare_comparison(store_directory, cache_manifest, checkpoints)
    target = configure_comparison_device(device, cuda_allocator_limit_gib)
    output.mkdir(parents=True)
    manifest.update(status="running", batch_size=batch_size, device=str(target), torch_version=str(torch.__version__),
                    cuda_allocator_limit_gib=cuda_allocator_limit_gib)
    write_json_atomic(output / "manifest.json", manifest)
    started = time.monotonic()
    for checkpoint in checkpoints:
        _evaluate_checkpoint(checkpoint, store, clips, output, manifest, device=target, batch_size=batch_size)
        gc.collect()
        if target.type == "cuda":
            torch.cuda.empty_cache()
    if store_hashes(store_directory) != manifest["store"]["sha256"] or dual_sha256(cache_manifest) != manifest["reference_cache"]["sha256"]:
        raise ValueError("Store or reference cache changed during comparison")
    for checkpoint in checkpoints:
        checkpoint.verify()
    # Require the same source pixels for all checkpoints, even if files changed between models.
    shard_identities = [{r["clip"]["clip_id"]: r["jpeg_shard_sha256"] for r in result["clips"]} for result in manifest["results"]]
    if any(identity != shard_identities[0] for identity in shard_identities[1:]):
        raise ValueError("Checkpoints did not read identical JPEG shards")
    manifest.update(status="complete", seconds=time.monotonic() - started, decision=select_winner(manifest["results"]))
    if target.type == "cuda":
        manifest["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(target)
        manifest["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(target)
    write_json_atomic(output / "manifest.json", manifest)
    (output / "comparison.md").write_text(comparison_markdown(manifest), encoding="utf-8")
    return output
