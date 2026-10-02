"""One r17 queue workload, with CPU preflight and a bounded CUDA allocator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import time
from dataclasses import asdict
from pathlib import Path
from types import FrameType
from typing import Any

import cv2
import torch
from omegaconf import OmegaConf

from src.tasks.ball_detection.data.store import BallFrameStore, shard_name
from src.tasks.ball_detection.inference.checkpoint import load_ball_checkpoint
from src.tasks.ball_detection.model_io.contracts import BallCandidateConfig
from src.tasks.ball_refiner.data.evidence_cache import (
    EvidenceCache,
    generate_evidence_cache,
)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def preflight(plan: dict[str, Any]) -> dict[str, Any]:
    for raw_path, expected in plan["input_sha256"].items():
        if digest(Path(raw_path)) != expected:
            raise ValueError(f"Input checksum changed: {raw_path}")
    output, report = Path(plan["output"]), Path(plan["report"])
    if output.exists() or report.exists():
        raise FileExistsError("Cache and report must both be new directories")
    if shutil.disk_usage(output.parent).free < plan["disk_budget_bytes"]:
        raise RuntimeError("Insufficient disk headroom for the declared run budget")
    available = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
                     if line.startswith("MemAvailable:"))
    if available < 8 * 1024 ** 3:
        raise RuntimeError("Require 8 GiB available before launch to retain 6 GiB host headroom")
    store = BallFrameStore(Path(plan["store"]))
    old = EvidenceCache(Path(plan["reference_manifest"]).parent, store)
    if old.manifest["selection"]["sources"] != plan["sources"] or old.manifest["selection"]["splits"] != plan["splits"]:
        raise ValueError("Source/split selection differs from r3")
    expected = old.manifest["detector"]
    for key, value in plan["settings"].items():
        if expected[key] != value:
            raise ValueError(f"Generation setting differs from r3: {key}")
    loaded = load_ball_checkpoint(Path(plan["checkpoint"]), strict=True, weights_only=False)
    actual = {"model_config": OmegaConf.to_container(loaded.config.model, resolve=True),
              "image_size_hw": list(loaded.config.data.image_size),
              "image_normalization": asdict(loaded.image_normalization),
              "window_length": int(loaded.config.model.num_frames)}
    # The persisted manifest serializes normalization tuples as JSON arrays.
    actual = json.loads(json.dumps(actual))
    for key, value in actual.items():
        if expected[key] != value:
            raise ValueError(f"Selected checkpoint input contract differs from r3: {key}")
    del loaded
    for record in old.manifest["clips"]:
        clip = store.clip_by_id(record["clip"]["clip_id"])
        shard = store.directory / "shards" / shard_name(clip.index)
        if digest(shard) != record["jpeg_shard_sha256"]:
            raise ValueError(f"JPEG identity differs from r3: {clip.clip_id}")
    return {"status": "passed", "cuda_used": False, "checkpoint_strict_cpu_load": True,
            "reference_manifest_sha256": digest(Path(plan["reference_manifest"])),
            "checkpoint_sha256": digest(Path(plan["checkpoint"])), "clips": len(old.clip_ids),
            "frames": sum(r["clip"]["frame_count"] for r in old.manifest["clips"]),
            "jpeg_shards_rehashed": len(old.clip_ids), "checkpoint_input_contract": actual,
            "ram_available_bytes": available}


def interrupted(signum: int, frame: FrameType | None) -> None:
    raise SystemExit(128 + signum)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    if args.check_only:
        print(json.dumps(preflight(plan), indent=2))
        return
    started = time.monotonic()
    report = Path(plan["report"])
    # Input/CPU validation happens before CUDA is initialized.
    checked = preflight(plan)
    report.mkdir(parents=True, exist_ok=False)
    write_json(report / "preflight.json", checked)
    shutil.copyfile(args.plan, report / "plan.json")
    signal.signal(signal.SIGTERM, interrupted)
    device: torch.device | None = None
    status = "failed"
    try:
        device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_per_process_memory_fraction(
            plan["allocator_limit_bytes"] / torch.cuda.get_device_properties(device).total_memory, device,
        )
        torch.cuda.reset_peak_memory_stats(device)
        settings = plan["settings"]
        output = generate_evidence_cache(
            store_directory=Path(plan["store"]), checkpoint=Path(plan["checkpoint"]), output=Path(plan["output"]),
            sources=tuple(plan["sources"]), splits=tuple(plan["splits"]), device=str(device),
            stride=settings["stride"], batch_size=settings["batch_size"], subpixel_refine=settings["subpixel_refine"],
            candidates=BallCandidateConfig(**settings["candidates"]),
        )
        cache = EvidenceCache(output, BallFrameStore(Path(plan["store"])))
        old = json.loads(Path(plan["reference_manifest"]).read_text())
        if cache.manifest["selection"] != old["selection"] or cache.manifest["detector"]["sha256"] != checked["checkpoint_sha256"]:
            raise ValueError("Generated cache differs from selected checkpoint or r3 coverage")
        if cache.manifest["context"] != {"pose": "not_generated", "court": "not_generated"}:
            raise ValueError("Unexpected context in ball-only cache")
        files = []
        for record, old_record in zip(cache.manifest["clips"], old["clips"], strict=True):
            if record["clip"] != old_record["clip"] or record["jpeg_shard_sha256"] != old_record["jpeg_shard_sha256"]:
                raise ValueError("Generated clip identity differs from r3")
            cache.load(record["clip"]["clip_id"])
            path = output / record["file"]
            files.append({"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size})
        # Record a final immutable manifest snapshot and every generated NPZ hash.
        shutil.copyfile(output / "manifest.json", report / "manifest.json")
        write_json(report / "artifact_hashes.json", {
            "manifest_sha256": digest(output / "manifest.json"),
            "checkpoint_sha256": checked["checkpoint_sha256"], "files": files,
            "files_bytes": sum(f["bytes"] for f in files), "validated_clips": len(files),
            "validated_frames": checked["frames"], "plan_sha256": digest(args.plan),
        })
        status = "complete"
    finally:
        write_json(report / "resource_usage.json", {
            "status": status, "seconds": time.monotonic() - started,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device is not None else None,
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device is not None else None,
            "pytorch_allocator_limit_bytes": plan["allocator_limit_bytes"],
            "queue_job": os.environ.get("TENNIS_RUN_ID"), "output": plan["output"],
        })


if __name__ == "__main__":
    main()
