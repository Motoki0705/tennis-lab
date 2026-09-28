"""CPU integrity audit; run from the captured checkout with PYTHONPATH=."""

import json
from collections import Counter
from pathlib import Path

import torch

from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_refiner.data.evidence_cache import EvidenceCache
from src.utils.checksum import dual_sha256


def main() -> None:
    torch.set_num_threads(4)
    root = Path("/home/kamimura/projects/tennis-lab")
    store = BallFrameStore(root / "data/ball_detection/ball-mix-v1")
    directory = root / "data/ball_refiner/detector-ft-e13-trainval-r3-20260928"
    cache = EvidenceCache(directory, store)
    counts: dict[str, Counter[str]] = {}
    for clip_id in cache.clip_ids:
        clip = store.clip_by_id(clip_id)
        evidence = cache.load(clip_id)
        row = counts.setdefault(f"{clip.source}/{clip.split}", Counter())
        row["clips"] += 1
        row["frames"] += len(evidence.frame_index)
        row["frames_without_candidates"] += int((~evidence.candidates.valid.any(-1)).sum())
        row["valid_candidates"] += int(evidence.candidates.valid.sum())
    report = {
        "cache": str(directory), "manifest_sha256": dual_sha256(directory / "manifest.json"),
        "status": cache.manifest["status"], "counts": counts,
        "clips": len(cache.clip_ids), "frames": sum(c["frames"] for c in counts.values()),
        "verified": "EvidenceCache.load: every NPZ checksum, contract, frame, PTS and seconds",
        "not_verified": "Original media/annotations and JPEG shards not rehashed in this recovery audit; no accuracy measurement",
    }
    Path(__file__).with_name("cache_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
