"""Verify Meiji 4-camera multiview-v4 benchmark, SceneResult v4 readback, and non-event parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_sums(path: Path) -> tuple[dict[str, str], int]:
    entries: dict[str, str] = {}
    duplicates = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        if rel in entries:
            duplicates += 1
        entries[rel] = digest
    return entries, duplicates


def verify(
    *,
    v3_root: Path,
    v4_root: Path,
    v1_dir: Path,
    v2_dir: Path,
) -> dict[str, Any]:
    from src.tasks.ball_refiner_3d.post_training.dataset import MeijiBenchmark
    from src.tasks.ball_refiner_3d.post_training.labels import Annotations
    from src.tennis_scene.ball_event_schema import (
        EVENT_CLASSES_V4,
        EVENT_SUBTYPES_V4,
    )
    from src.tennis_scene.pipeline.imports.scene_events import verify_replayed_scene

    v3_sums, v3_dup = _parse_sums(v3_root / "SHA256SUMS")
    v4_sums, v4_dup = _parse_sums(v4_root / "SHA256SUMS")
    if v3_dup != 0 or v4_dup != 0:
        raise ValueError(f"Duplicate entries in SHA256SUMS: v3={v3_dup}, v4={v4_dup}")

    from concurrent.futures import ThreadPoolExecutor

    def _check_entry(item: tuple[str, str]) -> str:
        rel, expected = item
        p4 = v4_root / rel
        if not p4.is_file():
            return f"missing:{rel}"
        p3 = v3_root / rel
        if rel in v3_sums and v3_sums[rel] == expected and p3.is_file():
            s3 = p3.stat()
            s4 = p4.stat()
            if (s3.st_dev, s3.st_ino) == (s4.st_dev, s4.st_ino):
                return "hardlink"
        return "rehash" if _sha256(p4) == expected else f"digest:{rel}"

    hardlinked_ok = 0
    rehashed_ok = 0
    mismatches: list[str] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for outcome in pool.map(_check_entry, v4_sums.items()):
            if outcome == "hardlink":
                hardlinked_ok += 1
            elif outcome == "rehash":
                rehashed_ok += 1
            else:
                mismatches.append(outcome)

    if mismatches:
        raise ValueError(f"SHA256SUMS verification failed ({len(mismatches)} errors): {mismatches[:5]}")

    v4_manifest_sha = _sha256(v4_root / "manifest.json")
    bench_v4 = MeijiBenchmark.load(v4_root, v4_manifest_sha)
    clip_split = {
        cid: split_name
        for split_name, cids in bench_v4.splits.items()
        for cid in cids
    }

    kind_counts: Counter[str] = Counter()
    subtype_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    split_counts: dict[str, dict[str, int]] = {}
    non_event_verified_clips = 0
    total_adopted = 0
    total_excluded = 0
    total_v1 = 0
    spotlight: dict[str, Any] = {}

    for clip_id in sorted(bench_v4.clips):
        row = bench_v4.clips[clip_id]
        clip_name = clip_id.split("/")[-1]
        v1_doc = json.loads((v1_dir / f"{clip_name}_ball_events.json").read_text(encoding="utf-8"))
        v2_doc = json.loads((v2_dir / f"{clip_name}_ball_events.json").read_text(encoding="utf-8"))
        v1_events = v1_doc["events"]
        v2_events = v2_doc["events"]
        v2_excluded = v2_doc["excluded_events"]

        total_v1 += len(v1_events)
        total_adopted += len(v2_events)
        total_excluded += len(v2_excluded)

        v1_ids = {ev["id"]: ev for ev in v1_events}
        v2_adopted_ids = {ev["id"] for ev in v2_events}
        v2_excluded_ids = {ev["id"] for ev in v2_excluded}
        if v2_adopted_ids & v2_excluded_ids:
            raise ValueError(f"{clip_id}: overlap between adopted and excluded IDs")
        if (v2_adopted_ids | v2_excluded_ids) != set(v1_ids):
            raise ValueError(f"{clip_id}: v2 events do not partition v1 events 1-to-1")

        # Verify byte-for-byte non-event array parity against immutable v2 scene_assembly.
        v3_scene_json = v3_root / row["scene"]["path"]
        v4_scene_json = v4_root / row["scene"]["path"]
        scene = verify_replayed_scene(v3_scene_json, v4_scene_json, expected_version=4)
        non_event_verified_clips += 1

        if scene.schema_version != 4:
            raise ValueError(f"{clip_id}: expected schema_version=4, got {scene.schema_version}")
        if scene.ball_event_in_point is not None:
            raise ValueError(f"{clip_id}: SceneResult v4 must omit ball_event_in_point")
        assert scene.ball_event_frame is not None
        assert scene.ball_event_kind is not None
        assert scene.ball_event_subtype is not None

        if len(scene.ball_event_frame) != len(v2_events):
            raise ValueError(f"{clip_id}: scene event count mismatch")

        for idx, ev in enumerate(v2_events):
            if int(scene.ball_event_frame[idx]) != ev["frame"]:
                raise ValueError(f"{clip_id} {ev['id']}: frame mismatch")
            if ev["frame"] != v1_ids[ev["id"]]["frame"]:
                raise ValueError(f"{clip_id} {ev['id']}: original v1 frame mismatch")
            kind_name = EVENT_CLASSES_V4[int(scene.ball_event_kind[idx])]
            subtype_name = EVENT_SUBTYPES_V4[int(scene.ball_event_subtype[idx])]
            if kind_name != ev["kind"] or subtype_name != ev["subtype"]:
                raise ValueError(f"{clip_id} {ev['id']}: readback mismatch {kind_name}/{subtype_name}")
            if subtype_name in {"other", "none"}:
                raise ValueError(f"{clip_id} {ev['id']}: forbidden subtype {subtype_name}")
            kind_counts[kind_name] += 1
            subtype_counts[subtype_name] += 1
            status_counts[ev["status"]] += 1

        # Verify post_training Annotations.from_scene readback.
        pt_ann = Annotations.from_scene(scene)
        if len(pt_ann.events) != len(v2_events):
            raise ValueError(f"{clip_id}: post_training event count mismatch")
        if any(not e.in_point for e in pt_ann.events):
            raise ValueError(f"{clip_id}: post_training v4 events must all have in_point=True")

        split_name = clip_split[clip_id]
        sp = split_counts.setdefault(split_name, {"clips": 0, "events": 0, "shot": 0, "bounce": 0, "net": 0})
        sp["clips"] += 1
        sp["events"] += len(v2_events)
        for ev in v2_events:
            sp[ev["kind"]] += 1

        by_eid = {ev["id"]: ev for ev in v2_events}
        ex_eid = {ev["id"]: ev for ev in v2_excluded}
        if clip_name in {"clip_009", "clip_012", "clip_023", "clip_035"}:
            lets = [
                {"event_id": e["id"], "frame": e["frame"], "subtype": e["subtype"]}
                for e in v2_events
                if e["subtype"] == "service_let"
            ]
            let_bounces = [
                {
                    "event_id": e["id"],
                    "frame": e["frame"],
                    "subtype": e["subtype"],
                    "review_source": e["review_source"],
                }
                for e in v2_events
                if e["review_source"] == "service_let_first_bounce"
            ]
            spotlight[f"{clip_name}_service_let"] = {"lets": lets, "first_bounces": let_bounces}
        if clip_name == "clip_026":
            spotlight["clip_026_net_in"] = {
                "e006": {"frame": by_eid["e006"]["frame"], "kind": by_eid["e006"]["kind"], "subtype": by_eid["e006"]["subtype"]},
                "e007": {
                    "frame": by_eid["e007"]["frame"],
                    "kind": by_eid["e007"]["kind"],
                    "subtype": by_eid["e007"]["subtype"],
                    "review_source": by_eid["e007"]["review_source"],
                },
            }
        if clip_name == "clip_035":
            spotlight["clip_035_e041_old_other"] = {
                "adopted": "e041" in by_eid,
                "frame": by_eid["e041"]["frame"],
                "kind": by_eid["e041"]["kind"],
                "subtype": by_eid["e041"]["subtype"],
            }
        if clip_name == "clip_056":
            spotlight["clip_056_e004_old_other"] = {
                "adopted": "e004" in by_eid,
                "excluded": "e004" in ex_eid,
                "frame": ex_eid["e004"]["frame"],
                "reason": ex_eid["e004"]["reason"],
            }
        if clip_name == "clip_000":
            spotlight["clip_000_e004_volley_center"] = {
                "frame": by_eid["e004"]["frame"],
                "kind": by_eid["e004"]["kind"],
                "subtype": by_eid["e004"]["subtype"],
            }

    return {
        "benchmark_id": "meiji_4cam/multiview-v4",
        "scene_schema_version": bench_v4.scene_schema_version,
        "clips_verified": len(bench_v4.clips),
        "non_event_byte_identical_clips": non_event_verified_clips,
        "manifest_sha256": v4_manifest_sha,
        "sha256sums_sha256": _sha256(v4_root / "SHA256SUMS"),
        "sha256sums_check": {
            "total_entries": len(v4_sums),
            "hardlinked_v3_files": hardlinked_ok,
            "newly_hashed_v4_files": rehashed_ok,
            "duplicates": v4_dup,
            "mismatches": len(mismatches),
        },
        "event_counts": {
            "v1_total": total_v1,
            "v2_adopted": total_adopted,
            "v2_excluded": total_excluded,
            "bijection_ok": total_v1 == total_adopted + total_excluded,
        },
        "kind_counts": dict(sorted(kind_counts.items())),
        "subtype_counts": dict(sorted(subtype_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "split_counts": split_counts,
        "spotlight_checks": spotlight,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Meiji 4-camera multiview-v4 benchmark.")
    parser.add_argument(
        "--v3-root",
        type=Path,
        default=Path("data/tennis_multiview/processed/meiji_4cam/benchmarks/multiview-v3"),
    )
    parser.add_argument(
        "--v4-root",
        type=Path,
        default=Path("data/tennis_multiview/processed/meiji_4cam/benchmarks/multiview-v4"),
    )
    parser.add_argument(
        "--v1-dir",
        type=Path,
        default=Path("data/tennis_multiview/processed/meiji_4cam/dataset/videos/video_000/clip_000-068_ball_events"),
    )
    parser.add_argument(
        "--v2-dir",
        type=Path,
        default=Path("data/tennis_multiview/processed/meiji_4cam/dataset/videos/video_000/clip_000-068_ball_events_v2"),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = verify(
        v3_root=args.v3_root,
        v4_root=args.v4_root,
        v1_dir=args.v1_dir,
        v2_dir=args.v2_dir,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
