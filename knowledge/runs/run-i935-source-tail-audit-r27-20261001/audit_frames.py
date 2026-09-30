"""CPU audit of the frozen clip_010 source/cache evidence (no inference or gate change)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from src.tasks.ball_detection.data.store import BallFrameStore, shard_name
from src.tasks.ball_refiner.data.evidence_cache import EvidenceCache
from src.tasks.ball_refiner.data.temporal import window_owners, window_starts
from src.utils.checksum import dual_sha256

ROOT = Path(__file__).resolve().parent
PLAN = ROOT.parent / "run-i935-source-check-retry-r25-20260930/plan.json"
GATE = ROOT.parent / "run-i935-source-b-gate-r26-20261001/results"
STORE = Path("/home/kamimura/projects/tennis-lab/data/ball_detection/ball-mix-v1")
CACHE = Path("/home/kamimura/projects/tennis-lab/data/ball_refiner/detector-mixed-e9-trainval-r17-20260930")


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def anchor_order(coords: np.ndarray, scores: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Mirror score-desc/y-asc/x-asc assignment, including invalid slots at the end."""
    return cast(np.ndarray, np.lexsort((coords[..., 0], coords[..., 1], -np.where(valid, scores, -np.inf)), axis=-1))


def match_candidates(a: np.ndarray, b: np.ndarray, av: np.ndarray, bv: np.ndarray,
                     radius: float = 20.) -> np.ndarray:
    """Maximum-cardinality spatial pairing then minimum distance; -1 means unmatched.

    This is a diagnostic, not physical candidate identity or an accuracy gate.
    Dummy columns prevent distant/invalid peaks from acquiring a counterpart.
    """
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 2 or av.shape != a.shape[:1] or bv.shape != av.shape:
        raise ValueError("Aligned K,2 coordinates and K masks required")
    if radius <= 0 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Finite coordinates and a positive radius required")
    k = len(a)
    distance = np.linalg.norm(a[:, None] - b[None], axis=-1)
    penalty = (k + 1) * radius + 1
    cost = np.full((k, 2 * k), penalty)
    cost[:, :k] = np.where(av[:, None] & bv[None] & (distance <= radius), distance, 2 * penalty)
    rows, cols = linear_sum_assignment(cost)
    result: np.ndarray = np.full(k, -1, np.int64)
    keep = cols < k
    result[rows[keep]] = cols[keep]
    return result


class Inputs:
    """Pin all read artifacts; verify expected history hashes and final immutability."""

    def __init__(self) -> None:
        self.hashes: dict[str, str] = {}

    def pin(self, path: Path, expected: str | None = None) -> Path:
        digest = dual_sha256(path)
        if expected is not None and digest != expected:
            raise ValueError(f"Input checksum mismatch: {path}")
        self.hashes[str(path)] = digest
        return path

    def finish(self, output: Path) -> None:
        for name, digest in self.hashes.items():
            if dual_sha256(Path(name)) != digest:
                raise ValueError(f"Input changed during audit: {name}")
        write(output, self.hashes)


def source_evidence(plan: dict[str, Any], camera: str, inputs: Inputs) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = read(inputs.pin(Path(plan["report"]) / camera / "execute.json"))
    ref = summary["references"][f"ball_detection/{camera}"]
    path = inputs.pin(Path(summary["scene_index"]).parent / ref["path"], ref["sha256"])
    doc = read(path)
    evidence = doc["payload"]["evidence"]
    result = {}
    for key, value in evidence.items():
        if isinstance(value, dict) and set(value) == {"array"}:
            name = value["array"]
            result[key] = np.load(inputs.pin(path.parent / name, doc["arrays"][name]["sha256"]), allow_pickle=False)
        else:
            result[key] = value
    return result, doc["identity"]["settings"]


def main() -> None:
    torch.set_num_threads(1)
    out = ROOT / "frames"
    out.mkdir(exist_ok=False)
    inputs = Inputs()
    plan = read(inputs.pin(PLAN))
    bundle = read(inputs.pin(Path(plan["bundle"]) / "manifest.json"))
    for name in ("metadata.json", "index.npz"):
        inputs.pin(STORE / name)
    inputs.pin(CACHE / "manifest.json", bundle["provenance"]["evidence_manifest_sha256"])
    store = BallFrameStore(STORE)
    cache = EvidenceCache(CACHE, store)
    frames, candidates, components, summaries = [], [], [], []
    with inputs.pin(GATE / "observed-frames.csv").open() as stream:
        gt = {(r["camera"], int(r["frame"])): r for r in csv.DictReader(stream)}
    for row in plan["cameras"]:
        cam = row["camera"]
        clip = store.clip_by_id(f"{plan['clip_id']}/{cam}")
        inputs.pin(STORE / "shards" / shard_name(clip.index))
        cache_record = next(r for r in cache.manifest["clips"] if r["clip"]["clip_id"] == clip.clip_id)
        inputs.pin(CACHE / cache_record["file"], cache_record["sha256"])
        evidence = cache.load(clip.clip_id)
        source, settings = source_evidence(plan, cam, inputs)
        inputs.pin(Path(row["cached_prediction"]), row["cached_sha256"])
        with np.load(row["cached_prediction"], allow_pickle=False) as z:
            cached = dict(z)
        with np.load(inputs.pin(Path(plan["report"]) / cam / "execute.npz"), allow_pickle=False) as z:
            ref = dict(z)
        for key, value in {"frame_indices": evidence.frame_index, "pts": evidence.pts,
                           "timestamps_seconds": evidence.timestamps_seconds,
                           "detector_window_start": evidence.window_start, "detector_time_index": evidence.time_index}.items():
            np.testing.assert_array_equal(ref[key], value)
        for length, stride, prefix in ((8, 4, "detector"), (33, 16, "refiner")):
            starts = window_starts(270, length, stride)
            expected = np.asarray(starts)[window_owners(270, starts, length)]
            np.testing.assert_array_equal(ref[f"{prefix}_window_start"], expected)
            np.testing.assert_array_equal(ref[f"{prefix}_time_index"], np.arange(270) - expected)
        np.testing.assert_array_equal(source["selected_window_start"], evidence.window_start)
        np.testing.assert_array_equal(source["selected_time_index"], evidence.time_index)
        np.testing.assert_array_equal(cached["frame_index"], evidence.frame_index)
        np.testing.assert_array_equal(cached["pts"], evidence.pts)
        scale = np.asarray([clip.source_width - 1, clip.source_height - 1], np.float32)
        c = evidence.candidates
        cp, sp = c.coords[0].numpy() * scale, source["candidate_uv_px"]
        cs, ss = c.scores[0].numpy(), source["candidate_scores"]
        cv, sv = c.valid[0].numpy(), source["candidate_valid"]
        ca, sa = anchor_order(cp, cs, cv), anchor_order(sp, ss, sv)
        cw, sw = cached["mixture_logits"].argmax(-1), ref["mixture_logits"].argmax(-1)
        cweight = torch.from_numpy(cached["mixture_logits"]).softmax(-1).numpy()
        sweight = torch.from_numpy(ref["mixture_logits"]).softmax(-1).numpy()
        camera_rows = []
        for f in range(270):
            match = match_candidates(cp[f], sp[f], cv[f], sv[f])
            native_equal = np.all(c.cells[0, f].numpy() == source["candidate_cells"][f], axis=-1) & cv[f] & sv[f]
            ci, si = int(cw[f]), int(sw[f])
            cslot, sslot = (int(ca[f, ci]) if ci < 3 else -1), (int(sa[f, si]) if si < 3 else -1)
            anchored = cslot >= 0 and sslot >= 0
            anchor_changed = bool(match[cslot] != sslot) if anchored else None
            label = gt.get((cam, f))
            record = {"camera": cam, "frame": f, "pts": int(evidence.pts[f]),
                      "detector_window_start": int(evidence.window_start[f]), "detector_time_index": int(evidence.time_index[f]),
                      "refiner_window_start": int(ref["refiner_window_start"][f]), "refiner_time_index": int(ref["refiner_time_index"][f]),
                      "matched_peaks_20px": int((match >= 0).sum()),
                      "same_rank_native_cells": int(native_equal.sum()),
                      "matched_rank_changes": int(((match >= 0) & (match != np.arange(8))).sum()),
                      "top1_distance_px": float(np.linalg.norm(cp[f, 0] - sp[f, 0])),
                      "top1_spatial_match": int(match[0]),
                      "cached_winner": ci, "source_winner": si, "winner_changed": ci != si,
                      "cached_winner_anchor_slot": cslot, "source_winner_anchor_slot": sslot,
                      "winner_anchor_changed_20px": anchor_changed,
                      "winner_mean_distance_px": float(np.linalg.norm((cached["means"][f, ci] - ref["means"][f, si]) * scale)),
                      "cached_error_px": float(label["cached_error_px"]) if label else None,
                      "source_error_px": float(label["source_error_px"]) if label else None}
            frames.append(record)
            camera_rows.append(record)
            for k in range(8):
                candidates.append({"camera": cam, "frame": f, "rank": k,
                    "cache_x": float(cp[f, k, 0]), "cache_y": float(cp[f, k, 1]), "cache_score": float(cs[f, k]), "cache_valid": bool(cv[f, k]),
                    "source_x": float(sp[f, k, 0]), "source_y": float(sp[f, k, 1]), "source_score": float(ss[f, k]), "source_valid": bool(sv[f, k]),
                    "matched_source_rank_20px": int(match[k])})
            for k in range(4):
                components.append({"camera": cam, "frame": f, "component": k,
                    "cached_x": float(cached["means"][f, k, 0] * scale[0]), "cached_y": float(cached["means"][f, k, 1] * scale[1]),
                    "source_x": float(ref["means"][f, k, 0] * scale[0]), "source_y": float(ref["means"][f, k, 1] * scale[1]),
                    "cached_weight": float(cweight[f, k]), "source_weight": float(sweight[f, k]),
                    "cached_anchor_slot": int(ca[f, k]) if k < 3 else -1, "source_anchor_slot": int(sa[f, k]) if k < 3 else -1})
        arrays: dict[str, Any] = {**{f"cache_{k}": v for k, v in evidence.arrays().items()},
                            **{f"source_{k}": v for k, v in source.items() if isinstance(v, np.ndarray)},
                            **{f"cached_refiner_{k}": cached[k] for k in ("means", "scale_tril", "mixture_logits", "presence_logits")},
                            **{f"source_refiner_{k}": ref[k] for k in ("means", "scale_tril", "mixture_logits", "presence_logits")}}
        np.savez_compressed(out / f"{cam}.npz", **arrays)
        summaries.append({"camera": cam, "frames": 270, "timeline_windows_equal": True,
            "winner_component_changes": sum(r["winner_changed"] for r in camera_rows),
            "both_winners_anchored": sum(r["winner_anchor_changed_20px"] is not None for r in camera_rows),
            "winner_anchor_changes_20px": sum(r["winner_anchor_changed_20px"] is True for r in camera_rows),
            "top1_changed_20px": sum(r["top1_spatial_match"] != 0 for r in camera_rows),
            "frames_with_matched_rank_changes": sum(r["matched_rank_changes"] > 0 for r in camera_rows),
            "cache_free_winner": int((cw == 3).sum()), "source_free_winner": int((sw == 3).sum()),
            "matched_peaks_20px": sum(r["matched_peaks_20px"] for r in camera_rows),
            "candidate_slots": 2160, "same_rank_native_cells": sum(r["same_rank_native_cells"] for r in camera_rows),
            "pipeline_settings": settings, "cache_settings": cache.manifest["detector"]})
    csv_rows(out / "frames.csv", frames)
    csv_rows(out / "candidates.csv", candidates)
    csv_rows(out / "components.csv", components)
    write(out / "summary.json", {"status": "complete", "matching": "maximum cardinality within 20 source px, then minimum total distance; diagnostic only",
                                  "cameras": summaries, "frame_rows": len(frames), "candidate_rows": len(candidates), "component_rows": len(components)})
    inputs.finish(out / "input_sha256.json")
    print(json.dumps([{k: v for k, v in s.items() if not k.endswith("settings")} for s in summaries], indent=2))


if __name__ == "__main__":
    main()
