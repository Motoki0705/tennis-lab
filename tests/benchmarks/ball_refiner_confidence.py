"""CPU-only Meiji validation selection; clip_000 and video_001 are never opened."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from src.tasks.ball_refiner.pipeline_options import E9_ANCHORED_S42
from src.tasks.ball_refiner.refiner_2d.calibration import load_covariance_calibration
from src.tasks.ball_refiner.refiner_2d.distribution import BallGMM2D
from src.utils.checksum import dual_sha256

if TYPE_CHECKING or __package__ == "tests.benchmarks":
    from tests.benchmarks.legacy_ball_confidence import (
        PointConfidenceRule,
        point_confidence,
    )
else:
    from legacy_ball_confidence import PointConfidenceRule, point_confidence


def eligible(clip_id: str) -> bool:
    return any(clip_id == f"meiji/video_000/clip_{i:03d}/cam{c}" for i in range(1, 12) for c in range(3))


def quantiles(values: Any) -> dict[str, Any]:
    return {"n": len(values), "median_px": None if len(values) == 0 else float(np.median(values)),
            "p90_px": None if len(values) == 0 else float(np.quantile(values, .9))}


def summarize(rows: list[dict[str, Any]], rule: PointConfidenceRule) -> dict[str, Any]:
    cameras: dict[str, Any] = {}
    multiview: dict[str, Any] = {}
    for camera in ("cam0", "cam1", "cam2"):
        selected = [r for r in rows if r["camera"] == camera]
        mask = np.concatenate([rule.rejection_codes(r["presence"], r["area"]) == 0 for r in selected])
        observed = np.concatenate([r["observed"] for r in selected])
        error = np.concatenate([r["error"] for r in selected])
        cameras[camera] = {"frames": len(mask), "kept": int(mask.sum()), "kept_fraction": float(mask.mean()),
                           "kept_observed": quantiles(error[mask & observed]),
                           "dropped_observed": quantiles(error[~mask & observed])}
    for clip in sorted({r["clip"] for r in rows}):
        selected = sorted((r for r in rows if r["clip"] == clip), key=lambda r: r["camera"])
        if len(selected) != 3 or any(not np.array_equal(r["frame_index"], selected[0]["frame_index"]) for r in selected):
            raise ValueError("Require three frame-aligned cameras per validation clip")
        masks = np.stack([rule.rejection_codes(r["presence"], r["area"]) == 0 for r in selected])
        multiview[clip] = {"frames": masks.shape[1], "at_least_two": int((masks.sum(0) >= 2).sum()),
                           "all_three": int(masks.all(0).sum()),
                           "pairs": {f"{a}-{b}": int((masks[a] & masks[b]).sum()) for a, b in ((0, 1), (0, 2), (1, 2))}}
    passed = (all(c["kept_observed"]["n"] >= 30 and c["kept_observed"]["median_px"] <= 5
                  and c["kept_observed"]["p90_px"] <= 20 and c["kept_fraction"] >= .05 for c in cameras.values())
              and all(m["at_least_two"] >= 8 for m in multiview.values()))
    return {"min_presence": rule.min_presence, "max_area_px2": rule.max_area_px2, "passes": passed,
            "kept": sum(c["kept"] for c in cameras.values()), "cameras": cameras, "multiview": multiview}


def run(plan_path: Path, calibration_path: Path, metadata_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    plan = json.loads(plan_path.read_text())
    source = Path(plan["source"])
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if dual_sha256(metadata_path) != manifest["input_sha256"][str(metadata_path)]:
        raise ValueError("Store metadata differs from the saved prediction inputs")
    best_path = Path(manifest["training_run"]) / "best.json"
    if (json.loads(best_path.read_text())["checkpoint_sha256"] != E9_ANCHORED_S42.checkpoint_sha256
            or dual_sha256(best_path) != manifest["input_sha256"][str(best_path)]):
        raise ValueError("Saved predictions do not bind the frozen anchored seed42 checkpoint")
    calibration = load_covariance_calibration(calibration_path, expected_sha256=E9_ANCHORED_S42.calibration_sha256,
                                              checkpoint_sha256=E9_ANCHORED_S42.checkpoint_sha256)
    if calibration.covariance_multiplier != plan["scale"]:
        raise ValueError("Plan covariance scale mismatch")
    records = {r["clip_id"]: r for r in json.loads(metadata_path.read_text())["clips"] if eligible(r["clip_id"])}
    entries = [r for r in manifest["artifacts"] if r["condition"] == "observed" and eligible(r["clip_id"])]
    if len(records) != 33 or len(entries) != 33 or {r["clip_id"] for r in entries} != set(records):
        raise ValueError("Exactly clip_001..011 x three cameras are required; no substitution")
    hashes = {str(p): dual_sha256(p) for p in (plan_path, manifest_path, metadata_path, best_path, calibration_path)}
    rows: list[dict[str, Any]] = []
    for entry in entries:
        record = records[entry["clip_id"]]
        if record["split"] != "val" or entry["source"] != "meiji":
            raise ValueError("Confidence selection may only read Meiji val")
        path = source / entry["path"]
        hashes[str(path)] = dual_sha256(path)
        if hashes[str(path)] != entry["sha256"]:
            raise ValueError("Saved predictions changed")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {k: archive[k] for k in archive.files}
        distribution = BallGMM2D(**{f.name: torch.from_numpy(arrays[f.name])[None] for f in fields(BallGMM2D)})
        size = (record["source_width"], record["source_height"])
        point, presence, area = point_confidence(calibration.apply(distribution), size)
        error = np.linalg.norm(point[0] - arrays["target_uv"] * (np.array(size) - 1), axis=-1)
        observed = arrays["target_reason"] == 0
        np.testing.assert_allclose(error[observed], arrays["error_px"][observed], rtol=1e-4, atol=1e-4)
        rows.append({"clip": entry["clip_id"].rsplit("/", 1)[0], "camera": entry["camera"],
                     "frame_index": arrays["frame_index"], "presence": presence[0], "area": area[0],
                     "point": point[0], "error": error, "observed": observed})
    candidates = [summarize(rows, PointConfidenceRule(float(p), float(a)))
                  for p, a in product(plan["presence_grid"], plan["area_grid_px2"])]
    feasible = sorted((c for c in candidates if c["passes"]), key=lambda c: (-c["kept"], c["max_area_px2"], -c["min_presence"]))
    report = {"schema": "i935.confidence_selection.v1", "status": "PASS" if feasible else "FAIL",
              "selected": feasible[0] if feasible else None, "candidates": candidates, "input_sha256": hashes,
              "excluded": plan["exclude"], "scored": "observed annotations only; cached JPEG-input predictions, not MP4 deployment accuracy"}
    output.mkdir(parents=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    for r in rows:
        np.savez_compressed(output / f"{r['clip'].rsplit('/', 1)[1]}-{r['camera']}.npz",
                            **{k: v for k, v in r.items() if k not in {"clip", "camera"}})
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = run(args.plan, args.calibration, args.metadata, args.output)
    print(json.dumps({"status": result["status"], "selected": result["selected"]}, indent=2))
