"""Replay #932's held-out safety bench with a frozen empirical confidence filter.

The original ball/calibration perturbations and RNG stream remain identical.
Confidence comes from contiguous, synchronized Meiji validation blocks, sampled
independently of synthetic errors; this tests filtering robustness, not learned
GMM accuracy on synthetic video. Confident false detections remain possible.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from omegaconf import OmegaConf

from src.tasks.ball_refiner.refiner_2d.confidence import PointConfidenceRule
from src.tasks.court_side.benchmark import (
    judge,
    load_blcs_scene,
    make_trials,
    outcome_row,
)
from src.tasks.court_side.hypothesis import CourtSideConfig
from src.tasks.court_side.scripts.benchmark_synthetic import CONDITIONS
from src.utils.checksum import dual_sha256


def replay_mask(
    blocks: list[tuple[NDArray[np.float64], NDArray[np.float64]]], *, frames: int, views: int,
    size: tuple[int, int], rule: PointConfidenceRule, rng: np.random.Generator,
) -> NDArray[np.bool_]:
    """Match ~30 fps with stride 2 of 59.94 fps validation; preserve first 3-view correlation.

    A fourth view uses an independent block from one empirical camera. No wrapping,
    padding or labels: blocks too short for the requested window are ineligible.
    """
    eligible = [b for b in blocks if b[0].shape[1] >= frames]
    if not eligible or views not in {3, 4}:
        raise ValueError("No complete confidence block or unsupported camera count")
    probability, area = eligible[int(rng.integers(len(eligible)))]
    start = int(rng.integers(probability.shape[1] - frames + 1))
    order = rng.permutation(3)
    p, a = probability[order, start:start + frames], area[order, start:start + frames]
    if views == 4:
        extra_p, extra_a = eligible[int(rng.integers(len(eligible)))]
        extra_start = int(rng.integers(extra_p.shape[1] - frames + 1))
        camera = int(rng.integers(3))
        p = np.vstack((p, extra_p[camera, extra_start:extra_start + frames]))
        a = np.vstack((a, extra_a[camera, extra_start:extra_start + frames]))
    # All empirical moments used 1920x1080 source pixels; transform ellipse area
    # by the determinant of the source-pixel coordinate scaling.
    area_scale = ((size[0] - 1) * (size[1] - 1)) / (1919 * 1079)
    return rule.rejection_codes(p, a * area_scale) == 0


def run(dataset: Path, confidence: Path, rule_path: Path, original: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    fixed = json.loads(original.read_text())
    selected = fixed["selected"]
    cfg = CourtSideConfig(**{**selected, "height_range_m": tuple(selected["height_range_m"])})
    if cfg.min_margin != .15 or (cfg.min_frames, cfg.max_cost, cfg.min_support, cfg.min_motion_px) != (8, .8, .2, 5.):
        raise ValueError("The original frozen court-side thresholds must be unchanged")
    rule = PointConfidenceRule(**dict(OmegaConf.load(rule_path)))
    selection_report = json.loads((confidence / "report.json").read_text())
    if selection_report["status"] != "PASS" or any(asdict(rule)[k] != selection_report["selected"][k] for k in asdict(rule)):
        raise ValueError("Confidence rule must equal the previously selected/pushed rule")
    hashes = {str(p): dual_sha256(p) for p in (original, rule_path, confidence / "report.json", dataset / "test.txt")}
    blocks = []
    for clip in range(1, 12):
        cameras = []
        for camera in range(3):
            path = confidence / f"clip_{clip:03d}-cam{camera}.npz"
            hashes[str(path)] = dual_sha256(path)
            with np.load(path, allow_pickle=False) as data:
                cameras.append((data["presence"][::2], data["area"][::2]))
        blocks.append((np.stack([c[0] for c in cameras]), np.stack([c[1] for c in cameras])))
    names = sorted((dataset / "test.txt").read_text().split())[600:1000]
    if len(names) != 400:
        raise ValueError("Require the original 400 held-out test scenes")
    scenes = [load_blcs_scene(dataset / "scenes" / name) for name in names]
    rng, confidence_rng = np.random.default_rng(1), np.random.default_rng(29001)
    report: dict[str, Any] = {"schema": "i935.court_side_filtered_safety.v1", "status": "running",
                            "thresholds": asdict(cfg), "rule": asdict(rule), "scenes": names,
                            "seed": 1, "confidence_seed": 29001, "conditions": {}, "input_sha256": hashes,
                            "scope": "original 28 conditions x 400 scenes; independent empirical confidence block replay, not refiner inference"}
    started = time.monotonic()
    with (output / "evidence.jsonl").open("w") as stream:
        for condition in CONDITIONS:
            pairs = []
            for scene in scenes:
                frames = min(condition.window_frames, len(scene.ball_xyz) - condition.sync_offset_frames)
                mask = replay_mask(blocks, frames=frames, views=condition.cameras, size=scene.image_size,
                                   rule=rule, rng=confidence_rng)
                pair = make_trials(scene, condition, (cfg, cfg), rng, observation_masks=(None, mask))
                pairs.append(pair)
                for label, trial in zip(("original", "filtered"), pair, strict=True):
                    stream.write(json.dumps({"scene": trial.scene_id, "condition": condition.name, "path": label,
                                             "expected": trial.expected, "frames": trial.evidence.frames,
                                             "pair_frames": trial.evidence.pair_record(),
                                             "hypotheses": [asdict(h) for h in trial.evidence.hypotheses]}) + "\n")
            baseline = outcome_row(judge([p[0] for p in pairs], cfg))
            treatment = outcome_row(judge([p[1] for p in pairs], cfg))
            if baseline != fixed["conditions"][condition.name]["selected"]:
                raise ValueError(f"Original benchmark reproduction differs: {condition.name}")
            report["conditions"][condition.name] = {"original": baseline, "filtered": treatment}
            report["elapsed_seconds"] = time.monotonic() - started
            (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            stream.flush()
            print(condition.name, baseline["wrong"], treatment["wrong"], baseline["stop_rate"], treatment["stop_rate"], flush=True)
    report["wrong"] = sum(c["filtered"]["wrong"] for c in report["conditions"].values())
    report["original_stop_rate"] = float(np.mean([c["original"]["stop_rate"] for c in report["conditions"].values()]))
    report["filtered_stop_rate"] = float(np.mean([c["filtered"]["stop_rate"] for c in report["conditions"].values()]))
    report["status"] = "PASS" if report["wrong"] == 0 else "FAIL"
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("dataset", "confidence", "rule", "original", "output"):
        parser.add_argument(f"--{argument}", type=Path, required=True)
    args = parser.parse_args()
    run(args.dataset, args.confidence, args.rule, args.original, args.output)
