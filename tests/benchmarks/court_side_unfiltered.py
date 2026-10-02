"""Replay r30's 28 x 400 cases through the production unfiltered point component."""
from __future__ import annotations

import argparse
import gzip
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from numpy.typing import NDArray

from src.tasks.ball_refiner.inference import SequencePrediction
from src.tasks.ball_refiner.refiner_2d.distribution import BallGMM2D
from src.tasks.court_side import benchmark
from src.tasks.court_side.hypothesis import CourtSideConfig, collect_side_evidence
from src.tasks.court_side.scripts.benchmark_synthetic import CONDITIONS
from src.tennis_scene.pipeline.components.ball_points import BallPointsModule
from src.tennis_scene.pipeline.components.ball_refiner import BallRefiner2DOutput
from src.utils.checksum import dual_sha256

if TYPE_CHECKING or __package__ == "tests.benchmarks":
    from tests.benchmarks.court_side_correlated import ResidualBank, capture, record
else:
    from court_side_correlated import ResidualBank, capture, record


def production_points(distribution: BallGMM2D, size: tuple[int, int], camera_ids: tuple[str, ...]) -> np.ndarray:
    """Run each camera's complete GMM through the actual point component."""
    frames: NDArray[np.int64] = np.arange(distribution.means.shape[1], dtype=np.int64)
    zeros = np.zeros_like(frames)
    points: list[NDArray[np.float32]] = []
    for view, camera in enumerate(camera_ids):
        gmm = BallGMM2D(distribution.means[view:view + 1], distribution.scale_tril[view:view + 1],
                        distribution.mixture_logits[view:view + 1], distribution.presence_logits[view:view + 1])
        source = BallRefiner2DOutput(camera, size, frames, frames, "1/30", (frames / 30).astype(np.float32),
                                    SequencePrediction(gmm, frames, zeros, 1), frames, zeros, 1, "uncalibrated")
        points.append(BallPointsModule(distribution_version=1).process(source).uv_px)
    result: NDArray[np.float32] = np.stack(points)
    return cast(np.ndarray, result)


def run(dataset: Path, bank_directory: Path, original: Path, previous: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    fixed, prior = json.loads(original.read_text()), json.loads(previous.read_text())
    cfg = CourtSideConfig(**{**fixed["selected"], "height_range_m": tuple(fixed["selected"]["height_range_m"])})
    if asdict(cfg) != {**prior["thresholds"], "height_range_m": tuple(prior["thresholds"]["height_range_m"])} or cfg.min_margin != .15:
        raise ValueError("Original geometry/side thresholds changed")
    bank = ResidualBank(bank_directory)
    names = sorted((dataset / "test.txt").read_text().split())[600:1000]
    if len(names) != 400 or len(CONDITIONS) != 28 or names != prior["scenes"]:
        raise ValueError("Require the original 28 x 400 bench")
    inputs = (bank_directory / "bank.npz", bank_directory / "calibration.json", original, dataset / "test.txt")
    hashes = {str(path): dual_sha256(path) for path in inputs}
    for path, digest in hashes.items():
        if prior["input_sha256"].get(path) != digest:
            raise ValueError(f"Original input hash changed: {path}")
    scenes = [benchmark.load_blcs_scene(dataset / "scenes" / name) for name in names]
    output.mkdir(parents=True)
    rng, joint_rng = np.random.default_rng(1), np.random.default_rng(30001)
    report: dict[str, Any] = {"schema": "i935.unfiltered_production_safety.v1", "status": "running",
        "thresholds": asdict(cfg), "seed": 1, "residual_seed": 30001, "scenes": names,
        "input_sha256": {**hashes, str(previous): dual_sha256(previous)}, "conditions": {}, "wrong_cases": [],
        "pass_rule": "zero unfiltered wrong decisions across all 11200; report stop rates"}
    started = time.monotonic()
    with gzip.open(output / "evidence.jsonl.gz", "wt") as stream:
        for condition in CONDITIONS:
            trials: dict[str, list[Any]] = {"original": [], "joint_unfiltered": []}
            for scene in scenes:
                base, obs = capture(scene, condition, cfg, rng, score=True)
                rows = bank.sample(obs["uv"].shape[1], len(obs["cameras"]), joint_rng)
                distribution = bank.transport(obs["uv"], scene.image_size, rows)
                point = production_points(distribution, scene.image_size, tuple(camera.camera_id for camera in obs["cameras"]))
                evidence = collect_side_evidence(obs["cameras"], obs["reference"], point, obs["visible"], obs["config"])
                trial = replace(base, evidence=evidence)
                for label, value in (("original", base), ("joint_unfiltered", trial)):
                    trials[label].append(value)
                    stream.write(json.dumps(record(value, label)) + "\n")
                if benchmark.judge([trial], cfg).wrong:
                    report["wrong_cases"].append(record(trial, "joint_unfiltered"))
            summary = {label: benchmark.outcome_row(benchmark.judge(value, cfg)) for label, value in trials.items()}
            if summary["original"] != fixed["conditions"][condition.name]["selected"]:
                raise ValueError(f"Original condition mismatch: {condition.name}")
            if summary["joint_unfiltered"] != prior["conditions"][condition.name]["joint_unfiltered"]:
                raise ValueError(f"Unfiltered r30 condition mismatch: {condition.name}")
            report["conditions"][condition.name] = summary
            report["elapsed_seconds"] = time.monotonic() - started
            (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            stream.flush()
            print(condition.name, {k: (v["wrong"], v["stop_rate"]) for k, v in summary.items()}, flush=True)
    report["totals"] = {label: {"wrong": sum(c[label]["wrong"] for c in report["conditions"].values()),
        "stop_rate": float(np.mean([c[label]["stop_rate"] for c in report["conditions"].values()]))}
        for label in ("original", "joint_unfiltered")}
    report["status"] = "PASS" if report["totals"]["joint_unfiltered"]["wrong"] == 0 else "FAIL"
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("dataset", "bank", "original", "previous", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.dataset, args.bank, args.original, args.previous, args.output)
