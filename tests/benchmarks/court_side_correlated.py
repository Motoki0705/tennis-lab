"""Pre-registered joint anchored residual/confidence transport onto #932 stresses."""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

import src.tasks.court_side.benchmark as benchmark
from src.tasks.ball_refiner.refiner_2d.distribution import BallGMM2D
from src.tasks.court_side.hypothesis import (
    BallSideEvidence,
    CourtSideConfig,
    collect_side_evidence,
)
from src.tasks.court_side.scripts.benchmark_synthetic import CONDITIONS
from src.utils.checksum import dual_sha256

if TYPE_CHECKING or __package__ == "tests.benchmarks":
    from tests.benchmarks.legacy_ball_confidence import (
        PointConfidenceRule,
        point_confidence,
    )
else:
    from legacy_ball_confidence import PointConfidenceRule, point_confidence

BANK_SHA = "0697fe921daf79c7960d616ed0437efecd3b7195f3f8a7dd6188048dd8ecc858"
CALIBRATION_SHA = "2c9cd7ba9a1d63addeaecb808441fbae0447ad21e4a3a279ed1694fcbf67ae01"


def capture(scene: Any, condition: Any, config: CourtSideConfig, rng: np.random.Generator,
            *, score: bool) -> tuple[Any, dict[str, Any]]:
    """Observe the unmodified benchmark's exact inputs without forking its generator."""
    observed: dict[str, Any] = {}

    def collect(cameras: Any, reference: str, uv: Any, visible: Any, scaled: Any) -> BallSideEvidence:
        observed.update(cameras=cameras, reference=reference, uv=uv.copy(), visible=visible.copy(), config=scaled)
        if score:
            return collect_side_evidence(cameras, reference, uv, visible, scaled)
        return BallSideEvidence(tuple(c.camera_id for c in cameras), reference, 0, np.zeros((len(cameras), len(cameras)), np.int64), ())

    with patch.object(benchmark, "collect_side_evidence", collect):
        trial = benchmark.make_trial(scene, condition, config, rng)
    return trial, observed


class ResidualBank:
    def __init__(self, directory: Path) -> None:
        if dual_sha256(directory / "bank.npz") != BANK_SHA or dual_sha256(directory / "calibration.json") != CALIBRATION_SHA:
            raise ValueError("Require the pre-registered calibrated residual bank hashes")
        report = json.loads((directory / "calibration.json").read_text())
        with np.load(directory / "bank.npz", allow_pickle=False) as archive:
            self.arrays = {k: archive[k] for k in archive.files}
        eligible = {i for i, row in enumerate(report["source_predictions"])
                    if row["condition"] == "observed" and row["source"] == "meiji"
                    and "/video_000/" in row["clip_id"] and "/clip_000/" not in row["clip_id"]}
        allowed = np.isin(self.arrays["source_artifact"], sorted(eligible))
        self.runs: dict[int, list[np.ndarray]] = {}
        for camera in range(3):
            rows = np.flatnonzero(allowed & (self.arrays["camera_index"] == camera))
            cut = np.flatnonzero((np.diff(rows) != 1) | ~self.arrays["continues"][rows[1:]]
                | (np.diff(self.arrays["source_frame"][rows]) != 1)
                | (np.diff(self.arrays["source_artifact"][rows]) != 0)) + 1
            self.runs[camera] = [r for r in np.split(rows, cut) if len(r)]
            if not self.runs[camera]:
                raise ValueError("Every empirical camera needs observed residuals")

    def sample(self, frames: int, views: int, rng: np.random.Generator) -> np.ndarray:
        if frames < 1 or views not in {3, 4}:
            raise ValueError("Invalid synthetic camera/window dimensions")
        cameras = list(rng.permutation(3))
        if views == 4:
            cameras.append(int(rng.integers(3)))
        result: np.ndarray = np.empty((views, frames), np.int64)
        for view, camera in enumerate(cameras):
            runs = self.runs[camera]
            lengths = np.array([len(r) for r in runs], np.float64)
            cursor = 0
            while cursor < frames:
                run = runs[int(rng.choice(len(runs), p=lengths / lengths.sum()))]
                start = int(rng.integers(len(run)))
                block = run[start::2][:min(30, frames - cursor)]
                result[view, cursor:cursor + len(block)] = block
                cursor += len(block)
        return result

    def transport(self, uv: np.ndarray, size: tuple[int, int], rows: np.ndarray) -> BallGMM2D:
        if rows.shape != uv.shape[:-1]:
            raise ValueError("Bank row selection must match every camera/frame")
        a = self.arrays
        means = torch.from_numpy(np.clip(uv[..., None, :] / (np.array(size) - 1) + a["error_uv"][rows], 0, 1).astype(np.float32))
        tril = torch.from_numpy(a["scale_tril_uv"][rows])
        logits = torch.from_numpy(a["mixture_logits"][rows])
        presence = torch.from_numpy(a["presence_logits"][rows])
        return BallGMM2D(means=means, scale_tril=tril, mixture_logits=logits, presence_logits=presence)


def record(trial: Any, label: str) -> dict[str, Any]:
    return {"scene": trial.scene_id, "condition": trial.condition, "path": label, "expected": trial.expected,
            "cameras": trial.evidence.camera_ids, "frames": trial.evidence.frames,
            "pair_frames": trial.evidence.pair_record(), "hypotheses": [asdict(h) for h in trial.evidence.hypotheses]}


def run(dataset: Path, bank_directory: Path, rule_path: Path, original: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    fixed = json.loads(original.read_text())
    cfg = CourtSideConfig(**{**fixed["selected"], "height_range_m": tuple(fixed["selected"]["height_range_m"])})
    rule = PointConfidenceRule(**dict(OmegaConf.load(rule_path)))
    if (cfg.min_margin, cfg.min_frames, cfg.max_cost, cfg.min_support, cfg.min_motion_px) != (.15, 8, .8, .2, 5.) \
            or asdict(rule) != {"min_presence": .9, "max_area_px2": 30000.}:
        raise ValueError("Pre-registered fixed rules changed")
    bank = ResidualBank(bank_directory)
    names = sorted((dataset / "test.txt").read_text().split())[600:1000]
    if len(names) != 400 or len(CONDITIONS) != 28:
        raise ValueError("Require the original 28 x 400 bench")
    scenes = [benchmark.load_blcs_scene(dataset / "scenes" / n) for n in names]
    output.mkdir(parents=True)
    rng, joint_rng = np.random.default_rng(1), np.random.default_rng(30001)
    report: dict[str, Any] = {"schema": "i935.joint_confidence_safety.v1", "status": "running", "rule": asdict(rule),
        "thresholds": asdict(cfg), "conditions": {}, "seed": 1, "residual_seed": 30001, "scenes": names,
        "input_sha256": {str(p): dual_sha256(p) for p in (bank_directory / "bank.npz", bank_directory / "calibration.json", rule_path, original, dataset / "test.txt")},
        "pass_rule": "zero filtered wrong decisions across all 11200; report stop rates", "wrong_cases": []}
    errors: dict[str, list[np.ndarray]] = {"kept_total": [], "dropped_total": [], "kept_empirical": [], "dropped_empirical": []}
    started = time.monotonic()
    with gzip.open(output / "evidence.jsonl.gz", "wt") as stream:
        for condition in CONDITIONS:
            trials: dict[str, list[Any]] = {label: [] for label in ("original", "joint_unfiltered", "joint_filtered")}
            for scene in scenes:
                clean_rng = copy.deepcopy(rng)
                base, obs = capture(scene, condition, cfg, rng, score=True)
                _, clean = capture(scene, replace(condition, false_rate=0., pixel_sigma_px=0.), cfg, clean_rng, score=False)
                rows = bank.sample(obs["uv"].shape[1], len(obs["cameras"]), joint_rng)
                distribution = bank.transport(obs["uv"], scene.image_size, rows)
                point, presence, area = point_confidence(distribution, scene.image_size)
                mask = rule.rejection_codes(presence, area) == 0
                for label, visibility in (("joint_unfiltered", obs["visible"]), ("joint_filtered", obs["visible"] & mask)):
                    evidence = collect_side_evidence(obs["cameras"], obs["reference"], point.astype(np.float32), visibility, obs["config"])
                    trial = replace(base, evidence=evidence)
                    trials[label].append(trial)
                    stream.write(json.dumps(record(trial, label)) + "\n")
                    if label == "joint_filtered" and benchmark.judge([trial], cfg).wrong:
                        item = record(trial, label)
                        report["wrong_cases"].append(item)
                        np.savez_compressed(output / f"wrong-{condition.name}-{scene.scene_id}.npz", uv=point, visible=obs["visible"],
                            keep=mask, presence=presence, area=area, rows=rows, truth_uv=clean["uv"])
                trials["original"].append(base)
                stream.write(json.dumps(record(base, "original")) + "\n")
                # Total point error includes unchanged synthetic stresses. Empirical
                # transport delta excludes them and stays paired with its confidence.
                for key, error in (("total", np.linalg.norm(point - clean["uv"], axis=-1)),
                                   ("empirical", np.linalg.norm(point - obs["uv"], axis=-1))):
                    errors[f"kept_{key}"].append(error[obs["visible"] & mask].astype(np.float32))
                    errors[f"dropped_{key}"].append(error[obs["visible"] & ~mask].astype(np.float32))
            summary = {label: benchmark.outcome_row(benchmark.judge(value, cfg)) for label, value in trials.items()}
            if summary["original"] != fixed["conditions"][condition.name]["selected"]:
                raise ValueError(f"Original 28-condition reproduction mismatch: {condition.name}")
            report["conditions"][condition.name] = summary
            report["elapsed_seconds"] = time.monotonic() - started
            (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            stream.flush()
            print(condition.name, {k: (v["wrong"], v["stop_rate"]) for k, v in summary.items()}, flush=True)
    report["totals"] = {label: {"wrong": sum(c[label]["wrong"] for c in report["conditions"].values()),
        "stop_rate": float(np.mean([c[label]["stop_rate"] for c in report["conditions"].values()]))}
        for label in ("original", "joint_unfiltered", "joint_filtered")}
    report["point_error_px"] = {}
    for key, values in errors.items():
        array = np.concatenate(values)
        report["point_error_px"][key] = {"n": len(array), "median": float(np.median(array)), "p90": float(np.quantile(array, .9))}
    report["status"] = "PASS" if report["totals"]["joint_filtered"]["wrong"] == 0 else "FAIL"
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("dataset", "bank", "rule", "original", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.dataset, args.bank, args.rule, args.original, args.output)
