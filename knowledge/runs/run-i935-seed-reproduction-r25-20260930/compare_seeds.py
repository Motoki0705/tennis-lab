"""Apply the run-24 gate and report fixed-scale diagnostics without fitting."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from src.tasks.ball_refiner.evaluation.cached_comparison import METRICS
from src.tasks.ball_refiner.evaluation.covariance_calibration import (
    _check_hashes,
    _load_saved,
)
from src.tasks.ball_refiner.evaluation.paired_metrics import gmm_rows, summarize
from src.tasks.ball_refiner.refiner_2d.calibration import CovarianceCalibration
from src.tasks.ball_refiner.training.configuration import PilotConfig
from src.tennis_scene.pipeline.artifacts import write_json_atomic
from src.utils.checksum import dual_sha256

LEVELS = (0.5, 0.9, 0.95)
COLUMNS = ("median_error_px", "p90_error_px", "p95_error_px", "mean_nll_px",
           "coverage_0.5", "coverage_0.9", "coverage_0.95",
           "area_px2_0.5", "area_px2_0.9", "area_px2_0.95")


def gate(seeds: dict[str, Any], detector: dict[str, Any], absolute: dict[str, Any]) -> dict[str, Any]:
    """Ten strict comparisons; one failure blocks reproduction, missing is unknown."""
    comparisons = []
    for seed in ("43", "44"):
        for condition, metric, reference in (
            ("observed", "median_error_px", detector),
            ("observed", "p90_error_px", detector),
            ("observed", "p95_error_px", detector),
            ("observed", "mean_nll_px", absolute),
            ("evidence_gap", "mean_nll_px", absolute),
        ):
            key = f"meiji/{condition}/observed"
            row = seeds.get(seed, {}).get(key, {})
            ref = reference.get(key, {})
            value, bound = row.get(metric), ref.get(metric)
            denominator = "error_px_frames" if "error_px" in metric else "nll_px_frames"
            complete = (row.get("frames", 0) > 0 and row.get("frames") == ref.get("frames")
                        and row.get(denominator) == row.get("frames")
                        and ref.get(denominator) == ref.get("frames"))
            finite = all(isinstance(v, (float, int)) and math.isfinite(v) for v in (value, bound))
            status = ("pass" if value < bound else "fail") if complete and finite else "unconfirmed"
            comparisons.append({"seed": int(seed), "condition": condition, "metric": metric,
                                "value": value, "reference": bound, "frames": row.get("frames"),
                                "status": status})
    statuses = {r["status"] for r in comparisons}
    return {"status": "unconfirmed" if "unconfirmed" in statuses else "fail" if "fail" in statuses else "pass",
            "passed": sum(r["status"] == "pass" for r in comparisons), "total": 10,
            "comparisons": comparisons}


def summary(parts: list[dict[str, Any]]) -> dict[str, Any]:
    for row in parts:
        for key in METRICS:
            if not np.isfinite(row[key]).all():
                raise ValueError(f"Observed scored values must be finite: {key}")
    result: dict[str, Any] = summarize(parts, LEVELS)
    errors = np.concatenate([row["error_px"] for row in parts])
    result["p90_error_px"] = float(np.quantile(errors, 0.9)) if len(errors) else None
    return result


def spread_table(seeds: dict[str, Any]) -> list[dict[str, Any]]:
    if set(seeds) != {"42", "43", "44"} or len({tuple(sorted(v)) for v in seeds.values()}) != 1:
        raise ValueError("All three seeds must cover exactly the same groups")
    result = []
    for key in sorted(seeds["42"]):
        frames = {seeds[s][key]["frames"] for s in seeds}
        if len(frames) != 1:
            raise ValueError(f"Unequal denominators: {key}")
        for metric in COLUMNS:
            values = [seeds[s][key][metric] for s in ("42", "43", "44")]
            finite = all(v is not None and math.isfinite(v) for v in values)
            result.append({"group": key, "metric": metric, "frames": seeds["42"][key]["frames"],
                           **dict(zip(("seed42", "seed43", "seed44"), values, strict=True)),
                           "min": min(values) if finite else None, "max": max(values) if finite else None,
                           "spread": max(values) - min(values) if finite else None})
    return result


def run(plan_path: Path, output: Path) -> None:
    started = time.monotonic()
    plan = json.loads(plan_path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    hashes = {str(plan_path): dual_sha256(plan_path), **plan["input_sha256"]}
    _check_hashes(hashes)
    artifact = json.loads(Path(plan["calibration_artifact"]).read_text())
    scale = artifact["covariance_multiplier"]
    if scale != 1.8125148752087792:
        raise ValueError("The predeclared covariance multiplier changed")
    raw, fixed, selections = {}, {}, {}
    reference_axes: dict[tuple[str, str], dict[str, Any]] = {}
    detector: dict[str, Any] = {}
    absolute: dict[str, Any] = {}
    for seed, source_text in plan["predictions"].items():
        source = Path(source_text)
        manifest = json.loads((source / "manifest.json").read_text())
        if json.loads((source / "run_state.json").read_text())["status"] != "complete":
            raise ValueError(f"Incomplete evaluation: {seed}")
        expected_seed = 42 if seed == "absolute" else int(seed)
        if manifest["settings"] != dict(chunk_size=32, levels=list(LEVELS), samples=2048, seed=1729, uniform_weight=1e-6):
            raise ValueError("HDR/evaluation settings changed")
        training = Path(manifest["training_run"])
        config = PilotConfig.from_config(OmegaConf.load(training / "config.yaml"))
        if config.seed != expected_seed or config.partition_seed != 42:
            raise ValueError("Training or partition seed changed")
        # r21 manifests predate these metadata fields; the hashed training config
        # is authoritative for every run, and r23/r24 must agree with it.
        if seed in ("43", "44") and (manifest["training_seed"] != config.seed
                                      or manifest["partition_seed"] != config.partition_seed):
            raise ValueError("Evaluation seed metadata differs from training")
        best = json.loads((training / "best.json").read_text())
        curve = [json.loads(line) for line in (training / "learning_curve.jsonl").read_text().splitlines()]
        selected = min(curve, key=lambda row: (row["selection_nll_uv"], row["epoch"]))
        if selected["epoch"] != best["epoch"] or selected["selection_nll_uv"] != best["selection_nll_uv"]:
            raise ValueError("Existing checkpoint does not match the frozen selection rule")
        selections[seed] = best
        additions = {**manifest["input_sha256"],
                     str(training / "learning_curve.jsonl"): dual_sha256(training / "learning_curve.jsonl"),
                     **{str(source / name): dual_sha256(source / name) for name in ("manifest.json", "metrics.json", "run_state.json")},
                     **{str(source / a["path"]): a["sha256"] for a in manifest["artifacts"]}}
        _check_hashes(additions)
        hashes.update(additions)
        if hashes[str(training / best["checkpoint"])] != best["checkpoint_sha256"]:
            raise ValueError("Selected checkpoint differs from evaluation")
        saved = _load_saved(source, manifest, config)
        reported = json.loads((source / "metrics.json").read_text())
        before: dict[str, list[dict[str, Any]]] = defaultdict(list)
        after: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for index, item in enumerate(saved):
            axis = {k: item.arrays[k] for k in ("frame_index", "pts", "target_uv", "target_reason", "gap_mask")}
            axis_key = (item.record.clip_id, item.condition)
            if seed == "42":
                reference_axes[axis_key] = axis
            else:
                for key, value in axis.items():
                    np.testing.assert_array_equal(value, reference_axes[axis_key][key])
            for group in item.groups:
                before[f"{group}/{item.condition}/observed"].append({k: item.arrays[k][item.scored] for k in METRICS})
            if seed != "absolute":
                adjusted = CovarianceCalibration(scale, best["checkpoint_sha256"]).apply(item.distribution)
                for name in ("means", "mixture_logits", "presence_logits"):
                    if not torch.equal(getattr(adjusted, name), getattr(item.distribution, name)):
                        raise ValueError(f"Fixed covariance scaling changed {name}")
                values = gmm_rows(adjusted, item.target, (item.record.source_width, item.record.source_height),
                                  levels=LEVELS, samples=2048, seed=1729, chunk_size=32,
                                  clip_id=item.record.clip_id, condition=item.condition, device=torch.device("cpu"))
                for group in item.groups:
                    after[f"{group}/{item.condition}/observed"].append({k: values[k][item.scored] for k in METRICS})
            if index % 20 == 0:
                print(json.dumps({"seed": seed, "clips_done": index + 1, "seconds": time.monotonic() - started}), flush=True)
        current = {key: summary(parts) for key, parts in sorted(before.items())}
        for key, row in current.items():
            for metric in (*COLUMNS, "frames", "error_px_frames", "nll_px_frames"):
                np.testing.assert_allclose(row[metric], reported[f"variant/{key}"][metric], rtol=0, atol=1e-12)
        if seed == "absolute":
            absolute = current
        else:
            raw[seed] = current
            fixed[seed] = {key: summary(parts) for key, parts in sorted(after.items())}
            candidate_detector = {key: reported[f"new_detector/{key}"] for key in current}
            if detector and candidate_detector != detector:
                raise ValueError("Detector baseline differs between seeds")
            detector = candidate_detector
            write_json_atomic(output / f"seed{seed}.json", {"raw": raw[seed], "fixed": fixed[seed]})
    result = gate(raw, detector, absolute)
    write_json_atomic(output / "gate.json", result)
    for name, values in (("raw", raw), ("fixed", fixed)):
        rows = spread_table(values)
        with (output / f"{name}-spread.csv").open("x") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    _check_hashes(hashes)
    write_json_atomic(output / "manifest.json", {"input_sha256": hashes, "selections": selections,
                                               "scale": scale, "settings": manifest["settings"],
                                               "scope": "Fixed seed42 full-fit scale on all three seeds; no refit, no OOF, no test",
                                               "criterion": plan["criterion"]})
    write_json_atomic(output / "run_state.json", {"status": "complete", "seconds": time.monotonic() - started,
                                                "gate": result["status"], "cuda_used": False})
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args.plan, args.output)
