"""Apply the pre-fixed B gate to saved source/cache outputs; CPU, no inference or fit."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_refiner.data.targets import TargetReason, project_store_targets
from src.tasks.ball_refiner.deployment import load_inference_bundle
from src.tasks.ball_refiner.pipeline_options import select_ball_path
from src.tasks.ball_refiner.refiner_2d.distribution import BallGMM2D
from src.utils.checksum import dual_sha256

B_GATE = "https://github.com/Motoki0705/tennis-lab/issues/935#issuecomment-5910704846"
LIMITS = {"median_error_px": 0.5, "p90_error_px": 5.0, "mean_nll_px": 0.1}
RUNS = Path(__file__).resolve().parent.parent
PLAN = RUNS / "run-i935-source-check-retry-r25-20260930/plan.json"
COLLECTION = PLAN.parent / "collected-r26"


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def score(
    prediction: BallGMM2D, target_uv: np.ndarray, observed: np.ndarray, size_wh: tuple[int, int],
) -> dict[str, np.ndarray]:
    """Only observed GT; top-weight component error and full-mixture conditional NLL."""
    n = prediction.means.shape[1]
    if (prediction.means.shape[0] != 1 or target_uv.shape != (n, 2)
            or observed.shape != (n,) or observed.dtype != np.bool_ or not observed.any()
            or min(size_wh) <= 1 or not np.isfinite(target_uv[observed]).all()):
        raise ValueError("Require aligned observed targets and source dimensions")
    pred = BallGMM2D(**{f.name: getattr(prediction, f.name)[:, observed].double() for f in fields(BallGMM2D)})
    target = torch.from_numpy(target_uv[observed]).double()[None]
    scale = np.asarray(size_wh, np.float64) - 1
    means = pred.means[0, torch.arange(int(observed.sum())), pred.mixture_logits[0].argmax(-1)].numpy()
    return {"error_px": np.linalg.norm((means - target_uv[observed].astype(np.float64)) * scale, axis=-1),
            "nll_px": -pred.log_prob(target)[0].numpy() + np.log(scale).sum()}


def summarize(rows: list[dict[str, np.ndarray]]) -> dict[str, float | int]:
    if not rows or any(r["error_px"].ndim != 1 or r["error_px"].shape != r["nll_px"].shape for r in rows):
        raise ValueError("Require aligned one-dimensional errors and NLLs")
    error, nll = [np.concatenate([r[key] for r in rows]) for key in ("error_px", "nll_px")]
    if not len(error) or not np.isfinite(error).all() or not np.isfinite(nll).all() or (error < 0).any():
        raise ValueError("Cannot drop missing/nonfinite observed predictions from the gate")
    return {"frames": len(error), "median_error_px": float(np.median(error)),
            "p90_error_px": float(np.quantile(error, .9)), "mean_nll_px": float(nll.mean())}


def accuracy_gate(cached: list[dict[str, np.ndarray]], source: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    if len(cached) != len(source) or any(
        a["error_px"].shape != b["error_px"].shape for a, b in zip(cached, source, strict=True)
    ):
        raise ValueError("The two paths must score the same camera/frame population")
    before, after = summarize(cached), summarize(source)
    tests = {key: {"delta": after[key] - before[key], "maximum_worsening": limit,
                   "passed": after[key] - before[key] <= limit} for key, limit in LIMITS.items()}
    return {"cached": before, "source": after, "tests": tests, "passed": all(t["passed"] for t in tests.values())}


def main(output: Path) -> None:
    if not output.is_absolute() or output.exists():
        raise ValueError("Use a new absolute output directory")
    torch.set_num_threads(1)
    plan = read(PLAN)
    collected = read(COLLECTION / "collection.json")
    source_root = Path(plan["report"])
    hashes: dict[str, str] = {}

    def verify(path: Path, expected: str) -> None:
        if dual_sha256(path) != expected:
            raise ValueError(f"Saved input checksum mismatch: {path}")
        hashes[str(path)] = expected

    verify(PLAN, collected["plan_sha256"])
    for relative, record in read(COLLECTION / "output_sha256.json").items():
        verify(source_root / relative, record["sha256"])
    for key in ("bundle", "calibration_artifact"):
        paths = [Path(plan[key])] if key == "calibration_artifact" else list(Path(plan[key]).iterdir())
        for path in paths:
            verify(path, plan["input_sha256"][str(path)])
    calibration = select_ball_path(plan["ball_path"], load_inference_bundle(Path(plan["bundle"])),
                                   Path(plan["calibration_artifact"]))
    if calibration is None:
        raise ValueError("Require the pinned covariance artifact")
    fixed = calibration.load()
    cached_root = Path(plan["cameras"][0]["cached_prediction"]).parent
    manifest_path = cached_root / "manifest.json"
    verify(manifest_path, plan["input_sha256"][str(manifest_path)])
    manifest = read(manifest_path)
    reference_path = Path(manifest["reference"]) / "manifest.json"
    verify(reference_path, manifest["input_sha256"][str(reference_path)])
    store_path = Path(read(reference_path)["recipe"]["store"])
    for name in ("metadata.json", "index.npz"):
        verify(store_path / name, manifest["input_sha256"][str(store_path / name)])
    store = BallFrameStore(store_path)
    cameras, cached_rows, source_rows, label_rows = [], [], [], []
    for row in plan["cameras"]:
        camera = row["camera"]
        record = store.clip_by_id(f"{plan['clip_id']}/{camera}")
        if record.split != "val" or record.frame_count != 270:
            raise ValueError("Only the predeclared full validation clip is allowed")
        verify(Path(row["video"]), record.media_sha256)
        verify(Path(row["cached_prediction"]), row["cached_sha256"])
        with np.load(source_root / camera / "execute.npz", allow_pickle=False) as z:
            source = dict(z)
        with np.load(row["cached_prediction"], allow_pickle=False) as z:
            cached = dict(z)
        target = project_store_targets(store, record)
        np.testing.assert_array_equal(target.frame_index, np.arange(270))
        for key, value in {"target_uv": target.uv, "target_reason": target.reason, "frame_index": target.frame_index,
                           "pts": target.pts, "presence": target.presence, "presence_valid": target.presence_valid}.items():
            np.testing.assert_equal(cached[key], value)
        if cached["gap_mask"].any():
            raise ValueError("B compares ordinary evidence, not artificial gaps")
        for key, value in {"frame_indices": target.frame_index, "pts": target.pts,
                           "timestamps_seconds": target.timestamps_seconds, "time_base": record.time_base,
                           "source_size_wh": (record.source_width, record.source_height), "camera_id": camera}.items():
            np.testing.assert_equal(source[key], value)
        before, after = [BallGMM2D(**{f.name: torch.from_numpy(data[f.name][None].copy())
                                    for f in fields(BallGMM2D)}) for data in (cached, source)]
        before = fixed.apply(before)  # Same fixed scale as the source path; never applied twice there.
        size = (record.source_width, record.source_height)
        c, s = [score(p, target.uv, target.position_valid, size) for p in (before, after)]
        cached_rows.append(c)
        source_rows.append(s)
        cameras.append({"camera": camera, "total_frames": record.frame_count,
                        "label_counts": {reason.name: int((target.reason == reason).sum()) for reason in TargetReason},
                        **accuracy_gate([c], [s])})
        label_rows.append({"frame_index": target.frame_index, "pts": target.pts,
                           "target_uv": target.uv, "target_reason": target.reason})
    pooled = accuracy_gate(cached_rows, source_rows)
    expected = [(c, p) for c in ("cam0", "cam1", "cam2") for p in ("execute", "load")]
    resource = read(source_root / "resource_usage.json")
    execute_passed = ([(p["camera"], p["phase"]) for p in resource["phases"]] == expected
                      and resource["seconds"] <= plan["timeout_seconds"]
                      and [c["frames"] for c in collected["cameras"]] == [270, 270, 270])
    load_passed = all(c["source_and_references_equal"] and len(c["bit_identical_fields"]) == 17
                      for c in collected["cameras"]) and len(collected["cameras"]) == 3
    result = {"gate_source": B_GATE, "covariance_multiplier": fixed.covariance_multiplier,
              "accuracy_population": "all 654 observed GT frames, no presence/score/error filtering; frame-pooled quantiles",
              "nll": "conditional full GMM position density, natural log in source px, float64; excludes presence",
              "execution_passed": execute_passed, "fresh_process_load_passed": load_passed,
              "cameras": cameras, "pooled": pooled,
              "b_gate_passed": execute_passed and load_passed and pooled["passed"],
              "strict_diagnostic": read(source_root / "comparison.json"),
              "default_switch": "allowed" if execute_passed and load_passed and pooled["passed"] else "forbidden"}
    for name, expected_hash in hashes.items():
        if dual_sha256(Path(name)) != expected_hash:
            raise ValueError(f"Input changed during scoring: {name}")
    output.mkdir(parents=True)
    write(output / "gate.json", result)
    write(output / "input_sha256.json", hashes)
    with (output / "observed-frames.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["camera", "frame", "pts", "cached_error_px", "source_error_px", "cached_nll_px", "source_nll_px"])
        for camera, labels, cached, source in zip(cameras, label_rows, cached_rows, source_rows, strict=True):
            mask = labels["target_reason"] == TargetReason.OBSERVED
            for i, frame in enumerate(np.flatnonzero(mask)):
                writer.writerow([camera["camera"], frame, labels["pts"][frame], cached["error_px"][i], source["error_px"][i],
                                 cached["nll_px"][i], source["nll_px"][i]])
            np.savez_compressed(output / f"{camera['camera']}-labels.npz", **labels)
    lines = ["# 固定Bゲート", "", f"判定の正本: {B_GATE}", "", "| camera | observed n | median cache → source (Δ px) | p90 cache → source (Δ px) | NLL cache → source (Δ nat) |",
             "|---|---:|---:|---:|---:|"]
    for row in [*cameras, {"camera": "pooled", **pooled}]:
        metrics = [f"{row['cached'][k]:.6f} → {row['source'][k]:.6f} ({row['tests'][k]['delta']:+.6f})" for k in LIMITS]
        lines.append(f"| {row['camera']} | {row['cached']['frames']} | " + " | ".join(metrics) + " |")
    lines += ["", f"実行: {execute_passed}、fresh-load: {load_passed}、精度: {pooled['passed']}、B: {result['b_gate_passed']}。",
              "許容悪化 median ≤0.5 px、p90 ≤5 px、observed位置NLL ≤0.1 nat。判定はpooledのみ。",
              "", "## strict field診断（ゲートではない）", "", "| camera | field | atol | max abs | exceeded / elements |", "|---|---|---:|---:|---:|"]
    for row in result["strict_diagnostic"]["cameras"]:
        for name, val in row["fields"].items():
            lines.append(f"| {row['camera']} | {name} | {val['atol']:.8g} | {val['max_abs']:.10g} | {val['exceeded']} / {val['elements']} |")
    (output / "gate.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"b_gate_passed": result["b_gate_passed"], "pooled": pooled}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
