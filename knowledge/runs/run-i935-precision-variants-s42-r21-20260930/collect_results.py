"""Collect the retry without inference; verify checkpoints, frames and reductions."""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_refiner.data.gaps import fixed_gap_mask
from src.tasks.ball_refiner.data.targets import project_store_targets
from src.tasks.ball_refiner.evaluation.cached_comparison import METRICS
from src.tasks.ball_refiner.evaluation.detector_selection import validation_clips
from src.tasks.ball_refiner.evaluation.paired_metrics import Array, strata, summarize
from src.utils.checksum import dual_sha256

BUNDLE = Path(__file__).resolve().parent
ROOT = Path("/home/kamimura/projects/tennis-lab")
JOB = "1790739175028391561_2386039_i935-precision-variants-s42-r21-20260930"


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def write(name: str, value: Any) -> None:
    (BUNDLE / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def copy_metadata(source: Path, destination: Path) -> None:
    destination.mkdir(exist_ok=True)
    for path in source.iterdir():
        if path.is_file() and path.suffix in (".json", ".jsonl", ".yaml", ".txt", ".sh", ".patch"):
            shutil.copy2(path, destination / path.name)


def main() -> None:
    torch.set_num_threads(2)
    plan = read(BUNDLE / "plan.json")
    queue = ROOT / ".training_queue"
    state = (queue / "state" / f"{JOB}.state").read_text()
    assert "state=done\n" in state and (queue / "done" / f"{JOB}.job").is_file()
    worker = [x for x in (queue / "worker.log").read_text().splitlines() if JOB in x]
    assert any(" done: " in x for x in worker)
    report = Path(plan["report"])
    resource = read(report / "resource_usage.json")
    assert resource["status"] == "complete" and resource["device_monitor"]["failure"] is None
    assert read(report / "plan.json") == plan
    # Run before the architecture fix; record the exact historical inputs.
    hashes = dict(plan["input_sha256"])
    baseline = Path(plan["baseline"])
    baseline_data = read(baseline / "data_manifest.json")
    reference = Path(plan["reference"])
    reference_manifest = read(reference / "manifest.json")
    refs = {(a["clip_id"], a["method"], a["condition"]): a for a in reference_manifest["artifacts"]}
    store = BallFrameStore(Path(reference_manifest["recipe"]["store"]))
    records = {r.clip_id: r for r in validation_clips(store)}
    assert len(records) == 70
    targets = {k: project_store_targets(store, r) for k, r in records.items()}
    artifacts, checkpoint_rows, summaries = [], [], {}
    reference_paths = set()
    combined = {}
    for variant in plan["variants"]:
        name = variant["name"]
        cfg = OmegaConf.load(variant["training_config"])
        training = ROOT / "outputs" / cfg.run.output_dir
        curve = [json.loads(x) for x in (training / "learning_curve.jsonl").read_text().splitlines()]
        epochs = int(cfg.training.epochs)
        assert epochs in (12, 48)
        assert [r["epoch"] for r in curve] == list(range(epochs))
        assert [r["step"] for r in curve] == list(range(250, epochs * 250 + 1, 250))
        assert len(list(training.glob("epoch-*.pt"))) == epochs
        assert read(training / "data_manifest.json") == baseline_data
        data_hash = dual_sha256(training / "data_manifest.json")
        checkpoints = []
        for row in curve:
            path = training / f"epoch-{row['epoch']:03d}.pt"
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            assert ckpt["epoch"] == row["epoch"] and ckpt["step"] == row["step"]
            assert ckpt["data_manifest_sha256"] == data_hash
            assert ckpt["selection_nll_uv"] == row["selection_nll_uv"]
            assert abs(row["selection_nll_uv"] - .5 * (row["observed"]["position_nll_uv"] + row["evidence_gap"]["position_nll_uv"])) < 1e-12
            assert all(torch.isfinite(x).all() for x in ckpt["state_dict"].values())
            checkpoints.append({"variant": name, "path": str(path), "sha256": dual_sha256(path),
                                "bytes": path.stat().st_size, "epoch": row["epoch"], "step": row["step"],
                                "selection_nll_uv": row["selection_nll_uv"]})
        best = read(training / "best.json")
        selected = min(curve, key=lambda r: r["selection_nll_uv"])
        assert best["epoch"] == selected["epoch"]
        assert best["checkpoint_sha256"] == checkpoints[best["epoch"]]["sha256"]
        assert read(training / "run_state.json") == {"best_epoch": best["epoch"], "selection_nll_uv": best["selection_nll_uv"],
                                                     "status": "complete", "steps": int(cfg.training.max_steps)}
        checkpoint_rows.extend(checkpoints)
        evaluation = Path(variant["evaluation_output"])
        manifest = read(evaluation / "manifest.json")
        hashes.update(manifest["input_sha256"])
        assert read(evaluation / "run_state.json") == {"clips": 70, "files": 140, "status": "complete"}
        assert read(evaluation / "progress.json")["artifacts"] == manifest["artifacts"]
        assert len(list(evaluation.glob("*.npz"))) == len(manifest["artifacts"]) == 140
        assert manifest["partition"] == reference_manifest["recipe"]["partition"]
        groups: dict[str, list[dict[str, Array]]] = defaultdict(list)
        seen = set()
        for entry in manifest["artifacts"]:
            identity = (entry["clip_id"], entry["condition"])
            assert identity not in seen
            seen.add(identity)
            record, target = records[entry["clip_id"]], targets[entry["clip_id"]]
            assert entry["frames"] == record.frame_count
            path = evaluation / entry["path"]
            assert dual_sha256(path) == entry["sha256"]
            artifacts.append({**entry, "variant": name, "path": str(path), "bytes": path.stat().st_size})
            with np.load(path, allow_pickle=False) as saved:
                arrays = dict(saved)
            gap = fixed_gap_mask(record.frame_count, clip_id=record.clip_id, block_length=int(cfg.data.window_length),
                                 lengths=tuple(cfg.training.gap_lengths), seed=int(cfg.data.partition_seed))
            if entry["condition"] == "observed":
                gap[:] = False
            fixed = {"frame_index": target.frame_index, "pts": target.pts, "target_reason": target.reason,
                     "target_uv": target.uv, "presence": target.presence, "presence_valid": target.presence_valid, "gap_mask": gap}
            for key, value in fixed.items():
                np.testing.assert_equal(arrays[key], value)
            located = np.isin(target.reason, [0, 5, 6])
            for key in ("nll_uv", "nll_px", "coverage", "area_px2"):
                assert np.isfinite(arrays[key][located]).all() and np.isnan(arrays[key][~located]).all()
            for key in ("means", "scale_tril", "mixture_logits", "presence_logits"):
                assert np.isfinite(arrays[key]).all()
            assert (arrays["scale_tril"][:, :, (0, 1), (0, 1)] > 0).all()
            assert np.isfinite(arrays["presence_nll"][target.presence_valid]).all()
            assert np.isnan(arrays["presence_nll"][~target.presence_valid]).all()
            methods = {"variant": arrays}
            for method in ("new_refiner", "new_detector"):
                ref = refs[(record.clip_id, method, entry["condition"])]
                rp = reference / ref["path"]
                hashes[str(rp)] = ref["sha256"]
                reference_paths.add(str(rp))
                with np.load(rp, allow_pickle=False) as saved:
                    methods[method] = dict(saved)
                for key, value in fixed.items():
                    np.testing.assert_equal(methods[method][key], value)
            labels = [record.source]
            if record.camera_id is not None:
                labels.append(f"{record.source}/{record.camera_id}")
            if record.source == "meiji":
                half = "selection" if record.clip_id in manifest["partition"]["selection"] else "calibration"
                labels.extend([f"meiji/{half}", f"meiji/{half}/{record.camera_id}"])
            for method, values in methods.items():
                for label, chosen in strata(target, gap, entry["condition"]).items():
                    for group in labels:
                        groups[f"{method}/{group}/{entry['condition']}/{label}"].append({k: values[k][chosen] for k in METRICS})
        assert seen == {(c, cond) for c in records for cond in ("observed", "evidence_gap")}
        metrics = {}
        for key, rows in sorted(groups.items()):
            reduced = summarize(rows, tuple(manifest["settings"]["levels"]))
            errors = np.concatenate([r["error_px"] for r in rows])
            errors = errors[np.isfinite(errors)]
            reduced["p90_error_px"] = float(np.quantile(errors, .9)) if len(errors) else None
            metrics[key] = reduced
        assert metrics == read(evaluation / "metrics.json")
        combined[name] = {k.removeprefix("variant/"): v for k, v in metrics.items() if k.startswith("variant/")}
        for method, alias in (("new_refiner", "r18_refiner"), ("new_detector", "e9_detector")):
            value = {k.removeprefix(method + "/"): v for k, v in metrics.items() if k.startswith(method + "/")}
            assert alias not in combined or combined[alias] == value
            combined[alias] = value
        summaries[name] = {"checkpoints": epochs, "steps": int(cfg.training.max_steps), "best": best,
                           "files": len(seen), "metric_groups_exactly_reproduced": len(metrics)}
        copy_metadata(training, BUNDLE / f"training-{name}")
        copy_metadata(evaluation, BUNDLE / f"evaluation-{name}")
    assert len(checkpoint_rows) == 108 and len(artifacts) == 420 and len(reference_paths) == 280
    for path, digest in hashes.items():
        assert dual_sha256(Path(path)) == digest, path
    assert combined == read(report / "comparison.json")
    copy_metadata(queue / "repro" / JOB, BUNDLE / "queue_repro")
    shutil.copy2(queue / "logs" / f"{JOB}.log", BUNDLE / "queue.log")
    shutil.copy2(queue / "done" / f"{JOB}.job", BUNDLE / "queue.job")
    (BUNDLE / "queue.state").write_text(state)
    (BUNDLE / "worker_excerpt.log").write_text("\n".join(worker) + "\n")
    for name in ("resource_usage.json", "comparison.json", "comparison.md"):
        shutil.copy2(report / name, BUNDLE / name)
    write("artifact_hashes.json", artifacts + checkpoint_rows)
    write("verified_input_sha256.json", hashes)
    write("collection.json", {"status": "verified", "queue_job": JOB, "queue_state": "done", "exit_code": 0,
                              "exit_evidence": "worker done branch requires rc=0", "variants": summaries,
                              "checkpoints": len(checkpoint_rows), "variant_npz": len(artifacts), "reference_npz": len(reference_paths),
                              "clips": len(records), "frames_per_method_condition": sum(r.frame_count for r in records.values()),
                              "input_files": len(hashes), "all_reductions_exact": True,
                              "disk_bytes_at_job_end": sum(resource["output_bytes"].values()),
                              "limitations": ["HDR Monte Carlo not rerun; saved arrays/hash/masks and all reductions verified.",
                                              "Media hashes inherited from store; evaluated shard/input files checked.",
                                              "Single seed, val only; no test or context ablation."]})
    print(json.dumps({"status": "verified", "checkpoints": 108, "npz": 420, "variants": summaries}))


if __name__ == "__main__":
    main()

