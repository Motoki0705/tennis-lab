"""CPU-only collection of the immutable run25 source-video check, before code edits."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

JOB = "1790774709010605164_1159028_i935-e9-anchored-source3cam-r25-20260930"
QUEUE = Path("/home/kamimura/projects/tennis-lab/.training_queue")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def collect(plan_path: Path, output: Path) -> None:
    plan = read(plan_path)
    root = Path(plan["report"])
    if read(root / "plan.json") != plan:
        raise ValueError("The executed plan differs from the frozen plan")
    for name, expected in plan["input_sha256"].items():
        if sha256(Path(name)) != expected:
            raise ValueError(f"Frozen input changed before collection: {name}")
    inventory = {str(p.relative_to(root)): {"sha256": sha256(p), "bytes": p.stat().st_size}
                 for p in sorted(root.rglob("*")) if p.is_file()}
    resources = read(root / "resource_usage.json")
    comparison = read(root / "comparison.json")
    expected_phases = [(row["camera"], phase) for row in plan["cameras"] for phase in ("execute", "load")]
    if [(p["camera"], p["phase"]) for p in resources["phases"]] != expected_phases:
        raise ValueError("Not all six execution/load phases completed")
    if resources["seconds"] > plan["timeout_seconds"]:
        raise ValueError("The check exceeded its predeclared runtime limit")
    cameras = []
    verified_arrays = 0
    for row in plan["cameras"]:
        camera = row["camera"]
        directory = root / camera
        receipts = [read(directory / f"{phase}.json") for phase in ("execute", "load")]
        for receipt, status in zip(receipts, ("executed", "loaded"), strict=True):
            if receipt["status"] != "complete" or receipt["components"] != {
                f"ball_detection/{camera}": status, f"ball_refiner_2d/{camera}": status,
            }:
                raise ValueError(f"Incomplete phase: {camera}")
            if receipt["source"]["videos"][0]["num_frames"] != 270:
                raise ValueError("Incomplete source clip")
        if receipts[0]["source"] != receipts[1]["source"] or receipts[0]["references"] != receipts[1]["references"]:
            raise ValueError("Fresh load changed source or artifact references")
        with np.load(directory / "execute.npz", allow_pickle=False) as execute, np.load(
            directory / "load.npz", allow_pickle=False,
        ) as loaded:
            if set(execute.files) != set(loaded.files):
                raise ValueError("Snapshot field sets differ")
            for key in execute.files:
                a, b = execute[key], loaded[key]
                if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
                    raise ValueError(f"Not bit-identical: {camera}/{key}")
            np.testing.assert_array_equal(execute["frame_indices"], np.arange(270))
            if execute["means"].shape != (270, 4, 2):
                raise ValueError("Incomplete GMM output")
            fields = sorted(execute.files)
        for reference in receipts[0]["references"].values():
            path = directory / "store" / reference["path"]
            if sha256(path) != reference["sha256"]:
                raise ValueError(f"Artifact manifest checksum mismatch: {path}")
            artifact = read(path)
            for name, record in artifact["arrays"].items():
                array_path = path.parent / name
                if sha256(array_path) != record["sha256"]:
                    raise ValueError(f"Artifact array checksum mismatch: {array_path}")
                array = np.load(array_path, allow_pickle=False, mmap_mode="r")
                if list(array.shape) != record["shape"] or str(array.dtype) != record["dtype"]:
                    raise ValueError(f"Artifact array schema mismatch: {array_path}")
                verified_arrays += 1
        compared = next(c for c in comparison["cameras"] if c["camera"] == camera)
        if compared["compared_frames"] != 270 or not compared["source_load_arrays_bit_equal"]:
            raise ValueError("Incomplete saved comparison")
        cameras.append({"camera": camera, "frames": 270, "bit_identical_fields": fields,
                        "source_and_references_equal": True, "strict_diagnostic_passed": compared["passed"]})
    output.mkdir(parents=True, exist_ok=False)
    for name in ("resource_usage.json", "comparison.json", "preflight.json"):
        shutil.copyfile(root / name, output / name)
    for row in plan["cameras"]:
        target = output / row["camera"]
        target.mkdir()
        for phase in ("execute", "load"):
            for suffix in ("json", "log", "npz"):
                shutil.copyfile(root / row["camera"] / f"{phase}.{suffix}", target / f"{phase}.{suffix}")
    shutil.copyfile(QUEUE / "logs" / f"{JOB}.log", output / "queue.log")
    shutil.copyfile(QUEUE / "state" / f"{JOB}.state", output / "queue.state")
    shutil.copytree(QUEUE / "repro" / JOB, output / "queue_repro")
    worker = [line for line in (QUEUE / "worker.log").read_text(errors="replace").splitlines() if JOB in line]
    (output / "worker_excerpt.log").write_text("\n".join(worker) + "\n")
    write(output / "output_sha256.json", inventory)
    write(output / "collection.json", {
        "collected_at": datetime.now().astimezone().isoformat(), "queue_job": JOB,
        "queue_exit_code": 1, "check_commit": read(output / "queue_repro/run.json")["commit"],
        "input_files_verified": len(plan["input_sha256"]), "artifact_arrays_verified": verified_arrays,
        "plan_sha256": sha256(plan_path), "source_output_root": str(root),
        "final_output_files": len(inventory), "final_output_bytes": sum(v["bytes"] for v in inventory.values()),
        "cameras": cameras, "resources": resources,
        "failure_location": "check_pipeline.py:266, after all six child phases, three comparisons and final verify_inputs",
        "load_call_guards": ["InferenceBundle.load_model", "RefinerEvidenceModule.load", "BallRefiner2DInputAssembler.assemble"],
        "load_evidence": "Each phase was a new subprocess; guarded calls raise AssertionError on the load side.",
        "b_gate": "Execution and fresh-process load passed; same-frame GT accuracy still to evaluate. Strict tolerances are diagnostic only.",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    collect(args.plan, args.output)
