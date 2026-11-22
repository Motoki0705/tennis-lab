"""Promote bounded C1 evidence; retain full recorder spans in owned outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[3]
DESTINATION = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def promote(root: Path) -> None:
    registration = json.loads((root / "queue-registration.json").read_text())
    job = registration["registration"].strip().removeprefix("queued: ").removesuffix(".job")
    common = Path(subprocess.check_output(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=REPOSITORY, text=True).strip())
    queue = common.parent / ".training_queue"
    output_relative = Path("outputs") / root.relative_to((REPOSITORY / "outputs").resolve())
    if not (queue / "repro" / job).is_dir():
        raise FileNotFoundError(queue / "repro" / job)
    evidence = []
    rows = []
    for path in sorted((root / "recordings").glob("*.json")):
        record = json.loads(path.read_text())
        total = record["total_wall_seconds"]
        if not math.isclose(sum(record["stage_wall_seconds"].values()), total, abs_tol=1e-8):
            raise ValueError("Wall-time partition does not sum to the measured interval")
        if record["measurement_errors"]:
            raise ValueError("Resource sampling failed")
        events = record["events"]
        for event in events:
            if not 0 <= event["start_ns"] <= event["end_ns"] <= round(total * 1e9):
                raise ValueError("Event outside the measurement interval")
        row = {k: record[k] for k in ("metadata", "status", "error_type", "total_wall_seconds",
               "stage_wall_seconds", "stage_inclusive_seconds", "models", "batch_shapes", "resources")}
        row["stage_wall_percent"] = {k: v / total * 100 for k, v in record["stage_wall_seconds"].items()}
        row["event_count"] = len(events)
        row["recorder_json"] = {"path": str(output_relative / path.relative_to(root)), "sha256": sha(path), "bytes": path.stat().st_size}
        row["model_instances"] = {}
        row["inference_by_model"] = {}
        for event in events:
            if event["stage"] in {"model_load", "model_unload"}:
                instance_key = json.dumps({k: event["attributes"].get(k) for k in ("clip_id", "node", "model", "instance")}, sort_keys=True)
                actions = row["model_instances"].setdefault(instance_key, [])
                actions.append(event["stage"])
            if event["stage"] == "inference":
                attrs = event["attributes"]
                stats = row["inference_by_model"].setdefault(attrs.get("model", "unknown"), {"calls": 0, "inclusive_seconds": 0., "batch_items": 0, "tensor_leading_items": 0})
                stats["calls"] += 1
                stats["inclusive_seconds"] += (event["end_ns"] - event["start_ns"]) / 1e9
                stats["batch_items"] += attrs.get("batch_items", 0)
                stats["tensor_leading_items"] += attrs.get("batch_shape", [0])[0]
        write(DESTINATION / "lifecycle" / path.name, [e for e in events if e["stage"] in {"node", "model_load", "model_unload"}])
        evidence.append(row["recorder_json"])
        rows.append(row)
    if not rows:
        raise ValueError("No measurement records")
    write(DESTINATION / "metrics.json", rows)
    for name in ("stopped.json", "queue-registration.json"):
        path = root / name
        if path.exists():
            shutil.copyfile(path, DESTINATION / name)
    for directory in ("receipts", "outcomes"):
        for path in sorted((root / directory).glob("*.json")):
            destination = DESTINATION / directory / path.name
            destination.parent.mkdir(exist_ok=True)
            shutil.copyfile(path, destination)
    for name in ("run.json", "git_status.txt", "uncommitted.patch", "repro.sh"):
        source = queue / "repro" / job / name
        destination = DESTINATION / "queue" / name
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, destination)
    for path in sorted((root / "datasets").rglob("tennis_scene.failure.json")):
        destination = DESTINATION / "failures" / (path.parents[1].name + ".json")
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, destination)
    # Raw recorder JSONs stay in outputs; each promoted file has a separate hash.
    write(DESTINATION / "evidence-files.json", {"outputs": evidence, "promoted": [
        {"path": str(p.relative_to(DESTINATION)), "sha256": sha(p), "bytes": p.stat().st_size}
        for p in sorted(DESTINATION.rglob("*")) if p.is_file() and p.name != "evidence-files.json"]})
    files = [p for p in DESTINATION.rglob("*") if p.is_file()]
    if any(p.stat().st_size > 256 * 1024 for p in files) or sum(p.stat().st_size for p in files) > 2 * 1024**2:
        raise ValueError("Promoted evidence exceeds knowledge storage bounds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    if not root.is_relative_to((REPOSITORY / "outputs").resolve()):
        raise ValueError("Evidence must be below this worktree's outputs")
    promote(root)
