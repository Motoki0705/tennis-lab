"""Replay the verified smoke bundles on new Drive paths; no rendering or GPU."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import shutil
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def union_seconds(events: list[dict[str, Any]]) -> float:
    end, total = 0.0, 0.0
    for event in sorted(events, key=lambda v: v["start"]):
        start, stop = event["start"], event["start"] + event["seconds"]
        total += max(0, stop - max(end, start))
        end = max(end, stop)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--drive-path", required=True)
    parser.add_argument("--baseline-drive-path")
    args = parser.parse_args()
    checkout = Path.cwd()
    sys.path.insert(0, str(checkout))
    from src.synthetic_data_generation.dataset.court.scale.durability import (
        fetch_completed_dataset,
        restore_inputs,
    )
    from src.synthetic_data_generation.dataset.court.scale.durable_storage import (
        BundleStore,
        SkillDriveFiles,
    )
    from src.synthetic_data_generation.dataset.court.scale.publication import (
        load_dataset_version,
    )
    from src.utils.data.image_record_store import file_sha256

    output = args.output.resolve()
    if not output.is_relative_to(checkout / "tmp"):
        raise ValueError("Benchmark scratch must stay under this checkout's tmp/.")
    output.mkdir(parents=True, exist_ok=False)
    real_rclone = shutil.which("rclone")
    if real_rclone is None:
        raise ValueError("rclone is required.")
    version = subprocess.check_output([real_rclone, "version"], text=True).splitlines()
    wrapper_dir = output / "bin"
    wrapper_dir.mkdir()
    wrapper = wrapper_dir / "rclone"
    wrapper.write_text(
        "#!/bin/sh\nexec "
        + shlex.join(
            [
                sys.executable,
                str(Path(__file__).with_name("trace_process.py").resolve()),
            ]
        )
        + ' "$@"\n'
    )
    wrapper.chmod(0o700)
    os.environ.update(
        RCLONE_BIN=str(wrapper),
        PATH=f"{wrapper_dir}:{os.environ['PATH']}",
        COURT_SESSION_RCLONE=real_rclone,
        COURT_SESSION_TRACE=str(output / "process-starts.jsonl"),
        COURT_SESSION_PHASE="initialization",
    )
    previous = checkout / "knowledge/runs/run-court-scale-colab-resume-l4-20261023"
    receipts = {
        path.relative_to(previous / "drive-receipts").as_posix(): json.loads(
            path.read_text()
        )
        for path in (previous / "drive-receipts").rglob("*.json")
    }
    inputs = receipts["request.json"]
    request_sha = inputs["request_sha256"]
    origin_path = inputs["archive_drive"]["path"].split("/objects/")[0]
    cache_paths = [
        "caches/" + item["cache_key"] + ".json"
        for item in json.loads((previous / "resumed-attempt.json").read_text())[
            "caches"
        ]
    ]
    assert len(cache_paths) == 24 and len(set(cache_paths)) == 24
    events: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    lock = threading.Lock()
    summary: dict[str, Any] = {
        "schema": "court_scale_transport_benchmark_v1",
        "started_at": datetime.now(UTC).isoformat(),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "python": sys.version.split()[0],
        "rclone_version": version,
        "host": "local CPU; existing Colab L4 bundles replayed without rendering",
        "source_drive_path": origin_path,
        "drive_path": args.drive_path,
        "baseline_drive_path": args.baseline_drive_path,
        "source_request_sha256": request_sha,
        "cache_count": 24,
        "steps": steps,
        "source_archive_bytes": sum(v["archive_bytes"] for v in receipts.values()),
        "http_request_count": None,
        "http_request_count_reason": "RC calls are measured, not credential-bearing Google HTTP traffic.",
    }

    @contextmanager
    def timed(name: str) -> Iterator[None]:
        os.environ["COURT_SESSION_PHASE"] = name
        start_index, started = len(events), time.perf_counter()
        try:
            yield
        finally:
            selected = events[start_index:]
            record = {
                "name": name,
                "seconds": time.perf_counter() - started,
                "rc_calls": dict(Counter(e["operation"] for e in selected)),
                "rc_busy_seconds": union_seconds(selected),
                "listing_busy_seconds": union_seconds(
                    [e for e in selected if e["operation"] == "operations/list"]
                ),
                "copy_busy_seconds": union_seconds(
                    [e for e in selected if e["operation"] == "operations/copyfile"]
                ),
            }
            steps.append(record)
            write(output / f"{name}-rc.json", selected)
            write(output / "measurement.json", summary)
            print(json.dumps(record), flush=True)

    @contextmanager
    def remote(path: str, name: str, *, fresh: bool = False) -> Iterator[Any]:
        os.environ["COURT_SESSION_PHASE"] = name
        with SkillDriveFiles(path) as adapter:
            original = adapter.session.call

            def measured(endpoint: str, **params: Any) -> dict[str, Any]:
                started = time.perf_counter()
                try:
                    result: dict[str, Any] = original(endpoint, **params)
                    return result
                finally:
                    with lock:
                        events.append(
                            {
                                "operation": endpoint,
                                "start": started,
                                "seconds": time.perf_counter() - started,
                            }
                        )

            adapter.session.call = measured
            if fresh:
                found = adapter.folders.lookup(adapter.drive_path)
                if found is not None and found[1] is not None:
                    raise ValueError(
                        "Benchmark destination must be a completely new Drive path."
                    )
            yield adapter

    source = output / "source"
    with timed("source-fetch"), remote(origin_path, "source-fetch") as adapter:
        source_version = fetch_completed_dataset(
            source, adapter, request_sha256=request_sha
        )
    verified_version = load_dataset_version(source_version, verify_images=True)
    summary["dataset_version_sha256"] = file_sha256(source_version)
    summary["accepted_images"] = sum(
        s["counts"]["qa_accepted"] for s in verified_version["scenes"]
    )

    transport_fields = {
        "schema",
        "archive_path",
        "archive_sha256",
        "archive_bytes",
        "archive_drive",
        "files",
        "verified_at",
    }

    def identity(receipt: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in receipt.items() if k not in transport_fields}

    def publish(store: Any, path: str) -> None:
        expected = receipts[path]
        saved = store.publish(
            "receipts/" + path, expected["files"], identity=identity(expected)
        )
        assert saved["archive_sha256"] == expected["archive_sha256"]
        assert (
            saved["files"] == expected["files"]
            and saved["schema"] == expected["schema"]
        )

    if args.baseline_drive_path:
        with (
            timed("baseline-setup"),
            remote(args.baseline_drive_path, "baseline-setup", fresh=True) as adapter,
        ):
            for relative in ("objects/probe.tar", "receipts/caches/probe.json"):
                adapter.folders.lookup(adapter._path(relative), create_parents=True)
        code_path = (
            "src/synthetic_data_generation/dataset/court/scale/durable_storage.py"
        )
        baseline_commit = "174ebef9cbf2876f602408f8de77964f45aba098"
        code = subprocess.check_output(
            ["git", "show", f"{baseline_commit}:{code_path}"], text=True
        )
        name = "src.synthetic_data_generation.dataset.court.scale._baseline_durable_storage"
        spec = importlib.util.spec_from_loader(name, loader=None)
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        exec(compile(code, f"{baseline_commit}:{code_path}", "exec"), module.__dict__)
        baseline = module.SkillDriveFiles(args.baseline_drive_path)
        with timed("baseline-new-cache"):
            assert not baseline.exists("receipts/" + cache_paths[0])
            publish(module.BundleStore(source, baseline), cache_paths[0])
        with timed("baseline-cache-recovery"):
            old_store = module.BundleStore(output / "baseline-recovered", baseline)
            receipt = old_store.receipt("receipts/" + cache_paths[0])
            assert receipt is not None
            old_store.restore(receipt)
        summary["baseline_commit"] = baseline_commit

    with (
        timed("initial-session"),
        remote(args.drive_path, "initial-session", fresh=True) as adapter,
    ):
        store = BundleStore(source, adapter)
        with timed("input-publish"):
            publish(store, "request.json")
        with timed("new-cache-00"):
            assert store.receipt("receipts/" + cache_paths[0]) is None
            publish(store, cache_paths[0])
    recovered = output / "resumed"
    with timed("resume-session"), remote(args.drive_path, "resume-session") as adapter:
        store = BundleStore(recovered, adapter)
        with timed("resume-input-restore"):
            restore_inputs(recovered, adapter, request_sha256=request_sha)
        with timed("resume-input-republish"):
            publish(store, "request.json")
        with timed("resume-cache-restore"):
            receipt = store.receipt("receipts/" + cache_paths[0])
            assert receipt is not None
            store.restore(receipt)

        def stage_files(path: str) -> None:
            for name in receipts[path]["files"]:
                target = recovered / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with (source / name).open("rb") as src, target.open("xb") as dst:
                    shutil.copyfileobj(src, dst)

        for i, path in enumerate(cache_paths[1:], start=1):
            stage_files(path)
            with timed(f"new-cache-{i:02d}"):
                assert store.receipt("receipts/" + path) is None
                publish(store, path)
        stage_files("completed.json")
        with timed("publication-publish"):
            publish(store, "completed.json")
    with (
        timed("completed-fetch"),
        remote(args.drive_path, "completed-fetch") as adapter,
    ):
        fetched = fetch_completed_dataset(
            output / "fetched", adapter, request_sha256=request_sha
        )
        assert file_sha256(fetched) == file_sha256(source_version)
        assert load_dataset_version(fetched, verify_images=True) == verified_version
    summary["finished_at"] = datetime.now(UTC).isoformat()
    summary["strict_fetch_matches_source"] = True
    summary["new_cache_seconds"] = {
        "median": statistics.median(
            s["seconds"] for s in steps if s["name"].startswith("new-cache-")
        ),
        "steady_median": statistics.median(
            s["seconds"]
            for s in steps
            if s["name"].startswith("new-cache-") and s["name"] != "new-cache-00"
        ),
    }
    process_events = [
        json.loads(line)
        for line in (output / "process-starts.jsonl").read_text().splitlines()
    ]
    summary["rclone_processes_by_phase"] = dict(
        Counter(v["phase"] for v in process_events)
    )
    summary["rclone_operations"] = dict(Counter(v["operation"] for v in process_events))
    write(output / "measurement.json", summary)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for step in steps:
        groups["cache" if step["name"].startswith("new-cache-") else "other"].append(
            step
        )
    print(json.dumps({k: v for k, v in summary.items() if k != "steps"}), flush=True)


if __name__ == "__main__":
    main()
