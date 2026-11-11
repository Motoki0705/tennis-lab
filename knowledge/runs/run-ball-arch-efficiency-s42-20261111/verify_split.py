"""Isolated CUDA qualification of the shared-split implementation.

No scientific updates or original benchmark artifacts are modified. The common
queue must own the whole GPU. Each CUDA stage runs in a fresh subprocess.
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import protocol
from protocol import CONFIG, REMOTE, ROOT, read, write

from src.tasks.ball_detection.checkpoints.provenance import source_identity
from src.utils.artifact_store import RcloneSnapshotStore

BASE = Path(__file__).resolve().parent
DESTINATION = ROOT / "split-prototype"


def child(arguments, log):
    with log.open("w") as stream:
        return subprocess.run(
            [sys.executable, __file__, *arguments],
            stdout=stream,
            stderr=subprocess.STDOUT,
        ).returncode


def main():
    if (
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
        ).strip()
        != "NVIDIA L4"
    ):
        raise RuntimeError("Single L4 required")
    if (
        not os.environ.get("TENNIS_LAB_DINO_EXTENSION")
        or not Path(os.environ["TENNIS_LAB_DINO_EXTENSION"]).is_file()
    ):
        raise RuntimeError("Explicit verified native binary required")
    data = read(ROOT / "data-verified.json")
    if not data["complete"] or data["verified_files"] != 2925:
        raise RuntimeError("Fixed snapshot not verified")
    DESTINATION.mkdir(parents=True, exist_ok=True)
    write(DESTINATION / "data-verified.json", data)
    write(DESTINATION / "source.json", source_identity())
    shutil.copyfile(ROOT / "level-decision.json", DESTINATION / "level-decision.json")
    logs = DESTINATION / "logs"
    logs.mkdir(exist_ok=True)
    with (logs / "cuda-tests.log").open("w") as stream:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-n",
                "0",
                "tests/integration/tasks/ball_detection/test_multiscale_query.py",
                "--junitxml=" + str(DESTINATION / "cuda-tests.xml"),
            ],
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    import xml.etree.ElementTree as ET

    suites = ET.parse(DESTINATION / "cuda-tests.xml").getroot().iter("testsuite")
    counts = [
        {k: int(s.attrib.get(k, 0)) for k in ("tests", "failures", "errors", "skipped")}
        for s in suites
    ]
    if sum(s["tests"] for s in counts) != 5 or any(
        s[k] for s in counts for k in ("failures", "errors", "skipped")
    ):
        raise RuntimeError("All five real CUDA/Inductor checks must pass")
    feasibility = {}
    for case in ("T1", "T2"):
        for levels, chunk in ((4, 1), (3, 4)):
            path = (
                DESTINATION
                / "qualification"
                / f"memory-{case}-{levels}-chunk{chunk}.json"
            )
            if not path.exists():
                result = child(
                    [
                        "memory",
                        "--case",
                        case,
                        "--levels",
                        str(levels),
                        "--chunk",
                        str(chunk),
                    ],
                    logs / f"memory-{case}-{levels}.log",
                )
            else:
                result = 0 if read(path)["status"] == "passed" else 1
            report = read(path)
            if report["status"] not in ("passed", "oom") or (
                levels == 3 and (result or report["status"] != "passed")
            ):
                raise RuntimeError("Split qualification failed: " + str(path))
            feasibility[f"{case}-{levels}"] = report
    if any(feasibility[f"{case}-4"]["status"] == "passed" for case in ("T1", "T2")):
        raise RuntimeError(
            "Four-level feasibility changed: review the level decision before applying"
        )
    cases = {}
    for case in ("T1", "T2"):
        path = DESTINATION / "benchmarks/block-0" / f"full-{case}.json"
        if not path.exists() and child(
            ["measure", "--case", case], logs / f"full-{case}.log"
        ):
            raise RuntimeError("Split measurement failed")
        measurement = read(path)
        previous = read(ROOT / "benchmarks/block-0" / f"full-{case}.json")
        cases[case] = dict(
            inference_ms=measurement["summary"]["inference_cuda_ms"],
            update_ms=measurement["summary"]["update_cuda_ms"],
            prior_update_ms=previous["summary"]["update_cuda_ms"],
            update_ratio=measurement["summary"]["update_cuda_ms"]["p50"]
            / previous["summary"]["update_cuda_ms"]["p50"],
            peak_allocated=max(
                row["peak_allocated"] for row in measurement["raw"]["training"]
            ),
            control_scope="One diagnostic block after identical warmup; final scientific efficiency table requires three blocks",
        )
    proof = dict(
        source=source_identity(),
        cuda_tests=counts,
        levels=feasibility,
        cases=cases,
        scientific_updates=0,
    )
    write(DESTINATION / "qualification.json", proof)
    store = RcloneSnapshotStore(
        DESTINATION,
        remote="mounted_drive",
        remote_root=REMOTE + "/split-prototype",
        config_path=CONFIG,
    )
    receipt = store.publish("qualified-" + str(time.time_ns()))
    with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
        readback = RcloneSnapshotStore(
            Path(temporary) / "readback",
            remote="mounted_drive",
            remote_root=store.remote_root,
            config_path=CONFIG,
        ).restore(receipt["generation"], manifest_sha256=receipt["manifest_sha256"])
    from serial import publish_index

    publish_index(
        ROOT / "split-prototype-receipt.json", dict(receipt=receipt, readback=readback)
    )
    print(json.dumps(cases), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "memory", "measure"))
    parser.add_argument("--case", choices=("T1", "T2"))
    parser.add_argument("--levels", type=int, choices=(3, 4))
    parser.add_argument("--chunk", type=int)
    args = parser.parse_args()
    if args.mode == "run":
        main()
    else:
        protocol.ROOT = DESTINATION
        if args.mode == "memory":
            sys.argv = [
                "memory.py",
                "--case",
                args.case,
                "--levels",
                str(args.levels),
                "--chunk",
                str(args.chunk),
            ]
            runpy.run_path(str(BASE / "memory.py"), run_name="__main__")
        else:
            sys.argv = [
                "measure.py",
                "one",
                "--case",
                "full-" + args.case,
                "--block",
                "0",
            ]
            runpy.run_path(str(BASE / "measure.py"), run_name="__main__")
