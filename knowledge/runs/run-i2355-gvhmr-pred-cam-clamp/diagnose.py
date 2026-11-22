"""Build static GVHMR subgraphs until the first failing transformer prefix."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import onnx

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent
REPO = HERE.parents[2]
COMMON = RUNS / "run-i2297-clipreid-tensorrt"
GVHMR = RUNS / "run-i2305-gvhmr-tensorrt"
for dependency in (REPO, COMMON, GVHMR, HERE):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from benchmark import write_json  # noqa: E402
from convert import require_queue  # noqa: E402
from graph_slices import slice_graph  # noqa: E402

BUILD = HERE / "build_engine.py"
FP16 = onnx.TensorProto.FLOAT16
BOOL = onnx.TensorProto.BOOL
BATCH = 2
FRAMES = 145
WIDTH = 512


def child_command(*arguments: object) -> list[str]:
    return [
        sys.executable,
        "-B",
        "-X",
        "faulthandler",
        str(BUILD),
        *map(str, arguments),
    ]


def run_build(
    report: dict[str, object], root: Path, source: Path, name: str
) -> dict[str, object]:
    destination = root / "cases" / name
    plan = destination / "model.plan"
    receipt = destination / "build.json"
    log = root / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = child_command(
        "--onnx",
        source,
        "--plan",
        plan,
        "--report",
        receipt,
        "--builder-optimization-level",
        3,
    )
    started = time.monotonic()
    with log.open("x") as stream:
        print(subprocess.list2cmdline(command), file=stream, flush=True)
        completed = subprocess.run(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
            env={
                **os.environ,
                "PYTHONPATH": os.pathsep.join(map(str, (REPO, COMMON, GVHMR, HERE))),
            },
        )
    text = log.read_text(errors="replace")
    timing_runners = [
        line.strip() for line in text.splitlines() if "Timing Runner:" in line
    ]
    result = {
        "exit_code": completed.returncode,
        "wall_seconds": time.monotonic() - started,
        "log": str(log.relative_to(root)),
        "plan_published": plan.is_file(),
        "receipt_published": receipt.is_file(),
        "last_timing_runner": timing_runners[-1] if timing_runners else None,
    }
    report.setdefault("builds", {})[name] = result
    write_json(root / "diagnosis.json", report)
    print(f"{name}: exit {completed.returncode}", flush=True)
    return result


def prepare_slice(
    report: dict[str, object],
    root: Path,
    source: Path,
    name: str,
    output_name: str,
    output_type: int,
    output_shape: tuple[int | str, ...],
    *,
    promoted_inputs: dict[str, tuple[int, tuple[int | str, ...]]] | None = None,
) -> Path:
    target = root / "graphs" / f"{name}.onnx"
    evidence = slice_graph(
        source,
        target,
        output_name=output_name,
        output_type=output_type,
        output_shape=output_shape,
        promoted_inputs=promoted_inputs,
    )
    report.setdefault("graphs", {})[name] = evidence
    write_json(root / "diagnosis.json", report)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require_queue()
    source = args.source.resolve(strict=True)
    root = args.output.resolve()
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    (Path(os.environ["TENNIS_REPRO_DIR"]) / "output_dir.txt").write_text(f"{root}\n")
    report: dict[str, object] = {
        "schema_version": 1,
        "queue_job": os.environ["TENNIS_RUN_ID"],
        "source": str(source),
        "fixed_shape": {"batch": BATCH, "length": FRAMES},
        "graphs": {},
        "builds": {},
    }
    write_json(root / "diagnosis.json", report)

    fixed_cases = (
        ("mask-only", "mask_out", BOOL, (BATCH, FRAMES), None),
        (
            "embedding-only",
            "/network/Add_3_output_0",
            FP16,
            (BATCH, FRAMES, WIDTH),
            None,
        ),
        (
            "pred-cam-head-only",
            "pred_cam",
            FP16,
            (BATCH, FRAMES, 3),
            {
                "/network/blocks.11/Add_1_output_0": (
                    FP16,
                    (BATCH, FRAMES, WIDTH),
                )
            },
        ),
        ("full-pred-cam", "pred_cam", FP16, (BATCH, FRAMES, 3), None),
    )
    for name, output_name, output_type, output_shape, promoted in fixed_cases:
        graph = prepare_slice(
            report,
            root,
            source,
            name,
            output_name,
            output_type,
            output_shape,
            promoted_inputs=promoted,
        )
        run_build(report, root, graph, name)

    first_failed = None
    for block in range(12):
        name = f"prefix-block-{block:02d}"
        graph = prepare_slice(
            report,
            root,
            source,
            name,
            f"/network/blocks.{block}/Add_1_output_0",
            FP16,
            (BATCH, FRAMES, WIDTH),
        )
        result = run_build(report, root, graph, name)
        if result["exit_code"] != 0:
            first_failed = block
            break
    report["first_failed_prefix_block"] = first_failed
    report["status"] = "done"
    write_json(root / "diagnosis.json", report)


if __name__ == "__main__":
    main()
