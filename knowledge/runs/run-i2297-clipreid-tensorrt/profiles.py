"""Capture and summarize the same one-inference CUDA range with Nsight."""

import csv
import math
import sqlite3
import subprocess
from collections import Counter
from contextlib import closing

PROFILE_RANGE = "steady_inference"
DURATION = "gpu__time_duration.sum"
READ = "dram__bytes_op_read.sum"
WRITE = "dram__bytes_op_write.sum"
TENSOR = "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"
STALL = {
    reason: f"smsp__warp_issue_stalled_{reason}_per_warp_active.pct"
    for reason in ("long_scoreboard", "short_scoreboard", "lg_throttle", "mio_throttle")
}
NCU_METRICS = [DURATION, READ, WRITE, TENSOR, *STALL.values()]


def read_nsys(path, name=PROFILE_RANGE):
    """Count actual GPU kernel activities inside the one synchronized NVTX span."""
    with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as db:
        ranges = db.execute(
            """SELECT n.start,n.end FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id
               WHERE coalesce(n.text,s.value)=? AND n.end>n.start""",
            (name,),
        ).fetchall()
        if len(ranges) != 1:
            raise ValueError(f"Expected one NVTX range {name!r}, found {len(ranges)}")
        if not db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='CUPTI_ACTIVITY_KIND_KERNEL'"
        ).fetchone():
            raise ValueError("Nsight Systems has no CUDA GPU kernel activity table")
        kernels = db.execute(
            """SELECT s.value,k.end-k.start FROM CUPTI_ACTIVITY_KIND_KERNEL k
               JOIN StringIds s ON k.shortName=s.id WHERE k.start>=? AND k.end<=?
               ORDER BY k.start,k.correlationId""",
            ranges[0],
        ).fetchall()
    if not kernels or any(duration <= 0 for _, duration in kernels):
        raise ValueError("Nsight Systems has no valid GPU kernels in the NVTX range")
    return {
        "gpu_kernels": len(kernels),
        "sum_kernel_duration_ns": sum(duration for _, duration in kernels),
        "kernel_names": dict(Counter(name for name, _ in kernels)),
    }


def read_ncu(path):
    """Sum DRAM bytes and time-weight per-kernel stall/tensor percentages."""
    with path.open(newline="") as stream:
        rows = csv.reader(stream)
        try:
            header = next(line for line in rows if line and line[0] == "ID")
        except StopIteration as exc:
            raise ValueError("Nsight Compute returned no raw metric table") from exc
        missing = set(NCU_METRICS) - set(header)
        if missing:
            raise ValueError(f"Missing Nsight Compute columns: {sorted(missing)}")
        columns = {name: header.index(name) for name in NCU_METRICS}
        values = []
        for row in rows:
            if not row or not row[0].isdigit():
                continue
            if len(row) != len(header):
                raise ValueError(
                    f"Nsight Compute CSV row {row[0]} has {len(row)} columns, "
                    f"expected {len(header)}"
                )
            values.append(row)
    if not values:
        raise ValueError("Nsight Compute returned no kernel rows")
    if len({row[0] for row in values}) != len(values):
        raise ValueError("Duplicate Nsight Compute kernel IDs")
    metrics = {
        name: [float(row[index].replace(",", "")) for row in values]
        for name, index in columns.items()
    }
    if not all(
        math.isfinite(value) and value >= 0
        for series in metrics.values()
        for value in series
    ):
        raise ValueError("Invalid Nsight Compute metric values")
    time = metrics[DURATION]
    if min(time) <= 0:
        raise ValueError("Invalid Nsight Compute GPU duration")

    def weighted(metric):
        return sum(
            t * value for t, value in zip(time, metrics[metric], strict=True)
        ) / sum(time)

    return {
        "profiled_kernels": len(values),
        "dram_read_bytes": int(sum(metrics[READ])),
        "dram_write_bytes": int(sum(metrics[WRITE])),
        "tensor_active_pct_time_weighted": weighted(TENSOR),
        "stall_pct_time_weighted": {
            reason: weighted(name) for reason, name in STALL.items()
        },
    }


def capture(tool, executable, command, output, *, ncu_timeout=900, ncu_replay="kernel"):
    log = output / f"{tool}.log"
    report = output / ("nsys.sqlite" if tool == "nsys" else "ncu.csv")
    try:
        with log.open("x") as stream:
            if tool == "nsys":
                trace = output / "nsys"
                subprocess.run(
                    [
                        executable,
                        "profile",
                        "--trace=cuda-sw,nvtx",
                        "--sample=none",
                        "--cpuctxsw=none",
                        "--capture-range=cudaProfilerApi",
                        "--capture-range-end=stop",
                        "-o",
                        str(trace),
                        *command,
                    ],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=120,
                )
                subprocess.run(
                    [
                        executable,
                        "export",
                        "--type=sqlite",
                        "-o",
                        str(report),
                        str(trace.with_suffix(".nsys-rep")),
                    ],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=120,
                )
                metrics = read_nsys(report)
            else:
                replay = (
                    ["--app-replay-match", "all", "--app-replay-mode", "strict"]
                    if ncu_replay == "application"
                    else []
                )
                subprocess.run(
                    [
                        executable,
                        "--profile-from-start",
                        "off",
                        "--nvtx",
                        "--nvtx-include",
                        f"{PROFILE_RANGE}/",
                        "--replay-mode",
                        ncu_replay,
                        *replay,
                        "--cache-control",
                        "none",
                        "--clock-control",
                        "none",
                        "--metrics",
                        ",".join(NCU_METRICS),
                        "--csv",
                        "--page",
                        "raw",
                        "--print-units",
                        "base",
                        "--log-file",
                        str(report),
                        *command,
                    ],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=ncu_timeout,
                )
                metrics = read_ncu(report)
    except (OSError, ValueError, sqlite3.Error, subprocess.SubprocessError) as exc:
        context = "\n".join(
            path.read_text(errors="replace")
            for path in (log, report)
            if path.suffix != ".sqlite" and path.is_file()
        )
        if "ERR_NVGPUCTRPERM" in context:
            detail = next(
                line for line in context.splitlines() if "ERR_NVGPUCTRPERM" in line
            )
        else:
            detail = str(exc)
        return {"status": "unavailable", "reason": detail}
    return {"status": "measured", **metrics}
