"""CPU checks for the fixed-input, profiler and strict CLI contracts."""

import csv
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest
from benchmark import accuracy
from convert import precision_layers
from inputs import CAMERAS, check_split
from profiles import (
    DURATION,
    NCU_METRICS,
    PROFILE_RANGE,
    READ,
    STALL,
    TENSOR,
    WRITE,
    read_ncu,
    read_nsys,
)


def _clip(clip, split, interval):
    return {
        "clip": clip,
        "split": split,
        "streams": [
            {
                "camera": camera,
                "sha256": f"{clip}/{camera}",
                "source_frames_inclusive": interval,
            }
            for camera in CAMERAS
        ],
    }


def test_raw_source_overlap_is_rejected_even_if_media_and_clip_ids_differ():
    evaluation = _clip("video_000/clip_001", "eval", [10, 20])
    calibration = _clip("video_000/clip_002", "calibration", [20, 30])
    with pytest.raises(ValueError, match="Shared raw source frames"):
        check_split({"clips": [evaluation, calibration]})
    for stream in calibration["streams"]:
        stream["source_frames_inclusive"] = [21, 30]
    assert check_split({"clips": [evaluation, calibration]})["calibration"] == [
        "video_000/clip_002"
    ]


def test_engine_inspection_uses_exact_precision_with_trt_dtype_spellings():
    formats = (
        "Half",
        "Row major linear FP16 format",
        "BFloat16",
        "UInt8",
        "Int8",
        "FP8",
    )
    details = {
        "Layers": [
            {"Name": dtype, "Inputs": [{"Format/Datatype": dtype}]} for dtype in formats
        ]
    }
    assert precision_layers(details, "fp16") == list(formats[:2])
    assert precision_layers(details, "int8") == ["Int8"]
    assert precision_layers(details, "fp8") == ["FP8"]


def test_nsys_counts_only_real_kernels_inside_the_range(tmp_path):
    trace = tmp_path / "trace.sqlite"
    with closing(sqlite3.connect(trace)) as conn, conn:
        conn.executescript("""
            CREATE TABLE StringIds(id INTEGER, value TEXT);
            CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, text TEXT, textId INTEGER);
            INSERT INTO NVTX_EVENTS VALUES (10, 70, NULL, 1);
            INSERT INTO StringIds VALUES (1, 'steady_inference'), (2, 'gemm'), (3, 'fill');
        """)
    with pytest.raises(ValueError, match="no CUDA GPU kernel activity"):
        read_nsys(trace)
    with closing(sqlite3.connect(trace)) as conn, conn:
        conn.executescript("""
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER, end INTEGER, shortName INTEGER, correlationId INTEGER);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (1, 4, 3, 1), (12, 20, 2, 2), (26, 39, 2, 3), (63, 73, 3, 4);
        """)
    assert read_nsys(trace, PROFILE_RANGE) == {
        "gpu_kernels": 2,
        "sum_kernel_duration_ns": 21,
        "kernel_names": {"gemm": 2},
    }


def test_ncu_sums_bytes_and_weights_percentages_by_gpu_duration(tmp_path):
    report = tmp_path / "ncu.csv"
    with report.open("w", newline="") as stream:
        stream.write("==PROF== Connected to process\n")
        writer = csv.DictWriter(stream, fieldnames=["ID", *NCU_METRICS])
        writer.writeheader()
        writer.writerow({name: "%" for name in NCU_METRICS})
        for i, (duration, read, write, tensor, stall) in enumerate(
            [(100, 1024, 16, 20, 80), (300, 2048, 24, 60, 0)]
        ):
            writer.writerow(
                {
                    "ID": i,
                    DURATION: duration,
                    READ: read,
                    WRITE: write,
                    TENSOR: tensor,
                    **dict.fromkeys(STALL.values(), stall),
                }
            )
    result = read_ncu(report)
    assert result["profiled_kernels"] == 2
    assert result["dram_read_bytes"] == 3072
    assert result["dram_write_bytes"] == 40
    assert result["tensor_active_pct_time_weighted"] == 50
    assert result["stall_pct_time_weighted"]["long_scoreboard"] == 20


def test_ncu_rejects_a_shifted_row_even_with_numeric_metric_cells(tmp_path):
    report = tmp_path / "ncu.csv"
    with report.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["ID", "NVTX Range", *NCU_METRICS])
        writer.writerow([0, "outer", "split payload", 10, 32, 64, 25, 1, 2, 3, 4])
    with pytest.raises(ValueError, match=r"CSV row 0 has 11 columns, expected 10"):
        read_ncu(report)


def test_clip_local_id_decisions_and_cosine_use_held_out_camera():
    addresses = [
        (clip, camera, index, player)
        for clip in ("clip_0", "clip_1")
        for camera in ("cam0", "cam1")
        for index, player in [(10, 0), (20, 1)]
    ]
    baseline = np.array(
        [[1, 0], [0, 1], [1, 0], [0, 1], [0, 1], [1, 0], [0, 1], [1, 0]],
        dtype=np.float32,
    )
    candidate = baseline.copy()
    candidate[2] = [0, 1]
    report = accuracy(candidate, baseline, addresses)
    assert report["cosine_mean"] == 7 / 8
    assert report["id_matches_fp32"] == 6
    assert report["id_changed_examples"] == [
        {"address": addresses[1], "fp32_id": 1, "condition_id": 0},
        {"address": addresses[2], "fp32_id": 0, "condition_id": 1},
    ]


def test_invalid_condition_fails_before_creating_a_run(tmp_path):
    output = tmp_path / "run"
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            str(Path(__file__).with_name("benchmark.py")),
            "--model",
            "clipreid_vitb16_market1501",
            "--condition",
            "bf16",
            "--inputs",
            str(tmp_path),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 2
    assert "invalid choice" in result.stderr
    assert not output.exists()
