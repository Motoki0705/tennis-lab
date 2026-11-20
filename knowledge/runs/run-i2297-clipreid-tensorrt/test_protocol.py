"""CPU checks for the fixed-input, profiler and strict CLI contracts."""

import csv
import errno
import json
import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import benchmark
import numpy as np
import profiles
import pytest
from convert import PLAN, set_profile_shapes, verify_compute_precision
from dino_accuracy import compare_dino_frame
from inputs import CAMERAS, check_split, sha256_file
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


def _gemm(dtype, tactic):
    return {
        "Name": "attention_projection",
        "LayerType": "gemm",
        "Inputs": [{"Format/Datatype": dtype}],
        "Outputs": [{"Format/Datatype": "Float"}],
        "TacticName": tactic,
        "Metadata": "[ONNX Layer: /projection/MatMul]",
    }


@pytest.mark.parametrize(
    ("precision", "dtype", "tactic"),
    [
        ("fp16", "Half", "sm80_xmma_gemm_f16f16_f16f32_f32_tn"),
        ("fp16", "Row major linear FP16 format", "sm80_xmma_gemm_f16f16_f16f32_f32_tn"),
        ("int8", "Int8", "sm80_xmma_gemm_i8f32_i8i32_f32_tn"),
        ("fp8", "FP8", "sm89_xmma_gemm_e4m3f16_e4m3f32_f32_tn"),
    ],
)
def test_engine_inspection_checks_operand_precision_and_allows_float_outputs(
    precision, dtype, tactic
):
    evidence = verify_compute_precision({"Layers": [_gemm(dtype, tactic)]}, precision)
    assert evidence["engine_layers_by_precision"] == {precision: 1}
    for other_dtype in ("BFloat16", "UInt8"):
        with pytest.raises(RuntimeError, match=f"expected {precision} operands"):
            verify_compute_precision(
                {"Layers": [_gemm(other_dtype, tactic)]}, precision
            )


@pytest.mark.parametrize(
    ("precision", "dtype"), [("fp16", "Half"), ("int8", "Int8"), ("fp8", "FP8")]
)
def test_engine_inspection_rejects_fp32_gemm_with_low_precision_cast(precision, dtype):
    details = {
        "Layers": [
            {
                "Name": "cast",
                "LayerType": "Reformat",
                "Outputs": [{"Format/Datatype": dtype}],
            },
            _gemm("Float", "sm80_xmma_gemm_f32f32_f32f32_f32_tn"),
        ]
    }
    with pytest.raises(
        RuntimeError, match=f"attention_projection expected {precision} operands"
    ):
        verify_compute_precision(details, precision)


def test_engine_inspection_rejects_fp32_tactic_with_fp16_input():
    layer = _gemm("Half", "sm80_xmma_gemm_f32f16_f32f32_f32_tn")
    with pytest.raises(
        RuntimeError, match="attention_projection expected fp16 operands"
    ):
        verify_compute_precision({"Layers": [layer]}, "fp16")


def test_engine_inspection_requires_every_declared_onnx_operation():
    layer = _gemm("Half", "sm80_xmma_gemm_f16f16_f16f32_f32_tn")
    expected = {"/projection/MatMul": "fp16", "/patch/Conv": "fp16"}
    with pytest.raises(RuntimeError, match=r"Missing compute evidence.*?/patch/Conv"):
        verify_compute_precision(
            {"Layers": [layer]}, "fp16", expected_operations=expected
        )


@pytest.mark.parametrize(
    ("precision", "dtype"), [("fp16", "Half"), ("int8", "Int8"), ("fp8", "FP8")]
)
def test_fused_attention_checks_qkv_around_two_boolean_masks(precision, dtype):
    layer = {
        "Name": "_gemm_mha_v2_myl0_35",
        "LayerType": "kgen",
        "TacticName": "_gemm_mha_v2_example",
        "Inputs": [
            {"Format/Datatype": value}
            for value in (dtype, dtype, "Bool", "Bool", dtype)
        ],
        "Metadata": "[ONNX Layer: /attn/MatMul]\x1f[ONNX Layer: /attn/MatMul_1]",
    }
    expected = {"/attn/MatMul": precision, "/attn/MatMul_1": precision}
    passed = verify_compute_precision(
        {"Layers": [layer]}, precision, expected_operations=expected
    )
    assert passed["onnx_operations_by_precision"] == {precision: 2}
    layer["Inputs"][-1]["Format/Datatype"] = "Float"
    with pytest.raises(RuntimeError, match=f"expected {precision} operands"):
        verify_compute_precision(
            {"Layers": [layer]}, precision, expected_operations=expected
        )


@pytest.mark.parametrize("epilogue", ("FcCastAddMulAdd", "FcCastAddAdd"))
def test_int8_residual_fusion_requires_matching_qdq_and_int8_tactic_operands(epilogue):
    layer = _gemm("Half", "sm80_xmma_gemm_i8f32_i8i32_f32_tn_by_fusion_tactic")
    layer["Name"] = f"__myl_{epilogue}_myl0_36"
    layer["LayerType"] = "fusion"
    layer["Inputs"][0]["Dimensions"] = [1, 900, 256]
    layer["Inputs"].append({"Format/Datatype": "Int8", "Dimensions": [1, 900, 1024]})
    layer["Outputs"][0]["Dimensions"] = [1, 900, 256]
    layer["Metadata"] += (
        "\x1f[ONNX Layer: weight_QuantizeLinear]\x1f[ONNX Layer: weight_DequantizeLinear]"
    )
    expected = {"/projection/MatMul": "int8"}
    assert verify_compute_precision(
        {"Layers": [layer]}, "int8", expected_operations=expected
    )["engine_layers_by_precision"] == {"int8": 1}
    for broken in (
        {**layer, "TacticName": layer["TacticName"].replace("i8f32", "f32f32")},
        {
            **layer,
            "Metadata": layer["Metadata"].replace(
                "weight_DequantizeLinear", "other_DequantizeLinear"
            ),
        },
    ):
        with pytest.raises(RuntimeError, match="expected int8 operands"):
            verify_compute_precision(
                {"Layers": [broken]}, "int8", expected_operations=expected
            )
    if epilogue == "FcCastAddAdd":
        layer["Outputs"][0]["Dimensions"] = [1, 900, 1024]
        with pytest.raises(RuntimeError, match="expected int8 operands"):
            verify_compute_precision(
                {"Layers": [layer]}, "int8", expected_operations=expected
            )


def test_int8_small_k_convolution_tactic_for_linear_layer():
    layer = _gemm(
        "Int8", "sm80_xmma_conv_fprop_smallk_i8i8_i32_f32_nchw_vect_c_32kcrs_vect_c_32"
    )
    details = {"Layers": [layer]}
    assert verify_compute_precision(details, "int8")["int8_small_k_layers"] == 1
    layer["TacticName"] = layer["TacticName"].replace("_i8i8_", "_f16f16_")
    with pytest.raises(RuntimeError, match="expected int8 operands"):
        verify_compute_precision(details, "int8")


def test_profile_uses_shape_readback_instead_of_set_shape_return():
    class Profile:
        def __init__(self):
            self.shapes = {}

        def set_shape(self, name, minimum, optimum, maximum):
            self.shapes[name] = (minimum, optimum, maximum)

        def get_shape(self, name):
            return list(self.shapes[name])

    shapes = {
        "length": [[1], [1], [1]],
        "obs": [[1, 117, 17, 3], [1, 145, 17, 3], [1, 177, 17, 3]],
    }
    profile = Profile()
    set_profile_shapes(profile, shapes)
    assert profile.get_shape("obs") == [
        (1, 117, 17, 3),
        (1, 145, 17, 3),
        (1, 177, 17, 3),
    ]
    profile.get_shape = lambda _: []
    with pytest.raises(ValueError, match="TensorRT profile length"):
        set_profile_shapes(profile, shapes)


def test_engine_inspection_checks_shifted_window_attention_operands():
    layer = {
        "Name": "shifted_window_attention",
        "LayerType": "kgen",
        "TacticName": "_gemm_mha_v2_observed",
        "Inputs": [
            {"Name": "mask", "Dimensions": [7, 1, 144], "Format/Datatype": "Half"},
            {"Name": "mask", "Dimensions": [7, 144, 1], "Format/Datatype": "Half"},
            {"Dimensions": [168, 144, 32], "Format/Datatype": "FP8"},
            {"Dimensions": [168, 32, 144], "Format/Datatype": "FP8"},
            {"Dimensions": [168, 144, 32], "Format/Datatype": "FP8"},
        ],
    }
    assert (
        verify_compute_precision({"Layers": [layer]}, "fp8")["masked_attention_layers"]
        == 1
    )

    layer["Inputs"][4]["Format/Datatype"] = "Float"
    with pytest.raises(RuntimeError, match="expected fp8 operands"):
        verify_compute_precision({"Layers": [layer]}, "fp8")
    layer["Inputs"][4]["Format/Datatype"] = "FP8"
    layer["Inputs"][0]["Dimensions"] = [7, 2, 144]
    with pytest.raises(RuntimeError, match="expected fp8 operands"):
        verify_compute_precision({"Layers": [layer]}, "fp8")


def test_engine_inspection_checks_named_constant_gemm_tactic_without_runtime_inputs():
    layer = _gemm("Half", "sm80_xmma_gemm_f16f16_f16f32_f32_tn")
    layer["Inputs"] = []
    details = {"Layers": [layer]}
    expected = {"/projection/MatMul": "fp16"}
    with pytest.raises(RuntimeError, match="expected fp16 operands"):
        verify_compute_precision(details, "fp16", expected_operations=expected)

    def verify():
        return verify_compute_precision(
            details,
            "fp16",
            expected_operations=expected,
            constant_operations=frozenset(expected),
        )

    assert verify()["constant_operand_operations"] == list(expected)
    layer["TacticName"] = "sm80_xmma_gemm_f32f32_f32f32_f32_tn"
    with pytest.raises(RuntimeError, match="expected fp16 operands"):
        verify()


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


def test_ncu_suffix_requires_full_ordered_nsys_launch_coverage(tmp_path):
    prefix, suffix = (tmp_path / name for name in ("prefix.csv", "suffix.csv"))
    for path, grid, duration, dram, tensor in (
        (prefix, "(32, 1, 1)", 100, 1024, 20),
        (suffix, "(8, 2, 1)", 300, 2048, 60),
    ):
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["ID", "Grid Size", "Block Size", *NCU_METRICS])
            writer.writerow(
                [0, grid, "(128, 1, 1)", duration, dram, 0, tensor, 0, 0, 0, 0]
            )
    reference = [(32, 1, 1, 128, 1, 1), (8, 2, 1, 128, 1, 1)]
    result = profiles.stitch_ncu(prefix, suffix, reference, tmp_path / "joined.csv")
    assert result["profiled_kernels"] == 2
    assert (result["prefix_kernels"], result["suffix_kernels"]) == (1, 1)
    assert result["dram_read_bytes"] == 3072
    assert result["tensor_active_pct_time_weighted"] == 50
    with pytest.raises(ValueError, match="ordered nsys range"):
        profiles.stitch_ncu(prefix, suffix, reference[::-1], tmp_path / "mismatch.csv")


@pytest.mark.parametrize("mode", ("kernel", "application"))
def test_ncu_uses_the_requested_replay_and_requires_strict_application_matching(
    mode, tmp_path, monkeypatch
):
    def inspect_command(command, *, timeout, **_kwargs):
        assert command[command.index("--replay-mode") + 1] == mode
        assert command[command.index("--metrics") + 1].split(",") == NCU_METRICS
        assert timeout == 720
        if mode == "application":
            assert command[command.index("--app-replay-mode") + 1] == "strict"
            assert command[command.index("--app-replay-match") + 1] == "all"
        else:
            assert "--app-replay-mode" not in command

    monkeypatch.setattr(profiles, "_run_ncu", inspect_command)
    monkeypatch.setattr(profiles, "read_ncu", lambda _path: {"profiled_kernels": 4})
    assert profiles.capture(
        "ncu", "ncu", ["runner"], tmp_path, ncu_timeout=720, ncu_replay=mode
    ) == {"status": "measured", "profiled_kernels": 4}


def test_ncu_timeout_stops_orphaned_writer_before_returning(tmp_path):
    fake_ncu = tmp_path / "fake-ncu"
    marker = tmp_path / "writer-group"
    gate = tmp_path / "write-late"
    os.mkfifo(gate)
    fake_ncu.write_text(
        f"""#!{sys.executable}
import os
import signal
import sys
from pathlib import Path

if os.fork():
    signal.pause()
else:
    report = Path(sys.argv[sys.argv.index("--log-file") + 1])
    report.write_text("partial report")
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    gate = os.open({str(gate)!r}, os.O_RDWR)
    Path({str(marker)!r}).write_text(str(os.getpgrp()))
    os.read(gate, 1)
    os.close(gate)
    with report.open("a") as stream:
        stream.write("late row")
"""
    )
    fake_ncu.chmod(0o700)

    result = profiles.capture("ncu", str(fake_ncu), [], tmp_path, ncu_timeout=1)
    try:
        gate_writer = os.open(gate, os.O_WRONLY | os.O_NONBLOCK)
    except OSError as exc:
        assert exc.errno == errno.ENXIO  # There is no report-writing reader left.
    else:
        try:
            os.write(
                gate_writer, b"x"
            )  # Release a surviving fake writer before failing.
        finally:
            os.close(gate_writer)
        pytest.fail("ncu returned while its report writer was still alive")

    assert int(marker.read_text()) == os.getpgrp()
    assert result["status"] == "unavailable"
    assert "timed out after 1 seconds" in result["reason"]
    assert (tmp_path / "ncu.csv").read_text() == "partial report"


def test_ncu_followup_fails_when_reused_nsys_kernel_count_differs(
    tmp_path, monkeypatch
):
    source = tmp_path / "measured"
    source.mkdir()
    output = tmp_path / "followup"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_bytes(b"checkpoint")
    (source / PLAN).write_bytes(b"plan")
    (source / "features.npy").write_bytes(b"features")
    checkpoint_sha256 = sha256_file(checkpoint)
    monkeypatch.setattr(benchmark, "CHECKPOINT_SHA256", checkpoint_sha256)
    tensors = {"eval": {"sha256": "fixed"}}
    address = benchmark.PROFILE_ADDRESS
    monkeypatch.setattr(
        benchmark, "verify_inputs", lambda _: ({"tensors": tensors}, [address])
    )
    benchmark.write_json(
        source / "result.json",
        {
            "identity": {
                "model": benchmark.MODEL,
                "condition": "2",
                "checkpoint_sha256": checkpoint_sha256,
                "tensors": tensors,
                "profile_address": list(address),
            },
            "features_sha256": sha256_file(source / "features.npy"),
            "engine_sha256": sha256_file(source / PLAN),
            "conversion": {},
            "profiles": {"nsys": {"status": "measured", "gpu_kernels": 114}},
        },
    )

    def capture_one_kernel(tool, executable, command, directory, *, ncu_timeout):
        assert tool == "ncu"
        assert ncu_timeout == 1800
        assert command[command.index("--plan") + 1] == str(source / PLAN)
        return {"status": "measured", "profiled_kernels": 1}

    monkeypatch.setattr(benchmark, "capture", capture_one_kernel)
    monkeypatch.setenv("TENNIS_RUN_ID", "cpu-regression")
    monkeypatch.setenv("TENNIS_REPRO_DIR", str(tmp_path))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.py",
            "--model",
            benchmark.MODEL,
            "--condition",
            "2",
            "--checkpoint",
            str(checkpoint),
            "--inputs",
            str(tmp_path),
            "--output",
            str(output),
            "--profiles",
            "ncu",
            "--profile-from",
            str(source),
            "--ncu-timeout",
            "1800",
        ],
    )
    assert benchmark.main() == 3
    assert json.loads((output / "status.json").read_text()) == {
        "status": "profile_incomplete"
    }
    result = json.loads((output / "result.json").read_text())
    assert result["nsys_comparison"]["gpu_kernels"] == 114
    assert result["profiles"]["ncu"]["status"] == "coverage_mismatch"
    assert result["profiles"]["ncu"]["profiled_kernels"] == 1
    assert result["profiles"]["ncu"]["timeout_seconds"] == 1800


def test_dino_matching_counts_same_score_box_replacements_as_missing_and_extra():
    reference = [[0, 0, 101, 10]]
    replacement = [[99, 0, 200, 10]]  # IoU 0.01
    reports = [
        compare_dino_frame(
            reference, [0.9], replacement if i in (17, 93) else reference, [0.9]
        )
        for i in range(100)
    ]
    assert sum(report["missing"] for report in reports) == 2
    assert sum(report["extra"] for report in reports) == 2
    np.testing.assert_array_equal(
        np.concatenate([report["iou"] for report in reports]), np.ones(98)
    )


def test_dino_matching_maximizes_eligible_pairs_and_accepts_iou_half():
    report = compare_dino_frame(
        [[0, 0, 12, 10], [2, 0, 14, 10]],
        [0.8, 0.9],
        [[0, 0, 12, 10], [-4, 0, 8, 10]],
        [0.6, 0.85],
    )
    assert report["missing"] == report["extra"] == 0
    np.testing.assert_allclose(report["iou"], [0.5, 10 / 14])
    np.testing.assert_array_equal(
        report["xyxy_abs_error_px"], [[4, 0, 4, 0], [2, 0, 2, 0]]
    )
    np.testing.assert_allclose(report["score_abs_error"], [0.05, 0.3])


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
    report = benchmark.accuracy(candidate, baseline, addresses)
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
